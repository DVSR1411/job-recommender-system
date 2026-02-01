import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time
import threading
import config

class UserManager:
    def __init__(self, db_url=None):
        self.engine = create_engine(db_url or config.DB_URL)
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        self.email_user = config.EMAIL_USER
        self.email_password = config.EMAIL_PASSWORD
        self.notification_running = False
        self.notification_thread = None

    def start_notification_scheduler(self):
        """Start the notification scheduler"""
        if self.notification_running:
            return
        
        self.notification_running = True
        schedule.every().day.at("13:00").do(self.check_new_jobs_and_notify)
        
        self.notification_thread = threading.Thread(target=self._notification_loop, daemon=True)
        self.notification_thread.start()
        print(f"[{datetime.now()}] Notification scheduler started - will run daily at 13:00")
    
    def _notification_loop(self):
        """Run the notification scheduler loop"""
        while self.notification_running:
            schedule.run_pending()
            time.sleep(60)

    def create_user(self, google_id, username, email):
        """Create a new user from Google OAuth data"""
        with self.engine.connect() as conn:
            stmt = text("""
                INSERT INTO users (google_id, username, email)
                VALUES (:google_id, :username, :email)
                ON CONFLICT (google_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    email = EXCLUDED.email
                RETURNING id
            """)
            result = conn.execute(stmt, {
                'google_id': google_id,
                'username': username,
                'email': email
            })
            conn.commit()
            return result.fetchone()[0]

    def get_user_by_google_id(self, google_id):
        """Get user by Google ID"""
        with self.engine.connect() as conn:
            stmt = text("SELECT * FROM users WHERE google_id = :google_id")
            result = conn.execute(stmt, {'google_id': google_id})
            return result.fetchone()

    def save_user_preferences(self, user_id, preferences):
        """Save user preferences directly in users table"""
        with self.engine.connect() as conn:
            stmt = text("""
                UPDATE users SET
                    location = :location,
                    role_name = :role_name,
                    skills = :skills
                WHERE id = :user_id
            """)
            conn.execute(stmt, {
                'user_id': user_id,
                'location': preferences.get('location'),
                'role_name': preferences.get('role_name'),
                'skills': preferences.get('skills', [])
            })
            conn.commit()

    def get_user_preferences(self, user_id):
        """Get user preferences from users table"""
        with self.engine.connect() as conn:
            stmt = text("SELECT location, role_name, skills FROM users WHERE id = :user_id")
            result = conn.execute(stmt, {'user_id': user_id})
            return result.fetchone()

    def save_user_resume(self, user_id, filename, resume_blob, extracted_skills):
        """Save resume directly in users table"""
        with self.engine.connect() as conn:
            stmt = text("""
                UPDATE users SET
                    resume_blob = :resume_blob,
                    skills = :extracted_skills
                WHERE id = :user_id
            """)
            conn.execute(stmt, {
                'user_id': user_id,
                'resume_blob': resume_blob,
                'extracted_skills': extracted_skills
            })
            conn.commit()

    def save_job(self, user_id, job_id):
        """Save a job for user"""
        with self.engine.connect() as conn:
            stmt = text("""
                INSERT INTO saved_jobs (user_id, job_id)
                VALUES (:user_id, :job_id)
                ON CONFLICT (user_id, job_id) DO NOTHING
            """)
            conn.execute(stmt, {'user_id': user_id, 'job_id': job_id})
            conn.commit()

    def get_saved_jobs(self, user_id):
        """Get user's saved jobs"""
        with self.engine.connect() as conn:
            stmt = text("""
                SELECT j.* FROM jobs j
                JOIN saved_jobs sj ON j.id = sj.job_id
                WHERE sj.user_id = :user_id
                ORDER BY sj.saved_at DESC
            """)
            result = conn.execute(stmt, {'user_id': user_id})
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def send_email_notification(self, to_email, subject, body):
        """Send email notification"""
        if not self.email_user or not self.email_password:
            print("Email credentials not configured")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False

    def check_new_jobs_and_notify(self):
        """Check for new jobs and send top 5 notifications using job_rag logic"""
        from job_rag import JobRAG
        
        rag = JobRAG()
        
        with self.engine.connect() as conn:
            users_stmt = text("""
                SELECT id, email, username, location, role_name, skills
                FROM users 
                WHERE location IS NOT NULL OR role_name IS NOT NULL
            """)
            users_result = conn.execute(users_stmt)
            users = users_result.fetchall()

            for user in users:
                # Build query from user preferences
                query_parts = []
                if user.role_name:
                    query_parts.append(f"{user.role_name} jobs")
                if user.location:
                    query_parts.append(f"in {user.location}")
                if user.skills:
                    query_parts.append(f"using {', '.join(user.skills[:3])}")
                
                query = " ".join(query_parts) if query_parts else "software jobs"
                
                # Build filters
                filters = {
                    'location': user.location,
                    'role_type': user.role_name,
                    'resume_skills': set(user.skills) if user.skills else set()
                }
                
                # Get top 5 jobs using RAG logic
                top_jobs = rag.search_jobs(query, filters, limit=5)
                
                if not top_jobs.empty:
                    # Filter out already notified jobs
                    new_jobs = []
                    for _, job in top_jobs.iterrows():
                        check_stmt = text("""
                            SELECT 1 FROM job_notifications 
                            WHERE user_id = :user_id AND job_id = :job_id
                        """)
                        result = conn.execute(check_stmt, {
                            'user_id': user.id,
                            'job_id': job['id']
                        })
                        if not result.fetchone():
                            new_jobs.append(job)
                    
                    if new_jobs:
                        self._send_job_notification_email(user, new_jobs)
                        
                        # Mark jobs as notified
                        for job in new_jobs:
                            notify_stmt = text("""
                                INSERT INTO job_notifications (user_id, job_id)
                                VALUES (:user_id, :job_id)
                                ON CONFLICT (user_id, job_id) DO NOTHING
                            """)
                            conn.execute(notify_stmt, {
                                'user_id': user.id,
                                'job_id': job['id']
                            })
            
            conn.commit()

    def _send_job_notification_email(self, user, jobs):
        """Send job notification email to user"""
        subject = f"🎯 Top {len(jobs)} Job Matches Found!"
        
        jobs_html = ""
        for job in jobs:
            score = job.get('final_score', 0) * 100 if hasattr(job, 'get') else 0
            matched_skills = job.get('matched_skills', []) if hasattr(job, 'get') else []
            
            jobs_html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3 style="color: #2563eb;">{job['title'] if hasattr(job, '__getitem__') else job.title}</h3>
                <p><strong>Location:</strong> {job['location'] if hasattr(job, '__getitem__') else job.location}</p>
                <p><strong>Experience:</strong> {job['experience'] if hasattr(job, '__getitem__') else job.experience}</p>
                <p><strong>Match Score:</strong> {score:.0f}%</p>
                {f'<p><strong>Matched Skills:</strong> {', '.join(matched_skills)}</p>' if matched_skills else ''}
                <p><strong>Description:</strong> {(job['description'] if hasattr(job, '__getitem__') else job.description)[:200]}...</p>
                {f'<a href="{job['apply_url'] if hasattr(job, '__getitem__') else job.apply_url}" style="background: #2563eb; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px;">Apply Now</a>' if (job.get('apply_url') if hasattr(job, 'get') else getattr(job, 'apply_url', None)) else ''}
            </div>
            """

        body = f"""
        <html>
        <body>
            <h2>Hi {user.username}! 👋</h2>
            <p>We found {len(jobs)} top job opportunities that match your preferences:</p>
            
            <div style="background: #f8fafc; padding: 15px; border-radius: 5px; margin: 15px 0;">
                <strong>Your Preferences:</strong><br>
                Role: {user.role_name or 'Any'}<br>
                Location: {user.location or 'Any'}<br>
                Skills: {', '.join(user.skills) if user.skills else 'None specified'}
            </div>
            
            <h3>Top Job Matches:</h3>
            {jobs_html}
            
            <p style="margin-top: 30px;">
                <a href="http://localhost:8501" style="background: #16a34a; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
                    View All Jobs
                </a>
            </p>
            
            <p style="color: #666; font-size: 12px; margin-top: 20px;">
                You're receiving this because you enabled job notifications. 
                You can update your preferences in the app.
            </p>
        </body>
        </html>
        """
        return self.send_email_notification(user.email, subject, body)