import requests
import json
from sqlalchemy import create_engine, text
import pandas as pd
import re
import config

class JobRAG:
    def __init__(self, db_url=None, ollama_url=None):
        self.engine = create_engine(db_url or config.DB_URL)
        self.ollama_url = ollama_url or config.OLLAMA_URL
        self.skill_patterns = {
            "spark": r"\b(py)?spark\b",
            "power bi": r"\bpower\s*bi\b",
            "machine learning": r"\b(machine\s*learning|ml)\b",
            "deep learning": r"\b(deep\s*learning|dl)\b",
            "javascript": r"\b(java\s*script|javascript|js)\b",
            "typescript": r"\b(type\s*script|typescript|ts)\b",
            "postgresql": r"\b(postgres|postgresql)\b",
            "mysql": r"\b(my\s*sql|mysql)\b",
            "c++": r"\b(c\+\+)\b",
            "c#": r"\b(c#|c sharp)\b",
        }
        self.tech_skills = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby',
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'fastapi', 'spring', 'asp.net',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'pytorch', 'tensorflow',
            'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv',
            'git', 'github', 'gitlab', 'jira', 'confluence', 'agile', 'scrum',
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind',
            'linux', 'unix', 'bash', 'shell', 'powershell',
            'data analysis', 'data engineering', 'data visualization', 'big data', 'hadoop', 'spark'
        }

    def get_embedding(self, text):
        try:
            response = requests.post(f"{self.ollama_url}/api/embeddings", 
                                   json={"model": "nomic-embed-text:v1.5", "prompt": text},
                                   timeout=30)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def extract_skills(self, text):
        if not text:
            return set()
        text_lower = text.lower()
        found_skills = set()
        for skill, pattern in self.skill_patterns.items():
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        for skill in self.tech_skills:
            if skill in self.skill_patterns:
                continue
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        return found_skills

    def calculate_skill_match(self, job_desc, user_skills):
        if not user_skills:
            return 0.0, set()
        job_skills = self.extract_skills(job_desc)
        if not job_skills:
            return 0.0, set()
        common_skills = user_skills.intersection(job_skills)
        if not common_skills:
            return 0.0, set()
        score = len(common_skills) / len(user_skills)
        return min(score, 1.0), common_skills

    def search_jobs(self, query, filters=None, limit=20):
        filters = filters or {}
        query_embedding = self.get_embedding(query)
        
        where_clauses = []
        params = {"limit": limit * 3}  # Get more to ensure we have enough after scoring
        
        # Always try to get all jobs first, then score them
        if query_embedding:
            where_clauses.append("embedding IS NOT NULL")
            params["query_embedding"] = str(query_embedding)
            vector_select = "1 - (embedding <=> :query_embedding) as vector_score"
        else:
            vector_select = "0.1 as vector_score"
        
        if filters.get('location'):
            loc = filters['location'].lower()
            where_clauses.append("LOWER(location) LIKE :location")
            params['location'] = f"%{loc}%"

        if filters.get('experience') and filters['experience'] != 'Any':
            exp = filters['experience'].lower()
            if 'fresher' in exp:
                where_clauses.append("(LOWER(experience) LIKE '%fresher%' OR LOWER(experience) LIKE '%0%')")
            else:
                where_clauses.append("LOWER(experience) LIKE :experience")
                params['experience'] = f"%{exp}%"
            
        where_str = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        with self.engine.connect() as conn:
            stmt = text(f"""
                SELECT id, title, role, location, experience, description, 
                       listing_url, apply_url, posted_date,
                       {vector_select}
                FROM jobs 
                WHERE {where_str}
                ORDER BY vector_score DESC
                LIMIT :limit
            """)
            
            result = conn.execute(stmt, params)
            jobs_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        if jobs_df.empty:
            return pd.DataFrame()

        # Priority scoring: Role > Title > Vector similarity
        role_score = 0.0
        title_score = 0.0
        
        if filters.get('role_type'):
            role_term = filters['role_type'].lower()
            jobs_df['title_match'] = jobs_df['title'].fillna('').str.lower().str.contains(role_term, na=False).astype(float) * 0.8
            jobs_df['role_match'] = jobs_df['role'].fillna('').str.lower().str.contains(role_term, na=False).astype(float) * 0.5
            jobs_df['final_score'] = jobs_df['title_match'] + jobs_df['role_match'] + (jobs_df['vector_score'] * 0.3)
        else:
            jobs_df['final_score'] = jobs_df['vector_score']
            jobs_df['role_match'] = 0.0
            jobs_df['title_match'] = 0.0

        # Add skill matching
        user_skills = filters.get('resume_skills', set())
        if not user_skills:
            user_skills = self.extract_skills(query)

        if user_skills:
            skill_scores = []
            matched_skills_list = []
            
            for desc in jobs_df['description']:
                score, matched = self.calculate_skill_match(desc, user_skills)
                skill_scores.append(score)
                matched_skills_list.append(list(matched))
            
            jobs_df['skill_score'] = skill_scores
            jobs_df['matched_skills'] = matched_skills_list
            jobs_df['final_score'] += jobs_df['skill_score'] * 0.3
        else:
            jobs_df['skill_score'] = 0.0
            jobs_df['matched_skills'] = [[] for _ in range(len(jobs_df))]

        jobs_df['final_score'] = jobs_df['final_score'].clip(upper=1.0)
        jobs_df = jobs_df.sort_values('final_score', ascending=False).head(limit)
        
        return jobs_df

    def generate_response(self, query, jobs_df, user_skills=None):
        if not user_skills:
            user_skills = set()
            
        if jobs_df.empty:
            return "No jobs found matching your criteria."
        
        user_skills_str = ", ".join(sorted(user_skills)) if user_skills else "None provided"
        top_jobs = jobs_df.head(4)
        
        # Find skills mentioned in job descriptions that user doesn't have
        all_job_skills = set()
        for _, row in top_jobs.iterrows():
            job_skills = self.extract_skills(row['description'])
            all_job_skills.update(job_skills)
        
        missing_skills = all_job_skills - user_skills
        missing_skills_str = ", ".join(sorted(list(missing_skills)[:5])) if missing_skills else "None identified"
        
        context = "\n".join([
            f"- {row['title']} at {row['location']} (Score: {row['final_score']:.2f}). Matched skills: {', '.join(row['matched_skills'])}"
            for _, row in top_jobs.iterrows()
        ])

        prompt = f"""User Current Skills: {user_skills_str}

Top 4 Job Matches:
{context}

Skills found in jobs but NOT in user's current skills: {missing_skills_str}

IMPORTANT: Only suggest skills from the "NOT in user's current skills" list above. Do NOT suggest skills the user already has.

Provide a brief analysis:
1. Match Summary: Why these top 4 jobs fit the user's profile
2. Missing Skills: From the missing skills list only, suggest 2-3 most important ones for long-term career growth
3. Alternative Roles: 2 related job titles

Keep response under 150 words."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 150,
                        "temperature": 0.3,
                        "num_threads": 4
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "Analysis unavailable.")
        except Exception as e:
            print(f"LLM Error: {e}")
            total_jobs = len(jobs_df)
            top_job = jobs_df.iloc[0]
            top_matched = top_job.get('matched_skills', [])
            
            response = f"Found {total_jobs} relevant positions. "
            response += f"Best match: '{top_job['title']}' with {top_job['final_score']*100:.0f}% compatibility. "
            if top_matched:
                response += f"Strong skills match: {', '.join(top_matched[:3])}. "
            response += "Consider developing related technologies for better matches."
            return response

    def chat(self, query, filters=None):
        """Main chat interface"""
        try:
            filters = filters or {}
            jobs = self.search_jobs(query, filters)
            
            if jobs.empty:
                return {"response": "No relevant jobs found matching your criteria.", "jobs": []}
            
            user_skills = filters.get('resume_skills', set())
            if not user_skills:
                user_skills = self.extract_skills(query)
            
            response = self.generate_response(query, jobs, user_skills)
            return {
                "response": response,
                "jobs": jobs.to_dict('records')
            }
        except Exception as e:
            return {"response": f"Error: {str(e)}", "jobs": []}

if __name__ == "__main__":
    rag = JobRAG()
    
    print("Testing Job Search...")
    filters = {
        'location': 'Bangalore',
        'role_type': 'Data Scientist',
        'resume_skills': {'python', 'machine learning', 'sql'}
    }
    result = rag.chat("Looking for data scientist jobs", filters)
    print("Response:", result["response"])
    print(f"\nFound {len(result['jobs'])} jobs")