import requests
import json
from sqlalchemy import create_engine, text
import pandas as pd
import re
class JobRAG:
    def __init__(self, db_url, ollama_url="http://localhost:11434"):
        self.engine = create_engine(db_url)
        self.ollama_url = ollama_url
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
            raise Exception(f"Embedding failed: {str(e)}")

    def extract_skills(self, text):
        if not text:
            return set()
        text_lower = text.lower()
        found_skills = set()
        for skill in self.tech_skills:
            pattern = re.compile(r'\b' + re.escape(skill) + r'\b')
            if pattern.search(text_lower):
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
        
        
        where_clauses = ["embedding IS NOT NULL"]
        params = {
            "query_embedding": str(query_embedding),
            "limit": limit * 2  
        }
        
        
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

        
        if filters.get('role_type'):
            
            
            where_clauses.append("(LOWER(role) LIKE :role_type OR LOWER(title) LIKE :role_type)")
            params['role_type'] = f"%{filters['role_type'].lower()}%"
            
        where_str = " AND ".join(where_clauses)
        
        
        with self.engine.connect() as conn:
            stmt = text(f"""
                SELECT id, title, role, location, experience, description, 
                       listing_url, apply_url, posted_date,
                       1 - (embedding <=> :query_embedding) as vector_score
                FROM jobs 
                WHERE {where_str}
                ORDER BY vector_score DESC
                LIMIT :limit
            """)
            
            result = conn.execute(stmt, params)
            jobs_df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        if jobs_df.empty:
            return pd.DataFrame()

        
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
            jobs_df['final_score'] = (jobs_df['vector_score'] * 0.7) + (jobs_df['skill_score'] * 0.3)
            if filters.get('role_type'):
                 jobs_df['final_score'] += 0.15
            matches_skills = jobs_df['skill_score'] > 0.5
            jobs_df.loc[matches_skills, 'final_score'] += 0.1
            has_any_skill = jobs_df['skill_score'] > 0
            jobs_df.loc[has_any_skill, 'final_score'] += 0.05

        else:
            
            jobs_df['final_score'] = jobs_df['vector_score'] * 1.2
            jobs_df['matched_skills'] = [[] for _ in range(len(jobs_df))]

        
        jobs_df['final_score'] = jobs_df['final_score'].clip(upper=1.0)

        
        jobs_df = jobs_df.sort_values('final_score', ascending=False).head(limit)
        
        return jobs_df
    
    def generate_response(self, query, jobs_df, user_skills=None):
        if not user_skills:
            user_skills = set()
            
        user_skills_str = ", ".join(user_skills) if user_skills else "None provided"
        context = "\n".join([
            f"- {row['title']} ({row['location']}). Matched: {', '.join(row['matched_skills'])}"
            for _, row in jobs_df.iterrows()
        ])

        prompt = f"""User Skills: {user_skills_str}
Job Matches:
{context}

Task:
1. Match Analysis: Briefly explain why these fit.
2. Missing Skills: List 2-3 skills NOT in the User Skills list.
3. Alternative Roles: Suggest 2 related titles.
Keep it very concise."""

        try:
            # Optimized parameters for qwen3:0.6b to prevent timeouts
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen3:0.6b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 250,
                        "num_ctx": 2048,
                        "temperature": 0.3,
                        "num_thread": 4
                    }
                },
                timeout=120 # Sufficient for 3b model on most hardware
            )
            response.raise_for_status()
            return response.json().get("response", "Analysis unavailable.")
        except Exception as e:
            print(f"LLM Error: {e}")
            return "The analysis timed out or failed. Please try a more specific search."

    def chat(self, query, filters=None):
        """Unified chat interface"""
        try:
            filters = filters or {}
            jobs = self.search_jobs(query, filters)
            
            if jobs.empty:
                return {"response": "No relevant jobs found matching your criteria.", "jobs": []}
            
            # Get user skills for analysis
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
    DB_URL = "postgresql+psycopg2://postgres:dbda123@localhost:35432/postgres"
    rag = JobRAG(DB_URL)
    
    print("Testing Hybrid Search...")
    filters = {
        'location': 'Bangalore',
        'resume_skills': {'python', 'django', 'sql'}
    }
    result = rag.chat("Looking for python developer jobs", filters)
    print("Response:", result["response"])
    print(f"\nFound {len(result['jobs'])} jobs")