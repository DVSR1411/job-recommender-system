import requests
import pandas as pd
from sqlalchemy import create_engine, text
import re
from concurrent.futures import ThreadPoolExecutor
import config
ROLE_PATTERNS = [
    (re.compile(r"\b(ai|artificial\s+intelligence)[/\s]+(ml|machine\s+learning)\s+(engineer|scientist|specialist|developer)\b", re.I), "AI/ML Engineer"),
    (re.compile(r"\b(machine\s+learning|ml)\s+(engineer|scientist|specialist|developer)\b", re.I), "ML Engineer"),
    (re.compile(r"\b(ai|artificial\s+intelligence)\s+(engineer|scientist|specialist|developer)\b", re.I), "AI Engineer"),
    (re.compile(r"\bdata\s+(scientist|science)\b", re.I), "Data Scientist"),
    (re.compile(r"\bdata\s+(analyst|analytics)\b", re.I), "Data Analyst"),
    (re.compile(r"\bdata\s+(engineer|engineering)\b", re.I), "Data Engineer"),
    (re.compile(r"\b(backend|back[\s-]?end)\s+(developer|engineer)\b", re.I), "Backend Developer"),
    (re.compile(r"\b(frontend|front[\s-]?end)\s+(developer|engineer)\b", re.I), "Frontend Developer"),
    (re.compile(r"\bfull[\s-]?stack\s+(developer|engineer)\b", re.I), "Full Stack Developer"),
    (re.compile(r"\bmobile\s+(developer|engineer|app)\b", re.I), "Mobile Developer"),
    (re.compile(r"\bandroid\s+(developer|engineer)\b", re.I), "Android Developer"),
    (re.compile(r"\bios\s+(developer|engineer)\b", re.I), "iOS Developer"),
    (re.compile(r"\bweb\s+(developer|engineer)\b", re.I), "Web Developer"),
    (re.compile(r"\bdevops\s+(engineer|specialist)\b", re.I), "DevOps Engineer"),
    (re.compile(r"\bcloud\s+(engineer|architect|specialist)\b", re.I), "Cloud Engineer"),
    (re.compile(r"\bsystem\s+(administrator|admin|engineer)\b", re.I), "System Administrator"),
    (re.compile(r"\bnetwork\s+(engineer|administrator)\b", re.I), "Network Engineer"),
    (re.compile(r"\b(qa|quality\s+assurance)\s+(engineer|analyst|tester)\b", re.I), "QA Engineer"),
    (re.compile(r"\btest\s+(engineer|analyst|automation)\b", re.I), "Test Engineer"),
    (re.compile(r"\bautomation\s+(engineer|tester)\b", re.I), "Automation Engineer"),
    (re.compile(r"\b(security|cyber\s*security)\s+(engineer|analyst|specialist)\b", re.I), "Security Engineer"),
    (re.compile(r"\b(tech|technical)\s+(lead|leader)\b", re.I), "Tech Lead"),
    (re.compile(r"\b(engineering|development)\s+manager\b", re.I), "Engineering Manager"),
    (re.compile(r"\bproject\s+manager\b", re.I), "Project Manager"),
    (re.compile(r"\bproduct\s+manager\b", re.I), "Product Manager"),
    (re.compile(r"\bbusiness\s+(analyst|intelligence)\b", re.I), "Business Analyst"),
    (re.compile(r"\bsales\s+(engineer|executive|representative)\b", re.I), "Sales"),
    (re.compile(r"\bcustomer\s+(support|success|service)\b", re.I), "Customer Support"),
    (re.compile(r"\bmarketing\s+(specialist|manager|executive)\b", re.I), "Marketing"),
    (re.compile(r"\bsoftware\s+(developer|programmer)\b", re.I), "Software Developer"),
    (re.compile(r"\bsoftware\s+engineer\b", re.I), "Software Engineer"),
    (re.compile(r"\bdata\s+analyst\b", re.I), "Data Analyst"),
    (re.compile(r"\bsystem\s+analyst\b", re.I), "System Analyst"),
    (re.compile(r"\bbusiness\s+analyst\b", re.I), "Business Analyst"),
    (re.compile(r"\bfinancial\s+analyst\b", re.I), "Financial Analyst"),
    (re.compile(r"\b(developer|programmer)\b", re.I), "Software Developer"),
    (re.compile(r"\bengineer\b", re.I), "Software Engineer"),
    (re.compile(r"\banalyst\b", re.I), "Data Analyst"),
    (re.compile(r"\bspecialist\b", re.I), "Specialist")
]

class JobProcessor:
    def __init__(self, db_url=None, ollama_url=None):
        self.engine = create_engine(
            db_url or config.DB_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        self.ollama_url = ollama_url or config.OLLAMA_URL

    def get_embedding(self, text):
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": "nomic-embed-text:v1.5",
                "prompt": text
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embedding"]
    def extract_role(self, title, description):
        text = f"{title} {description or ''}"

        for pattern, role in ROLE_PATTERNS:
            if pattern.search(text):
                return role

        return "Software Engineer"

    def process_single_job(self, job):
        try:
            combined_text = f"{job['title']} {job['description'] or ''}"

            embedding = self.get_embedding(combined_text)
            role = self.extract_role(job["title"], job["description"])

            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE jobs
                        SET embedding = :embedding,
                            role = :role,
                            updated_at = NOW()
                        WHERE id = :job_id
                    """),
                    {
                        "embedding": embedding,
                        "role": role,
                        "job_id": job["id"]
                    }
                )

            print(f"OK {job['title'][:35]} -> {role}")

        except Exception as e:
            print(f"ERROR ({job['title'][:30]}): {e}")

    def process_jobs_parallel(self, max_workers=8, limit=200):
        query = f"""
            SELECT id, title, description
            FROM jobs
            WHERE embedding IS NULL OR role IS NULL
            LIMIT {limit}
        """

        jobs = pd.read_sql(query, self.engine)
        if jobs.empty:
            print("No jobs to process")
            return

        print(f"Processing {len(jobs)} jobs with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(
                self.process_single_job,
                jobs.to_dict("records")
            )
if __name__ == "__main__":
    processor = JobProcessor()
    processor.process_jobs_parallel(max_workers=8, limit=200)