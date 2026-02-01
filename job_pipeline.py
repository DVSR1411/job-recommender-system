import sys
import os
import schedule
import time
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), 'scrapers'))
from scrapers.jobsnet_scraper import scrape_jobsnet
from scrapers.freshersnow_scraper import scrape_freshersnow
from scrapers.freshersrecruitment_scraper import scrape_freshersrecruitment
from job_processor import JobProcessor
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import config

def run_all_scrapers():
    """Run all scrapers in parallel"""
    print("=" * 60)
    print("STARTING JOB SCRAPING PIPELINE (PARALLEL)")
    print("=" * 60)
    
    scrapers = [
        ("JobsNet", scrape_jobsnet),
        ("FreshersNow", scrape_freshersnow),
        ("FreshersRecruitment", scrape_freshersrecruitment)
    ]
    
    total_jobs = 0
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scraper_func): name for name, scraper_func in scrapers}
        
        for future in futures:
            name = futures[future]
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                df = future.result()
                if df is not None:
                    jobs_count = len(df)
                    total_jobs += jobs_count
                    print(f"OK {name}: {jobs_count} jobs scraped")
                else:
                    print(f"FAIL {name}: Failed to scrape")
            except Exception as e:
                print(f"ERROR {name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE: {total_jobs} total jobs")
    print(f"{'='*60}")
    
    return total_jobs

def process_embeddings_and_roles():
    print("\n" + "=" * 60)
    print("PROCESSING EMBEDDINGS AND ROLES (PARALLEL)")
    print("=" * 60)
    try:
        processor = JobProcessor()
        processor.process_jobs_parallel(max_workers=8)  # Process 8 jobs at once
        print("OK Embeddings and roles processed successfully")
    except Exception as e:
        print(f"ERROR processing embeddings and roles: {e}")
def main():
    """Main pipeline execution"""
    # Step 1: Run all scrapers
    total_jobs = run_all_scrapers()
    if total_jobs > 0:
        # Step 2: Process embeddings and roles
        process_embeddings_and_roles()
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("Data is ready for RAG system")
        print("Run 'python job_rag.py' to test search")
        print("Run 'python web_interface.py' for web interface")
        print("=" * 60)
    else:
        print("\nNo jobs scraped. Pipeline stopped.")

class JobPipelineScheduler:
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start_scheduler(self):
        """Start the job pipeline scheduler"""
        if self.running:
            return
        
        self.running = True
        schedule.every().day.at("09:00").do(self._run_pipeline)
        
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.thread.start()
        print(f"[{datetime.now()}] Job pipeline scheduler started - will run daily at 09:00")
    
    def stop_scheduler(self):
        """Stop the job pipeline scheduler"""
        self.running = False
        schedule.clear()
        print(f"[{datetime.now()}] Job pipeline scheduler stopped")
    
    def _scheduler_loop(self):
        """Run the scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def _run_pipeline(self):
        """Run the job pipeline"""
        try:
            print(f"[{datetime.now()}] Starting scheduled job pipeline...")
            main()
            print(f"[{datetime.now()}] Scheduled job pipeline completed")
        except Exception as e:
            print(f"[{datetime.now()}] Error in scheduled pipeline: {e}")

# Global scheduler instance
_pipeline_scheduler = None

def start_job_pipeline_scheduler():
    """Start the global job pipeline scheduler"""
    global _pipeline_scheduler
    if _pipeline_scheduler is None:
        _pipeline_scheduler = JobPipelineScheduler()
        _pipeline_scheduler.start_scheduler()
    return _pipeline_scheduler

def stop_job_pipeline_scheduler():
    """Stop the global job pipeline scheduler"""
    global _pipeline_scheduler
    if _pipeline_scheduler:
        _pipeline_scheduler.stop_scheduler()
        _pipeline_scheduler = None
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Job Pipeline')
    parser.add_argument('--schedule', action='store_true', help='Run with scheduler')
    args = parser.parse_args()
    
    if args.schedule:
        print("Starting job pipeline with scheduler...")
        scheduler = start_job_pipeline_scheduler()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down scheduler...")
            stop_job_pipeline_scheduler()
    else:
        main()