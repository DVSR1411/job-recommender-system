import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
from sqlalchemy import create_engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/87.0.4280.141 Safari/537.36 '
                  'Edg/87.0.664.75'
}

BLOCKED_KEYWORDS = {'telegram', 'freshersrecruitment', 'whatsapp'}
DAYS_BACK = 30

def clean_job_data(df):
    """Clean and normalize job data"""
    for idx, row in df.iterrows():
        location = str(row.get('location', '')).strip()
        experience = str(row.get('experience', '')).strip()
        description = str(row.get('description', '')).strip()
        title = str(row.get('title', '')).strip()
        
        # Truncate title to fit database constraints
        if len(title) > 500:
            df.at[idx, 'title'] = title[:497] + '...'
        
        # Clean location - handle multiple locations, null/empty values, and long text
        if not location or location.lower() in ['nan', 'none', '', 'null'] or location == '':
            df.at[idx, 'location'] = 'Pan India'
        elif 'multiple locations' in location.lower() or len(location) > 200:
            # Move long location or multiple locations to description
            if not description or description.lower() in ['nan', 'none', '', 'null']:
                df.at[idx, 'description'] = f"Location: {location}"
            else:
                df.at[idx, 'description'] = f"{description}\n\nLocation: {location}"
            df.at[idx, 'location'] = 'Pan India'
        
        # Check if experience is too long (likely description)
        valid_exp_keywords = ['fresher', '0', '1', '2', '3', '4', '5', 'year', 'batch', '2024', '2025', 'experience']
        
        # Handle null/empty experience first
        if not experience or experience.lower() in ['nan', 'none', '', 'null'] or experience == '':
            df.at[idx, 'experience'] = 'Freshers'
        elif len(experience) > 100 or (experience and not any(keyword in experience.lower() for keyword in valid_exp_keywords)):
            # Move invalid experience to description
            if not description or description.lower() in ['nan', 'none', '', 'null']:
                df.at[idx, 'description'] = experience
            else:
                df.at[idx, 'description'] = f"{description}\n\n{experience}"
            df.at[idx, 'experience'] = 'Freshers'
        elif len(experience) > 100:
            df.at[idx, 'experience'] = experience[:97] + '...'
        
        # Handle missing descriptions
        final_desc = str(df.at[idx, 'description']).strip()
        if not final_desc or final_desc.lower() in ['nan', 'none', '', 'null']:
            # Check if experience has useful info
            exp_text = str(df.at[idx, 'experience']).strip()
            if len(exp_text) > 30:
                df.at[idx, 'description'] = exp_text
                df.at[idx, 'experience'] = 'Freshers'
            else:
                df.at[idx, 'description'] = 'No detailed description available. Please visit the apply link for more information.'
        
        # Clean description - remove 'nan' at end
        final_desc = str(df.at[idx, 'description']).strip()
        if final_desc.endswith('\n\nnan'):
            df.at[idx, 'description'] = final_desc[:-5].strip()
    
    return df

def scrape_freshersrecruitment():
    print(f"Starting FreshersRecruitment scraper... (Last {DAYS_BACK} days)")
    
    today = datetime.today().date()
    cutoff_date = today - timedelta(days=DAYS_BACK)
    
    page_num = 1
    stop_pagination = False
    job_details = []
    
    while not stop_pagination:
        page_url = f"https://freshersrecruitment.co.in/category/jobs/page/{page_num}/"
        print(f"Fetching: {page_url}")
        
        soup = None
        for attempt in range(3):
            try:
                response = requests.get(page_url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to fetch page {page_num}: {e}")
                time.sleep(2 ** attempt)
        
        if not soup:
            break
        
        articles = soup.find_all("article")
        if not articles:
            break
        
        for article in articles:
            job = {}
            
            title_tag = article.find("h2", class_="entry-title")
            if title_tag:
                a_tag = title_tag.find("a", href=True)
                if a_tag:
                    job["title"] = a_tag.get_text(strip=True)
                    job["listing_url"] = a_tag["href"]
            
            time_tag = article.find("time")
            if time_tag:
                try:
                    posted_date = datetime.strptime(
                        time_tag.get_text(strip=True), "%B %d, %Y"
                    ).date()
                    job["posted_date"] = posted_date
                except ValueError:
                    continue
            
            if job.get("posted_date") and job["posted_date"] < cutoff_date:
                print(f"Reached cutoff date: {job['posted_date']}")
                stop_pagination = True
                break
            
            if job.get("listing_url"):
                job_details.append(job)
        
        page_num += 1
    
    print(f"Found {len(job_details)} job listings. Enriching details...")
    
    enriched_jobs = []
    
    for i, job in enumerate(job_details, 1):
        print(f"Processing job {i}/{len(job_details)}: {job['title'][:50]}...")
        
        job_soup = None
        for attempt in range(3):
            try:
                response = requests.get(job["listing_url"], headers=HEADERS, timeout=10)
                response.raise_for_status()
                job_soup = BeautifulSoup(response.content, "html.parser")
                break
            except Exception:
                time.sleep(2 ** attempt)
        
        if not job_soup:
            continue
        
        job_info = job.copy()
        
        apply_urls = set()
        for a in job_soup.find_all("a", href=True):
            anchor_text = a.get_text(strip=True).lower()
            if any(x in anchor_text for x in ("apply here", "click here", "apply now")):
                apply_url = a["href"]
                if not any(b in apply_url.lower() for b in BLOCKED_KEYWORDS):
                    apply_urls.add(apply_url)
        
        if apply_urls:
            job_info["apply_urls"] = list(apply_urls)
        
        for ul in job_soup.find_all("ul", class_="wp-block-list"):
            for li in ul.find_all("li"):
                strong = li.find("strong")
                if strong:
                    label = strong.get_text(strip=True).lower()
                    value = li.get_text(strip=True).replace(
                        strong.get_text(strip=True), ""
                    ).lstrip(": ").strip()
                    
                    if "location" in label:
                        job_info["location"] = value
                    elif "experience" in label:
                        job_info["experience"] = value
        
        descriptions = []
        
        for ul in job_soup.find_all("ul", class_="wp-block-list"):
            for li in ul.find_all("li"):
                txt = li.get_text(strip=True)
                if txt:
                    descriptions.append(txt)
        
        for p in job_soup.find_all("p"):
            txt = p.get_text(strip=True)
            if txt.startswith(("•", "-", "–", "*")):
                descriptions.append(txt.lstrip("•-–* ").strip())
        
        if descriptions:
            job_info["description"] = " ".join(dict.fromkeys(descriptions))
        else:
            job_info["description"] = (
                "No detailed description available. Please visit the apply link for more information."
            )
        
        enriched_jobs.append(job_info)
    
    rows = []
    
    for job in enriched_jobs:
        urls = job.get("apply_urls", [])
        if isinstance(urls, list) and urls:
            for apply_url in urls:
                if isinstance(apply_url, str) and not any(
                    b in apply_url.lower() for b in BLOCKED_KEYWORDS
                ):
                    new_job = job.copy()
                    new_job["apply_url"] = apply_url
                    new_job.pop("apply_urls", None)
                    rows.append(new_job)
        else:
            new_job = job.copy()
            new_job["apply_url"] = job.get("listing_url")
            new_job.pop("apply_urls", None)
            rows.append(new_job)
    
    df = pd.DataFrame(rows)
    
    df = clean_job_data(df)
    df["apply_url"] = df["apply_url"].fillna(df["listing_url"])
    df["source"] = "freshersrecruitment"
    
    print(f"Scraping completed. Total jobs: {len(df)}")
    
    # Save to database
    engine = create_engine(config.DB_URL)
    df.to_sql(
        name="jobs",
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=500
    )
    
    return df

if __name__ == "__main__":
    df = scrape_freshersrecruitment()
    print("\nFirst 5 jobs:")
    if "posted_date" in df.columns:
        print(df[["title", "posted_date", "location", "experience"]].head())
    else:
        print(df[["title", "location", "experience"]].head())