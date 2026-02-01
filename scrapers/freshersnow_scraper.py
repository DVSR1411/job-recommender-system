import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
from sqlalchemy import create_engine

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 '
                  'Edg/87.0.664.75'
}

BLOCKED_KEYWORDS = {'telegram', 'freshersnow', 'whatsapp'}
DAYS_BACK = 40
DB_URL = "postgresql+psycopg2://postgres:dbda123@localhost:35432/postgres"

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

def scrape_freshersnow():
    print(f"Starting FreshersNow scraper... (Last {DAYS_BACK} days)")
    
    soup = None
    for attempt in range(3):
        try:
            response = requests.get(
                "https://www.freshersnow.com/freshers-jobs/",
                headers=HEADERS,
                timeout=10
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            break
        except Exception as e:
            if attempt == 2:
                print(f"Failed to fetch main page: {e}")
            time.sleep(2 ** attempt)
    
    if not soup:
        print("Exiting: main page not available")
        return None
    
    job_details = []
    
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", class_="hidden-xs")
        if len(tds) < 6:
            continue
        
        company = tds[0].get_text(strip=True)
        role = tds[1].get_text(strip=True)
        experience = tds[3].get_text(strip=True)
        location = tds[4].get_text(strip=True)
        
        if tds[5].find("a"):
            apply_url = tds[5].find("a")["href"]
        else:
            apply_url = tds[5].get_text(strip=True)
        
        job_details.append({
            "title": f"{company} | {role}",
            "experience": experience,
            "location": location,
            "listing_url": apply_url
        })
    
    print(f"Found {len(job_details)} job listings on main page. Enriching details...")
    
    today = datetime.today().date()
    cutoff_date = today - timedelta(days=DAYS_BACK)
    
    enriched_jobs = []
    
    for i, job in enumerate(job_details, 1):
        print(f"Processing job {i}/{len(job_details)}: {job['title'][:50]}...")
        
        job_soup = None
        for attempt in range(3):
            try:
                response = requests.get(
                    job["listing_url"],
                    headers=HEADERS,
                    timeout=10
                )
                response.raise_for_status()
                job_soup = BeautifulSoup(response.content, "html.parser")
                break
            except Exception:
                time.sleep(2 ** attempt)
        
        if not job_soup:
            continue
        
        job_info = job.copy()
        
        title_tag = job_soup.find("h1", class_="entry-title")
        if title_tag:
            job_info["name"] = title_tag.get_text(strip=True)
        
        apply_urls = set()
        for a in job_soup.find_all("a", href=True):
            anchor_text = a.get_text(strip=True).lower()
            if any(x in anchor_text for x in ("apply here", "click here", "apply now")):
                url = a["href"]
                if not any(b in url.lower() for b in BLOCKED_KEYWORDS):
                    apply_urls.add(url)
        
        if apply_urls:
            job_info["apply_urls"] = list(apply_urls)
        
        descriptions = []
        
        for heading in job_soup.find_all(["h2", "h3"]):
            ul = heading.find_next_sibling("ul")
            if ul:
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
            job_info["description"] = "No detailed description available. Please visit the apply link for more information."
        
        time_tag = job_soup.find("time")
        if time_tag:
            job_date = pd.to_datetime(time_tag.get_text(strip=True), errors="coerce")
            if not pd.isna(job_date):
                job_date = job_date.date()
                job_info["posted_date"] = job_date
                if job_date < cutoff_date:
                    print(f"Reached cutoff date: {job_date}")
                    break
        
        enriched_jobs.append(job_info)
    
    rows = []
    
    for job in enriched_jobs:
        urls = job.get("apply_urls", [])
        if isinstance(urls, list) and urls:
            for url in urls:
                if isinstance(url, str) and not any(b in url.lower() for b in BLOCKED_KEYWORDS):
                    new_job = job.copy()
                    new_job["apply_url"] = url
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
    df["source"] = "freshersnow"
    
    if "name" in df.columns:
        df["title"] = df["name"]
        df.drop(columns=["name"], inplace=True)
    
    print(f"Scraping completed. Total jobs: {len(df)}")
    
    # Save to database
    engine = create_engine(DB_URL)
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
    df = scrape_freshersnow()
    if df is not None:
        print("\nFirst 5 jobs:")
        if "posted_date" in df.columns:
            print(df[["title", "posted_date", "location", "experience"]].head())
        else:
            print(df[["title", "location", "experience"]].head())