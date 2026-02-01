import streamlit as st
import pandas as pd
from job_rag import JobRAG
import time
import PyPDF2
import docx
import requests

# Page configuration
st.set_page_config(
    page_title="Job Recommender System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minor tweaks (still useful for general styling)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

# Initialize State
if 'rag' not in st.session_state:
    DB_URL = "postgresql+psycopg2://postgres:dbda123@localhost:35432/postgres"
    st.session_state.rag = JobRAG(DB_URL)

if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Resume Upload in Sidebar
    st.subheader("📄 Resume (Optional)")
    uploaded_file = st.file_uploader("Upload for Auto-Fill", type=['pdf', 'docx', 'txt'])
    
    # Extract skills if resume uploaded
    resume_skills = set()
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif "word" in uploaded_file.type:
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            resume_text = str(uploaded_file.read(), "utf-8")
        
        with st.spinner("Analyzing Resume..."):
            resume_skills = st.session_state.rag.extract_skills(resume_text)
        st.success(f"Resume Loaded: {len(resume_skills)} skills found")
        
        with st.expander("Detected Skills"):
            st.write(", ".join(resume_skills))
    
    st.divider()
    
    # DB Status
    try:
        with st.session_state.rag.engine.connect() as conn:
            st.success("Database: Connected")
        requests.get("http://localhost:11434", timeout=1)
        st.success("Ollama: Online")
    except Exception:
        st.error("System Status: Connection Issues")

# Main Content
st.markdown('<h1 class="main-header">🎯 Job Recommender System</h1>', unsafe_allow_html=True)

# Unified Search Form
with st.container():
    st.markdown("### 🔍 Search Criteria")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        role_val = st.text_input("Job Role", placeholder="Data Scientist, Java Dev...")
    with col2:
        loc_val = st.text_input("Location", placeholder="Bangalore, Mumbai...")
    with col3:
        exp_val = st.selectbox("Experience", ["Any", "Fresher (0-1y)", "Junior (1-3y)", "Mid (3-5y)", "Senior (5y+)"])

    # Skills Input (Pre-filled if resume exists)
    default_skills = ", ".join(resume_skills) if resume_skills else ""
    skills_input = st.text_area("Skills (Comma Separated)", value=default_skills, placeholder="python, sql, aws...")

    if st.button("🚀 Find Matching Jobs", type="primary"):
        # Process inputs
        final_skills = {s.strip() for s in skills_input.split(',')} if skills_input else set()
        
        filters = {
            'location': loc_val,
            'experience': exp_val,
            'role_type': role_val,
            'resume_skills': final_skills
        }
        
        # Build Query
        parts = []
        if role_val: parts.append(f"{role_val} jobs")
        if loc_val: parts.append(f"in {loc_val}")
        if final_skills: parts.append(f"using {', '.join(list(final_skills)[:5])}")
        
        query_text = " ".join(parts) if parts else "Software Engineering jobs"
        
        with st.spinner(f"Searching for: {query_text}..."):
            result = st.session_state.rag.chat(query_text, filters)
            st.session_state.last_results = result
            
# Results Display
if 'last_results' in st.session_state:
    res = st.session_state.last_results
    jobs = res.get('jobs', [])
    response = res.get('response', '')
    
    st.divider()
    st.markdown(f"### 📊 Results Found: {len(jobs)}")
    
    # AI Analysis Display
    if response:
        with st.container():
            st.info(f"**🤖 AI Analysis:**\n\n{response}")
    
    if jobs:
        for i, job in enumerate(jobs):
            score = job.get('final_score', 0) * 100
            
            # Determine color/label based on score
            if score >= 80:
                score_label = "Excellent Match"
                delta_color = "normal" 
            elif score >= 60:
                score_label = "Good Match"
                delta_color = "off"
            else:
                score_label = "Potential Match"
                delta_color = "inverse"
            
            with st.container():
                # Use standard Streamlit columns for clean layout
                # Layout: Title/Meta (Left) | Score (Right)
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    st.subheader(f"{i+1}. {job['title']}")
                    st.caption(f"🏢 {job.get('role', 'N/A')} •📍 {job['location']} • 💼 {job['experience']}")
                    
                with c2:
                    st.metric("Match Score", f"{score:.0f}%", score_label, delta_color=delta_color)
                
                # Match Reasoning
                if job.get('matched_skills'):
                    st.markdown("**✅ Matched Skills:**")
                    st.caption(", ".join(job['matched_skills']))
                
                # Description Expander
                with st.expander("Show Job Description"):
                    st.markdown(job['description'])
                
                # Action Buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if job.get('apply_url'):
                        st.link_button("👉 Apply Now", job['apply_url'], use_container_width=True)
                with col2:
                    st.button("💾 Save", key=f"save_{job['id']}", use_container_width=True)
                
                st.divider()
    else:
        st.warning("No jobs found matching your criteria. Try relaxing your filters.")