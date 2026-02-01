# Job Recommender System

AI-powered job recommendation system with Google OAuth, smart matching, and automated notifications.

## 🚀 Features

- **Google Sign-In** - Secure OAuth authentication
- **Smart Job Matching** - AI-powered recommendations using RAG
- **Resume Analysis** - Automatic skill extraction
- **Email Notifications** - Daily top 5 job matches
- **Automated Scraping** - Daily job collection from multiple sources
- **Saved Jobs** - Bookmark interesting opportunities

## 📋 Prerequisites

1. **PostgreSQL** with pgvector extension
2. **Ollama** with embedding model
3. **Google Cloud Console** project
4. **Gmail** account with app password

## 🛠️ Setup Instructions

### 1. Database Setup

```bash
# Install PostgreSQL and pgvector
# Then create database and run schemas:

psql -U postgres -d postgres -f schema.sql
psql -U postgres -d postgres -f user_schema.sql
```

### 2. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull nomic-embed-text:v1.5
ollama pull llama3.2:3b
```

### 3. Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials:
   - Application type: Web application
   - Authorized redirect URI: `http://localhost:8501`
5. Copy Client ID and Client Secret

### 4. Gmail App Password

1. Enable 2-Factor Authentication on Gmail
2. Go to Google Account → Security → App passwords
3. Generate app password for "Mail"
4. Copy the 16-character password

### 5. Environment Configuration

```bash
# Copy template
cp .env.example .env

# Edit .env with your values:
```

```env
# Database
DB_URL=postgresql+psycopg2://postgres:dbda123@localhost:35432/postgres

# Google OAuth
GOOGLE_CLIENT_ID=your_actual_client_id
GOOGLE_CLIENT_SECRET=your_actual_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8501

# Email Notifications
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_16_char_app_password

# Ollama
OLLAMA_URL=http://localhost:11434
```

### 6. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the System

### Option 1: Full System (Recommended)
```bash
streamlit run app.py
```
This starts:
- Web interface at http://localhost:8501
- Daily job scraping at 09:00
- Daily notifications at 13:00

### Option 2: Manual Job Pipeline
```bash
# One-time scraping
python job_pipeline.py

# Scheduled scraping
python job_pipeline.py --schedule
```

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Web Scrapers  │───▶│  PostgreSQL  │◀───│  Streamlit  │
│   (3 sources)   │    │  + pgvector  │    │     App     │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │                     │
                              ▼                     ▼
                       ┌──────────────┐    ┌─────────────┐
                       │    Ollama    │    │   Google    │
                       │  (Embeddings │    │   OAuth     │
                       │   + LLM)     │    └─────────────┘
                       └──────────────┘
```

## 🎯 Usage Flow

1. **Sign In** with Google account
2. **Set Preferences** (role, location)
3. **Upload Resume** (optional, for skill extraction)
4. **Search Jobs** with AI-powered matching
5. **Save Jobs** you're interested in
6. **Get Notifications** - top 5 daily matches via email at 1 PM

## 📧 Automated Features

- **Daily 09:00**: Scrape new jobs from 3 sources
- **Daily 09:00**: Process embeddings and extract roles
- **Daily 13:00**: Send personalized job notifications
- **Smart Deduplication**: No duplicate jobs in database
- **Notification Tracking**: No spam, only new matches

## 🔧 Configuration

### Database Tables
- **jobs** - Job listings with vector embeddings
- **users** - User profiles and preferences
- **saved_jobs** - User bookmarks
- **job_notifications** - Email tracking

### Key Files
- **app.py** - Main Streamlit application
- **job_rag.py** - AI search and matching
- **job_pipeline.py** - Automated scraping
- **user_manager.py** - User management and notifications
- **config.py** - Environment configuration

## 🚨 Troubleshooting

### Common Issues

**Google OAuth Error**
```bash
# Check your .env file
GOOGLE_CLIENT_ID=your_actual_client_id  # Not placeholder
GOOGLE_CLIENT_SECRET=your_actual_secret
```

**Email Not Sending**
```bash
# Use Gmail app password (16 characters)
EMAIL_PASSWORD=abcd efgh ijkl mnop  # App password, not regular password
```

**Database Connection Error**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check pgvector extension
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Ollama Not Working**
```bash
# Check Ollama is running
ollama list

# Pull models if missing
ollama pull nomic-embed-text:v1.5
ollama pull llama3.2:3b
```

### Verify Setup
```python
# Test configuration
python -c "import config; print('✅ Config loaded')"

# Test database
python -c "from job_rag import JobRAG; rag = JobRAG(); print('✅ Database connected')"

# Test Ollama
python -c "from job_rag import JobRAG; rag = JobRAG(); print('✅ Ollama:', rag.get_embedding('test') is not None)"
```

## 📈 Performance

- **2-3 Users**: Optimized schema and minimal indexes
- **Smart Caching**: Vector embeddings cached in database
- **Efficient Scraping**: Parallel processing with deduplication
- **Fast Search**: pgvector for similarity search

## 🔒 Security

- **OAuth Only**: No password storage
- **Environment Variables**: Secrets in .env file
- **SQL Injection Protection**: Parameterized queries
- **Unique Constraints**: Prevent data corruption

## 📄 License

MIT License - Feel free to use and modify!

---

**Need Help?** Check the troubleshooting section or verify your .env configuration.