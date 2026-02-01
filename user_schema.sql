-- Simple User Management Schema (for 2-3 users)

-- Users table - basic info only
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    google_id VARCHAR(100) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    location VARCHAR(200),
    role_name VARCHAR(100),
    skills TEXT[],
    resume_blob BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Simple saved jobs
CREATE TABLE IF NOT EXISTS saved_jobs (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(user_id, job_id)
);

-- Simple notification tracking
CREATE TABLE IF NOT EXISTS job_notifications (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(user_id, job_id)
);

-- Only essential index for OAuth lookup
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);