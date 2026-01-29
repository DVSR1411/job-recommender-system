-- Job Recommender System - PostgreSQL Schema with pgvector
-- Run this script to create the database schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id UUID DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    location VARCHAR(200) DEFAULT 'Pan India',
    experience VARCHAR(100) DEFAULT 'Freshers',
    listing_url TEXT,
    apply_url TEXT,
    posted_date DATE,
    source VARCHAR(50),
    role VARCHAR(100),
    embedding vector(768),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);