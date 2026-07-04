🌟 Introduction

Careerly is a comprehensive AI-powered career assistant built using Flask, Large Language Models (LLMs), Speech Recognition, Computer Vision, and Machine Learning technologies.

Instead of focusing solely on interview preparation, Careerly provides a complete career development ecosystem that enables users to:

Practice realistic interviews
Receive intelligent AI feedback
Improve speaking confidence
Analyze body language
Tailor resumes
Generate cover letters
Search for jobs
Build personalized learning plans
Create long-term career roadmaps

The platform combines multiple AI capabilities into one seamless workflow, making it an all-in-one assistant for students, graduates, and professionals preparing for internships or full-time roles.

✨ Features
🎤 AI Mock Interview System

Careerly generates interview questions dynamically based on:

Job Role
Experience Level
Interview Type
Difficulty Level

Supported interview modes include:

✅ Text Interviews
✅ Audio Interviews
✅ Video Interviews

🤖 AI Interview Evaluation

Every response is evaluated using Large Language Models across multiple dimensions.

Evaluation Criteria:

Communication
Technical Knowledge
Clarity
Confidence
Structure
Relevance
Problem Solving
Conciseness

The AI also provides:

Strengths
Weaknesses
Suggestions for improvement
Overall score
Detailed written feedback

🎙 Speech Analysis

For audio interviews, Careerly performs automatic speech analysis.

Features include:

Speech-to-text transcription
Speaking speed
Pause detection
Hesitation analysis
Filler word detection
Fluency analysis

Powered by:

OpenAI Whisper

🎥 Video Interview Analysis

Video interviews are analyzed using Computer Vision techniques.

Behavioral analysis includes:

Eye contact detection
Facial emotion recognition
Head pose estimation
Body posture
Hand movement
Fidgeting detection
Engagement analysis

Libraries used:

OpenCV
MediaPipe
FER

📄 Resume & CV Optimization

Users can upload their CV and a target Job Description.

The system can:

Tailor resumes
Improve ATS compatibility
Highlight relevant skills
Optimize formatting
Suggest missing keywords
✉ AI Cover Letter Generator

Automatically generates personalized cover letters using:

Resume
Job Description
Company Information

💼 Job Search Automation

Careerly can automatically:

Extract job titles
Search job postings
Scrape multiple job platforms
Present relevant opportunities

📚 Personalized Learning Assistant

Based on interview performance and target role, the platform recommends:

Online courses
Weekly study plans
Skill improvement roadmap
Career roadmap

👤 User Management

Complete authentication system with:

Registration
Login
Logout
Profile Management
Password Update
Interview History
Session Reports

🏗 System Architecture
                    User
                      │
                      ▼
               Flask Web Application
                      │
 ┌────────────────────┼────────────────────┐
 │                    │                    │
 ▼                    ▼                    ▼
Authentication   Interview Engine   Career Automation
 │                    │                    │
 ▼                    ▼                    ▼
Database        AI Question Generator   Resume Tailoring
                     │
                     ▼
              Interview Responses
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 Speech Analysis  Video Analysis  LLM Evaluation
      │              │              │
      └──────────────┼──────────────┘
                     ▼
              Performance Report
                     │
                     ▼
          Career Recommendations
          
🧠 AI Components

Careerly integrates several AI technologies.

Google Gemini

Used for:

Interview question generation
Answer evaluation
Resume analysis
Career roadmap generation
Cover letter generation
Whisper

Used for:

Audio transcription
Speech analysis
Computer Vision

Used for:

Emotion detection
Eye contact analysis
Gesture analysis
Body posture estimation
Machine Learning

Used for:

Behavioral metrics
Feedback generation
Recommendation logic

🛠 Tech Stack
Backend
Python
Flask
Flask-Login
Flask-Session
SQLAlchemy
Database
MySQL
PyMySQL
Artificial Intelligence
Google Gemini API
OpenAI Whisper
Computer Vision
OpenCV
MediaPipe
FER
Machine Learning
NumPy
Pandas
Scikit-learn
Frontend
HTML5
Tailwind CSS
JavaScript
Fetch API
Document Processing
python-docx
PyPDF2
BeautifulSoup
Selenium

📂 Project Structure
Careerly/
│
├── main.py                     # Main Flask application
├── auth_routes.py              # Authentication routes
├── agents.py                   # AI Agents
├── database.py                 # Database configuration
├── models.py                   # SQLAlchemy models
├── video.py                    # Video analysis
│
├── templates/                  # HTML Templates
├── static/                     # CSS, JS, Media
├── uploads/                    # Uploaded resumes
├── outputs/                    # Generated files
├── feedback_data/              # Feedback CSVs
│
├── requirements.txt
├── init_db.py
├── env.example
└── README.md

⚙ Installation
Clone Repository
git clone https://github.com/yourusername/Careerly.git

cd Careerly
Create Virtual Environment

Windows

python -m venv venv

venv\Scripts\activate

Linux/macOS

python3 -m venv venv

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file.

SECRET_KEY=

DATABASE_URL=mysql+pymysql://username:password@localhost/careerly

GEMINI_API_KEY=

GEMINI_API_KEY_SECONDARY=
🗄 Database Setup

Create a MySQL database.

CREATE DATABASE careerly;

Initialize tables.

python init_db.py
▶ Running the Application
python main.py

Application runs on:

http://127.0.0.1:5002

🔄 Interview Workflow
Login

↓

Select Interview Settings

↓

AI Generates Questions

↓

User Answers

↓

Speech / Video Analysis

↓

LLM Evaluation

↓

Performance Report

↓

Career Suggestions

🚀 Career Automation Workflow
Upload Resume

↓

Upload Job Description

↓

Resume Tailoring

↓

Cover Letter Generation

↓

Strength Analysis

↓

Course Suggestions

↓

Weekly Learning Plan

↓

Career Roadmap

🌐 Main API Endpoints
Endpoint	Description
POST /api/upload-cv	Upload Resume
POST /api/tailor-cv	Tailor Resume
POST /api/generate-cover-letter	Generate Cover Letter
POST /api/scrape-jobs	Search Jobs
POST /api/analyze-strengths-weaknesses	Skill Analysis
POST /api/suggest-courses	Course Recommendation
POST /api/weekly-learning-plan	Weekly Study Plan
POST /api/career-roadmap	Career Roadmap
GET /api/health	Health Check

📈 Future Improvements
CI/CD Pipeline
AI Voice Interviewer
Live Coding Interviews
ATS Resume Score
Multi-language Support
OAuth Login
Admin Dashboard
Analytics Dashboard
Cloud Deployment
Background Task Queue (Celery)
Real-time Interview Analytics

🤝 Contributing

Contributions are welcome!

Fork the repository.
Create a new feature branch.
Commit your changes.
Push your branch.
Open a Pull Request.
📄 License

This project is licensed under the MIT License.
