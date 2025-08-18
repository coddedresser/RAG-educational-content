# 🎓 Educational Content RAG with Learning Path Generation

A comprehensive Retrieval-Augmented Generation (RAG) system focused on PDF Q&A, with additional pages for learning paths, content search, student profiles, progress tracking, and system analytics.

## 📋 Table of Contents

- [Features](#features)
- [Current Pages](#current-pages)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Configuration](#configuration)

## ✨ Features

### Core Functionality
- **PDF Document Q&A (RAG)**: Upload a PDF and ask questions about its content
- **Authentication-first UX**: Embedded sign-in/sign-up shown immediately on Home if not authenticated
- **User Sessions**: Local session and user storage in `data/sessions.json` and `data/users.json`
- **Groq API (optional)**: AI-generated answers when `GROQ_API_KEY` is set; intelligent fallback otherwise
- **Learning Tools**: Learning Path, Content Search, Student Profile, Progress Tracking, System Analytics

## Current Pages
- **Home** (`app/Home.py`): Authentication embedded; primary PDF Q&A experience
- **User Settings** (`app/pages/0_User_Settings.py`): Standalone authentication page (also embedded on Home)
- **Learning Path** (`app/pages/1_Learning_Path.py`)
- **Content Search** (`app/pages/2_Content_search.py`)
- **Student Profile** (`app/pages/3_Student_Profile.py`)
- **Progress Tracking** (`app/pages/4_Progress_Tracking.py`)
- **System Analytics** (`app/pages/5_System_analytics.py`)

## 🏗️ System Architecture

```
educational-rag-system/
├── app/
│   ├── Home.py                     # Streamlit entry point (Home)
│   ├── config.py                   # Configuration/settings
│   ├── components/
│   │   ├── auth.py                 # Authentication (users/sessions)
│   │   ├── llm_service.py          # Free LLM service helpers
│   │   ├── data_processor.py       # (optional) Content processing
│   │   ├── embeddings.py           # (optional) Embedding generation
│   │   ├── retriever.py            # (optional) Retrieval helpers
│   │   ├── learning_path.py        # Learning path helpers
│   │   ├── student_profile.py      # Student profile helpers
│   │   └── progress_tracker.py     # Progress tracking helpers
│   └── pages/
│       ├── 0_User_Settings.py      # Authentication page
│       ├── 1_Learning_Path.py
│       ├── 2_Content_search.py
│       ├── 3_Student_Profile.py
│       ├── 4_Progress_Tracking.py
│       └── 5_System_analytics.py
├── data/
│   ├── users.json                  # Local user store
│   └── sessions.json               # Local session store
├── models/                         # Local DBs / artifacts
├── vector_db/                      # Vector storage (if used)
├── requirements.txt
└── README.md
```

### Technology Stack
- **Framework**: Streamlit
- **Optional AI**: Groq API (via `GROQ_API_KEY`); built-in fallback available
- **Visualization**: Plotly
- **Data**: Local JSON/SQLite artifacts in `data/` and `models/`

## 🚀 Installation

### Prerequisites
- Python 3.9+
- 4GB+ RAM recommended

### Quick Start
1. Clone the repository
```bash
git clone https://github.com/your-username/educational-rag-system.git
cd educational-rag-system
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. (Optional) Set Groq API key for AI answers
```bash
# Powershell
$env:GROQ_API_KEY = "your_key"
# or bash
export GROQ_API_KEY=your_key
```
4. Run the application
```bash
streamlit run app/Home.py
```
The app will be available at `http://localhost:8501`.

## 📖 Usage

1. **Authentication-first on Home**
   - Home shows the sign-in/sign-up UI if not authenticated
   - Create an account or sign in (stored locally in `data/`)
2. **Upload a PDF**
   - Use “Document Upload” on Home
   - Process the PDF to enable Q&A
3. **Ask Questions**
   - Enter a question about the uploaded PDF
   - If `GROQ_API_KEY` is set, answers are AI-generated; otherwise, enhanced fallback is used
4. **Explore Pages**
   - Learning Path, Content Search, Student Profile, Progress Tracking, System Analytics

## 🚀 Deployment

### Local Development
```bash
streamlit run app/Home.py
```

### Docker (example)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Enables Groq AI answers
GROQ_API_KEY=your_groq_api_key
```

- Without `GROQ_API_KEY`, the system uses a local enhanced fallback for answers.
- Users and sessions are stored locally under `data/`.

---

If you need a deeper API/engineering reference, see the modules under `app/components/`. This README reflects the current app structure, page names, and the authentication-first user flow with Home as the entry point.