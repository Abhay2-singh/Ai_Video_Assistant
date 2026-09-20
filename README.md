# 🎬 AI Video Assistant & Meeting Intelligence

> **Transform any video or meeting recording into structured insights, conversational intelligence, and adaptive mock interviews.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Mistral AI](https://img.shields.io/badge/LLM-Mistral%20AI-FF7000)](https://mistral.ai/)
[![OpenAI Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![Sarvam AI](https://img.shields.io/badge/STT-Sarvam%20AI-0052CC)](https://www.sarvam.ai/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-FC60A8)](https://www.trychroma.com/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain%20LCEL-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why This Tech Stack? (A to Z Breakdown)](#-why-this-tech-stack-a-to-z-breakdown)
- [Key Features](#-key-features)
  - [1. Video Intelligence & Meeting Analytics](#1-video-intelligence--meeting-analytics)
  - [2. Interactive RAG (Chat with Meeting)](#2-interactive-rag-chat-with-meeting)
  - [3. Adaptive AI Mock Interviewer](#3-adaptive-ai-mock-interviewer)
  - [4. Multi-format Export (PDF & TXT)](#4-multi-format-export-pdf--txt)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [Repository Structure](#-repository-structure)
- [Prerequisites & System Setup](#-prerequisites--system-setup)
- [Quick Start Guide](#-quick-start-guide)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [Streamlit Cloud Deployment Guide](#-streamlit-cloud-deployment-guide)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 💡 Overview

Consuming hours of recorded webinars, team syncs, technical tutorials, and lecture videos is time-consuming. Finding specific decisions or retaining technical knowledge often requires re-watching large sections of media.

**AI Video Assistant** solves this by providing a unified pipeline that:
1. **Ingests** videos from YouTube URLs or local audio/video uploads (`.mp4`, `.mkv`, `.mp3`, `.wav`, etc.).
2. **Transcribes** speech using a dual-engine ASR approach (local **OpenAI Whisper** for English and **Sarvam AI** for Indian code-mixed Hinglish/Hindi speech).
3. **Distills** the recording into executive summaries, action items (with owners and deadlines), key decisions, and unresolved questions.
4. **Indexes** the transcript into a local **ChromaDB vector store** for real-time semantic search and conversation.
5. **Generates an Adaptive Mock Interview** grounded directly in concepts taught in the video, evaluating candidate responses with granular metrics and generating detailed feedback reports.

---

## 🔬 Why This Tech Stack? (A to Z Breakdown)

Every dependency in this project was selected for distinct performance, privacy, and architectural reasons:

| Component / Tool | Technology Used | Why We Chose It Over Alternatives |
| :--- | :--- | :--- |
| **Media Extraction** | `yt-dlp` | Faster, actively maintained, and handles modern YouTube streaming formats without throttling or breakage compared to older `pytube`. |
| **Audio Processing** | `pydub` + `ffmpeg` | Converts arbitrary audio/video formats into uniform 16kHz mono WAV files (the sweet spot for automatic speech recognition) and handles memory-efficient chunking. |
| **English Transcription** | `openai-whisper` | Runs locally offline without external API costs or data privacy leaks. Robust against background noise, accents, and varying audio fidelity. |
| **Hinglish/Hindi Transcription** | `Sarvam AI` (`saaras:v2.5`) | Standard Whisper models often hallucinate or produce phonetic garble on code-mixed Hinglish. Sarvam AI specializes in Indic accents and delivers simultaneous translation into clean English text. |
| **LLM Reasoning & Generation** | `ChatMistralAI` (`open-mistral-nemo`) | Provides state-of-the-art 128k context reasoning, high multilingual precision, and strict instruction adherence at a fraction of GPT-4 costs. |
| **Resilience & Fault Tolerance** | `tenacity` | Protects LLM API calls with exponential backoff and jitter, preventing pipeline failures caused by temporary rate limits or network drops. |
| **Text Chunking** | `RecursiveCharacterTextSplitter` | Preserves semantic paragraph and sentence boundaries when splitting long transcripts, preventing fragmented context in RAG retrieval. |
| **Embedding Generation** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Compact, ultra-fast 384-dimensional embedding model that runs smoothly on standard CPUs without requiring expensive GPU compute. |
| **Vector Database** | `ChromaDB` (`langchain-chroma`) | Lightweight, embedded vector database with persistent SQLite storage. Requires zero external server management, docker containers, or cloud subscriptions. |
| **Orchestration Layer** | `LangChain LCEL` | Declarative, composable pipelines (`prompt \| llm \| StrOutputParser()`) that simplify chain maintenance, streaming, and context passing. |
| **User Interface** | `Streamlit` | Enables reactive, stateful web applications in pure Python with real-time UI updates, custom CSS glassmorphism, and seamless session state control. |
| **Document Export** | `reportlab` & `fpdf2` | Generates formatted, downloadable PDF summaries and transcripts directly in-browser. |

---

## 🚀 Key Features

### 1. Video Intelligence & Meeting Analytics
- **Multi-source Ingestion**: Paste a YouTube link or supply a local file path.
- **Automated Preprocessing**: Extracts audio, normalizes channels and sampling rate, and splits long files into 10-minute chunks to prevent out-of-memory errors.
- **Smart Title Generation**: Creates clear, professional titles capturing the essence of the session.
- **Executive Summaries**: Delivers structured markdown summaries highlighting core topics and takeaways.
- **Structured Extraction**:
  - 📝 **Action Items**: Identifies actionable tasks, designated owners, and deadlines.
  - 🔑 **Key Decisions**: Extracts strategic and architectural decisions made during the discussion.
  - ❓ **Open Questions**: Isolates unanswered questions and follow-ups.

### 2. Interactive RAG (Chat with Meeting)
- Chat with your meeting recording in natural language.
- Every answer is grounded directly in the transcript using ChromaDB similarity search, minimizing hallucinations.
- Query historical context, timestamps, or specific technical definitions without scanning through hours of video.

### 3. Adaptive AI Mock Interviewer
- **Automated Concept Extraction**: Scans the transcript to identify core technical concepts, definitions, and practical significance.
- **Grounded Questioning**: Generates technical interview questions tied strictly to concepts present in the video.
- **Dynamic Difficulty**: Choose from **Beginner**, **Intermediate**, or **Advanced**. The interviewer adapts subsequent questions dynamically based on the candidate's performance.
- **Multi-Dimensional Evaluation**:
  - Concept Understanding (0–100)
  - Technical Accuracy (0–100)
  - Clarity & Depth (0–100)
  - Constructive feedback and a model reference answer
- **Final Performance Scorecard**: Provides an overall score, strengths, areas for improvement, and concept-by-concept analysis.
- **Persistent History**: Saves past interview sessions in `data/interviews.json` for progress tracking over time.

### 4. Multi-format Export (PDF & TXT)
- Download structured meeting summaries, action items, or full transcripts as clean PDF documents or plain text files.

---

## 🏗 System Architecture & Data Flow

```
                                  [ Input Source ]
                         (YouTube URL or Local File)
                                      │
                                      ▼
                        [ utils/audio_processor.py ]
                        • yt-dlp audio download
                        • ffmpeg conversion (16kHz WAV)
                        • 10-minute chunking
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [ Language: English ]                 [ Language: Hinglish ]
         OpenAI Whisper (Local)                Sarvam AI API (saaras:v2.5)
         (Chunk-by-chunk ASR)                  (25s pieces with translation)
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                             [ Full Transcript ]
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
[ core/summarizer.py ]      [ core/extractor.py ]        [ core/vector_store.py ]
• Generate Title            • Action Items               • Recursive Chunking
• Structured Summary        • Key Decisions              • all-MiniLM-L6-v2 Embeddings
(Mistral AI + LCEL)         • Open Questions             • ChromaDB Storage
                                                                   │
                                                                   ▼
                                                         [ core/rag_engine.py ]
                                                         • Similarity Retriever
                                                         • Conversational Q&A
                                                                   │
                                                                   ▼
                                                         [ core/interviewer.py ]
                                                         • Technical Concept Extraction
                                                         • Adaptive Interview Engine
                                                         • Multi-metric Evaluation & Report
```

---

## 📂 Repository Structure

```
AI-Video-Assistant-/
├── app.py                     # Streamlit multi-page web application (UI & navigation)
├── main.py                    # Command-line interface (CLI) execution pipeline
├── test.py                    # Standalone end-to-end verification script
├── requirements.txt           # Python package dependencies
├── packages.txt               # OS-level Linux packages for deployment (ffmpeg)
├── .env                       # Environment variables & API keys (git-ignored)
├── .gitignore                 # Git ignore rules
│
├── core/                      # Core business logic & AI orchestration
│   ├── transcriber.py         # Whisper & Sarvam AI dual-engine transcription
│   ├── summarizer.py          # Session title & executive summary generators
│   ├── extractor.py           # Extraction of action items, decisions & questions
│   ├── vector_store.py        # ChromaDB setup, embedding models & text chunking
│   ├── rag_engine.py          # Retrieval-Augmented Generation query chain
│   ├── interviewer.py         # Mock interview generation, evaluation & scoring
│   ├── interviewer_ui.py      # Streamlit UI components for the Interviewer mode
│   └── interview_storage.py   # JSON persistence for interview records & videos
│
├── utils/                     # Utility modules
│   └── audio_processor.py     # Audio downloading, format conversion & chunking
│
├── data/                      # Local JSON storage (created automatically)
│   ├── processed_videos.json  # Catalog of processed videos and extracted concepts
│   └── interviews.json        # History of completed mock interview sessions
│
├── vector_db/                 # ChromaDB vector database storage directory
└── downloades/                # Temporary directory for downloaded/processed audio
```

---

## ⚙️ Prerequisites & System Setup

### 1. Python Version
- **Recommended**: **Python 3.10** or **Python 3.11**
- *Note for Python 3.13 / 3.14*: Python 3.13 removed the standard library `audioop` module. This repository includes `audioop-lts; python_version >= '3.13'` to maintain compatibility, but Python 3.10/3.11 is strongly recommended for optimal PyTorch and ChromaDB wheel stability.

### 2. FFmpeg (Required System Dependency)
FFmpeg is required by `pydub` and `yt-dlp` to decode and convert audio streams.

- **Windows**:
  - Download from [gyan.dev FFmpeg](https://www.gyan.dev/ffmpeg/builds/) or install via winget:
    ```powershell
    winget install Gyan.FFmpeg
    ```
  - Verify by opening PowerShell and typing: `ffmpeg -version`
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```

---

## 📦 Quick Start Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abhay2-singh/Ai_Video_Assistant.git
cd Ai_Video_Assistant
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=open-mistral-nemo

# Optional: Needed only for Hinglish / Hindi transcription
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_STT_MODEL=saaras:v2.5

# Optional: Whisper model size (tiny, base, small, medium, large)
WHISPER_MODEL=base
```

### Step 5: Launch the Application

#### Option A: Interactive Web UI (Recommended)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### Option B: Terminal CLI
```bash
python main.py
```
Enter any YouTube URL or local file path when prompted, review summaries in the terminal, and chat with the transcript interactively.

---

## 🔑 Configuration & Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MISTRAL_API_KEY` | *(Required)* | API key from [Mistral AI Console](https://console.mistral.ai/). Used for summarization, extraction, RAG, and mock interview evaluation. |
| `MISTRAL_MODEL` | `open-mistral-nemo` | Mistral model identifier. Supports `open-mistral-nemo`, `mistral-small-latest`, or `mistral-large-latest`. |
| `SARVAM_API_KEY` | *(Optional)* | API key from [Sarvam AI](https://www.sarvam.ai/). Required only if transcribing Indian code-mixed Hinglish videos. |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam Speech-to-Text model version. |
| `WHISPER_MODEL` | `base` | Local OpenAI Whisper model weight. Options: `tiny`, `base`, `small`, `medium`, `large`. `base` offers a balance between speed and accuracy. |

---

## ☁️ Streamlit Cloud Deployment Guide

When deploying to [Streamlit Community Cloud](https://share.streamlit.io/):

1. **Repository Files**:
   - Ensure **`requirements.txt`** is lowercase (Linux is case-sensitive).
   - Ensure **`packages.txt`** exists in the repo root containing `ffmpeg`.
2. **Select Python Version**:
   - In your Streamlit Cloud App Settings, set the Python version to **`3.11`** or **`3.10`**.
3. **Configure Secrets**:
   - In Streamlit Cloud dashboard, navigate to **Settings** → **Secrets** and paste:
     ```toml
     MISTRAL_API_KEY = "your_actual_mistral_api_key"
     MISTRAL_MODEL = "open-mistral-nemo"

     # If using Sarvam:
     SARVAM_API_KEY = "your_sarvam_key"
     ```
4. **Deploy**: Streamlit Cloud will install OS-level `ffmpeg` from `packages.txt`, install Python wheels from `requirements.txt`, and boot `app.py`.

---

## ❓ Troubleshooting & FAQs

#### Q: `ModuleNotFoundError: No module named 'dotenv'` on deployment?
- Ensure the filename is `requirements.txt` with a lowercase `r`. Linux containers ignore `Requirements.txt`.

#### Q: `ModuleNotFoundError: No module named 'audioop'`?
- Python 3.13+ removed `audioop`. Add `audioop-lts; python_version >= '3.13'` to `requirements.txt`, or set your deployment environment to Python 3.11.

#### Q: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`?
- FFmpeg is not installed or not in your system's PATH. Install FFmpeg as outlined in [Prerequisites](#-prerequisites--system-setup). On Streamlit Cloud, verify that `packages.txt` contains `ffmpeg`.

#### Q: Can I run this without a GPU?
- Yes. The default Whisper model (`base`) and the embedding model (`all-MiniLM-L6-v2`) are lightweight and run smoothly on standard multi-core CPUs.

#### Q: Where are my processed videos and interview scorecards stored?
- Metadata and interview records are saved locally in the `data/` folder as `processed_videos.json` and `interviews.json`. Vector indexes are persisted in the `vector_db/` folder.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).