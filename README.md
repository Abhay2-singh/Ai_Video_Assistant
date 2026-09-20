🎬 AI Video Assistant & Meeting Intelligence

Transform any video or meeting recording into structured insights, conversational intelligence, and adaptive mock interviews.

AI Video Assistant is a Streamlit-based application that turns recorded videos, meetings, lectures, webinars, and technical sessions into useful and searchable information.

The application can transcribe videos, generate summaries, extract important information, allow users to chat with the transcript using RAG, and conduct an Adaptive AI Mock Interview based on the concepts actually discussed in the selected video.

✨ What This Project Does

Instead of watching a long video again and again to find important information, the application processes the recording and provides:

Video/audio transcription
Automatic session title generation
Structured summaries
Action items
Key decisions
Open questions
Transcript-based conversational Q&A
RAG-powered semantic search
Adaptive AI mock interviews
Interview performance evaluation
Concept-wise feedback
Previous interview history
PDF and TXT exports

The complete pipeline is built around the original video content, so the AI features are connected to the information extracted from the user's recording.

🚀 Key Features
1. Video Intelligence & Meeting Analytics

The application accepts both online and local media sources.

Supported Inputs
YouTube URLs
Local video files
Local audio files
.mp4
.mkv
.mp3
.wav
and other supported formats

The audio is automatically processed before transcription. Long recordings are divided into smaller chunks to reduce memory usage and make processing more reliable.

Automatic Processing

The system performs:

Audio extraction
Audio normalization
Format conversion
Audio chunking
Speech-to-text transcription
Title generation
Summary generation
Information extraction
Vector indexing
RAG-based question answering
📝 Transcription

The project uses two transcription approaches depending on the language of the recording.

English

OpenAI Whisper is used for local English transcription.

Hinglish / Hindi

Sarvam AI is used for Indian code-mixed speech and Hindi/Hinglish recordings.

This allows the application to handle technical videos where the speaker may switch between English and Hindi.

The transcription pipeline and language-specific processing are part of the core architecture.

🧠 AI-Powered Video Understanding

After transcription, the system uses the LLM to understand the content of the recording.

It can generate:

Session Title

Creates a short title representing the main topic of the recording.

Executive Summary

Provides a structured summary of the important points discussed.

Action Items

Identifies tasks, owners, and deadlines when they are present in the recording.

Key Decisions

Extracts important decisions or conclusions discussed during the session.

Open Questions

Identifies questions or topics that remain unresolved.

These operations are handled through the existing summarization and extraction components.

💬 Interactive RAG — Chat With Your Video

One of the main features of the project is the ability to interact with a processed recording.

After transcription, the transcript is split into meaningful chunks and converted into embeddings using:

all-MiniLM-L6-v2

The embeddings are stored in:

ChromaDB

When the user asks a question, the RAG engine retrieves the most relevant transcript sections and provides them as context to the LLM.

This means the user can ask questions such as:

What did the speaker explain about REST APIs?

Why was this architecture selected?

What was discussed about database normalization?

Explain the example given in the video.

The answer is generated using the relevant transcript context instead of requiring the user to manually search through the entire recording.

The existing RAG architecture uses a similarity retriever and conversational Q&A pipeline.

🎤 Adaptive AI Mock Interviewer

The Adaptive AI Mock Interviewer is an extension of the existing transcript + concept extraction + RAG + LLM pipeline.

Instead of asking a fixed list of interview questions, the interviewer generates questions from the concepts that are actually present in the selected video.

This makes the interview connected to what the user has learned from that specific recording.

How It Works
Processed Video
       ↓
Transcript
       ↓
Important Concept Extraction
       ↓
RAG Context
       ↓
Initial Interview Question
       ↓
User Answer
       ↓
AI Evaluation
       ↓
Concept / Knowledge Analysis
       ↓
Next Question
       ↓
Final Interview Report

The interviewer reuses the project's existing AI infrastructure rather than creating a separate processing pipeline.

🎯 Interview Difficulty

The user can select the difficulty before starting an interview.

Beginner

Focuses on:

Basic concepts
Definitions
Simple explanations
Fundamental understanding
Intermediate

Focuses on:

Deeper conceptual understanding
Why/how questions
Practical application
Connecting multiple concepts
Advanced

Focuses on:

Complex scenarios
Problem solving
Debugging
Deeper technical reasoning
Applying concepts to practical situations
❓ Dynamic Question Generation

The interviewer supports multiple question types:

Conceptual
Why / How
Practical
Debugging
Scenario-based
Follow-up

Questions are generated dynamically from the concepts available in the processed video.

The system does not rely on a hardcoded question bank.

For example, if a video explains:

REST API
HTTP methods
GET
POST
PUT
DELETE
Authentication

the interviewer can generate questions around those concepts.

It should not create a question about a completely unrelated topic that was never discussed in the recording.

🔄 Adaptive Interview Flow

The interview changes depending on the user's previous answer.

If the answer is weak

The interviewer can:

identify the missing concept
detect the knowledge gap
ask a follow-up question
return to the same concept from another angle
If the answer is strong

The interviewer can:

move to another concept
increase the difficulty
ask a practical or scenario-based question

This creates an interview that changes according to the user's demonstrated understanding instead of following a fixed sequence.

🧪 Answer Evaluation

After every answer, the LLM evaluates the response using multiple dimensions.

Evaluation includes:
Concept understanding
Technical correctness
Missing concepts
Misconceptions
Explanation quality
Clarity
Depth
Confidence

The existing README describes the evaluation using Concept Understanding, Technical Accuracy, and Clarity & Depth scores along with constructive feedback and a reference answer.

Example feedback:

Concept Understanding: 82/100
Technical Accuracy: 76/100
Clarity & Depth: 68/100

Missing Concepts:
- Stateless communication
- HTTP status codes

Feedback:
Your explanation correctly describes the basic purpose of REST APIs,
but it does not explain why stateless communication is important.

Suggested Improved Answer:
...
🛡️ Grounded Interviewing

A major part of the interviewer is grounding.

The interviewer uses:

Video Transcript
      +
Extracted Concepts
      +
RAG Retrieved Context
      ↓
LLM
      ↓
Question / Evaluation

The LLM is instructed to stay within the information available from the processed video.

For example, if the video discusses:

Python
Flask
REST APIs
SQL

the interviewer should not claim:

"The instructor explained Kubernetes..."

if Kubernetes was never present in the recording.

This keeps the interview connected to the user's actual learning material.

📊 Final Interview Report

After completing the interview, the system generates a final performance report.

The report contains:

Overall Performance

A combined score representing the interview performance.

Concept-wise Performance

Shows how the user performed across the concepts covered during the interview.

Strong Concepts

Topics where the user demonstrated good understanding.

Weak Concepts

Topics where the user's answers showed gaps or uncertainty.

Missed Concepts

Important concepts that were not included or properly explained in the user's answers.

Misconceptions

Incorrect interpretations or technically inaccurate statements identified during evaluation.

Recommended Topics

Topics that should be revised based on the interview performance.

Improved Answers

Provides examples of how weak or incomplete answers could be explained more clearly.

🖥️ Adaptive Interview UI

The application provides a dedicated interface for the interview flow.

The user can:

Select a processed video
Select interview difficulty
Start an interview
Read the generated question
Enter an answer
Submit the answer
View AI feedback
Continue to the next question
Complete the interview
View the final report
Review previous interviews
Interview Interface

Add the provided screenshot to your repository as assets/adaptive-ai-interviewer.png.

📚 Interview History

Completed interviews are stored locally so that users can review their previous performance.

Interview records are stored in:

data/interviews.json

Processed video information and extracted concepts are maintained in:

data/processed_videos.json

The project therefore supports reviewing previous interview sessions instead of losing the results after closing the application.

🏗️ System Architecture

The overall architecture is:

                    INPUT SOURCE
              YouTube URL / Local File
                         │
                         ▼
              Audio Processing Layer
                         │
                         ▼
               Speech Recognition
              ┌──────────┴──────────┐
              │                     │
          Whisper                Sarvam AI
          English               Hindi/Hinglish
              │                     │
              └──────────┬──────────┘
                         ▼
                   Full Transcript
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Summarizer        Extractor       Vector Store
        │                │                │
        │                │           Embeddings
        │                │                │
        │                │            ChromaDB
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                    RAG Engine
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Video Chat            AI Interviewer
                                    │
                                    ▼
                            Answer Evaluation
                                    │
                                    ▼
                           Adaptive Question
                                    │
                                    ▼
                            Final Interview
                               Report

The repository architecture separates transcription, summarization, extraction, vector storage, RAG, interviewing, UI, and interview persistence into individual components.

🧩 Technology Stack
Component	Technology
Programming Language	Python
UI	Streamlit
LLM	Mistral AI
LLM Orchestration	LangChain LCEL
English STT	OpenAI Whisper
Hindi/Hinglish STT	Sarvam AI
Audio Processing	pydub + FFmpeg
YouTube Extraction	yt-dlp
Embeddings	all-MiniLM-L6-v2
Vector Database	ChromaDB
PDF Generation	ReportLab / FPDF2
Persistence	JSON
Deployment	Streamlit Community Cloud

The project uses LangChain LCEL to compose the AI processing pipelines and ChromaDB for persistent local vector storage.

📂 Project Structure
AI-Video-Assistant/
│
├── app.py
├── main.py
├── test.py
├── requirements.txt
├── packages.txt
├── .env
├── .gitignore
│
├── core/
│   ├── transcriber.py
│   ├── summarizer.py
│   ├── extractor.py
│   ├── vector_store.py
│   ├── rag_engine.py
│   ├── interviewer.py
│   ├── interviewer_ui.py
│   └── interview_storage.py
│
├── utils/
│   └── audio_processor.py
│
├── data/
│   ├── processed_videos.json
│   └── interviews.json
│
├── vector_db/
│
└── downloades/
Core Modules

transcriber.py

Handles Whisper and Sarvam-based transcription.

summarizer.py

Generates titles and structured summaries.

extractor.py

Extracts action items, decisions, and open questions.

vector_store.py

Handles transcript chunking, embeddings, and ChromaDB storage.

rag_engine.py

Handles transcript retrieval and conversational Q&A.

interviewer.py

Handles concept extraction, adaptive interview generation, evaluation, and scoring.

interviewer_ui.py

Contains Streamlit UI components for the interview experience.

interview_storage.py

Stores interview sessions and results.

These modules correspond to the repository structure documented in the project.

🔬 Why This Architecture?

The main idea behind the architecture is to avoid creating separate AI pipelines for every feature.

The same processed transcript can be reused for:

Transcript
   │
   ├── Summary
   │
   ├── Information Extraction
   │
   ├── RAG Chat
   │
   └── Adaptive Interview

This reduces unnecessary processing and keeps the different features connected to the same source material.

For example:

User uploads video
        ↓
Video is transcribed once
        ↓
Transcript is indexed
        ↓
        ├── User asks questions
        │
        └── User starts interview
                    ↓
             Concepts retrieved
                    ↓
             Questions generated
⚙️ Installation
1. Clone the Repository
git clone https://github.com/Abhay2-singh/Ai_Video_Assistant.git

cd Ai_Video_Assistant
2. Create Virtual Environment
Windows
python -m venv venv

.\venv\Scripts\activate
Linux / macOS
python3 -m venv venv

source venv/bin/activate
3. Install Dependencies
pip install --upgrade pip

pip install -r requirements.txt

The repository recommends Python 3.10 or 3.11 for better compatibility with PyTorch and ChromaDB.

🎵 FFmpeg Setup

FFmpeg is required for audio/video processing.

Windows
winget install Gyan.FFmpeg

Verify:

ffmpeg -version
Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg
macOS
brew install ffmpeg
🔑 Environment Variables

Create a .env file in the project root.

MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=open-mistral-nemo

SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_STT_MODEL=saaras:v2.5

WHISPER_MODEL=base

SARVAM_API_KEY is only required when using Sarvam AI for Hindi/Hinglish transcription.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

Then open:

http://localhost:8501

The repository also provides a CLI pipeline:

python main.py

☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

For deployment:

Keep requirements.txt in the repository root.
Keep packages.txt in the root directory.
Add ffmpeg to packages.txt.
Use Python 3.10 or 3.11.
Add API keys through Streamlit Secrets.
Deploy app.py.

Example:

MISTRAL_API_KEY = "your_actual_mistral_api_key"
MISTRAL_MODEL = "open-mistral-nemo"

SARVAM_API_KEY = "your_sarvam_api_key"

The project's deployment configuration is designed around Streamlit Community Cloud and the FFmpeg system dependency.

🧪 Testing

The repository includes:

test.py

which is used for standalone end-to-end verification of the application pipeline.

The complete workflow can be tested as:

Input Video
   ↓
Audio Processing
   ↓
Transcription
   ↓
Summary / Extraction
   ↓
Vector Indexing
   ↓
RAG Chat
   ↓
Adaptive Interview
   ↓
Answer Evaluation
   ↓
Final Report
🛠️ Troubleshooting
ModuleNotFoundError: No module named 'audioop'

Python 3.13+ removed the standard audioop module.

The project supports the compatibility package:

audioop-lts

However, Python 3.10/3.11 is recommended.

ffmpeg not found

Make sure FFmpeg is installed and available in PATH.

For Streamlit Cloud, verify:

packages.txt

contains:

ffmpeg
Can the project run without a GPU?

Yes. The default Whisper model and MiniLM embedding model are designed to work on standard CPU systems.

🌟 What Makes the Adaptive Interviewer Different?

Traditional video summarization tools mainly stop after:

Video → Transcript → Summary

This project extends that workflow:

Video
  ↓
Transcript
  ↓
Understand the Content
  ↓
Ask Questions
  ↓
Evaluate User
  ↓
Identify Knowledge Gaps
  ↓
Adapt Next Question
  ↓
Generate Performance Report

The important part is that the interview is connected to the same content the user watched.

So instead of practicing with a generic question bank, the user can use the actual technical concepts from their learning material as the basis for the interview.

📈 Future Improvements

Possible future extensions include:

Voice-based interview answers
Real-time speech evaluation
Interview timer
More detailed confidence analysis
Topic-level progress tracking
Interview difficulty calibration
Personalized revision plans
Interview performance history over multiple videos
Support for multiple interview formats
📄 License

This project is open-source and available under the MIT License.

MIT License
README mein screenshot exactly kahan lagana hai?

Tumhari jo screenshot hai, usko Adaptive AI Mock Interviewer section ke immediately after lagana best rahega:

# 🎤 Adaptive AI Mock Interviewer

...

## 🖥️ Adaptive Interview UI

![Adaptive AI Interviewer](assets/adaptive-ai-interviewer.png)

Aur repo mein:

AI-Video-Assistant/
│
├── assets/
│   └── adaptive-ai-interviewer.png
│
├── app.py
├── main.py
├── core/
...
