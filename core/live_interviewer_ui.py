import os
import io
import time
import tempfile
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

from core.interviewer import (
    generate_live_question,
    evaluate_live_answer,
    select_adaptive_live_next,
    generate_live_final_report,
)
from core.interview_storage import save_interview_session, get_past_interviews

# Cache whisper model in memory so it doesn't reload on every answer
_whisper_cache = None


def _get_whisper():
    global _whisper_cache
    if _whisper_cache is None:
        import whisper
        _whisper_cache = whisper.load_model("base")
    return _whisper_cache


def _generate_tts_audio(text: str) -> bytes | None:
    """Generate spoken MP3 audio from question text using gTTS."""
    if not text:
        return None
    try:
        from gtts import gTTS
        clean_text = text.replace("**", "").replace("*", "").replace("`", "")
        # Limit to 300 chars for snappy TTS generation
        tts = gTTS(text=clean_text[:400], lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"TTS generation notice: {e}")
        return None


def _transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe candidate's recorded audio bytes using Whisper."""
    if not audio_bytes:
        return ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = _get_whisper()
        result = model.transcribe(tmp_path, fp16=False)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Audio transcription error: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─── Live Interview State Initialization ────────────────────────────────────────

def init_live_interview_state():
    """Ensure all required state variables for Live AI Interview exist."""
    defaults = {
        "live_step": "setup",  # 'setup', 'room', 'evaluation', 'report'
        "live_topic": "Python",
        "live_custom_topic": "",
        "live_difficulty": "Intermediate",
        "live_target_questions": 5,
        "live_history": [],
        "live_current_q": None,
        "live_last_eval": None,
        "live_final_report": None,
        "live_user_answer": "",
        "live_enable_camera": False,
        "live_tts_audio": None,
        "live_start_timestamp": None,
        "live_saved_session_id": None,
        "live_last_audio_sig": None,
        "live_ai_status": "Speaking",  # 'Speaking', 'Listening', 'Evaluating'
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Main Entry Point ───────────────────────────────────────────────────────────

def render_live_interviewer_interface():
    """Main router for the Live AI Interview experience."""
    init_live_interview_state()

    step = st.session_state.get("live_step", "setup")
    if step == "setup":
        _render_live_setup_screen()
    elif step == "room":
        _render_live_room_screen()
    elif step == "evaluation":
        _render_live_evaluation_screen()
    elif step == "report":
        _render_live_report_screen()


# ─── SCREEN 1: Setup ────────────────────────────────────────────────────────────

TOPIC_OPTIONS = [
    "Python",
    "C++",
    "Java",
    "SQL",
    "OOP",
    "DBMS",
    "Data Structures",
    "Computer Networks",
    "Operating Systems",
    "Web Development",
    "Django",
    "Flask",
    "AI / ML",
    "RAG / Generative AI",
    "General Technical Interview",
    "Custom Topic",
]


def _render_live_setup_screen():
    st.markdown('<div class="hero-title" style="font-size:2rem">🎙️ Live AI Technical Interview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time voice & camera mock interview with adaptive AI questioning</div>', unsafe_allow_html=True)
    st.markdown("---")

    col_setup, col_info = st.columns([3, 2], gap="large")

    with col_setup:
        st.markdown('<div class="card-title">🎯 Interview Topic & Domain</div>', unsafe_allow_html=True)
        cur_topic = st.session_state.live_topic
        topic_idx = TOPIC_OPTIONS.index(cur_topic) if cur_topic in TOPIC_OPTIONS else 0
        selected_topic = st.selectbox("Select Domain / Technology", TOPIC_OPTIONS, index=topic_idx)
        st.session_state.live_topic = selected_topic

        actual_topic = selected_topic
        if selected_topic == "Custom Topic":
            custom_input = st.text_input(
                "Enter Custom Topic / Specialization",
                value=st.session_state.live_custom_topic,
                placeholder="e.g. Distributed Systems, Kubernetes, Go, Rust, System Design..."
            )
            st.session_state.live_custom_topic = custom_input
            if custom_input.strip():
                actual_topic = custom_input.strip()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="card-title">📈 Starting Difficulty</div>', unsafe_allow_html=True)
            diff_options = ["Beginner", "Intermediate", "Advanced"]
            cur_diff = st.session_state.live_difficulty
            diff_idx = diff_options.index(cur_diff) if cur_diff in diff_options else 1
            difficulty = st.selectbox(
                "Initial Difficulty",
                diff_options,
                index=diff_idx,
                help="The AI interviewer will adapt questions up or down based on your performance."
            )
            st.session_state.live_difficulty = difficulty

        with col2:
            st.markdown('<div class="card-title">🔢 Interview Length</div>', unsafe_allow_html=True)
            len_options = [3, 5, 8, 10]
            cur_len = st.session_state.live_target_questions
            len_idx = len_options.index(cur_len) if cur_len in len_options else 1
            q_count = st.selectbox("Number of Questions", len_options, index=len_idx)
            st.session_state.live_target_questions = q_count

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎥 Video & Media Settings</div>', unsafe_allow_html=True)
        cam_enabled = st.checkbox(
            "📷 Enable Camera preview during interview",
            value=st.session_state.live_enable_camera,
            help="Enables candidate mirror feed. Evaluation is strictly based on spoken answers."
        )
        st.session_state.live_enable_camera = cam_enabled

        st.markdown("<br>", unsafe_allow_html=True)
        action_col1, action_col2 = st.columns([3, 1], gap="medium")
        with action_col1:
            if st.button("🚀  Start Live AI Interview", use_container_width=True):
                _start_new_live_interview(actual_topic, difficulty, q_count)
        with action_col2:
            if st.button("📜 Past Interviews", use_container_width=True, type="secondary"):
                st.session_state.interview_step = "history"
                st.rerun()

    with col_info:
        st.markdown(f"""
        <div class="card" style="padding:1.75rem; border-left:4px solid var(--accent)">
            <div style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700; color:var(--text); margin-bottom:0.75rem">
                🎙️ How the Live AI Interview Works
            </div>
            <div style="font-size:0.85rem; color:var(--text-muted); line-height:1.8">
                1. <strong>AI Spoken Question:</strong> The AI Interviewer asks an adaptive question generated specifically for <em>{actual_topic}</em>.<br>
                2. <strong>Listen or Read:</strong> Hear the AI voice or read the question prompt.<br>
                3. <strong>Voice / Text Answer:</strong> Use your microphone to speak your answer, or type into the response box.<br>
                4. <strong>Live Speech-to-Text:</strong> Your speech is automatically transcribed in real time via Whisper.<br>
                5. <strong>Multi-Dimensional Scoring:</strong> Get immediate feedback on technical accuracy, understanding, and clarity.<br>
                6. <strong>Dynamic Adaptation:</strong> Subsequent questions adapt dynamically based on your performance.
            </div>
            <div style="margin-top:1.5rem; display:flex; gap:0.5rem; flex-wrap:wrap">
                <span class="badge badge-purple">Speech-to-Text</span>
                <span class="badge badge-cyan">Text-to-Speech</span>
                <span class="badge badge-green">Adaptive AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _start_new_live_interview(topic: str, difficulty: str, target_questions: int):
    """Initialize state and generate Question 1."""
    st.session_state.live_history = []
    st.session_state.live_final_report = None
    st.session_state.live_last_eval = None
    st.session_state.live_user_answer = ""
    st.session_state.live_start_timestamp = time.time()
    st.session_state.live_saved_session_id = None
    st.session_state.live_ai_status = "Speaking"

    with st.spinner(f"Preparing AI Interviewer and generating Question 1 for {topic}…"):
        q1 = generate_live_question(
            topic=topic,
            difficulty=difficulty,
            question_type="Conceptual",
            previous_evaluation=None,
            previous_questions=[],
            question_number=1,
        )
        st.session_state.live_current_q = q1
        st.session_state.live_tts_audio = _generate_tts_audio(q1.get("question", ""))
        st.session_state.live_step = "room"
        st.rerun()


# ─── SCREEN 2: Live Room (Side-by-Side Studio) ──────────────────────────────────

def _render_live_room_screen():
    q = st.session_state.get("live_current_q", {})
    history = st.session_state.get("live_history", [])
    total_q = st.session_state.get("live_target_questions", 5)
    current_num = len(history) + 1
    topic = st.session_state.get("live_topic", "Technical")
    if topic == "Custom Topic" and st.session_state.get("live_custom_topic"):
        topic = st.session_state.get("live_custom_topic")
    diff = q.get("difficulty", st.session_state.get("live_difficulty", "Intermediate"))
    concept_name = q.get("concept_name", "Core Mechanism")

    # Header & Badges
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(f'<div class="hero-title" style="font-size:1.75rem">🎙️ Live Interview Room: {topic}</div>', unsafe_allow_html=True)
    with h_col2:
        st.markdown("""<div style="height:10px"></div>""", unsafe_allow_html=True)
        if st.button("⏹️ End Interview", type="secondary", use_container_width=True):
            if history:
                _finalize_live_interview_report()
            else:
                st.session_state.live_step = "setup"
                st.rerun()

    # Progress bar & Metadata Badges
    progress_val = min(1.0, current_num / total_q)
    st.progress(progress_val)

    diff_badge = "badge-green" if diff == "Beginner" else ("badge-purple" if diff == "Intermediate" else "badge-cyan")
    status_label = st.session_state.get("live_ai_status", "Speaking")
    status_dot_class = "dot-active" if status_label == "Speaking" else ("dot-done" if status_label == "Listening" else "dot-pending")

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin:0.75rem 0 1.25rem 0; flex-wrap:wrap; gap:0.5rem">
        <div style="display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap">
            <span class="badge {diff_badge}">🎯 {diff}</span>
            <span class="badge badge-purple">Question {current_num} of {total_q}</span>
            <span class="badge badge-cyan">📌 {concept_name}</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem; background:var(--surface-2); padding:0.35rem 0.8rem; border-radius:6px; border:1px solid var(--border); font-size:0.8rem">
            <div class="status-dot {status_dot_class}"></div>
            <span>Status: <strong>AI {status_label}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Dual Panel Studio ───
    left_col, right_col = st.columns([1, 1], gap="large")

    # ─── LEFT: AI Interviewer Panel ───
    with left_col:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem">
            <div style="font-size:1.8rem">🤖</div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.1rem; color:var(--text)">
                    AI Interviewer
                </div>
                <div style="font-size:0.75rem; color:var(--accent-glow)">
                    Senior Staff Technical Interviewer
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Question Box
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid var(--accent); min-height: 220px; display:flex; flex-direction:column; justify-content:center">
            <div class="card-title">Question {current_num}</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; line-height:1.6; color:var(--text)">
                {q.get('question')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Voice audio player
        tts_audio = st.session_state.get("live_tts_audio")
        if tts_audio:
            st.audio(tts_audio, format="audio/mp3", autoplay=True)

        audio_btn_col1, audio_btn_col2 = st.columns([1, 1])
        with audio_btn_col1:
            if st.button("🔊 Replay Question Voice", use_container_width=True, type="secondary"):
                st.session_state.live_ai_status = "Speaking"
                st.rerun()

    # ─── RIGHT: Candidate Panel ───
    with right_col:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem">
            <div style="display:flex; align-items:center; gap:0.75rem">
                <div style="font-size:1.8rem">👤</div>
                <div>
                    <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.1rem; color:var(--text)">
                        Candidate (You)
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-muted)">
                        Microphone Active • Camera Ready
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Optional Camera Feed
        if st.session_state.get("live_enable_camera"):
            # Embed live mirror feed
            components.html("""
            <div style="position:relative; width:100%; border-radius:10px; overflow:hidden; background:#111; border:1px solid #7c3aed">
                <video id="liveCam" autoplay playsinline muted style="width:100%; height:180px; object-fit:cover; transform:scaleX(-1)"></video>
                <div style="position:absolute; bottom:8px; right:10px; background:rgba(0,0,0,0.6); padding:2px 8px; border-radius:4px; font-size:11px; color:#10b981; font-family:monospace">
                    ● LIVE
                </div>
            </div>
            <script>
            navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                .then(stream => { document.getElementById('liveCam').srcObject = stream; })
                .catch(err => {
                    document.getElementById('liveCam').outerHTML = '<div style="height:180px;display:flex;align-items:center;justify-content:center;color:#7070a0;font-size:12px;font-family:sans-serif">Camera preview unavailable or permission denied</div>';
                });
            </script>
            """, height=195)
        else:
            st.markdown("""
            <div class="card" style="padding:1.5rem; text-align:center; height:180px; display:flex; flex-direction:column; justify-content:center; align-items:center">
                <div style="font-size:2rem; margin-bottom:0.4rem">🎙️</div>
                <div style="font-weight:700; font-size:0.95rem; color:var(--text)">Microphone Interview Mode</div>
                <div style="color:var(--text-muted); font-size:0.78rem; margin-top:0.3rem">
                    Speak your answer below using the audio recorder, or type your response.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Voice Input Component
        recorded_audio = st.audio_input("🎤 Record your Answer (Click to speak)", key=f"rec_audio_{current_num}")

        # When candidate speaks, transcribe audio via Whisper
        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()
            audio_sig = f"q_{current_num}_{len(audio_bytes)}"
            if st.session_state.get("live_last_audio_sig") != audio_sig:
                st.session_state.live_ai_status = "Listening"
                with st.spinner("Transcribing your spoken answer via Whisper…"):
                    transcript_text = _transcribe_audio_bytes(audio_bytes)
                    if transcript_text:
                        st.session_state.live_user_answer = transcript_text
                        st.session_state.live_last_audio_sig = audio_sig
                        st.rerun()

        # Transcript & Typed Fallback
        st.markdown('<div class="card-title" style="margin-top:0.75rem">📝 Answer Transcript (Spoken or Typed)</div>', unsafe_allow_html=True)
        user_answer = st.text_area(
            "Answer Input",
            value=st.session_state.get("live_user_answer", ""),
            height=120,
            placeholder="Your spoken answer will appear here automatically. You can also type directly or edit your transcript before submitting…",
            label_visibility="collapsed",
            key=f"text_ans_{current_num}"
        )
        st.session_state.live_user_answer = user_answer

        # Submit Answer Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡  Submit Answer", use_container_width=True):
            if not user_answer.strip():
                st.warning("Please speak into the microphone or type your answer before submitting.")
            else:
                _evaluate_submitted_live_answer(user_answer.strip(), q, topic, diff, concept_name)


def _evaluate_submitted_live_answer(user_answer: str, current_q: dict, topic: str, difficulty: str, concept_name: str):
    """Evaluate candidate answer using LLM and transition to evaluation screen."""
    st.session_state.live_ai_status = "Evaluating"
    with st.spinner("AI Interviewer is evaluating your answer…"):
        eval_result = evaluate_live_answer(
            topic=topic,
            concept_name=concept_name,
            difficulty=difficulty,
            question=current_q.get("question", ""),
            expected_points=current_q.get("expected_key_points", []),
            user_answer=user_answer,
        )

        history_item = {
            "question": current_q.get("question"),
            "question_type": current_q.get("question_type", "Conceptual"),
            "concept_name": concept_name,
            "difficulty": difficulty,
            "user_answer": user_answer,
            "evaluation": eval_result,
        }
        st.session_state.live_history.append(history_item)
        st.session_state.live_last_eval = eval_result
        st.session_state.live_user_answer = ""
        st.session_state.live_last_audio_sig = None
        st.session_state.live_step = "evaluation"
        st.rerun()


# ─── SCREEN 3: Live Evaluation ──────────────────────────────────────────────────

def _render_live_evaluation_screen():
    history = st.session_state.get("live_history", [])
    if not history:
        st.session_state.live_step = "setup"
        st.rerun()
        return

    last_item = history[-1]
    eval_res = last_item.get("evaluation", {})
    total_q = st.session_state.get("live_target_questions", 5)
    is_last = len(history) >= total_q

    score = eval_res.get("score", 5)
    cu = eval_res.get("concept_understanding_score", score)
    ta = eval_res.get("technical_accuracy_score", score)
    cd = eval_res.get("clarity_depth_score", score)
    rating = eval_res.get("rating", "Partially Correct")

    score_color = "var(--success)" if score >= 8 else ("var(--warning)" if score >= 5 else "var(--danger)")
    badge_class = "badge-green" if score >= 8 else ("badge-purple" if score >= 5 else "badge-cyan")

    st.markdown(f'<div class="hero-title" style="font-size:1.8rem">📊 Question {len(history)} Evaluation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{last_item.get("concept_name")} • {last_item.get("difficulty")} Level</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Score Banner
    st.markdown(f"""
    <div class="card" style="display:flex; justify-content:space-between; align-items:center; padding:1.5rem">
        <div>
            <div class="card-title">Interviewer Assessment</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:var(--text)">
                Rating: <span class="badge {badge_class}" style="font-size:0.9rem">{rating}</span>
            </div>
            <div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.3rem">
                {eval_res.get('explanation_quality', 'Evaluation complete.')}
            </div>
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:{score_color}">
            {score} <span style="font-size:1.2rem; color:var(--text-muted)">/ 10</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-Dimension Score Breakdown
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1rem">
            <div style="font-size:0.78rem; color:var(--text-muted)">Concept Understanding</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:var(--accent-2); margin-top:0.3rem">
                {cu} / 10
            </div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1rem">
            <div style="font-size:0.78rem; color:var(--text-muted)">Technical Accuracy</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:var(--accent-glow); margin-top:0.3rem">
                {ta} / 10
            </div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1rem">
            <div style="font-size:0.78rem; color:var(--text-muted)">Clarity & Depth</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:var(--success); margin-top:0.3rem">
                {cd} / 10
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Question & Candidate Answer recap
    with st.expander("🔎 Review Question & Spoken Answer", expanded=False):
        st.markdown(f"""
        <div style="margin-bottom:0.75rem">
            <strong style="color:var(--text)">Question:</strong> {last_item.get('question')}
        </div>
        <div class="transcript-box" style="max-height:140px">
            <strong>Candidate Answer:</strong> {last_item.get('user_answer')}
        </div>
        """, unsafe_allow_html=True)

    # Feedback Details
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        corrects = eval_res.get("correct_points", [])
        c_items = "".join([f"<li style='margin-bottom:0.4rem'>{p}</li>" for p in corrects]) if corrects else "<li>Basic engagement with topic</li>"
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid var(--success)">
            <div class="card-title" style="color:var(--success)">✓ What You Got Right</div>
            <ul style="padding-left:1.2rem; font-size:0.83rem; line-height:1.6; color:var(--text)">
                {c_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        missings = eval_res.get("missing_points", [])
        m_items = "".join([f"<li style='margin-bottom:0.4rem'>{p}</li>" for p in missings]) if missings else "<li>Covered core expectations thoroughly</li>"
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid var(--warning)">
            <div class="card-title" style="color:var(--warning)">⚠ Missing Points & Gaps</div>
            <ul style="padding-left:1.2rem; font-size:0.83rem; line-height:1.6; color:var(--text)">
                {m_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Misconceptions if any
    misconceptions = eval_res.get("misconceptions", [])
    if misconceptions:
        misc_items = "".join([f"<li style='margin-bottom:0.4rem'>{m}</li>" for m in misconceptions])
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid var(--danger); margin-top:0.75rem">
            <div class="card-title" style="color:var(--danger)">🧠 Misconceptions Corrected</div>
            <ul style="padding-left:1.2rem; font-size:0.83rem; line-height:1.6; color:var(--text)">
                {misc_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Interviewer Feedback Summary
    feedback_text = eval_res.get("feedback_summary", "")
    if feedback_text:
        st.markdown(f"""
        <div class="card" style="background:var(--surface-2); margin-top:0.75rem">
            <div class="card-title">💬 Interviewer Coaching Note</div>
            <div style="font-size:0.85rem; line-height:1.7; color:var(--text)">
                {feedback_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Action / Next Question Button
    st.markdown("<br>", unsafe_allow_html=True)
    btn_label = "📊 View Final Performance Scorecard →" if is_last else f"Next Adaptive Question ({len(history) + 1}/{total_q}) →"
    if st.button(btn_label, use_container_width=True):
        if is_last:
            _finalize_live_interview_report()
        else:
            _prepare_adaptive_next_live_question()


def _prepare_adaptive_next_live_question():
    """Adaptively generate next question based on candidate performance."""
    with st.spinner("AI Interviewer is adapting difficulty and generating next question…"):
        history = st.session_state.live_history
        last_eval = st.session_state.live_last_eval
        topic = st.session_state.get("live_topic", "Python")
        if topic == "Custom Topic" and st.session_state.get("live_custom_topic"):
            topic = st.session_state.get("live_custom_topic")
        current_diff = history[-1].get("difficulty", "Intermediate")

        next_concept, next_diff, q_type = select_adaptive_live_next(
            topic=topic,
            interview_history=history,
            last_evaluation=last_eval,
            current_difficulty=current_diff
        )

        prev_q_texts = [h.get("question") for h in history]

        next_q = generate_live_question(
            topic=topic,
            difficulty=next_diff,
            question_type=q_type,
            previous_evaluation=last_eval,
            previous_questions=prev_q_texts,
            question_number=len(history) + 1
        )

        st.session_state.live_current_q = next_q
        st.session_state.live_tts_audio = _generate_tts_audio(next_q.get("question", ""))
        st.session_state.live_ai_status = "Speaking"
        st.session_state.live_step = "room"
        st.rerun()


def _finalize_live_interview_report():
    """Generate final report and persist live interview session."""
    with st.spinner("Compiling comprehensive interview scorecard…"):
        history = st.session_state.live_history
        topic = st.session_state.get("live_topic", "Technical")
        if topic == "Custom Topic" and st.session_state.get("live_custom_topic"):
            topic = st.session_state.get("live_custom_topic")

        start_t = st.session_state.get("live_start_timestamp")
        duration_sec = int(time.time() - start_t) if start_t else 120
        d_m, d_s = divmod(duration_sec, 60)
        duration_str = f"{d_m}m {d_s}s"

        report = generate_live_final_report(
            interview_history=history,
            topic=topic,
            duration_str=duration_str
        )

        st.session_state.live_final_report = report

        # Persist session to history with interview_mode = "live"
        session_data = {
            "interview_mode": "live",
            "video_title": f"Live Interview: {topic}",
            "topic": topic,
            "difficulty": st.session_state.get("live_difficulty", "Intermediate"),
            "duration": duration_str,
            "history": history,
            "report": report,
            "created_at": datetime.now().isoformat(),
        }
        saved_rec = save_interview_session(session_data)
        st.session_state.live_saved_session_id = saved_rec.get("id")

        st.session_state.live_step = "report"
        st.rerun()


# ─── SCREEN 4: Final Report ─────────────────────────────────────────────────────

def _render_live_report_screen():
    report = st.session_state.get("live_final_report")
    history = st.session_state.get("live_history", [])
    topic = st.session_state.get("live_topic", "Technical")
    if topic == "Custom Topic" and st.session_state.get("live_custom_topic"):
        topic = st.session_state.get("live_custom_topic")

    if not report:
        st.session_state.live_step = "setup"
        st.rerun()
        return

    score = report.get("overall_score", 0.0)
    cu = report.get("concept_understanding_score", score)
    ta = report.get("technical_accuracy_score", score)
    cd = report.get("clarity_depth_score", score)
    duration = report.get("duration", "N/A")

    score_color = "var(--success)" if score >= 7.5 else ("var(--warning)" if score >= 5 else "var(--danger)")
    hire_recommendation = "Strong Hire" if score >= 8.5 else ("Hire / Promising" if score >= 7.0 else ("Borderline / Practice Required" if score >= 5.0 else "Needs Fundamental Study"))

    st.markdown('<div class="hero-title" style="font-size:2rem">🏆 Live Interview Performance Scorecard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">Domain: {topic} • Completed {len(history)} Adaptive Questions • Duration: {duration}</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Hero Result Card
    st.markdown(f"""
    <div class="card" style="display:flex; justify-content:space-between; align-items:center; padding:2rem; margin-bottom:1.5rem">
        <div>
            <span class="badge badge-purple" style="margin-bottom:0.5rem">Live AI Mock Interview</span>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:var(--text)">
                {hire_recommendation}
            </div>
            <div style="color:var(--text-muted); font-size:0.85rem; margin-top:0.4rem">
                Domain Mastery: <strong>{topic}</strong> • Questions Answered: <strong>{len(history)}</strong>
            </div>
        </div>
        <div style="text-align:right">
            <div style="font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800; color:{score_color}; line-height:1">
                {score}
            </div>
            <div style="font-size:0.85rem; color:var(--text-muted); font-weight:600">OUT OF 10</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-dimensional Metric Cards
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1.25rem">
            <div style="font-size:0.8rem; color:var(--text-muted)">Concept Understanding</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; color:var(--accent-2); margin-top:0.4rem">
                {cu} <span style="font-size:0.9rem; color:var(--text-muted)">/ 10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1.25rem">
            <div style="font-size:0.8rem; color:var(--text-muted)">Technical Accuracy</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; color:var(--accent-glow); margin-top:0.4rem">
                {ta} <span style="font-size:0.9rem; color:var(--text-muted)">/ 10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1.25rem">
            <div style="font-size:0.8rem; color:var(--text-muted)">Clarity & Depth</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; color:var(--success); margin-top:0.4rem">
                {cd} <span style="font-size:0.9rem; color:var(--text-muted)">/ 10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Strengths and Weaknesses
    col_str, col_weak = st.columns(2, gap="medium")
    with col_str:
        strongs = report.get("strong_concepts", [])
        if strongs:
            s_html = "".join([f"<div style='margin-bottom:0.4rem'><span class='badge badge-green' style='margin-right:0.4rem'>✓</span><strong>{s}</strong></div>" for s in strongs])
        else:
            s_html = "<div style='color:var(--text-muted); font-size:0.85rem'>Continue practicing to build solid strong areas.</div>"
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid var(--success)">
            <div class="card-title" style="color:var(--success)">🌟 Strong Areas Demonstrated</div>
            {s_html}
        </div>
        """, unsafe_allow_html=True)

    with col_weak:
        weaks = report.get("weak_concepts", [])
        if weaks:
            w_html = "".join([f"<div style='margin-bottom:0.4rem'><span class='badge badge-purple' style='margin-right:0.4rem'>⚠</span><strong>{w}</strong></div>" for w in weaks])
        else:
            w_html = "<div style='color:var(--text-muted); font-size:0.85rem'>No major weak concepts detected!</div>"
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid var(--warning)">
            <div class="card-title" style="color:var(--warning)">🎯 Areas for Focused Improvement</div>
            {w_html}
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    recs = report.get("recommended_topics", [])
    if recs:
        st.markdown('<div class="card-title" style="margin-top:1.5rem">📚 Actionable Practice Recommendations</div>', unsafe_allow_html=True)
        r_cols = st.columns(min(len(recs), 3))
        for i, rec in enumerate(recs):
            c_idx = i % len(r_cols)
            prio = rec.get("priority", "Medium")
            prio_badge = "badge-cyan" if prio == "High" else "badge-purple"
            with r_cols[c_idx]:
                st.markdown(f"""
                <div class="card" style="padding:1rem; height:100%">
                    <span class="badge {prio_badge}" style="margin-bottom:0.5rem">{prio} Priority</span>
                    <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; color:var(--text); margin-bottom:0.3rem">
                        {rec.get('topic')}
                    </div>
                    <div style="font-size:0.8rem; color:var(--text-muted); line-height:1.5">
                        {rec.get('action')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Question-by-Question breakdown
    st.markdown('<div class="card-title" style="margin-top:1.5rem">📋 Question-by-Question Breakdown</div>', unsafe_allow_html=True)
    for q_idx, item in enumerate(history):
        ev = item.get("evaluation", {})
        item_score = ev.get("score", 0)
        item_color = "var(--success)" if item_score >= 7.5 else ("var(--warning)" if item_score >= 5 else "var(--danger)")

        with st.expander(f"Q{q_idx + 1}: {item.get('concept_name')} — Score: {item_score}/10 ({item.get('difficulty')})", expanded=False):
            st.markdown(f"""
            <div style="margin-bottom:0.75rem">
                <strong style="color:var(--text)">Question:</strong> {item.get('question')}
            </div>
            <div style="background:var(--surface-2); padding:0.75rem; border-radius:6px; font-size:0.83rem; margin-bottom:0.75rem">
                <strong>Candidate:</strong> {item.get('user_answer')}
            </div>
            <div style="font-size:0.82rem; color:var(--accent-glow)">
                <strong>Interviewer Feedback:</strong> {ev.get('feedback_summary')}
            </div>
            """, unsafe_allow_html=True)

    # Bottom Actions
    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns(3, gap="medium")
    with b_col1:
        if st.button("🔄 Retake Live Interview", use_container_width=True):
            st.session_state.live_step = "setup"
            st.rerun()
    with b_col2:
        if st.button("📜 View Past Interviews", use_container_width=True, type="secondary"):
            st.session_state.interview_step = "history"
            st.rerun()
    with b_col3:
        if st.button("🎬 Video Intelligence", use_container_width=True, type="secondary"):
            st.session_state.nav_mode = "video_intelligence"
            st.rerun()
