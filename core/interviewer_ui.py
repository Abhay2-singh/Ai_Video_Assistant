import streamlit as st
import time
from core.interviewer import (
    extract_concepts,
    get_grounded_context,
    generate_interview_question,
    evaluate_user_answer,
    select_adaptive_next_concept,
    generate_final_report,
)
from core.interview_storage import (
    get_saved_videos,
    get_video_by_id,
    save_processed_video_record,
    update_video_concepts,
    save_interview_session,
    get_past_interviews,
)
from core.vector_store import load_vector_store, build_vector_store
from core.live_interviewer_ui import render_live_interviewer_interface


def init_interview_state():
    """Ensure all required interview session state variables exist."""
    defaults = {
        "nav_mode": "video_intelligence",
        "interview_submode": "video",  # 'video' or 'live'
        "selected_video_id": None,
        "interview_step": "setup",  # 'setup', 'question', 'evaluation', 'report', 'history'
        "interview_difficulty": "Intermediate",
        "interview_target_questions": 5,
        "interview_history": [],
        "interview_current_q": None,
        "interview_current_concept": None,
        "interview_current_context": "",
        "interview_current_timestamp": "",
        "interview_last_eval": None,
        "interview_final_report": None,
        "interview_user_answer": "",
        "interview_saved_session_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_interviewer_interface():
    """Main entry point for the Adaptive AI Interviewer UI."""
    init_interview_state()

    # Step routing
    step = st.session_state.get("interview_step", "setup")
    if step == "history":
        _render_history_screen()
        return

    # Dual Mode Selector Banner
    col_mode1, col_mode2 = st.columns(2, gap="medium")
    with col_mode1:
        is_vid = st.session_state.get("interview_submode", "video") == "video"
        if st.button("📹  Video-Based Interview", use_container_width=True, type="primary" if is_vid else "secondary"):
            st.session_state.interview_submode = "video"
            st.rerun()
    with col_mode2:
        is_live = st.session_state.get("interview_submode", "video") == "live"
        if st.button("🎙️  Live AI Interview (Voice & Camera)", use_container_width=True, type="primary" if is_live else "secondary"):
            st.session_state.interview_submode = "live"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Route to Live AI Interview
    if st.session_state.get("interview_submode", "video") == "live":
        render_live_interviewer_interface()
        return

    # Route to Video-Based Interview
    if step == "setup":
        _render_setup_screen()
    elif step == "question":
        _render_question_screen()
    elif step == "evaluation":
        _render_evaluation_screen()
    elif step == "report":
        _render_report_screen()


# ─── Helper: Get Available Video Data ───────────────────────────────────────────

def _get_active_video():
    """Retrieve video transcript and details either from current result or saved storage."""
    saved_videos = get_saved_videos()

    # If current pipeline has a result, ensure it's saved in storage
    if st.session_state.get("result"):
        res = st.session_state.result
        title = res.get("title", "Untitled Session")
        transcript = res.get("transcript", "")
        summary = res.get("summary", "")
        if transcript:
            rec = save_processed_video_record(title=title, transcript=transcript, summary=summary)
            if not st.session_state.get("selected_video_id"):
                st.session_state.selected_video_id = rec["id"]
            saved_videos = get_saved_videos()

    selected_id = st.session_state.get("selected_video_id")
    if selected_id:
        video = get_video_by_id(selected_id)
        if video:
            return video, saved_videos

    if saved_videos:
        return saved_videos[0], saved_videos

    return None, []


# ─── SCREEN 1: Setup & Video Selection ──────────────────────────────────────────

def _render_setup_screen():
    st.markdown('<div class="hero-title" style="font-size:2rem">🎙️ Adaptive AI Interviewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Dynamic technical mock interview grounded in video concepts</div>', unsafe_allow_html=True)
    st.markdown("---")

    active_video, saved_videos = _get_active_video()

    if not active_video:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem">
            <div style="font-size:3rem;margin-bottom:1rem">🎬</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
                No Processed Videos Found
            </div>
            <div style="color:var(--text-muted);font-size:0.85rem;max-width:440px;margin:0 auto 1.5rem auto;line-height:1.7">
                First process a YouTube video or audio file in <strong>Video Intelligence</strong> mode to extract transcript and concepts for your mock interview.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("← Go to Video Intelligence", use_container_width=True):
                st.session_state.nav_mode = "video_intelligence"
                st.rerun()
        with col_b:
            if st.button("📜 View Past Interview History", use_container_width=True):
                st.session_state.interview_step = "history"
                st.rerun()
        return

    # Video Selection Card
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📹 Selected Video for Interview</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            {active_video.get('title', 'Untitled Video')}
        </div>
        <div style="font-size:0.8rem;color:var(--text-muted)">
            Transcript length: {len(active_video.get('transcript', ''))} characters • Saved on: {active_video.get('created_at', '')[:10]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(saved_videos) > 1:
        options = {v["id"]: f"{v['title']} ({v.get('created_at', '')[:10]})" for v in saved_videos}
        current_idx = list(options.keys()).index(active_video["id"]) if active_video["id"] in options else 0
        new_sel = st.selectbox(
            "Switch to another processed video:",
            options=list(options.keys()),
            format_func=lambda k: options[k],
            index=current_idx
        )
        if new_sel != active_video["id"]:
            st.session_state.selected_video_id = new_sel
            st.rerun()

    # Concept Extraction
    concepts = active_video.get("concepts", [])
    if not concepts:
        with st.spinner("Extracting key concepts taught in this video…"):
            concepts = extract_concepts(active_video.get("transcript", ""), max_concepts=6)
            update_video_concepts(active_video["id"], concepts)
            active_video["concepts"] = concepts

    # Concept Display
    st.markdown("""
    <div class="card-title" style="margin-top:1.5rem">
        🧠 Extracted Technical Concepts from Video
    </div>
    """, unsafe_allow_html=True)

    c_cols = st.columns(min(len(concepts), 3) if concepts else 1)
    for i, concept in enumerate(concepts):
        col_idx = i % len(c_cols)
        diff = concept.get("difficulty", "Intermediate")
        badge_class = "badge-green" if diff == "Beginner" else ("badge-purple" if diff == "Intermediate" else "badge-cyan")
        with c_cols[col_idx]:
            st.markdown(f"""
            <div class="card" style="margin-bottom:0.8rem;padding:1rem">
                <span class="badge {badge_class}" style="margin-bottom:0.5rem">{diff}</span>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:var(--text);margin-bottom:0.3rem">
                    {concept.get('name')}
                </div>
                <div style="font-size:0.78rem;color:var(--text-muted);line-height:1.5">
                    {concept.get('description')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Configuration Form
    st.markdown("---")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="card-title">🎯 Starting Difficulty</div>', unsafe_allow_html=True)
        diff_options = ["Beginner", "Intermediate", "Advanced"]
        cur_diff = st.session_state.interview_difficulty
        difficulty = st.selectbox(
            "Initial Difficulty",
            diff_options,
            index=diff_options.index(cur_diff) if cur_diff in diff_options else 1,
            help="The interview will adapt up or down from here based on your answers."
        )
        st.session_state.interview_difficulty = difficulty

    with col2:
        st.markdown('<div class="card-title">🔢 Interview Length</div>', unsafe_allow_html=True)
        len_options = [3, 5, 8, 10]
        cur_len = st.session_state.interview_target_questions
        q_count = st.selectbox(
            "Number of Questions",
            len_options,
            index=len_options.index(cur_len) if cur_len in len_options else 1
        )
        st.session_state.interview_target_questions = q_count

    st.markdown("<br>", unsafe_allow_html=True)
    action_col1, action_col2 = st.columns([3, 1], gap="medium")

    with action_col1:
        if st.button("🚀  Start Adaptive Mock Interview", use_container_width=True):
            _start_new_interview(active_video, concepts, difficulty)

    with action_col2:
        if st.button("📜 Past Interviews", use_container_width=True, type="secondary"):
            st.session_state.interview_step = "history"
            st.rerun()


def _start_new_interview(video: dict, concepts: list, initial_difficulty: str):
    """Initialize interview state and generate Question 1."""
    st.session_state.interview_history = []
    st.session_state.interview_final_report = None
    st.session_state.interview_last_eval = None
    st.session_state.interview_user_answer = ""

    with st.spinner("Preparing RAG context and generating Question 1…"):
        # Ensure fresh vector store isolated to this specific video
        if "interview_vector_store" not in st.session_state or st.session_state.get("interview_vector_store_video_id") != video.get("id"):
            vector_store = build_vector_store(video.get("transcript", ""))
            st.session_state.interview_vector_store = vector_store
            st.session_state.interview_vector_store_video_id = video.get("id")
        else:
            vector_store = st.session_state.interview_vector_store

        first_concept = concepts[0]
        context, ts_range, _ = get_grounded_context(
            vector_store,
            first_concept["name"],
            video.get("transcript", "")
        )

        q1 = generate_interview_question(
            concept=first_concept,
            difficulty=initial_difficulty,
            rag_context=context,
            timestamp_range=ts_range,
            question_type="Conceptual"
        )

        st.session_state.interview_current_q = q1
        st.session_state.interview_current_concept = first_concept
        st.session_state.interview_current_context = context
        st.session_state.interview_current_timestamp = ts_range
        st.session_state.interview_step = "question"
        st.rerun()


# ─── SCREEN 2: Active Question Screen ──────────────────────────────────────────

def _render_question_screen():
    q = st.session_state.get("interview_current_q")
    if not q:
        st.session_state.interview_step = "setup"
        st.rerun()
        return

    history = st.session_state.get("interview_history", [])
    total_q = st.session_state.get("interview_target_questions", 5)
    current_num = len(history) + 1

    # Header Bar
    progress_val = (current_num - 1) / total_q
    st.progress(progress_val)

    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
    with top_col1:
        st.markdown(f'<div class="hero-sub">Question {current_num} of {total_q}</div>', unsafe_allow_html=True)
    with top_col2:
        diff = q.get("difficulty", "Intermediate")
        badge_class = "badge-green" if diff == "Beginner" else ("badge-purple" if diff == "Intermediate" else "badge-cyan")
        st.markdown(f'<div style="text-align:right"><span class="badge {badge_class}">Level: {diff}</span></div>', unsafe_allow_html=True)
    with top_col3:
        qtype = q.get("question_type", "Conceptual")
        st.markdown(f'<div style="text-align:right"><span class="badge badge-purple">{qtype}</span></div>', unsafe_allow_html=True)

    # Concept & Timestamp Grounding Tag
    concept_name = q.get("concept_name", "Core Topic")
    ts = q.get("timestamp_range", "00:00 – 05:00")
    st.markdown(f"""
    <div style="display:flex;gap:0.75rem;margin:0.75rem 0;flex-wrap:wrap">
        <span class="badge badge-cyan">📌 Concept: {concept_name}</span>
        <span class="badge badge-green">⏱️ Video Grounding: {ts}</span>
    </div>
    """, unsafe_allow_html=True)

    # Main Question Card
    st.markdown(f"""
    <div class="card" style="border-left: 4px solid var(--accent); margin-bottom: 1.5rem">
        <div class="card-title">Interviewer Question</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;line-height:1.6;color:var(--text)">
            {q.get('question')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Optional Grounding Context Expander
    with st.expander("🔎 View Grounding Video Context (Reference)", expanded=False):
        st.markdown(f"""
        <div class="transcript-box" style="max-height:180px">
{st.session_state.get('interview_current_context', 'Context retrieved from video transcript.')}
        </div>
        """, unsafe_allow_html=True)

    # Answer Input Box
    st.markdown('<div class="card-title" style="margin-top:1rem">✍️ Your Answer</div>', unsafe_allow_html=True)
    user_answer = st.text_area(
        "Candidate Answer",
        value=st.session_state.get("interview_user_answer", ""),
        height=180,
        placeholder="Type your explanation here. Mention why and how it works, technical mechanisms, and practical considerations…",
        label_visibility="collapsed"
    )

    btn_col1, btn_col2 = st.columns([4, 1], gap="medium")
    with btn_col1:
        submit_btn = st.button("⚡  Submit Answer", use_container_width=True)
    with btn_col2:
        early_finish = st.button("⏹️ Finish Early", use_container_width=True, type="secondary")

    if early_finish:
        if history:
            _finalize_interview_report()
        else:
            st.session_state.interview_step = "setup"
            st.rerun()

    if submit_btn:
        if not user_answer.strip():
            st.warning("Please provide an answer before submitting.")
        else:
            _evaluate_submitted_answer(user_answer.strip(), q)


def _evaluate_submitted_answer(user_answer: str, current_q: dict):
    """Evaluate user answer using LLM and transition to evaluation screen."""
    with st.spinner("Evaluating your response against video ground truth…"):
        eval_result = evaluate_user_answer(
            question=current_q.get("question", ""),
            expected_points=current_q.get("expected_key_points", []),
            user_answer=user_answer,
            concept_name=current_q.get("concept_name", ""),
            difficulty=current_q.get("difficulty", "Intermediate"),
            rag_context=st.session_state.get("interview_current_context", "")
        )

        history_entry = {
            "question": current_q.get("question"),
            "question_type": current_q.get("question_type"),
            "difficulty": current_q.get("difficulty"),
            "concept_name": current_q.get("concept_name"),
            "timestamp_range": current_q.get("timestamp_range"),
            "expected_key_points": current_q.get("expected_key_points", []),
            "user_answer": user_answer,
            "evaluation": eval_result,
        }
        st.session_state.interview_history.append(history_entry)
        st.session_state.interview_last_eval = eval_result
        st.session_state.interview_user_answer = ""
        st.session_state.interview_step = "evaluation"
        st.rerun()


# ─── SCREEN 3: Answer Evaluation & Adaptive Next Step ──────────────────────────

def _render_evaluation_screen():
    history = st.session_state.get("interview_history", [])
    if not history:
        st.session_state.interview_step = "setup"
        st.rerun()
        return

    last_item = history[-1]
    ev = last_item.get("evaluation", {})
    score = ev.get("score", 5)
    rating = ev.get("rating", "Partially Correct")
    total_q = st.session_state.get("interview_target_questions", 5)
    is_last = len(history) >= total_q

    # Color code score
    score_badge_class = "badge-green" if score >= 8 else ("badge-purple" if score >= 5 else "badge-cyan")

    # Header Card with Score and Adaptive Banner
    action = ev.get("recommended_difficulty_action", "maintain")
    if score >= 8:
        adapt_msg = "⚡ Excellent answer! Raising challenge level with deeper practical / advanced concepts."
        banner_border = "var(--success)"
    elif score >= 5:
        adapt_msg = "🔍 Good intuition, but missing important nuances. Follow-up will test this clarification."
        banner_border = "var(--warning)"
    else:
        adapt_msg = "💡 Foundational gaps detected. Adapting next question to reinforce core prerequisites."
        banner_border = "var(--danger)"

    st.markdown(f"""
    <div class="card" style="border-left: 4px solid {banner_border}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem">
            <span class="card-title" style="margin:0">Evaluation Feedback</span>
            <span class="badge {score_badge_class}" style="font-size:0.85rem;padding:0.3rem 0.8rem">
                Score: {score} / 10 • {rating}
            </span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.88rem;color:var(--text);line-height:1.6">
            {ev.get('feedback_summary')}
        </div>
        <div style="margin-top:0.75rem;font-size:0.78rem;color:var(--accent-glow);font-weight:600">
            {adapt_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # What you nailed vs Missing points
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        corrects = ev.get("correct_points", [])
        correct_html = "".join([f"<li style='margin-bottom:0.4rem'>{p}</li>" for p in corrects]) or "<li>Attempted question.</li>"
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="card-title" style="color:var(--success)">✓ What You Answered Correctly</div>
            <ul style="font-size:0.83rem;color:var(--text);padding-left:1.2rem;line-height:1.6">
                {correct_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        missings = ev.get("missing_points", [])
        missing_html = "".join([f"<li style='margin-bottom:0.4rem'>{p}</li>" for p in missings]) or "<li>No major omissions detected.</li>"
        st.markdown(f"""
        <div class="card" style="height:100%">
            <div class="card-title" style="color:var(--warning)">⚠ What Was Missing / Needed</div>
            <ul style="font-size:0.83rem;color:var(--text);padding-left:1.2rem;line-height:1.6">
                {missing_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Misconceptions if any
    misconceptions = ev.get("misconceptions", [])
    if misconceptions:
        misc_html = "".join([f"<li style='margin-bottom:0.4rem'>{m}</li>" for m in misconceptions])
        st.markdown(f"""
        <div class="card" style="border-left:4px solid var(--danger);margin-top:1rem">
            <div class="card-title" style="color:var(--danger)">🧠 Misconceptions Detected</div>
            <ul style="font-size:0.83rem;color:var(--text);padding-left:1.2rem;line-height:1.6">
                {misc_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Next Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    next_btn_label = "📊  View Final Interview Report →" if is_last else f"Next Question ({len(history) + 1}/{total_q}) →"

    if st.button(next_btn_label, use_container_width=True):
        if is_last:
            _finalize_interview_report()
        else:
            _prepare_adaptive_next_question()


def _prepare_adaptive_next_question():
    """Determine the next adaptive question based on previous performance."""
    with st.spinner("Adapting difficulty and generating next question…"):
        active_video, _ = _get_active_video()
        concepts = active_video.get("concepts", [])
        history = st.session_state.interview_history
        last_eval = st.session_state.interview_last_eval
        current_diff = history[-1].get("difficulty", "Intermediate")

        next_concept, next_diff, q_type = select_adaptive_next_concept(
            all_concepts=concepts,
            interview_history=history,
            last_evaluation=last_eval,
            current_difficulty=current_diff
        )

        # Reuse cached vector store for this video
        if "interview_vector_store" in st.session_state and st.session_state.get("interview_vector_store_video_id") == active_video.get("id"):
            vector_store = st.session_state.interview_vector_store
        else:
            vector_store = build_vector_store(active_video.get("transcript", ""))
            st.session_state.interview_vector_store = vector_store
            st.session_state.interview_vector_store_video_id = active_video.get("id")

        context, ts_range, _ = get_grounded_context(
            vector_store,
            next_concept["name"],
            active_video.get("transcript", "")
        )

        prev_q_texts = [h.get("question") for h in history]

        next_q = generate_interview_question(
            concept=next_concept,
            difficulty=next_diff,
            rag_context=context,
            timestamp_range=ts_range,
            question_type=q_type,
            previous_evaluation=last_eval,
            previous_questions=prev_q_texts
        )

        st.session_state.interview_current_q = next_q
        st.session_state.interview_current_concept = next_concept
        st.session_state.interview_current_context = context
        st.session_state.interview_current_timestamp = ts_range
        st.session_state.interview_step = "question"
        st.rerun()


def _finalize_interview_report():
    """Generate and persist the final interview report."""
    with st.spinner("Analyzing performance patterns and creating revision report…"):
        active_video, _ = _get_active_video()
        concepts = active_video.get("concepts", [])
        history = st.session_state.interview_history

        report = generate_final_report(
            interview_history=history,
            all_concepts=concepts,
            video_title=active_video.get("title", "Video Session")
        )

        session_record = {
            "video_title": active_video.get("title", "Video Session"),
            "video_id": active_video.get("id"),
            "target_questions": len(history),
            "history": history,
            "report": report
        }
        saved_session = save_interview_session(session_record)
        st.session_state.interview_saved_session_id = saved_session.get("id")
        st.session_state.interview_final_report = report
        st.session_state.interview_step = "report"
        st.rerun()


# ─── SCREEN 4: Final Interview Report ──────────────────────────────────────────

def _render_report_screen():
    report = st.session_state.get("interview_final_report")
    if not report:
        st.session_state.interview_step = "setup"
        st.rerun()
        return

    active_video, _ = _get_active_video()
    overall = report.get("overall_score", 0.0)

    # Hero Banner
    st.markdown('<div class="hero-title" style="font-size:2rem">📊 Final Interview Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{active_video.get("title", "Video Session")} • Comprehensive Performance Breakdown</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Overall Score Card
    badge_color = "var(--success)" if overall >= 7.5 else ("var(--warning)" if overall >= 5 else "var(--danger)")
    tier_desc = (
        "High Mastery — Ready for Senior Technical Discussion"
        if overall >= 7.5
        else ("Moderate Understanding — Specific Gaps to Address" if overall >= 5 else "Foundational Stage — Key Prerequisites Need Review")
    )

    st.markdown(f"""
    <div class="card" style="text-align:center;padding:2rem;border-top:4px solid {badge_color}">
        <div class="card-title" style="justify-content:center">Overall Technical Score</div>
        <div style="font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;color:{badge_color};line-height:1">
            {overall} <span style="font-size:1.5rem;color:var(--text-muted)">/ 10</span>
        </div>
        <div style="margin-top:0.75rem;font-size:0.9rem;font-weight:600;color:var(--text)">
            {tier_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Concept Performance Breakdown
    perf = report.get("concept_performance", {})
    if perf:
        st.markdown('<div class="card-title" style="margin-top:1.5rem">📈 Concept Performance Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(perf), 3))
        for i, (c_name, data) in enumerate(perf.items()):
            col_idx = i % len(cols)
            sc = data.get("score", 0)
            c_color = "var(--success)" if sc >= 8 else ("var(--warning)" if sc >= 5 else "var(--danger)")
            with cols[col_idx]:
                st.markdown(f"""
                <div class="card" style="padding:1rem;margin-bottom:0.8rem">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                        <span style="font-size:0.7rem;text-transform:uppercase;color:var(--text-muted)">{data.get('status')}</span>
                        <span style="font-weight:700;color:{c_color};font-size:0.9rem">{sc}/10</span>
                    </div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.92rem;color:var(--text)">
                        {c_name}
                    </div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.25rem">
                        Tested in {data.get('attempts', 1)} question(s)
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 4-Quadrant Breakdown
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        strongs = report.get("strong_concepts", [])
        strong_html = "".join([f"<li style='margin-bottom:0.3rem'>✓ {s}</li>" for s in strongs]) or "<li>No standout strong concepts yet.</li>"
        st.markdown(f"""
        <div class="card">
            <div class="card-title" style="color:var(--success)">✓ Strong Concepts</div>
            <ul style="font-size:0.85rem;color:var(--text);padding-left:1.2rem;line-height:1.6">
                {strong_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        weaks = report.get("weak_concepts", [])
        weak_html = "".join([f"<li style='margin-bottom:0.3rem'>⚠ {w}</li>" for w in weaks]) or "<li>No critical weak concepts!</li>"
        st.markdown(f"""
        <div class="card">
            <div class="card-title" style="color:var(--warning)">⚠ Needs Improvement</div>
            <ul style="font-size:0.85rem;color:var(--text);padding-left:1.2rem;line-height:1.6">
                {weak_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Misconceptions & Recommended Revision
    misconceptions = report.get("misconceptions", [])
    if misconceptions:
        misc_html = "".join([f"<li style='margin-bottom:0.3rem'>🧠 {m}</li>" for m in misconceptions])
        st.markdown(f"""
        <div class="card" style="border-left:4px solid var(--danger)">
            <div class="card-title" style="color:var(--danger)">🧠 Misconceptions Identified</div>
            <ul style="font-size:0.85rem;color:var(--text);padding-left:1.2rem;line-height:1.6">
                {misc_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Recommended Topics Roadmap
    recs = report.get("recommended_topics", [])
    if recs:
        st.markdown('<div class="card-title" style="margin-top:1.5rem">📚 Recommended Revision Plan</div>', unsafe_allow_html=True)
        for i, item in enumerate(recs):
            prio = item.get("priority", "Medium")
            prio_badge = "badge-purple" if prio == "High" else "badge-cyan"
            st.markdown(f"""
            <div class="card" style="padding:1rem;margin-bottom:0.6rem">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem">
                    <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;color:var(--text)">
                        {i + 1}. {item.get('topic')}
                    </span>
                    <span class="badge {prio_badge}">{prio} Priority</span>
                </div>
                <div style="font-size:0.83rem;color:var(--text-muted);line-height:1.5">
                    {item.get('action')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Full Q&A Review Expander
    history = st.session_state.get("interview_history", [])
    with st.expander("📝 View Full Interview Q&A Transcript", expanded=False):
        for idx, h in enumerate(history):
            sc = h.get("evaluation", {}).get("score", 0)
            sc_color = "var(--success)" if sc >= 8 else ("var(--warning)" if sc >= 5 else "var(--danger)")
            st.markdown(f"""
            <div class="card" style="margin-bottom:1rem">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem">
                    <span class="card-title">Q{idx + 1}: {h.get('concept_name')} ({h.get('difficulty')})</span>
                    <span style="font-weight:700;color:{sc_color}">Score: {sc}/10</span>
                </div>
                <div style="font-weight:600;margin-bottom:0.75rem;color:var(--text)">{h.get('question')}</div>
                <div style="background:var(--surface-2);padding:0.75rem;border-radius:6px;font-size:0.82rem;margin-bottom:0.75rem">
                    <strong>Candidate:</strong> {h.get('user_answer')}
                </div>
                <div style="font-size:0.8rem;color:var(--text-muted)">
                    <strong>Feedback:</strong> {h.get('evaluation', {}).get('feedback_summary')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Bottom Actions
    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns(3, gap="medium")

    with b_col1:
        if st.button("🔄  Retake Interview", use_container_width=True):
            st.session_state.interview_step = "setup"
            st.rerun()

    with b_col2:
        if st.button("📜  View Past Interviews", use_container_width=True, type="secondary"):
            st.session_state.interview_step = "history"
            st.rerun()

    with b_col3:
        if st.button("🎬  Video Intelligence", use_container_width=True, type="secondary"):
            st.session_state.nav_mode = "video_intelligence"
            st.rerun()


# ─── SCREEN 5: Past Interviews History ─────────────────────────────────────────

def _render_history_screen():
    st.markdown('<div class="hero-title" style="font-size:2rem">📜 Past Interview Sessions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Review past mock interviews and track progress over time</div>', unsafe_allow_html=True)
    st.markdown("---")

    past = get_past_interviews()

    h_col1, h_col2 = st.columns([1, 1], gap="medium")
    with h_col1:
        if st.button("← Back to Interview Setup", use_container_width=True, type="secondary"):
            st.session_state.interview_step = "setup"
            st.rerun()
    with h_col2:
        if st.button("🎬 Go to Video Intelligence", use_container_width=True, type="secondary"):
            st.session_state.nav_mode = "video_intelligence"
            st.rerun()

    if not past:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;margin-top:1rem">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">📂</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--text)">
                No Past Interviews Yet
            </div>
            <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.5rem">
                Completed interviews are automatically saved and displayed here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter tabs
    tab_all, tab_vid, tab_live = st.tabs(["All Sessions", "📹 Video Interviews", "🎙️ Live AI Interviews"])

    def _render_session_card(session):
        sid = session.get("id", "N/A")
        is_live = session.get("interview_mode") == "live"
        mode_badge = "<span class='badge badge-purple' style='margin-right:0.4rem'>🎙️ Live Voice</span>" if is_live else "<span class='badge badge-cyan' style='margin-right:0.4rem'>📹 Video Grounded</span>"
        v_title = session.get("video_title", "Untitled Session")
        created = session.get("created_at", "")[:16].replace("T", " ")
        rep = session.get("report", {})
        score = rep.get("overall_score", 0.0)
        history = session.get("history", [])

        try:
            score_num = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score_num = 0.0
        score_color = "var(--success)" if score_num >= 7.5 else ("var(--warning)" if score_num >= 5 else "var(--danger)")

        with st.expander(f"🎯 {v_title} — Score: {score}/10 ({created})", expanded=False):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
                <div>
                    <div style="margin-bottom:0.4rem">{mode_badge}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:var(--text)">
                        {v_title}
                    </div>
                    <div style="font-size:0.78rem;color:var(--text-muted)">
                        Session ID: {sid} • Completed on: {created} • Questions: {len(history)}
                    </div>
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:{score_color}">
                    {score} / 10
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Performance tags
            strongs = rep.get("strong_concepts", [])
            weaks = rep.get("weak_concepts", [])
            if strongs or weaks:
                tag_html = ""
                for s in strongs:
                    tag_html += f"<span class='badge badge-green' style='margin-right:0.4rem'>✓ {s}</span>"
                for w in weaks:
                    tag_html += f"<span class='badge badge-purple' style='margin-right:0.4rem'>⚠ {w}</span>"
                st.markdown(f"<div style='margin-bottom:1rem'>{tag_html}</div>", unsafe_allow_html=True)

            # Questions breakdown
            for q_idx, item in enumerate(history):
                ev = item.get("evaluation", {})
                item_score = ev.get("score", 0)
                try:
                    s_num = float(item_score) if item_score is not None else 0.0
                except (ValueError, TypeError):
                    s_num = 0.0
                item_color = "var(--success)" if s_num >= 7.5 else ("var(--warning)" if s_num >= 5 else "var(--danger)")

                st.markdown(f"""
                <div class="card" style="margin-bottom:0.75rem;padding:0.9rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                        <strong style="font-size:0.85rem">Q{q_idx + 1}: {item.get('concept_name', 'General')}</strong>
                        <span style="font-size:0.8rem;font-weight:700;color:{item_color}">{item_score}/10</span>
                    </div>
                    <div style="font-size:0.82rem;margin-bottom:0.5rem;color:var(--text)">{item.get('question', '')}</div>
                    <div style="font-size:0.78rem;color:var(--text-muted)">
                        <strong>Answer:</strong> {item.get('user_answer', '')}
                    </div>
                    <div style="font-size:0.78rem;color:var(--accent-glow);margin-top:0.3rem">
                        <strong>Feedback:</strong> {ev.get('feedback_summary', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_all:
        for s in past:
            _render_session_card(s)

    with tab_vid:
        vid_sessions = [s for s in past if s.get("interview_mode") != "live"]
        if vid_sessions:
            for s in vid_sessions:
                _render_session_card(s)
        else:
            st.info("No video-based interview sessions found yet.")

    with tab_live:
        live_sessions = [s for s in past if s.get("interview_mode") == "live"]
        if live_sessions:
            for s in live_sessions:
                _render_session_card(s)
        else:
            st.info("No live AI interview sessions found yet. Take a Live AI Interview to see your sessions here!")
