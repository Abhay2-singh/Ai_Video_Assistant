import os
import json
import re
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from tenacity import retry, wait_exponential, stop_after_attempt
from core.vector_store import build_vector_store, get_retriever, load_vector_store

load_dotenv()


def get_llm(temperature: float = 0.3, max_tokens: int = 2048):
    """Retrieve configured ChatMistralAI model with fallback to open-mistral-nemo."""
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "open-mistral-nemo"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens
    )


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True
)
def invoke_with_retry(chain, input_data):
    """Invoke LangChain runnable with exponential backoff retry."""
    return chain.invoke(input_data)


def clean_json_response(raw_text: str) -> dict | list:
    """Robustly parse JSON from LLM output, stripping code fences if present."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text, strict=False)
    except Exception:
        pass

    # Try finding the first { or [ to last } or ]
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = text.rfind("}")
        if last_brace != -1:
            try:
                return json.loads(text[first_brace : last_brace + 1], strict=False)
            except Exception:
                pass
    elif first_bracket != -1:
        last_bracket = text.rfind("]")
        if last_bracket != -1:
            try:
                return json.loads(text[first_bracket : last_bracket + 1], strict=False)
            except Exception:
                pass

    # Regex fallback for dictionary objects
    res_dict = {}
    q_match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
    if q_match:
        res_dict["question"] = q_match.group(1)
    s_match = re.search(r'"score"\s*:\s*(\d+)', text)
    if s_match:
        res_dict["score"] = int(s_match.group(1))
    if res_dict:
        return res_dict

    raise ValueError(f"Could not parse valid JSON from LLM response: {raw_text[:200]}")


# ─── 1. Concept Extraction ──────────────────────────────────────────────────────

def extract_concepts(transcript: str, max_concepts: int = 8) -> List[Dict]:
    """
    Extract key technical / conceptual topics taught in the video transcript.
    Returns a list of dicts with concept name, difficulty level, and brief description.
    """
    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert curriculum and technical interview designer.
Your task is to analyze the provided video transcript and extract the key concepts, topics, or lessons actually taught in the video.

Requirements:
1. ONLY extract concepts that are explicitly discussed or taught in the transcript. Do NOT invent concepts.
2. Categorize each concept into an initial difficulty level: 'Beginner', 'Intermediate', or 'Advanced'.
3. Provide a 1-sentence description and 2-3 search keywords for each concept.
4. Extract between 4 and {max_concepts} concepts.

Return ONLY a valid JSON array of objects with the exact schema:
[
  {{
    "name": "Concept Name",
    "difficulty": "Beginner | Intermediate | Advanced",
    "description": "Brief description of how it is explained in the video",
    "keywords": ["keyword1", "keyword2"]
  }}
]
Do NOT add any conversational explanation outside the JSON.
"""
        ),
        ("human", "Meeting/Video Transcript:\n\n{text}")
    ])

    chain = prompt | llm | StrOutputParser()

    # Pass first 12,000 characters to stay within context bounds
    sample_text = transcript[:12000] if len(transcript) > 12000 else transcript
    raw = invoke_with_retry(chain, {"text": sample_text, "max_concepts": max_concepts})

    try:
        concepts = clean_json_response(raw)
        if isinstance(concepts, list) and len(concepts) > 0:
            for i, c in enumerate(concepts):
                c["id"] = f"concept_{i + 1}"
                if c.get("difficulty") not in ["Beginner", "Intermediate", "Advanced"]:
                    c["difficulty"] = "Intermediate"
            return concepts
    except Exception as e:
        print(f"Error parsing concepts JSON: {e}, raw: {raw[:150]}")

    # Fallback default concepts from transcript snippet
    return [
        {
            "id": "concept_1",
            "name": "Core Principles & Overview",
            "difficulty": "Beginner",
            "description": "Foundational concepts and context covered in the session",
            "keywords": ["overview", "introduction"]
        },
        {
            "id": "concept_2",
            "name": "Key Implementations & Architecture",
            "difficulty": "Intermediate",
            "description": "Technical mechanisms and processes discussed",
            "keywords": ["implementation", "process"]
        },
        {
            "id": "concept_3",
            "name": "Practical Decisions & Trade-offs",
            "difficulty": "Advanced",
            "description": "Decisions, optimization, and edge cases highlighted",
            "keywords": ["decisions", "tradeoffs"]
        }
    ]


# ─── 2. RAG Grounding & Timestamp Estimation ───────────────────────────────────

def get_grounded_context(
    vector_store,
    query: str,
    full_transcript: str = "",
    k: int = 3
) -> Tuple[str, str, int]:
    """
    Retrieve matching chunks from the Chroma vector store.
    Estimates approximate timestamp in MM:SS based on chunk offset or transcript location.
    Returns: (context_text, timestamp_range_str, start_seconds)
    """
    if vector_store is None:
        # Fallback to transcript snippet if vector store is not available
        snippet = full_transcript[:1500] if full_transcript else "Video transcript context."
        return snippet, "00:00 - 05:00", 0

    try:
        docs = vector_store.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error during similarity search: {e}")
        docs = []

    if not docs:
        snippet = full_transcript[:1500] if full_transcript else ""
        return snippet, "00:00 - 05:00", 0

    context_parts = []
    chunk_indices = []

    for doc in docs:
        context_parts.append(doc.page_content)
        idx = doc.metadata.get("chunk_index")
        if idx is not None:
            chunk_indices.append(idx)

    combined_context = "\n\n---\n\n".join(context_parts)

    # Estimate timestamp:
    # Each chunk in build_vector_store is 500 chars with 50 overlap (~450 net chars).
    # Typical speaking rate is ~150 words/min = ~850 chars/min = ~14 chars/sec.
    # So each 450 chars represents ~32 seconds of video audio.
    if chunk_indices:
        min_idx = min(chunk_indices)
        max_idx = max(chunk_indices)
        start_sec = max(0, int(min_idx * 32))
        end_sec = int((max_idx + 1) * 32)
    elif full_transcript and docs[0].page_content in full_transcript:
        char_pos = full_transcript.find(docs[0].page_content)
        start_sec = max(0, int(char_pos / 14))
        end_sec = start_sec + 60
    else:
        start_sec = 0
        end_sec = 180

    start_m, start_s = divmod(start_sec, 60)
    end_m, end_s = divmod(end_sec, 60)
    timestamp_str = f"{start_m:02d}:{start_s:02d} – {end_m:02d}:{end_s:02d}"

    return combined_context, timestamp_str, start_sec


# ─── 3. Dynamic Question Generation ─────────────────────────────────────────────

QUESTION_TYPES = [
    "Conceptual",
    "Why/How",
    "Practical",
    "Scenario-based",
    "Debugging",
    "Code-related",
    "Follow-up"
]


def generate_interview_question(
    concept: Dict,
    difficulty: str,
    rag_context: str,
    timestamp_range: str,
    question_type: str = "Conceptual",
    previous_evaluation: Optional[Dict] = None,
    previous_questions: Optional[List[str]] = None,
) -> Dict:
    """
    Generate a dynamic, non-hardcoded question grounded strictly in the video transcript context.
    """
    llm = get_llm(temperature=0.4)

    prev_q_str = "\n".join([f"- {q}" for q in (previous_questions or [])])
    prev_feedback_str = ""
    if previous_evaluation:
        prev_feedback_str = f"""
Previous Question Feedback:
- Score: {previous_evaluation.get('score', 'N/A')}/10
- What was missing/weak: {', '.join(previous_evaluation.get('missing_points', []))}
- Misconceptions: {', '.join(previous_evaluation.get('misconceptions', []))}
"""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a senior technical interviewer conducting an adaptive mock interview based on a recorded technical video/meeting.

GROUNDING RULES:
1. Base the question STRICTLY on the provided video transcript context. Do NOT ask about topics not present in the context.
2. The question must test genuine understanding of: '{concept_name}'.
3. Target Difficulty: {difficulty}.
   - Beginner: Foundational concepts, definitions, core purpose ('What is X and why use it?').
   - Intermediate: Mechanics, interactions, trade-offs, why/how questions ('How does X work under the hood?').
   - Advanced: Edge cases, architectural implications, practical scenarios, debugging ('What happens if X fails in scenario Y?').
4. Question Type to use: {question_type}.
5. Do NOT repeat or rephrase any previously asked questions.
6. If previous feedback is provided, naturally adapt: either probe deeper into missing knowledge or advance to a new practical angle.

Previously asked questions (DO NOT REPEAT):
{prev_questions}

{prev_feedback}

Video Transcript Context (Ground Truth):
{context}

Return ONLY a valid JSON object with the exact schema:
{{
  "question": "The interview question text",
  "question_type": "{question_type}",
  "difficulty": "{difficulty}",
  "concept_name": "{concept_name}",
  "timestamp_range": "{timestamp_range}",
  "expected_key_points": [
    "Key point 1 expected in a good answer",
    "Key point 2 expected in a good answer"
  ]
}}
Do NOT output anything outside the JSON object.
"""
        ),
        ("human", "Generate the next interview question for concept: {concept_name}")
    ])

    chain = prompt | llm | StrOutputParser()

    raw = invoke_with_retry(chain, {
        "concept_name": concept.get("name", "Technical Concept"),
        "difficulty": difficulty,
        "question_type": question_type,
        "timestamp_range": timestamp_range,
        "prev_questions": prev_q_str if prev_q_str else "None yet.",
        "prev_feedback": prev_feedback_str,
        "context": rag_context[:3000]
    })

    try:
        parsed = clean_json_response(raw)
        if isinstance(parsed, dict) and "question" in parsed:
            parsed["timestamp_range"] = timestamp_range
            return parsed
    except Exception as e:
        print(f"Error parsing question JSON: {e}, raw: {raw[:150]}")

    # Fallback dynamically tailored question
    return {
        "question": f"Based on the video discussion on {concept.get('name')}, explain its main purpose and how it is applied.",
        "question_type": question_type,
        "difficulty": difficulty,
        "concept_name": concept.get("name", "General"),
        "timestamp_range": timestamp_range,
        "expected_key_points": ["Clear explanation of concept", "Practical application mentioned in video"]
    }


# ─── 4. Answer Evaluation ───────────────────────────────────────────────────────

def evaluate_user_answer(
    question: str,
    expected_points: List[str],
    user_answer: str,
    concept_name: str,
    difficulty: str,
    rag_context: str,
) -> Dict:
    """
    Evaluate user answer on correctness, understanding, missing points, and misconceptions.
    Returns structured feedback with a numerical score (1-10) and adaptive recommendation.
    """
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert technical interviewer evaluating a candidate's answer.

You must objectively evaluate the candidate's answer against the video transcript context and expected key points.

EVALUATION CRITERIA:
1. Correctness: Are the factual technical claims accurate?
2. Concept Understanding: Did the candidate grasp the core underlying mechanism?
3. Missing Concepts: What important aspects from the video context were omitted?
4. Misconceptions: Did the user state anything incorrect, confusing, or misleading?
5. Score: Provide a fair integer score from 1 to 10:
   - 8-10: Strong / Excellent (comprehensive, correct, explains why)
   - 5-7: Partially Correct (has the basic idea, but misses important details or nuances)
   - 1-4: Weak / Inaccurate (major misconceptions, vague, or incorrect)
6. Recommended Difficulty Action:
   - 'increase' if score >= 8
   - 'maintain' if score is 5 to 7
   - 'decrease' if score < 5

Ground Truth Context from Video:
{context}

Question Asked:
{question}

Expected Key Points:
{expected_points}

Candidate Answer:
{user_answer}

Return ONLY a valid JSON object with the exact schema:
{{
  "score": 7,
  "rating": "Strong | Partially Correct | Weak",
  "correct_points": ["Point 1 user explained correctly"],
  "missing_points": ["Point 1 user omitted or missed"],
  "misconceptions": ["Any incorrect assumption user made (empty list if none)"],
  "explanation_quality": "Concise comment on communication and depth",
  "feedback_summary": "2-3 sentences of constructive, direct interview feedback",
  "recommended_difficulty_action": "increase | maintain | decrease",
  "weak_concept_identified": "Specific sub-concept user struggled with, or None"
}}
Do NOT output anything outside the JSON object.
"""
        ),
        ("human", "Evaluate the user's answer now.")
    ])

    chain = prompt | llm | StrOutputParser()

    raw = invoke_with_retry(chain, {
        "context": rag_context[:3000],
        "question": question,
        "expected_points": "\n".join([f"- {p}" for p in expected_points]),
        "user_answer": user_answer.strip() if user_answer.strip() else "(No answer provided)"
    })

    try:
        eval_data = clean_json_response(raw)
        if isinstance(eval_data, dict) and "score" in eval_data:
            # Ensure score is integer within 1-10
            score = int(eval_data.get("score", 5))
            eval_data["score"] = max(1, min(10, score))
            if eval_data["score"] >= 8:
                eval_data["rating"] = "Strong"
            elif eval_data["score"] >= 5:
                eval_data["rating"] = "Partially Correct"
            else:
                eval_data["rating"] = "Weak"
            return eval_data
    except Exception as e:
        print(f"Error parsing evaluation JSON: {e}, raw: {raw[:150]}")

    # Fallback evaluation
    word_count = len(user_answer.strip().split())
    fallback_score = 6 if word_count > 15 else 3
    return {
        "score": fallback_score,
        "rating": "Partially Correct" if fallback_score >= 5 else "Weak",
        "correct_points": ["User attempted the question."],
        "missing_points": ["Detailed technical explanation with specifics."],
        "misconceptions": [],
        "explanation_quality": "Brief explanation.",
        "feedback_summary": "Your answer covers basic intuition, but needs more concrete technical grounding.",
        "recommended_difficulty_action": "maintain",
        "weak_concept_identified": concept_name
    }


# ─── 5. Adaptive Flow Controller ────────────────────────────────────────────────

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]


def get_next_difficulty(current_difficulty: str, action: str) -> str:
    """Adjust difficulty up or down based on adaptive evaluation."""
    idx = DIFFICULTY_LEVELS.index(current_difficulty) if current_difficulty in DIFFICULTY_LEVELS else 1
    if action == "increase" and idx < len(DIFFICULTY_LEVELS) - 1:
        return DIFFICULTY_LEVELS[idx + 1]
    elif action == "decrease" and idx > 0:
        return DIFFICULTY_LEVELS[idx - 1]
    return current_difficulty


def select_adaptive_next_concept(
    all_concepts: List[Dict],
    interview_history: List[Dict],
    last_evaluation: Dict,
    current_difficulty: str
) -> Tuple[Dict, str, str]:
    """
    Decides the next concept, difficulty, and question type adaptively:
    - If Strong (>=8): Step up difficulty, move to deeper/practical scenario or next concept.
    - If Partially Correct (5-7): Maintain difficulty, target missing concept with Follow-up/Why-How question.
    - If Weak (<5): Lower difficulty, ask simpler foundational/conceptual question on prerequisite.
    - If Repeated weakness: Focus on the specific concept user is struggling with.
    Returns: (selected_concept, next_difficulty, question_type)
    """
    last_score = last_evaluation.get("score", 5)
    action = last_evaluation.get("recommended_difficulty_action", "maintain")
    next_difficulty = get_next_difficulty(current_difficulty, action)

    # Detect repeat struggles
    concept_scores: Dict[str, List[int]] = {}
    for h in interview_history:
        c_name = h.get("concept_name")
        sc = h.get("evaluation", {}).get("score", 5)
        concept_scores.setdefault(c_name, []).append(sc)

    # Check if there is a concept with repeated low scores (<6)
    struggling_concept_name = None
    for c_name, scores in concept_scores.items():
        if len(scores) >= 2 and sum(scores) / len(scores) < 6:
            struggling_concept_name = c_name
            break

    last_q_concept_name = interview_history[-1].get("concept_name") if interview_history else None

    # Case 1: Partially correct -> Follow-up on same concept targeting missing points
    if 5 <= last_score <= 7 and last_q_concept_name:
        for c in all_concepts:
            if c.get("name") == last_q_concept_name:
                return c, next_difficulty, "Follow-up"

    # Case 2: Weak answer -> simpler question on same concept or foundational concept
    if last_score < 5:
        if struggling_concept_name:
            for c in all_concepts:
                if c.get("name") == struggling_concept_name:
                    return c, "Beginner", "Conceptual"
        if last_q_concept_name:
            for c in all_concepts:
                if c.get("name") == last_q_concept_name:
                    return c, next_difficulty, "Why/How"

    # Case 3: Strong answer -> select an untested concept or advanced practical/scenario question
    tested_concepts = set(concept_scores.keys())
    untested = [c for c in all_concepts if c.get("name") not in tested_concepts]

    if untested:
        # Pick the next matching or slightly higher difficulty concept
        selected = untested[0]
        q_type = "Practical" if next_difficulty == "Advanced" else "Why/How"
        return selected, next_difficulty, q_type

    # If all tested, cycle to concept with lowest average score to reinforce learning
    sorted_by_need = sorted(
        all_concepts,
        key=lambda c: sum(concept_scores.get(c.get("name"), [10])) / len(concept_scores.get(c.get("name"), [1]))
    )
    selected = sorted_by_need[0] if sorted_by_need else all_concepts[0]
    q_type = "Scenario-based" if next_difficulty == "Advanced" else "Practical"
    return selected, next_difficulty, q_type


# ─── 6. Final Interview Report Generation ────────────────────────────────────────

def generate_final_report(
    interview_history: List[Dict],
    all_concepts: List[Dict],
    video_title: str = ""
) -> Dict:
    """
    Generate comprehensive interview report:
    - Overall score
    - Concept performance breakdown
    - Strong concepts (✓)
    - Weak concepts (⚠)
    - Missing concepts (❌)
    - Misconceptions (🧠)
    - Recommended revision topics with actionable advice (📚)
    """
    if not interview_history:
        return {
            "overall_score": 0.0,
            "concept_performance": {},
            "strong_concepts": [],
            "weak_concepts": [],
            "missing_concepts": [],
            "misconceptions": [],
            "recommended_topics": []
        }

    # Aggregate scores per concept
    concept_map: Dict[str, List[int]] = {}
    all_misconceptions = []
    all_missing_points = []

    total_score = 0
    for item in interview_history:
        ev = item.get("evaluation", {})
        c_name = item.get("concept_name", "General")
        sc = ev.get("score", 5)
        total_score += sc
        concept_map.setdefault(c_name, []).append(sc)

        for m in ev.get("misconceptions", []):
            if m and m not in all_misconceptions:
                all_misconceptions.append(m)

        for miss in ev.get("missing_points", []):
            if miss and miss not in all_missing_points:
                all_missing_points.append(miss)

    overall_score = round(total_score / len(interview_history), 1)

    concept_performance = {}
    strong_concepts = []
    weak_concepts = []

    for c_name, scores in concept_map.items():
        avg = round(sum(scores) / len(scores), 1)
        concept_performance[c_name] = {
            "score": avg,
            "attempts": len(scores),
            "status": "Strong" if avg >= 8 else ("Average" if avg >= 5 else "Needs Improvement")
        }
        if avg >= 8:
            strong_concepts.append(c_name)
        elif avg < 6:
            weak_concepts.append(c_name)

    # Missing concepts: concepts from video that were either never tested or heavily missed
    tested_names = set(concept_map.keys())
    untested_concepts = [c.get("name") for c in all_concepts if c.get("name") not in tested_names]
    missing_concepts = untested_concepts + [f"{c_name} (omitted key details)" for c_name in weak_concepts]

    # Generate personalized recommendations using LLM
    llm = get_llm(temperature=0.2)
    rec_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a technical career mentor. Based on a candidate's mock interview results from a video session:
Generate 3 to 5 prioritized, actionable revision recommendations.

Video Topic: {video_title}
Overall Score: {overall_score}/10
Weak Concepts: {weak_concepts}
Misconceptions: {misconceptions}
Missing Points: {missing_points}

Return ONLY a valid JSON array of objects:
[
  {{
    "topic": "Topic Name",
    "action": "Concrete advice on what to study or practice from the video",
    "priority": "High | Medium"
  }}
]
Do NOT output any markdown outside the JSON.
"""
        ),
        ("human", "Generate revision recommendations now.")
    ])

    chain = rec_prompt | llm | StrOutputParser()

    recommended_topics = []
    try:
        raw_rec = invoke_with_retry(chain, {
            "video_title": video_title or "Technical Video",
            "overall_score": overall_score,
            "weak_concepts": ", ".join(weak_concepts) if weak_concepts else "None",
            "misconceptions": "; ".join(all_misconceptions[:5]) if all_misconceptions else "None",
            "missing_points": "; ".join(all_missing_points[:5]) if all_missing_points else "None"
        })
        parsed_recs = clean_json_response(raw_rec)
        if isinstance(parsed_recs, list):
            recommended_topics = parsed_recs
    except Exception as e:
        print(f"Error generating recommendations: {e}")

    # Fallback recommendations if LLM failed
    if not recommended_topics:
        for wc in weak_concepts:
            recommended_topics.append({
                "topic": wc,
                "action": f"Re-watch video sections explaining {wc} and review key mechanics.",
                "priority": "High"
            })
        if not recommended_topics:
            recommended_topics.append({
                "topic": "General Review",
                "action": "Solid performance! Review edge cases and real-world system trade-offs.",
                "priority": "Medium"
            })

    return {
        "overall_score": overall_score,
        "concept_performance": concept_performance,
        "strong_concepts": strong_concepts,
        "weak_concepts": weak_concepts,
        "missing_concepts": missing_concepts[:5],
        "misconceptions": all_misconceptions[:5],
        "recommended_topics": recommended_topics
    }
