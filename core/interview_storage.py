import json
import os
import uuid
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VIDEOS_FILE = os.path.join(DATA_DIR, "processed_videos.json")
INTERVIEWS_FILE = os.path.join(DATA_DIR, "interview_history.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(file_path: str, default: list) -> list:
    _ensure_data_dir()
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default


def _save_json(file_path: str, data: list):
    _ensure_data_dir()
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


# ─── Processed Videos Storage ───────────────────────────────────────────────────

def save_processed_video_record(
    title: str,
    transcript: str,
    summary: str = "",
    source: str = "",
    concepts: list = None
) -> dict:
    """Save or update a processed video record with transcript and concepts."""
    videos = _load_json(VIDEOS_FILE, [])

    # Check if video with same title already exists
    for v in videos:
        if v.get("title") == title:
            v["transcript"] = transcript
            v["summary"] = summary
            v["source"] = source
            if concepts:
                v["concepts"] = concepts
            v["updated_at"] = datetime.now().isoformat()
            _save_json(VIDEOS_FILE, videos)
            return v

    new_record = {
        "id": str(uuid.uuid4())[:8],
        "title": title or "Untitled Video",
        "source": source,
        "transcript": transcript,
        "summary": summary,
        "concepts": concepts or [],
        "created_at": datetime.now().isoformat(),
    }
    videos.append(new_record)
    _save_json(VIDEOS_FILE, videos)
    return new_record


def get_saved_videos() -> list:
    """Return all saved processed video records."""
    return _load_json(VIDEOS_FILE, [])


def get_video_by_id(video_id: str) -> dict:
    """Retrieve a specific video record by its id."""
    videos = get_saved_videos()
    for v in videos:
        if v.get("id") == video_id:
            return v
    return None


def update_video_concepts(video_id: str, concepts: list):
    """Save extracted concepts to a video record."""
    videos = _load_json(VIDEOS_FILE, [])
    for v in videos:
        if v.get("id") == video_id:
            v["concepts"] = concepts
            break
    _save_json(VIDEOS_FILE, videos)


# ─── Interview Sessions Storage ─────────────────────────────────────────────────

def save_interview_session(session_data: dict) -> dict:
    """Save a completed or active interview session."""
    interviews = _load_json(INTERVIEWS_FILE, [])

    session_id = session_data.get("id") or str(uuid.uuid4())[:8]
    session_data["id"] = session_id
    if "created_at" not in session_data:
        session_data["created_at"] = datetime.now().isoformat()

    # If updating existing
    updated = False
    for i, item in enumerate(interviews):
        if item.get("id") == session_id:
            interviews[i] = session_data
            updated = True
            break

    if not updated:
        interviews.insert(0, session_data)  # newest first

    _save_json(INTERVIEWS_FILE, interviews)
    return session_data


def get_past_interviews() -> list:
    """Return all past interview sessions sorted newest first."""
    return _load_json(INTERVIEWS_FILE, [])


def get_interview_by_id(interview_id: str) -> dict:
    """Retrieve a specific interview session by id."""
    interviews = get_past_interviews()
    for item in interviews:
        if item.get("id") == interview_id:
            return item
    return None
