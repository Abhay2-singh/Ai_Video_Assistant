import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s_%(title).50s.%(ext)s")
    
    # Try different player client configurations if YouTube blocks (HTTP 403 Forbidden)
    client_strategies = [
        ["android", "mweb", "web"],
        ["ios", "mweb"],
        ["mweb"],
        ["android"],
    ]
    
    last_error = None
    for clients in client_strategies:
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_path,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }
                ],
                "quiet": True,
                "no_warnings": True,
                "nocheckcertificate": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": clients
                    }
                },
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            }
            
            # Support cookie file or YOUTUBE_COOKIES secret if provided
            cookie_path = os.path.join(os.getcwd(), "cookies.txt")
            if os.path.exists(cookie_path):
                ydl_opts["cookiefile"] = cookie_path
            elif os.getenv("YOUTUBE_COOKIES"):
                temp_cookie = os.path.join(DOWNLOAD_DIR, "temp_cookies.txt")
                with open(temp_cookie, "w", encoding="utf-8") as f:
                    f.write(os.getenv("YOUTUBE_COOKIES"))
                ydl_opts["cookiefile"] = temp_cookie

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    continue
                prepared = ydl.prepare_filename(info)
                base = os.path.splitext(prepared)[0]
                wav_filename = f"{base}.wav"
                
                if os.path.exists(wav_filename):
                    return wav_filename
                
                # Check if extracted file has another extension
                for ext in [".wav", ".m4a", ".webm", ".opus", ".mp3", ".mp4"]:
                    cand = f"{base}{ext}"
                    if os.path.exists(cand):
                        return cand if ext == ".wav" else convert_to_wav(cand)
                
                return wav_filename
        except Exception as e:
            last_error = e
            continue
            
    raise RuntimeError(
        f"Unable to download YouTube audio. YouTube blocked the request (HTTP 403 Forbidden). "
        f"Details: {last_error}"
    )



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


