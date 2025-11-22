# transcrire_et_decouper.py
# - Transcribe local videos with Whisper
# - Segment into ~30s blocks + write SRT
# - Summarize and create a short title with OpenAI
# - Find YouTube URL (unlisted/private included) via OAuth and update urls.csv
# - Update urls.csv non-destructively (keeps existing fields) and insert at TOP

import os
import re
import json
import shutil
import subprocess
import shlex
import tempfile
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import argparse
import pandas as pd
import whisper
import requests
from tqdm import tqdm
from difflib import SequenceMatcher

# Optional Google imports (graceful fallback if missing)
try:
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except Exception:
    build = InstalledAppFlow = Request = Credentials = None

# ---------------- Paths and files ----------------
DOSSIER_VIDEOS = "videos"
DOSSIER_JSON   = "json"
DOSSIER_SRT    = "srt"
DOSSIER_RESUME = r"C:\Transcript\Dropbox (Personal)\resume"
DOSSIER_BLOCS  = "blocs"
GLOSSAIRE_PATH = "glossaire.csv"
URLS_CSV       = "urls.csv"

# ---------------- Whisper decode options ----------------
DECODE_OPTS = dict(
    language="fr",
    task="transcribe",
    temperature=0.0,
    beam_size=5,
    best_of=None,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    logprob_threshold=-0.5,
    no_speech_threshold=0.6,
)

# ---- early flag (--reauth) ----
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--reauth", action="store_true", help="Force a new Google OAuth (delete token.json)")
args, _ = ap.parse_known_args()
if args.reauth:
    try:
        os.remove("token.json")
        print("[OK] token.json deleted (--reauth).")
    except FileNotFoundError:
        pass
    os.environ["YT_FORCE_REAUTH"] = "1"

# ======================================================
#                   System helpers
# ======================================================
def ensure_ffmpeg_in_path():
    candidates = [
        r"C:\ffmpeg\bin",
        os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin"),
        r"C:\ProgramData\chocolatey\bin",
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg not found. Install it or add C:\\ffmpeg\\bin to PATH.")

def extract_wav_16k_mono(path_in: str) -> str:
    ensure_ffmpeg_in_path()
    tmp_wav = os.path.join(tempfile.gettempdir(), Path(path_in).stem + "_16k.wav")
    cmd = f'ffmpeg -y -i "{path_in}" -vn -ac 1 -ar 16000 -f wav "{tmp_wav}"'
    cp = subprocess.run(shlex.split(cmd), capture_output=True)
    if cp.returncode != 0 or not os.path.exists(tmp_wav):
        raise RuntimeError(f"ffmpeg failed: {cp.stderr.decode(errors='ignore')[:500]}")
    return tmp_wav

def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# ======================================================
#                  Robust CSV helpers
# ======================================================
BOM = "\ufeff"

def read_csv_safely(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise RuntimeError(f"Cannot read {path}. Last error: {last_err}")
    df.columns = [str(c).replace(BOM, "").strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).map(lambda x: x.replace(BOM, "").strip())
    return df

def write_csv(df: pd.DataFrame, path: str, enc: str = "utf-8-sig") -> None:
    df.to_csv(path, index=False, encoding=enc)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def normalize_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    base = os.path.splitext(s)[0]
    return base or s

def upsert_urls_row(csv_path: str, fichier: str, url: str, resume: str, titre: str, date_iso: Optional[str] = None):
    # Create or load DataFrame
    if os.path.exists(csv_path):
        df = read_csv_safely(csv_path)
    else:
        df = pd.DataFrame(columns=["fichier","url","resume","titre","date"])

    # Keep any extra columns present
    df = ensure_columns(df, ["fichier","url","resume","titre","date"])

    # Drop duplicate columns like 'fichier.1'
    dup_cols = [c for c in df.columns if c.lower().startswith("fichier.") and c != "fichier"]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    # Build index by normalized key
    key_to_row: Dict[str, int] = {}
    for i, v in enumerate(df["fichier"]):
        k = normalize_key(v)
        if k and k not in key_to_row:
            key_to_row[k] = i

    k = normalize_key(fichier)
    row_idx = key_to_row.get(k)

    # Non-destructive field merge
    def merge_val(old: str, new: str) -> str:
        old = (old or "").strip()
        new = (new or "").strip()
        return new if (not old and new) else old

    row_data = {
        "fichier": fichier,
        "url": (url or "").strip(),
        "resume": (resume or "").strip(),
        "titre": (titre or "").strip(),
        "date": (date_iso or datetime.now().date().isoformat()),
    }

    if row_idx is None:
        # 1) If there is an empty row at the very top, fill it first
        if len(df) > 0 and str(df.iloc[0]["fichier"]).strip() == "":
            for c in row_data:
                df.iat[0, df.columns.get_loc(c)] = row_data[c]
        else:
            # 2) Otherwise insert at TOP
            row_df = pd.DataFrame([row_data]).reindex(columns=df.columns, fill_value="")
            df = pd.concat([row_df, df], ignore_index=True)
    else:
        # Update existing row (non-destructive)
        df.at[row_idx, "url"]    = merge_val(df.at[row_idx, "url"], row_data["url"])
        df.at[row_idx, "resume"] = merge_val(df.at[row_idx, "resume"], row_data["resume"])
        df.at[row_idx, "titre"]  = merge_val(df.at[row_idx, "titre"], row_data["titre"])
        df.at[row_idx, "date"]   = merge_val(df.at[row_idx, "date"], row_data["date"])

    write_csv(df, csv_path, "utf-8-sig")

# ======================================================
#                OpenAI: short summary + title
# ======================================================
def summarize_and_title(text: str) -> (str, str):
    """Return (resume, titre_court). Returns '', '' if OPENAI_API_KEY missing or any error."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "", ""
    try:
        from openai import OpenAI
    except Exception:
        return "", ""

    client = OpenAI(api_key=api_key)

    # Short summary (1–2 sentences, <= 40 words), in French
    system_msg = (
        "Tu ecris des resumes tres courts en francais, 1 a 2 phrases, maximum 40 mots. "
        "Style informatif, neutre et clair. Pas de puces, pas de titres, pas d'emoji."
    )
    user_msg = (
        "Voici le contenu d'une formation (texte transcrit). "
        "Ecris un resume TRES court (1-2 phrases, <= 40 mots) qui synthetise toute la video.\n\n"
        f"TEXTE:\n{text}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        resume = (resp.choices[0].message.content or "").strip()
    except Exception:
        resume = ""

    # Very short title (<= 7 words), in French
    system_title = "Tu proposes un TRES court titre en francais, maximum 7 mots, descriptif, sans emoji."
    user_title = f"Propose un TRES court titre (<= 7 mots) pour la video suivante.\n\nCONTENU:\n{text}"
    try:
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_title},
                {"role": "user", "content": user_title},
            ],
        )
        titre = (resp2.choices[0].message.content or "").strip()
    except Exception:
        titre = ""

    return resume, titre

# ======================================================
#           YouTube OAuth: list uploads (incl. unlisted)
# ======================================================
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
_YT_INDEX = None  # {title: url}

def get_youtube_service_oauth():
    """Return a YouTube service authenticated via OAuth, or None if unavailable."""
    if build is None:
        print("[WARN] google-api-python-client not installed -> YouTube URL auto off.")
        return None

    token_path = "token.json"
    force = os.environ.get("YT_FORCE_REAUTH") == "1"
    if force and os.path.exists(token_path):
        try:
            os.remove(token_path)
            print("[INFO] token.json removed due to forced reauth.")
        except Exception:
            pass

    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("[OK] OAuth token refreshed.")
            except Exception as e:
                print(f"[WARN] Token refresh failed: {e}")
                creds = None
        if not creds:
            if not os.path.exists("client_secret.json"):
                print("[WARN] client_secret.json missing -> cannot fetch unlisted URLs.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
                try:
                    creds = flow.run_local_server(
                        port=0,
                        prompt="consent",
                        access_type="offline",
                        include_granted_scopes="true",
                        open_browser=True,
                    )
                except Exception as e1:
                    print(f"[WARN] Browser open failed ({e1}); trying console mode.")
                    creds = flow.run_console(
                        authorization_prompt_message="Open this URL: {url}",
                        code_verifier=None,
                        open_browser=False,
                    )
            except Exception as e:
                print(f"[WARN] OAuth flow failed: {e}")
                return None
        try:
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            print("[OK] OAuth token saved.")
        except Exception:
            pass

    try:
        return build("youtube", "v3", credentials=creds)
    except Exception as e:
        print(f"[WARN] Failed to build YouTube service: {e}")
        return None

def get_uploads_playlist_id(youtube):
    """Get the channel's uploads playlist id (requires OAuth, mine=true)."""
    try:
        resp = youtube.channels().list(part="contentDetails", mine=True).execute()
        items = resp.get("items", [])
        if not items:
            return None
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        print(f"[WARN] channels.list failed: {e}")
        return None

def build_my_uploads_index(youtube) -> Dict[str, str]:
    """Return {title: full_url} for all uploaded videos (incl. unlisted/private)."""
    pid = get_uploads_playlist_id(youtube)
    if not pid:
        return {}
    index = {}
    pageToken = None
    while True:
        try:
            pl = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=pid, maxResults=50, pageToken=pageToken
            ).execute()
        except Exception as e:
            print(f"[WARN] playlistItems.list failed: {e}")
            break
        for it in pl.get("items", []):
            vid   = it.get("contentDetails", {}).get("videoId")
            title = (it.get("snippet", {}).get("title") or "").strip()
            if vid and title:
                index[title] = f"https://www.youtube.com/watch?v={vid}"
        pageToken = pl.get("nextPageToken")
        if not pageToken:
            break
    return index

def normalize_key(s: str) -> str:
    import unicodedata, re, os
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\u2000-\u200B\u202F\u205F\u3000]", " ", s)
    s = re.sub(r"[\-‐‑‒–—−]+", "-", s)
    s = re.sub(r"[^a-z0-9\-\._ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    base = os.path.splitext(s)[0]
    return base or s

def _combined_score(target_norm: str, title_norm: str) -> float:
    ta = set(target_norm.split())
    tb = set(title_norm.split())
    jacc = len(ta & tb) / max(1, len(ta | tb))
    sm = SequenceMatcher(None, target_norm, title_norm).ratio()
    bonus = 0.15 if target_norm in title_norm or title_norm in target_norm else 0.0
    return 0.5 * jacc + 0.5 * sm + bonus

DEBUG_YT = os.environ.get("DEBUG_YT") == "1"

def find_youtube_url_for_file(filename_stem: str) -> str:
    """Fuzzy match local filename to your uploads titles and return the best URL."""
    global _YT_INDEX
    if _YT_INDEX is None:
        yt = get_youtube_service_oauth()
        if yt is None:
            _YT_INDEX = {}
            return ""
        _YT_INDEX = build_my_uploads_index(yt)

    if not _YT_INDEX:
        return ""

    target_raw = filename_stem.replace("_", " ").strip()
    target_norm = _normalize_text(target_raw)

    scored = []
    for title, url in _YT_INDEX.items():
        s = _combined_score(target_norm, _normalize_text(title))
        scored.append((s, title, url))
    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        return ""

    if DEBUG_YT:
        print("[DEBUG_YT] target:", target_raw, "=>", target_norm)
        for s, t, u in scored[:5]:
            print(f"[DEBUG_YT] {s:.3f} :: {t} :: {u}")

    best_score, best_title, best_url = scored[0]
    return best_url if best_score >= 0.40 else ""

# ======================================================
#                Optional glossary for prompt
# ======================================================
def load_glossary_prompt() -> Optional[str]:
    if os.path.exists(GLOSSAIRE_PATH):
        try:
            try:
                df_g = pd.read_csv(GLOSSAIRE_PATH, encoding="utf-8")
            except UnicodeDecodeError:
                df_g = pd.read_csv(GLOSSAIRE_PATH, encoding="cp1252")
            termes = df_g.get("mot", pd.Series([], dtype=str)).dropna().astype(str).tolist()
            print(f"[INFO] Glossary loaded ({len(termes)} terms).")
            if termes:
                return " ".join(termes[:200])
        except Exception as e:
            print(f"[WARN] Glossary load error: {e}")
    else:
        print("[INFO] No glossary found.")
    return None

# ======================================================
#                         MAIN
# ======================================================
def main():
    print("[INFO] Loading Whisper model ...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = whisper.load_model("large-v3", device=device)
        print(f"[OK] Whisper loaded on: {device}")
    except RuntimeError as e:
        if "no kernel image is available" in str(e) or "CUDA error" in str(e):
            print("[WARN] CUDA not available -> using CPU.")
            device = "cpu"
            model = whisper.load_model("large-v3", device=device)
            print("[OK] Whisper loaded on: cpu")
        else:
            raise

    # Ensure folders
    for d in (DOSSIER_JSON, DOSSIER_SRT, DOSSIER_RESUME, DOSSIER_BLOCS):
        os.makedirs(d, exist_ok=True)

    # Find videos (mp4, mkv, mov, m4a, mp3, wav)
    valid_ext = {".mp4", ".mkv", ".mov", ".m4a", ".mp3", ".wav"}
    videos = [p for p in Path(DOSSIER_VIDEOS).glob("*.*") if p.suffix.lower() in valid_ext]
    videos = sorted(videos)
    print("[INFO] Videos found:", [v.name for v in videos])

    prompt_glossaire = load_glossary_prompt()

    for chemin_video in videos:
        nom_video = chemin_video.stem
        json_path = Path(DOSSIER_JSON) / f"{nom_video}.json"

        if json_path.exists():
            print(f"[INFO] JSON already exists for {nom_video}, skipping transcription.")
        else:
            print(f"[RUN ] Transcribe: {nom_video}")
            try:
                audio_input = extract_wav_16k_mono(str(chemin_video))
            except Exception as e:
                print(f"[WARN] WAV extraction failed ({e}), will try the video file directly.")
                audio_input = str(chemin_video)

            result = model.transcribe(
                audio_input,
                verbose=True,
                fp16=(device == "cuda"),
                initial_prompt=prompt_glossaire,
                **DECODE_OPTS,
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[OK] JSON saved: {json_path}")

        # Load JSON and segments
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])

        # Write native SRT
        srt_path = Path(DOSSIER_SRT) / f"{nom_video}.srt"
        with open(srt_path, "w", encoding="utf-8") as f_srt:
            idx = 1
            for seg in segments:
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                start_srt = seconds_to_srt_time(seg["start"])
                end_srt   = seconds_to_srt_time(seg["end"])
                f_srt.write(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n\n")
                idx += 1
        print(f"[OK] SRT written: {srt_path}")

        # Re-segment into ~30s blocks
        resegmented = []
        buf_txt, buf_start, buf_end = "", None, None
        for seg in segments:
            t = (seg.get("text") or "").strip()
            if not t:
                continue
            if buf_start is None:
                buf_start = seg["start"]
            buf_txt += (" " if buf_txt else "") + t
            buf_end = seg["end"]
            if buf_end - buf_start >= 30.0:
                resegmented.append({"start": buf_start, "end": buf_end, "text": buf_txt.strip()})
                buf_txt, buf_start, buf_end = "", None, None
        if buf_txt:
            resegmented.append({"start": buf_start, "end": buf_end, "text": buf_txt.strip()})

        # Full text for summary/title
        full_text = " ".join(seg["text"] for seg in resegmented if seg.get("text"))

        # Save concat transcript as TXT (for reference)
        os.makedirs(DOSSIER_RESUME, exist_ok=True)
        resume_path = Path(DOSSIER_RESUME) / f"{nom_video}.txt"
        with open(resume_path, "w", encoding="utf-8") as f_resume:
            f_resume.write(full_text.strip())
        print(f"[OK] Transcript TXT written: {resume_path}")

        # Save blocks CSV
        os.makedirs(DOSSIER_BLOCS, exist_ok=True)
        blocs_path = Path(DOSSIER_BLOCS) / f"{nom_video}.csv"
        pd.DataFrame(resegmented).to_csv(blocs_path, index=False, encoding="utf-8")
        print(f"[OK] Blocks CSV written: {blocs_path}")

        # --- Summary + short title + YouTube URL via OAuth + upsert urls.csv ---
        resume, titre = summarize_and_title(full_text)
        url = find_youtube_url_for_file(nom_video)  # includes unlisted/private
        today = datetime.now().date().isoformat()

        upsert_urls_row(
            URLS_CSV,
            fichier=nom_video,
            url=url,
            resume=resume,
            titre=titre,
            date_iso=today,
        )
        print(f"[OK] urls.csv updated for: {nom_video}")

    print("\n[OK] Done for all videos.")

if __name__ == "__main__":
    main()
