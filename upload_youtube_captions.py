# upload_youtube_captions.py
# - Parcourt ./srt/*.srt
# - Trouve la vidéo YouTube correspondante (même nom que le .srt)
# - Uploade/Met à jour la piste de sous-titres via YouTube Data API v3
#
# Dépendances : google-api-python-client, google-auth-oauthlib
#   pip install --upgrade google-api-python-client google-auth-oauthlib

import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
import argparse
import sys
from typing import Dict, Optional, Tuple

# Google API
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# =========================
#  Configuration par défaut
# =========================
SRT_DIR = "srt"
DEFAULT_LANG = "fr"
TOKEN_PATH = "token.json"
CLIENT_SECRET = "client_secret.json"

# IMPORTANT: pour l'upload de sous-titres on a besoin d'un scope "manage"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# =========================
#  Helpers OAuth (calqué sur ton script)
# =========================
def get_youtube_service(reauth: bool = False):
    """Construit le service YouTube authentifié OAuth (token.json / client_secret.json).
       Re-auth si le scope requis n'est pas présent."""
    creds = None

    # forcer reauth si demandé
    if reauth and os.path.exists(TOKEN_PATH):
        try:
            os.remove(TOKEN_PATH)
            print("[INFO] token.json supprimé (--reauth).")
        except Exception:
            pass

    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
            # Attention: si l'ancien token a d'autres scopes (ex: readonly),
            # creds peut être invalide pour "force-ssl" => reauth ci-dessous.
            if not creds or (set(SCOPES) - set(creds.scopes or [])):
                creds = None
        except Exception:
            creds = None

    if not creds:
        if not os.path.exists(CLIENT_SECRET):
            print(f"[ERR] {CLIENT_SECRET} manquant.")
            sys.exit(1)
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
        try:
            creds = flow.run_local_server(
                port=0, prompt="consent", access_type="offline", include_granted_scopes="true"
            )
        except Exception as e:
            print(f"[ERR] OAuth local_server a échoué: {e}")
            creds = flow.run_console(
                authorization_prompt_message="Open this URL: {url}",
                code_verifier=None,
                open_browser=False,
            )
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        print("[OK] OAuth terminé, token.json enregistré.")

    # refresh si expiré
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            print("[OK] Token rafraîchi.")
        except Exception as e:
            print(f"[WARN] Refresh token a échoué: {e}")

    return build("youtube", "v3", credentials=creds)

# =========================
#  Récupérer l'index de tes vidéos (titre -> (videoId, url))
# =========================
def get_uploads_playlist_id(youtube) -> Optional[str]:
    try:
        r = youtube.channels().list(part="contentDetails", mine=True).execute()
        items = r.get("items", [])
        if not items:
            return None
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except Exception as e:
        print(f"[WARN] channels.list: {e}")
        return None

def build_uploads_index(youtube) -> Dict[str, Tuple[str, str]]:
    """Retourne {title: (videoId, url)} pour toutes tes vidéos (publique/non répertoriée/privée)."""
    pid = get_uploads_playlist_id(youtube)
    if not pid:
        return {}
    index = {}
    pageToken = None
    while True:
        try:
            pl = youtube.playlistItems().list(
                part="snippet,contentDetails", playlistId=pid, maxResults=50, pageToken=pageToken
            ).execute()
        except Exception as e:
            print(f"[WARN] playlistItems.list: {e}")
            break
        for it in pl.get("items", []):
            vid = it.get("contentDetails", {}).get("videoId")
            title = (it.get("snippet", {}).get("title") or "").strip()
            if vid and title:
                index[title] = (vid, f"https://www.youtube.com/watch?v={vid}")
        pageToken = pl.get("nextPageToken")
        if not pageToken:
            break
    return index

# =========================
#  Matching nom de fichier -> vidéo
# (exact puis fallback fuzzy, à la façon de ton script)
# =========================
def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def _combined_score(target_norm: str, title_norm: str) -> float:
    ta = set(target_norm.split())
    tb = set(title_norm.split())
    jacc = len(ta & tb) / max(1, len(ta | tb))
    sm = SequenceMatcher(None, target_norm, title_norm).ratio()
    bonus = 0.15 if target_norm in title_norm or title_norm in target_norm else 0.0
    return 0.5 * jacc + 0.5 * sm + bonus

def find_video_for_stem(stem: str, uploads: Dict[str, Tuple[str, str]]) -> Optional[str]:
    """Retourne videoId correspondant au nom de fichier (sans extension)."""
    # 1) exact, case-insensitive
    for title, (vid, _) in uploads.items():
        if stem.strip().lower() == title.strip().lower():
            return vid
    # 2) normalisé + fuzzy
    target_norm = _normalize_text(stem)
    best = (0.0, None)
    for title, (vid, _) in uploads.items():
        score = _combined_score(target_norm, _normalize_text(title))
        if score > best[0]:
            best = (score, vid)
    return best[1] if best[0] >= 0.40 else None

# =========================
#  Captions: list/insert/update
# =========================
def get_existing_caption_id(youtube, video_id: str, lang: str) -> Optional[str]:
    try:
        r = youtube.captions().list(part="snippet", videoId=video_id).execute()
        for item in r.get("items", []):
            snip = item.get("snippet", {})
            if (snip.get("language") or "").lower().startswith(lang.lower()):
                return item.get("id")
    except Exception as e:
        print(f"[WARN] captions.list({video_id}): {e}")
    return None

def upload_or_update_caption(youtube, video_id: str, srt_path: str, lang: str = DEFAULT_LANG, name: Optional[str] = None):
    name = name or Path(srt_path).stem
    media = MediaFileUpload(srt_path, mimetype="application/octet-stream", resumable=False)

    existing_id = get_existing_caption_id(youtube, video_id, lang)
    if existing_id:
        print(f"  ↻ Mise à jour sous-titres [{lang}] pour {video_id} …")
        try:
            youtube.captions().update(
                part="snippet",
                body={"id": existing_id, "snippet": {"isDraft": False}},
                media_body=media
            ).execute()
            print("  [OK] captions.update")
            return
        except Exception as e:
            print(f"  [WARN] captions.update a échoué ({e}), tentative insert…")

    print(f"  ⤴︎ Upload sous-titres [{lang}] pour {video_id} …")
    body = {"snippet": {"videoId": video_id, "language": lang, "name": name, "isDraft": False}}
    try:
        youtube.captions().insert(
            part="snippet", body=body, media_body=media
        ).execute()
        print("  [OK] captions.insert")
    except Exception as e:
        print(f"  [ERR] captions.insert a échoué: {e}")

# =========================
#  Main
# =========================
def main():
    p = argparse.ArgumentParser(description="Uploader des fichiers .srt vers YouTube (multisite)")
    p.add_argument("--dir", default=SRT_DIR, help="Dossier des SRT (par défaut: srt)")
    p.add_argument("--lang", default=DEFAULT_LANG, help="Code langue des sous-titres (fr, en, …)")
    p.add_argument("--reauth", action="store_true", help="Forcer une nouvelle autorisation OAuth")
    p.add_argument("--dry-run", action="store_true", help="Lister les correspondances sans uploader")
    args = p.parse_args()

    youtube = get_youtube_service(reauth=args.reauth)
    uploads = build_uploads_index(youtube)
    if not uploads:
        print("[ERR] Impossible de lister tes vidéos (vérifie l'API et les autorisations).")
        sys.exit(1)

    srt_dir = Path(args.dir)
    files = sorted(srt_dir.glob("*.srt"))
    if not files:
        print(f"[INFO] Aucun .srt trouvé dans {srt_dir.resolve()}")
        return

    print(f"[INFO] {len(files)} fichier(s) .srt trouvé(s).")
    for srt in files:
        stem = srt.stem
        vid = find_video_for_stem(stem, uploads)
        if not vid:
            print(f"[SKIP] Aucun match pour: {srt.name}")
            continue
        print(f"[MATCH] {srt.name}  →  videoId={vid}")
        if not args.dry_run:
            upload_or_update_caption(youtube, vid, str(srt), lang=args.lang, name=stem)

    print("\n[OK] Terminé.")

if __name__ == "__main__":
    main()
