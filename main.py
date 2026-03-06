"""
main.py
-------
Daemon that syncs liked tracks bidirectionally between Spotify and Yandex Music.
Runs the sync job every 2 hours using the `schedule` library.

Direction 1: Spotify liked tracks → Yandex Music likes
Direction 2: Yandex Music liked tracks → Spotify liked tracks

Tracks that have already been processed (found/liked) are recorded in cache
files so they are never searched or re-added again — even if the user later
removes the like on one side.

Required environment variables:
    SPOTIFY_CLIENT_ID       — Spotify application client ID
    SPOTIFY_CLIENT_SECRET   — Spotify application client secret
    SPOTIFY_REFRESH_TOKEN   — Long-lived refresh token (from get_spotify_token.py)
    YANDEX_TOKEN            — Yandex Music OAuth token

Optional environment variables:
    DATA_DIR                — Directory for persistent cache files
                              (default: /data)
                              Mount this as a Docker volume so files survive
                              container restarts.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import requests as http_client
import schedule
import spotipy
import yandex_music
from yandex_music import Client as YandexClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNC_INTERVAL_HOURS: int = 2
SPOTIFY_LIKED_LIMIT: int = 50       # Spotify API max per request
YANDEX_LIKED_LIMIT: int = 100       # How many recent Yandex likes to check
YANDEX_REQUEST_DELAY: float = 1.5   # seconds between Yandex API calls

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

# Directory for persistent cache files.
# Override via DATA_DIR env var; mount as a Docker/Coolify volume.
DATA_DIR: str = os.environ.get("DATA_DIR", "/data")

# Cache files — one per sync direction.
# Keys in spotify_to_yandex are normalized "artist|title" from Spotify.
# Keys in yandex_to_spotify are Yandex track IDs.
CACHE_SP_TO_YA: str = os.path.join(DATA_DIR, "synced_spotify_to_yandex.txt")
CACHE_YA_TO_SP: str = os.path.join(DATA_DIR, "synced_yandex_to_spotify.txt")

# Legacy cache file (v1 format — plain Yandex IDs)
LEGACY_CACHE: str = os.environ.get("SYNCED_IDS_PATH", os.path.join(DATA_DIR, "synced_ids.txt"))


def _require_env(name: str) -> str:
    """Return the value of an environment variable or exit with a clear error."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"FATAL: Required environment variable '{name}' is not set. "
            "Set it in Coolify and restart the container."
        )
    return value


# ---------------------------------------------------------------------------
# Helpers — normalisation & cache
# ---------------------------------------------------------------------------

def normalize_key(artist: str, title: str) -> str:
    """
    Build a stable cache key from artist + title.
    Lowercase, stripped, pipe-separated.
    """
    return f"{artist.strip().lower()}|{title.strip().lower()}"


def _load_cache(path: str) -> set[str]:
    """Load a set of strings from a cache file (one per line)."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        ids = {line.strip() for line in fh if line.strip()}
    logger.info("Cache: loaded %d entries from '%s'.", len(ids), path)
    return ids


def _save_cache(path: str, entries: set[str]) -> None:
    """Overwrite a cache file with the full set of entries."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(entries)) + "\n")
    logger.info("Cache: saved %d entries to '%s'.", len(entries), path)


def _append_to_cache(path: str, entry: str) -> None:
    """Append a single entry to a cache file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(entry + "\n")


# ---------------------------------------------------------------------------
# Spotify client — fully headless, no browser, no stdin
# ---------------------------------------------------------------------------

def _get_spotify_access_token() -> str:
    """
    Exchange a refresh token for a fresh Spotify access token via direct
    HTTP POST to the Spotify Accounts service.  No browser or web server
    needed — works perfectly in headless / Docker environments.
    """
    client_id = _require_env("SPOTIFY_CLIENT_ID")
    client_secret = _require_env("SPOTIFY_CLIENT_SECRET")
    refresh_token = _require_env("SPOTIFY_REFRESH_TOKEN")

    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode()
    ).decode()

    response = http_client.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=15,
    )
    response.raise_for_status()
    token_data = response.json()
    access_token: str = token_data["access_token"]
    scope: str = token_data.get("scope", "unknown")
    logger.info("Spotify: token scope = '%s'", scope)
    return access_token


def build_spotify_client() -> spotipy.Spotify:
    """
    Build a Spotipy client using a freshly obtained access token.
    The token is refreshed on every call to this function (i.e. every sync run),
    so it is always valid for the duration of the sync job.
    """
    access_token = _get_spotify_access_token()
    logger.info("Spotify: access token refreshed successfully.")
    return spotipy.Spotify(auth=access_token)


# ---------------------------------------------------------------------------
# Yandex Music client
# ---------------------------------------------------------------------------

def build_yandex_client() -> YandexClient:
    """Build and initialize the Yandex Music client."""
    token = _require_env("YANDEX_TOKEN")
    client = YandexClient(token=token)
    client.init()
    return client


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def get_spotify_liked_tracks(sp: spotipy.Spotify) -> list[dict]:
    """
    Fetch the most recently liked tracks from Spotify (up to SPOTIFY_LIKED_LIMIT).
    Returns a list of dicts with keys: 'artist', 'title'.
    """
    results = sp.current_user_saved_tracks(limit=SPOTIFY_LIKED_LIMIT)
    tracks: list[dict] = []

    for item in results.get("items", []):
        track = item.get("track")
        if not track:
            continue
        artists = ", ".join(a["name"] for a in track.get("artists", []))
        title: str = track.get("name", "")
        if artists and title:
            tracks.append({"artist": artists, "title": title})

    logger.info("Spotify: fetched %d liked tracks.", len(tracks))
    return tracks


def search_track_on_spotify(
    sp: spotipy.Spotify,
    artist: str,
    title: str,
) -> Optional[dict]:
    """
    Search for a track on Spotify and return a dict with 'id', 'artist', 'title'
    or None if not found.
    """
    query = f"artist:{artist} track:{title}"
    try:
        results = sp.search(q=query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if items:
            t = items[0]
            return {
                "id": t["id"],
                "artist": ", ".join(a["name"] for a in t.get("artists", [])),
                "title": t.get("name", ""),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Spotify search error for '%s - %s': %s", artist, title, exc)
    return None


def like_track_on_spotify(sp: spotipy.Spotify, track_id: str) -> bool:
    """
    Add a track to the user's Spotify Liked Songs.
    Returns True on success, False on failure.
    """
    try:
        sp.current_user_saved_tracks_add(tracks=[track_id])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to like track on Spotify (id=%s): %s", track_id, exc)
        return False


# ---------------------------------------------------------------------------
# Yandex helpers
# ---------------------------------------------------------------------------

def get_yandex_liked_tracks_info(yandex: YandexClient) -> list[dict]:
    """
    Fetch recently liked tracks from Yandex Music.
    Returns a list of dicts with keys: 'id', 'artist', 'title'.
    """
    liked = yandex.users_likes_tracks()
    if not liked:
        return []

    # liked is a list of TrackShort — we need to fetch full info for artist/title
    track_shorts = liked[:YANDEX_LIKED_LIMIT]

    tracks: list[dict] = []
    # Fetch full track info in bulk (the API supports it)
    track_ids = []
    for ts in track_shorts:
        if ts.id is not None:
            # TrackShort.id can be "123:456" (track_id:album_id) or just "123"
            track_ids.append(str(ts.id).split(":")[0])

    if not track_ids:
        return []

    try:
        time.sleep(YANDEX_REQUEST_DELAY)
        full_tracks = yandex.tracks(track_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch Yandex track details: %s", exc)
        return []

    for t in full_tracks:
        if t and t.title:
            artist = t.artists[0].name if t.artists else ""
            if artist:
                tracks.append({
                    "id": str(t.id),
                    "artist": artist,
                    "title": t.title,
                })

    logger.info("Yandex: fetched %d liked tracks info.", len(tracks))
    return tracks


def get_yandex_liked_ids(yandex: YandexClient) -> set[str]:
    """
    Fetch the IDs of all tracks already liked on Yandex Music.
    Returns a set of string IDs.
    """
    liked_tracks = yandex.users_likes_tracks()
    if not liked_tracks:
        return set()

    ids: set[str] = set()
    for lt in liked_tracks:
        if lt.id is not None:
            ids.add(str(lt.id).split(":")[0])
    logger.info("Yandex Music: %d tracks already liked.", len(ids))
    return ids


def search_track_on_yandex(
    yandex: YandexClient,
    query: str,
) -> Optional[yandex_music.Track]:
    """
    Search for a track on Yandex Music and return the best match, or None.
    Respects the rate-limit delay after the call.
    """
    time.sleep(YANDEX_REQUEST_DELAY)
    try:
        search_result = yandex.search(query, type_="track")
        if search_result and search_result.tracks and search_result.tracks.results:
            return search_result.tracks.results[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Yandex search error for '%s': %s", query, exc)
    return None


def like_track_on_yandex(
    yandex: YandexClient,
    track: yandex_music.Track,
) -> bool:
    """
    Like a track on Yandex Music.
    Returns True on success, False on failure.
    Respects the rate-limit delay before the call.
    """
    time.sleep(YANDEX_REQUEST_DELAY)
    try:
        track.like()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to like track '%s – %s' (id=%s): %s",
            track.artists[0].name if track.artists else "?",
            track.title,
            track.id,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Bootstrap — first-run snapshots
# ---------------------------------------------------------------------------

def bootstrap_spotify_to_yandex(
    sp: spotipy.Spotify,
    yandex: YandexClient,
) -> set[str]:
    """
    First-run bootstrap for Spotify→Yandex direction.
    Creates a cache of normalized "artist|title" keys for ALL current Spotify
    liked tracks, so they are treated as baseline and never re-synced.
    Also adds all currently liked Yandex track IDs (via old cache migration
    if available).
    """
    logger.info("Bootstrapping Spotify→Yandex cache …")

    cache: set[str] = set()

    # 1. Snapshot current Spotify liked tracks (all pages)
    offset = 0
    while True:
        results = sp.current_user_saved_tracks(limit=SPOTIFY_LIKED_LIMIT, offset=offset)
        items = results.get("items", [])
        if not items:
            break
        for item in items:
            track = item.get("track")
            if not track:
                continue
            artists = ", ".join(a["name"] for a in track.get("artists", []))
            title = track.get("name", "")
            if artists and title:
                cache.add(normalize_key(artists, title))
        offset += len(items)

    logger.info(
        "Bootstrap: %d Spotify liked tracks added to Spotify→Yandex cache.",
        len(cache),
    )

    _save_cache(CACHE_SP_TO_YA, cache)
    return cache


def bootstrap_yandex_to_spotify(yandex: YandexClient) -> set[str]:
    """
    First-run bootstrap for Yandex→Spotify direction.
    Creates a cache of all currently liked Yandex track IDs, so they are
    treated as baseline and never re-synced to Spotify.
    """
    logger.info("Bootstrapping Yandex→Spotify cache …")
    ids = get_yandex_liked_ids(yandex)
    _save_cache(CACHE_YA_TO_SP, ids)
    logger.info(
        "Bootstrap: %d Yandex liked tracks added to Yandex→Spotify cache.",
        len(ids),
    )
    return ids


# ---------------------------------------------------------------------------
# Migration from v1 cache format
# ---------------------------------------------------------------------------

def maybe_migrate_legacy_cache() -> None:
    """
    If the old synced_ids.txt exists but new cache files don't, log a message.
    We cannot migrate automatically because the v1 format stored Yandex IDs
    while the new Spotify→Yandex cache stores normalized 'artist|title' keys.
    The bootstrap will recreate everything from scratch.
    """
    if os.path.exists(LEGACY_CACHE) and not os.path.exists(CACHE_SP_TO_YA):
        logger.info(
            "Legacy cache '%s' found but new format cache is missing. "
            "A fresh bootstrap will be performed. The legacy file is preserved "
            "but will no longer be used.",
            LEGACY_CACHE,
        )


# ---------------------------------------------------------------------------
# Sync jobs
# ---------------------------------------------------------------------------

def sync_spotify_to_yandex(
    sp: spotipy.Spotify,
    yandex: YandexClient,
    cache: set[str],
) -> set[str]:
    """Sync Spotify liked tracks → Yandex Music likes."""
    logger.info("— Syncing Spotify → Yandex Music …")

    try:
        spotify_tracks = get_spotify_liked_tracks(sp)
    except Exception as exc:
        logger.error("Failed to fetch Spotify liked tracks: %s", exc)
        return cache

    added = skipped = not_found = errors = 0

    for track_info in spotify_tracks:
        key = normalize_key(track_info["artist"], track_info["title"])

        if key in cache:
            logger.debug("ALREADY PROCESSED: '%s'", key)
            skipped += 1
            continue

        query = f"{track_info['artist']} - {track_info['title']}"
        yandex_track = search_track_on_yandex(yandex, query)

        if yandex_track is None:
            logger.warning("NOT FOUND on Yandex: '%s'", query)
            not_found += 1
            # Still record the key so we don't re-search next time
            cache.add(key)
            _append_to_cache(CACHE_SP_TO_YA, key)
            continue

        success = like_track_on_yandex(yandex, yandex_track)
        if success:
            cache.add(key)
            _append_to_cache(CACHE_SP_TO_YA, key)
            artist_name = yandex_track.artists[0].name if yandex_track.artists else "?"
            logger.info("✅ LIKED on Yandex: '%s – %s'", artist_name, yandex_track.title)
            added += 1
        else:
            errors += 1

    logger.info(
        "Spotify→Yandex: Added: %d | Skipped: %d | Not found: %d | Errors: %d",
        added, skipped, not_found, errors,
    )
    return cache


def sync_yandex_to_spotify(
    sp: spotipy.Spotify,
    yandex: YandexClient,
    cache: set[str],
) -> set[str]:
    """Sync Yandex Music liked tracks → Spotify likes."""
    logger.info("— Syncing Yandex Music → Spotify …")

    try:
        yandex_tracks = get_yandex_liked_tracks_info(yandex)
    except Exception as exc:
        logger.error("Failed to fetch Yandex liked tracks: %s", exc)
        return cache

    added = skipped = not_found = errors = 0

    for track_info in yandex_tracks:
        yandex_id = track_info["id"]

        if yandex_id in cache:
            logger.debug("ALREADY PROCESSED (Yandex→Spotify): id=%s", yandex_id)
            skipped += 1
            continue

        spotify_result = search_track_on_spotify(
            sp, track_info["artist"], track_info["title"],
        )

        if spotify_result is None:
            logger.warning(
                "NOT FOUND on Spotify: '%s - %s'",
                track_info["artist"], track_info["title"],
            )
            not_found += 1
            # Record so we don't re-search
            cache.add(yandex_id)
            _append_to_cache(CACHE_YA_TO_SP, yandex_id)
            continue

        success = like_track_on_spotify(sp, spotify_result["id"])
        if success:
            cache.add(yandex_id)
            _append_to_cache(CACHE_YA_TO_SP, yandex_id)
            logger.info(
                "✅ LIKED on Spotify: '%s – %s'",
                spotify_result["artist"], spotify_result["title"],
            )
            added += 1
        else:
            errors += 1

    logger.info(
        "Yandex→Spotify: Added: %d | Skipped: %d | Not found: %d | Errors: %d",
        added, skipped, not_found, errors,
    )
    return cache


# ---------------------------------------------------------------------------
# Main sync job
# ---------------------------------------------------------------------------

def sync_job() -> None:
    """Main synchronization job: bidirectional Spotify ↔ Yandex Music."""
    logger.info("=" * 60)
    logger.info("Starting sync job …")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Build clients
    # ------------------------------------------------------------------
    try:
        sp = build_spotify_client()
        yandex = build_yandex_client()
    except Exception as exc:
        logger.error("Failed to initialize API clients: %s", exc)
        return

    # ------------------------------------------------------------------
    # Check for legacy cache migration
    # ------------------------------------------------------------------
    maybe_migrate_legacy_cache()

    # ------------------------------------------------------------------
    # Load / bootstrap the Spotify→Yandex cache
    # ------------------------------------------------------------------
    try:
        sp_to_ya_cache = _load_cache(CACHE_SP_TO_YA)
        if not sp_to_ya_cache:
            sp_to_ya_cache = bootstrap_spotify_to_yandex(sp, yandex)
    except Exception as exc:
        logger.error("Failed to load/bootstrap Spotify→Yandex cache: %s", exc)
        return

    # ------------------------------------------------------------------
    # Load / bootstrap the Yandex→Spotify cache
    # ------------------------------------------------------------------
    try:
        ya_to_sp_cache = _load_cache(CACHE_YA_TO_SP)
        if not ya_to_sp_cache:
            ya_to_sp_cache = bootstrap_yandex_to_spotify(yandex)
    except Exception as exc:
        logger.error("Failed to load/bootstrap Yandex→Spotify cache: %s", exc)
        return

    # ------------------------------------------------------------------
    # Run both directions
    # ------------------------------------------------------------------
    sp_to_ya_cache = sync_spotify_to_yandex(sp, yandex, sp_to_ya_cache)
    ya_to_sp_cache = sync_yandex_to_spotify(sp, yandex, ya_to_sp_cache)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Sync complete. Next run in %d hours.", SYNC_INTERVAL_HOURS)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Spotify ↔ Yandex Music bidirectional sync daemon starting up.")
    logger.info(
        "Sync interval: every %d hour(s). Running initial sync now …",
        SYNC_INTERVAL_HOURS,
    )

    # Run once immediately on startup so we don't wait 2 hours first
    sync_job()

    # Schedule subsequent runs
    schedule.every(SYNC_INTERVAL_HOURS).hours.do(sync_job)

    while True:
        schedule.run_pending()
        time.sleep(30)  # check the schedule every 30 seconds


if __name__ == "__main__":
    main()
