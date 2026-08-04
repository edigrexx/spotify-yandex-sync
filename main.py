"""
main.py
-------
Daemon that syncs liked tracks bidirectionally between Spotify and Yandex Music.

Direction 1: Spotify liked tracks → Yandex Music likes
Direction 2: Yandex Music liked tracks → Spotify liked tracks

State lives in a SQLite database (see store.py) that records, per direction and
per track, whether it was synced, not found, or failed transiently — so a
momentary API error no longer retires a track permanently.

Matching is handled by matching.py, which scores candidates on title, artist and
duration instead of blindly taking the first search result.

Required environment variables:
    SPOTIFY_CLIENT_ID       — Spotify application client ID
    SPOTIFY_CLIENT_SECRET   — Spotify application client secret
    SPOTIFY_REFRESH_TOKEN   — Long-lived refresh token (from get_spotify_token.py)
    YANDEX_TOKEN            — Yandex Music OAuth token (from get_yandex_token.py)

Optional environment variables:
    DATA_DIR                — Directory for the state database (default: /data).
                              Mount as a volume so it survives restarts.
    SYNC_INTERVAL_HOURS     — Hours between runs (default: 2)
    MATCH_THRESHOLD         — Minimum match confidence, 0..1 (default: 0.68)
    YANDEX_PROXY_URL        — Route Yandex API calls through a proxy. Needed only
                              if the host's IP is geo-blocked (HTTP 451).
    YANDEX_TIMEOUT          — Seconds per Yandex API call (default: 30)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional, Sequence

import requests as http_client
import schedule
import spotipy
from yandex_music import Client as YandexClient
from yandex_music.utils.request import Request as YandexRequest

import matching
from matching import TrackMeta
from store import SP_TO_YA, YA_TO_SP, SyncStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNC_INTERVAL_HOURS: int = _env_int("SYNC_INTERVAL_HOURS", 2)
MATCH_THRESHOLD: float = _env_float("MATCH_THRESHOLD", matching.DEFAULT_THRESHOLD)

SPOTIFY_PAGE_SIZE: int = 50          # Spotify API max per request
SPOTIFY_MAX_PAGES: int = 40          # hard stop, ~2000 tracks
# Stop paging once this many consecutive pages contain nothing new. Keeps the
# steady state cheap while still catching a burst of new likes.
SPOTIFY_QUIET_PAGES: int = 2

YANDEX_REQUEST_DELAY: float = 1.5    # seconds between Yandex API calls
YANDEX_TRACK_BATCH: int = 10         # tracks per metadata fetch
YANDEX_TIMEOUT: int = _env_int("YANDEX_TIMEOUT", 30)

SEARCH_CANDIDATES: int = 10          # results to score per query
MAX_QUERY_VARIANTS: int = 3          # query rewrites to try before giving up

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
STATE_DB: str = os.path.join(DATA_DIR, "sync.db")

# Flat caches written by earlier versions, imported once into the database.
LEGACY_SP_TO_YA: str = os.path.join(DATA_DIR, "synced_spotify_to_yandex.txt")
LEGACY_YA_TO_SP: str = os.path.join(DATA_DIR, "synced_yandex_to_spotify.txt")


def _require_env(name: str) -> str:
    """Return the value of an environment variable or exit with a clear error."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"FATAL: Required environment variable '{name}' is not set. "
            "Set it in Coolify and restart the container."
        )
    return value


def normalize_key(artist: str, title: str) -> str:
    """
    Build a stable cache key from artist + title.

    The exact formatting is load-bearing: keys imported from the pre-SQLite
    flat caches were written this way, and changing it would orphan every
    existing entry and re-sync the whole library.
    """
    return f"{artist.strip().lower()}|{title.strip().lower()}"


# ---------------------------------------------------------------------------
# Spotify client — fully headless, no browser, no stdin
# ---------------------------------------------------------------------------

def _get_spotify_access_token() -> str:
    """Exchange the refresh token for a fresh Spotify access token."""
    client_id = _require_env("SPOTIFY_CLIENT_ID")
    client_secret = _require_env("SPOTIFY_CLIENT_SECRET")
    refresh_token = _require_env("SPOTIFY_REFRESH_TOKEN")

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    response = http_client.post(
        SPOTIFY_TOKEN_URL,
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=15,
    )
    response.raise_for_status()
    token_data = response.json()

    scope: str = token_data.get("scope", "")
    if "user-library-modify" not in scope:
        logger.warning(
            "Spotify: token is missing the 'user-library-modify' scope — "
            "Yandex→Spotify likes will fail. Re-run get_spotify_token.py."
        )
    return token_data["access_token"]


def build_spotify_client() -> spotipy.Spotify:
    """Build a Spotipy client with a freshly minted access token."""
    client = spotipy.Spotify(auth=_get_spotify_access_token())
    logger.info("Spotify: access token refreshed successfully.")
    return client


# ---------------------------------------------------------------------------
# Yandex Music client
# ---------------------------------------------------------------------------

def _patch_yandex_artist_model() -> None:
    """
    Tolerate artist payloads that arrive without an `id`.

    yandex-music declares Artist.id as required, but the API does return
    artists without one. The whole batch of tracks then fails to deserialise
    with "Artist.__init__() missing 1 required positional argument: 'id'",
    and those tracks silently vanish from the sync.
    """
    from yandex_music import Artist

    if getattr(Artist, "_id_optional_patch", False):
        return

    original = Artist.de_json.__func__

    def de_json(cls, data, client):  # noqa: ANN001
        if isinstance(data, dict) and "id" not in data:
            data = dict(data, id=None)
        return original(cls, data, client)

    Artist.de_json = classmethod(de_json)
    Artist._id_optional_patch = True


def build_yandex_client() -> YandexClient:
    """
    Build and initialise the Yandex Music client, verifying it is authenticated.

    A token issued for a different Yandex application is not rejected outright:
    /account/status answers 200 with an anonymous account, init() raises
    nothing, and every later call fails with `ownerOtherwiseUserBindingError`
    against /users/None/likes/tracks. Checking for a uid here turns that silent
    misconfiguration into an immediate, explicit failure.
    """
    _patch_yandex_artist_model()

    token = _require_env("YANDEX_TOKEN").strip()
    for prefix in ("OAuth ", "Bearer "):
        if token.startswith(prefix):
            token = token[len(prefix):].strip()

    proxy_url = os.environ.get("YANDEX_PROXY_URL") or None
    request = YandexRequest(proxy_url=proxy_url, timeout=YANDEX_TIMEOUT)
    if proxy_url:
        logger.info("Yandex: routing API calls through a proxy.")

    client = YandexClient(token=token, request=request).init()

    account = getattr(client.me, "account", None)
    if not account or not account.uid:
        raise RuntimeError(
            "YANDEX_TOKEN is not authenticated for Yandex Music: /account/status "
            "returned an anonymous account with no uid. The token is probably "
            "issued for a different Yandex application, or it has expired. "
            "Run get_yandex_token.py to mint a working one."
        )

    logger.info("Yandex: authenticated as uid=%s (%s).", account.uid, account.login)
    return client


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

def _spotify_track_to_meta(track: dict) -> Optional[TrackMeta]:
    """Convert a Spotify track object into the matcher's representation."""
    if not track or not track.get("name"):
        return None
    artists = tuple(a["name"] for a in track.get("artists", []) if a.get("name"))
    if not artists:
        return None
    return TrackMeta(
        artists=artists,
        title=track["name"],
        duration_ms=track.get("duration_ms"),
        source_id=track.get("id"),
    )


def get_spotify_liked_tracks(sp: spotipy.Spotify, store: SyncStore) -> list[TrackMeta]:
    """
    Fetch Spotify liked tracks, newest first, stopping once the pages go quiet.

    The previous version read only the first 50, so liking more than that
    between runs silently dropped the overflow.
    """
    tracks: list[TrackMeta] = []
    quiet_pages = 0

    for page in range(SPOTIFY_MAX_PAGES):
        results = sp.current_user_saved_tracks(
            limit=SPOTIFY_PAGE_SIZE, offset=page * SPOTIFY_PAGE_SIZE
        )
        items = results.get("items", [])
        if not items:
            break

        page_has_new = False
        for item in items:
            meta = _spotify_track_to_meta(item.get("track"))
            if meta is None:
                continue
            tracks.append(meta)
            key = normalize_key(", ".join(meta.artists), meta.title)
            if store.should_attempt(SP_TO_YA, key):
                page_has_new = True

        quiet_pages = 0 if page_has_new else quiet_pages + 1
        if quiet_pages >= SPOTIFY_QUIET_PAGES or len(items) < SPOTIFY_PAGE_SIZE:
            break

    logger.info("Spotify: fetched %d liked tracks.", len(tracks))
    return tracks


def search_spotify_candidates(sp: spotipy.Spotify, query: str) -> list[TrackMeta]:
    """Run one Spotify search and return the results as candidates."""
    try:
        results = sp.search(q=query, type="track", limit=SEARCH_CANDIDATES)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Spotify search error for %r: %s", query, exc)
        raise

    candidates = []
    for track in results.get("tracks", {}).get("items", []):
        meta = _spotify_track_to_meta(track)
        if meta is not None:
            candidates.append(meta)
    return candidates


def like_track_on_spotify(sp: spotipy.Spotify, track_id: str) -> bool:
    """
    Save a track to the user's Spotify library and confirm it landed.

    Uses the documented PUT /v1/me/tracks endpoint, then reads the library back:
    a 2xx from the write says the request was accepted, not that the track is
    actually saved.
    """
    try:
        sp.current_user_saved_tracks_add([track_id])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to like track on Spotify (id=%s): %s", track_id, exc)
        return False

    try:
        contains = sp.current_user_saved_tracks_contains([track_id])
        if not contains or not contains[0]:
            logger.warning(
                "Spotify accepted the like for id=%s but the track is not in the "
                "library — treating as a failure so it is retried.", track_id,
            )
            return False
    except Exception as exc:  # noqa: BLE001
        # The write succeeded; only the read-back failed. Trust the write.
        logger.debug("Could not verify Spotify like for id=%s: %s", track_id, exc)

    return True


# ---------------------------------------------------------------------------
# Yandex helpers
# ---------------------------------------------------------------------------

def _yandex_track_to_meta(track) -> Optional[TrackMeta]:  # noqa: ANN001
    """Convert a Yandex track object into the matcher's representation."""
    if not track or not getattr(track, "title", None):
        return None
    artists = tuple(a.name for a in (track.artists or []) if getattr(a, "name", None))
    if not artists:
        return None

    # Yandex keeps "feat. X" and "Remastered" style qualifiers in a separate
    # `version` field; the matcher wants them attached to the title.
    title = track.title
    version = getattr(track, "version", None)
    if version:
        title = f"{title} ({version})"

    return TrackMeta(
        artists=artists,
        title=title,
        duration_ms=getattr(track, "duration_ms", None),
        source_id=str(track.id),
    )


def _short_track_id(track_short) -> str:  # noqa: ANN001
    """TrackShort ids look like "123:456" (track:album) or plain "123"."""
    return str(track_short.id).split(":")[0]


def get_yandex_liked_shorts(yandex: YandexClient) -> list:
    """
    Fetch every liked track reference, newest first.

    One cheap call returns the whole list, so the sync no longer works from a
    fixed-size window whose ordering the API never guaranteed — sorting by the
    like timestamp makes "most recent" explicit instead of assumed.
    """
    liked = yandex.users_likes_tracks()
    if not liked:
        return []
    shorts = [ts for ts in liked if ts.id is not None]
    shorts.sort(key=lambda ts: ts.timestamp or "", reverse=True)
    logger.info("Yandex: %d liked tracks in the library.", len(shorts))
    return shorts


def fetch_yandex_tracks(yandex: YandexClient, track_ids: Sequence[str]) -> list:
    """
    Fetch full track objects in batches, falling back to one-by-one.

    Reports the ids that could not be fetched at all, which the previous
    version discarded silently.
    """
    fetched: list = []
    missing: list[str] = []

    for start in range(0, len(track_ids), YANDEX_TRACK_BATCH):
        batch = list(track_ids[start:start + YANDEX_TRACK_BATCH])
        try:
            time.sleep(YANDEX_REQUEST_DELAY)
            result = yandex.tracks(batch)
            if result:
                fetched.extend(result)
                continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch fetch failed (%d tracks), retrying one by one: %s",
                           len(batch), exc)

        for track_id in batch:
            try:
                time.sleep(YANDEX_REQUEST_DELAY * 0.5)
                single = yandex.tracks([track_id])
                if single:
                    fetched.extend(single)
                else:
                    missing.append(track_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not fetch Yandex track %s: %s", track_id, exc)
                missing.append(track_id)

    if missing:
        logger.warning("Yandex: %d track(s) could not be fetched: %s",
                       len(missing), ", ".join(missing[:10]))
    return fetched


def search_yandex_candidates(yandex: YandexClient, query: str) -> list[TrackMeta]:
    """Run one Yandex search and return the results as candidates."""
    time.sleep(YANDEX_REQUEST_DELAY)
    result = yandex.search(query, type_="track")
    if not result or not result.tracks or not result.tracks.results:
        return []

    candidates = []
    for track in result.tracks.results[:SEARCH_CANDIDATES]:
        meta = _yandex_track_to_meta(track)
        if meta is not None:
            candidates.append((meta, track))
    # Keep the raw objects around so the caller can call .like() on the winner.
    return candidates


def like_track_on_yandex(yandex: YandexClient, track) -> bool:  # noqa: ANN001
    """Like a track on Yandex Music."""
    time.sleep(YANDEX_REQUEST_DELAY)
    try:
        return bool(track.like())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to like Yandex track id=%s: %s", track.id, exc)
        return False


# ---------------------------------------------------------------------------
# Matching driver
# ---------------------------------------------------------------------------

def find_match(source: TrackMeta, search, threshold: float):  # noqa: ANN001
    """
    Search for `source` using progressively looser queries.

    Returns (match, score, best_seen). `match` is None when nothing cleared the
    threshold; `best_seen` is the highest score any candidate reached, which
    makes near-misses visible in the logs instead of indistinguishable from
    "no results at all".

    Raises whatever the search callable raises, so the caller can tell a
    transient API failure apart from a genuine miss.
    """
    best_seen = 0.0
    for query in matching.query_variants(source)[:MAX_QUERY_VARIANTS]:
        candidates = search(query)
        if not candidates:
            continue
        metas = [c[0] if isinstance(c, tuple) else c for c in candidates]
        match, value = matching.pick_best(source, metas, threshold)
        best_seen = max(best_seen, value)
        if match is not None:
            for candidate in candidates:
                if (candidate[0] if isinstance(candidate, tuple) else candidate) is match:
                    return candidate, value, best_seen
    return None, 0.0, best_seen


# ---------------------------------------------------------------------------
# Sync directions
# ---------------------------------------------------------------------------

def sync_spotify_to_yandex(sp: spotipy.Spotify, yandex: YandexClient,
                           store: SyncStore) -> None:
    """Sync Spotify liked tracks → Yandex Music likes."""
    logger.info("— Syncing Spotify → Yandex Music …")

    try:
        spotify_tracks = get_spotify_liked_tracks(sp, store)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch Spotify liked tracks: %s", exc)
        return

    added = skipped = not_found = errors = 0

    for source in spotify_tracks:
        key = normalize_key(", ".join(source.artists), source.title)
        if not store.should_attempt(SP_TO_YA, key):
            skipped += 1
            continue

        try:
            match, value, best_seen = find_match(
                source, lambda q: search_yandex_candidates(yandex, q), MATCH_THRESHOLD
            )
        except Exception as exc:  # noqa: BLE001
            # A transient failure says nothing about the track. Record it as an
            # error so the next run tries again instead of retiring the track.
            logger.warning("Yandex search failed for %r: %s", source.label(), exc)
            store.mark_error(SP_TO_YA, key, str(exc))
            errors += 1
            continue

        if match is None:
            logger.info("NOT FOUND on Yandex: %r (best candidate scored %.2f)",
                        source.label(), best_seen)
            store.mark_not_found(SP_TO_YA, key, best_score=best_seen)
            not_found += 1
            continue

        meta, raw_track = match
        if like_track_on_yandex(yandex, raw_track):
            store.mark_synced(SP_TO_YA, key, src_id=source.source_id,
                              dst_id=meta.source_id, score=value)
            # Suppress the echo: this track is now liked on Yandex, and the
            # other direction must not treat it as a new Yandex like.
            store.mark_synced(YA_TO_SP, str(meta.source_id), dst_id=source.source_id,
                              score=value, detail="mirrored from Spotify→Yandex")
            logger.info("✅ LIKED on Yandex: %r (%.2f)", meta.label(), value)
            added += 1
        else:
            store.mark_error(SP_TO_YA, key, "like failed")
            errors += 1

    logger.info("Spotify→Yandex: Added: %d | Skipped: %d | Not found: %d | Errors: %d",
                added, skipped, not_found, errors)


def sync_yandex_to_spotify(sp: spotipy.Spotify, yandex: YandexClient,
                           store: SyncStore) -> None:
    """Sync Yandex Music liked tracks → Spotify likes."""
    logger.info("— Syncing Yandex Music → Spotify …")

    try:
        shorts = get_yandex_liked_shorts(yandex)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch Yandex liked tracks: %s", exc)
        return

    # Diff before fetching metadata: in the steady state this makes the whole
    # direction free, instead of paying for a hundred track lookups every run.
    pending = [_short_track_id(ts) for ts in shorts]
    pending = [tid for tid in pending if store.should_attempt(YA_TO_SP, tid)]
    skipped = len(shorts) - len(pending)

    if not pending:
        logger.info("Yandex→Spotify: nothing new (%d already processed).", skipped)
        return

    logger.info("Yandex→Spotify: %d new track(s) to look up.", len(pending))
    tracks = fetch_yandex_tracks(yandex, pending)

    added = not_found = errors = 0

    for track in tracks:
        source = _yandex_track_to_meta(track)
        if source is None:
            continue
        yandex_id = str(track.id)

        try:
            match, value, best_seen = find_match(
                source, lambda q: search_spotify_candidates(sp, q), MATCH_THRESHOLD
            )
        except Exception as exc:  # noqa: BLE001
            store.mark_error(YA_TO_SP, yandex_id, str(exc))
            errors += 1
            continue

        if match is None:
            logger.info("NOT FOUND on Spotify: %r (best candidate scored %.2f)",
                        source.label(), best_seen)
            store.mark_not_found(YA_TO_SP, yandex_id, best_score=best_seen)
            not_found += 1
            continue

        if like_track_on_spotify(sp, match.source_id):
            store.mark_synced(YA_TO_SP, yandex_id, src_id=yandex_id,
                              dst_id=match.source_id, score=value)
            # Mirror into the other direction so Spotify→Yandex skips it.
            store.mark_synced(SP_TO_YA,
                              normalize_key(", ".join(match.artists), match.title),
                              dst_id=yandex_id, score=value,
                              detail="mirrored from Yandex→Spotify")
            logger.info("✅ LIKED on Spotify: %r (%.2f)", match.label(), value)
            added += 1
        else:
            store.mark_error(YA_TO_SP, yandex_id, "like failed")
            errors += 1

    logger.info("Yandex→Spotify: Added: %d | Skipped: %d | Not found: %d | Errors: %d",
                added, skipped, not_found, errors)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(sp: spotipy.Spotify, yandex: YandexClient, store: SyncStore) -> None:
    """
    Prepare state on first run.

    Flat caches from earlier versions are imported so their history is kept.
    Only when there is nothing at all to import does a direction get a baseline
    snapshot of the existing library, which marks what is already there as
    handled rather than flooding the other service with a full back-catalogue.
    """
    store.import_legacy_cache(SP_TO_YA, LEGACY_SP_TO_YA)
    store.import_legacy_cache(YA_TO_SP, LEGACY_YA_TO_SP)

    if store.is_empty(SP_TO_YA):
        logger.info("Bootstrapping Spotify→Yandex baseline …")
        keys = []
        for page in range(SPOTIFY_MAX_PAGES):
            results = sp.current_user_saved_tracks(
                limit=SPOTIFY_PAGE_SIZE, offset=page * SPOTIFY_PAGE_SIZE
            )
            items = results.get("items", [])
            if not items:
                break
            for item in items:
                meta = _spotify_track_to_meta(item.get("track"))
                if meta is not None:
                    keys.append(normalize_key(", ".join(meta.artists), meta.title))
            if len(items) < SPOTIFY_PAGE_SIZE:
                break
        logger.info("Bootstrap: %d Spotify tracks recorded as baseline.",
                    store.bulk_mark_synced(SP_TO_YA, keys))

    if store.is_empty(YA_TO_SP):
        logger.info("Bootstrapping Yandex→Spotify baseline …")
        ids = [_short_track_id(ts) for ts in get_yandex_liked_shorts(yandex)]
        logger.info("Bootstrap: %d Yandex tracks recorded as baseline.",
                    store.bulk_mark_synced(YA_TO_SP, ids))


# ---------------------------------------------------------------------------
# Main sync job
# ---------------------------------------------------------------------------

def sync_job() -> None:
    """One full bidirectional synchronisation pass."""
    logger.info("=" * 60)
    logger.info("Starting sync job …")
    logger.info("=" * 60)

    try:
        sp = build_spotify_client()
        yandex = build_yandex_client()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise API clients: %s", exc)
        return

    try:
        with SyncStore(STATE_DB) as store:
            bootstrap(sp, yandex, store)
            sync_spotify_to_yandex(sp, yandex, store)
            sync_yandex_to_spotify(sp, yandex, store)
            logger.info("State: Spotify→Yandex %s | Yandex→Spotify %s",
                        store.counts(SP_TO_YA), store.counts(YA_TO_SP))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sync job failed: %s", exc)
        return

    logger.info("-" * 60)
    logger.info("Sync complete. Next run in %d hour(s).", SYNC_INTERVAL_HOURS)
    logger.info("=" * 60)


def _check_data_dir() -> None:
    """Fail fast if the state directory is not writable."""
    os.makedirs(DATA_DIR, exist_ok=True)
    probe = os.path.join(DATA_DIR, ".write-test")
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise SystemExit(
            f"FATAL: DATA_DIR '{DATA_DIR}' is not writable ({exc}). Sync state "
            "would be lost on every restart. Check the volume's ownership — the "
            "container runs as a non-root user."
        )


def main() -> None:
    logger.info("Spotify ↔ Yandex Music bidirectional sync daemon starting up.")
    logger.info("Sync interval: every %d hour(s). Match threshold: %.2f.",
                SYNC_INTERVAL_HOURS, MATCH_THRESHOLD)
    _check_data_dir()

    logger.info("Running initial sync now …")
    sync_job()

    schedule.every(SYNC_INTERVAL_HOURS).hours.do(sync_job)

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:  # noqa: BLE001
            # Never let one bad run kill the daemon.
            logger.exception("Scheduler error: %s", exc)
        time.sleep(30)


if __name__ == "__main__":
    main()
