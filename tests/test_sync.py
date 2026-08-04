"""
Integration tests for the sync directions, driven by fake API clients.

These cover the behaviours that broke in production: transient errors being
recorded as permanent misses, the Yandex direction paying for metadata it did
not need, echoes bouncing between the two services, and a Spotify like that is
accepted but does not actually land.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from store import SP_TO_YA, STATUS_ERROR, STATUS_NOT_FOUND, STATUS_SYNCED, YA_TO_SP, SyncStore  # noqa: E402


@pytest.fixture(autouse=True)
def no_rate_limit_delay(monkeypatch):
    monkeypatch.setattr(main, "YANDEX_REQUEST_DELAY", 0)
    # The flag is process-wide; reset it so tests do not leak into each other.
    monkeypatch.setattr(main, "_verify_likes", True)


@pytest.fixture()
def store(tmp_path):
    with SyncStore(str(tmp_path / "sync.db")) as s:
        yield s


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeArtist:
    def __init__(self, name):
        self.name = name


class FakeYandexTrack:
    def __init__(self, track_id, artists, title, duration_ms=200_000, version=None):
        self.id = track_id
        self.artists = [FakeArtist(a) for a in artists]
        self.title = title
        self.duration_ms = duration_ms
        self.version = version
        self.liked = False

    def like(self):
        self.liked = True
        return True


class FakeTrackShort:
    def __init__(self, track_id, timestamp):
        self.id = track_id
        self.timestamp = timestamp


class FakeYandex:
    """Minimal stand-in for yandex_music.Client."""

    def __init__(self, catalogue=None, liked=None, search_error=None):
        self.catalogue = catalogue or []
        self._liked = liked or []
        self.search_error = search_error
        self.search_calls: list[str] = []
        self.tracks_calls: list[list[str]] = []

    def users_likes_tracks(self):
        return self._liked

    def tracks(self, ids):
        self.tracks_calls.append(list(ids))
        wanted = {str(i) for i in ids}
        return [t for t in self.catalogue if str(t.id) in wanted]

    def search(self, query, type_="track"):
        self.search_calls.append(query)
        if self.search_error:
            raise self.search_error

        class Results:
            def __init__(self, results):
                self.results = results

        class Search:
            def __init__(self, results):
                self.tracks = Results(results) if results else None

        terms = {w for w in query.lower().split() if len(w) > 2}
        hits = [t for t in self.catalogue
                if terms & {w for w in
                            (t.title + " " + " ".join(a.name for a in t.artists)).lower().split()}]
        return Search(hits)


class FakeSpotify:
    """Minimal stand-in for spotipy.Spotify."""

    def __init__(self, liked=None, catalogue=None, search_error=None,
                 contains_result=True, library_error=None, tracks_add_error=None,
                 contains_error=None):
        self.liked = liked or []
        self.catalogue = catalogue or []
        self.search_error = search_error
        self.contains_result = contains_result
        self.contains_error = contains_error
        self.contains_calls = 0
        self.library_error = library_error
        self.tracks_add_error = tracks_add_error
        self.added: list[str] = []
        self.search_calls: list[str] = []
        self.put_calls: list[str] = []

    def current_user_saved_tracks(self, limit=50, offset=0):
        return {"items": [{"track": t} for t in self.liked[offset:offset + limit]]}

    def search(self, q, type="track", limit=10):  # noqa: A002
        self.search_calls.append(q)
        if self.search_error:
            raise self.search_error
        terms = {w for w in q.lower().split() if len(w) > 2}
        hits = [t for t in self.catalogue
                if terms & {w for w in
                            (t["name"] + " " +
                             " ".join(a["name"] for a in t["artists"])).lower().split()}]
        return {"tracks": {"items": hits[:limit]}}

    def _put(self, url):
        self.put_calls.append(url)
        if self.library_error:
            raise self.library_error
        self.added.append(url.rsplit(":", 1)[-1])

    def current_user_saved_tracks_add(self, ids):
        if self.tracks_add_error:
            raise self.tracks_add_error
        self.added.extend(ids)

    def current_user_saved_tracks_contains(self, ids):
        self.contains_calls += 1
        if self.contains_error:
            raise self.contains_error
        return [self.contains_result for _ in ids]


def sp_track(track_id, artists, name, duration_ms=200_000):
    return {"id": track_id, "name": name, "duration_ms": duration_ms,
            "artists": [{"name": a} for a in artists]}


# ---------------------------------------------------------------------------
# Spotify → Yandex
# ---------------------------------------------------------------------------

class TestSpotifyToYandex:
    def test_likes_a_matching_track(self, store):
        sp = FakeSpotify(liked=[sp_track("sp1", ["Papa Roach"], "Kill The Noise", 199_000)])
        yandex_track = FakeYandexTrack("ya1", ["Papa Roach"], "Kill The Noise", 199_000)
        ya = FakeYandex(catalogue=[yandex_track])

        main.sync_spotify_to_yandex(sp, ya, store)

        assert yandex_track.liked is True
        row = store.get(SP_TO_YA, "papa roach|kill the noise")
        assert row["status"] == STATUS_SYNCED
        assert row["dst_id"] == "ya1"

    def test_records_the_mirror_to_suppress_echo(self, store):
        # After liking on Yandex, the reverse direction must not treat the same
        # track as a brand new Yandex like and push it back to Spotify.
        sp = FakeSpotify(liked=[sp_track("sp1", ["Papa Roach"], "Kill The Noise", 199_000)])
        ya = FakeYandex(catalogue=[FakeYandexTrack("ya1", ["Papa Roach"],
                                                   "Kill The Noise", 199_000)])

        main.sync_spotify_to_yandex(sp, ya, store)

        assert store.should_attempt(YA_TO_SP, "ya1") is False

    def test_cross_script_match_is_found(self, store):
        # Spotify spells the artist in Latin, Yandex in Cyrillic.
        sp = FakeSpotify(liked=[sp_track("sp1", ["Basta"], "Медлячок", 215_000)])
        yandex_track = FakeYandexTrack("ya1", ["Баста"], "Медлячок", 215_000)
        ya = FakeYandex(catalogue=[yandex_track])

        main.sync_spotify_to_yandex(sp, ya, store)

        assert yandex_track.liked is True

    def test_wrong_artist_is_not_liked(self, store):
        sp = FakeSpotify(liked=[sp_track("sp1", ["Bring Me The Horizon"], "Doomed", 285_000)])
        impostor = FakeYandexTrack("ya9", ["Some Other Band"], "Doomed", 285_000)
        ya = FakeYandex(catalogue=[impostor])

        main.sync_spotify_to_yandex(sp, ya, store)

        assert impostor.liked is False
        assert store.get(SP_TO_YA, "bring me the horizon|doomed")["status"] == STATUS_NOT_FOUND

    def test_transient_error_is_not_recorded_as_not_found(self, store):
        # The production bug: a 451 during search permanently retired the track.
        sp = FakeSpotify(liked=[sp_track("sp1", ["Papa Roach"], "Kill The Noise")])
        ya = FakeYandex(search_error=RuntimeError("451 Unavailable For Legal Reasons"))

        main.sync_spotify_to_yandex(sp, ya, store)

        key = "papa roach|kill the noise"
        assert store.get(SP_TO_YA, key)["status"] == STATUS_ERROR
        assert store.should_attempt(SP_TO_YA, key) is True

    def test_error_then_recovery_syncs_the_track(self, store):
        sp = FakeSpotify(liked=[sp_track("sp1", ["Papa Roach"], "Kill The Noise", 199_000)])
        main.sync_spotify_to_yandex(sp, FakeYandex(search_error=RuntimeError("boom")), store)

        recovered = FakeYandexTrack("ya1", ["Papa Roach"], "Kill The Noise", 199_000)
        main.sync_spotify_to_yandex(sp, FakeYandex(catalogue=[recovered]), store)

        assert recovered.liked is True
        assert store.get(SP_TO_YA, "papa roach|kill the noise")["status"] == STATUS_SYNCED

    def test_already_synced_track_costs_no_search(self, store):
        store.mark_synced(SP_TO_YA, "papa roach|kill the noise")
        sp = FakeSpotify(liked=[sp_track("sp1", ["Papa Roach"], "Kill The Noise")])
        ya = FakeYandex(catalogue=[])

        main.sync_spotify_to_yandex(sp, ya, store)

        assert ya.search_calls == []


# ---------------------------------------------------------------------------
# Yandex → Spotify
# ---------------------------------------------------------------------------

class TestYandexToSpotify:
    def test_likes_a_matching_track(self, store):
        ya = FakeYandex(
            liked=[FakeTrackShort("ya1", "2026-08-04T10:00:00+00:00")],
            catalogue=[FakeYandexTrack("ya1", ["ENMY"], "Incomplete", 190_000)],
        )
        sp = FakeSpotify(catalogue=[sp_track("sp1", ["ENMY"], "Incomplete", 190_000)])

        main.sync_yandex_to_spotify(sp, ya, store)

        assert sp.added == ["sp1"]
        assert store.get(YA_TO_SP, "ya1")["status"] == STATUS_SYNCED

    def test_skips_metadata_fetch_when_nothing_is_new(self, store):
        # The expensive part is fetching full track objects; diffing the id list
        # first means a quiet run costs a single API call.
        store.mark_synced(YA_TO_SP, "ya1")
        ya = FakeYandex(
            liked=[FakeTrackShort("ya1", "2026-08-04T10:00:00+00:00")],
            catalogue=[FakeYandexTrack("ya1", ["ENMY"], "Incomplete")],
        )
        sp = FakeSpotify()

        main.sync_yandex_to_spotify(sp, ya, store)

        assert ya.tracks_calls == []
        assert sp.added == []

    def test_newest_likes_are_processed_first(self, store):
        # Ordering must come from the like timestamp, not from API response order.
        ya = FakeYandex(
            liked=[
                FakeTrackShort("old", "2020-01-01T00:00:00+00:00"),
                FakeTrackShort("new", "2026-08-04T10:00:00+00:00"),
            ],
            catalogue=[],
        )
        shorts = main.get_yandex_liked_shorts(ya)
        assert [s.id for s in shorts] == ["new", "old"]

    def test_composite_track_id_is_split(self, store):
        ya = FakeYandex(
            liked=[FakeTrackShort("123:456", "2026-08-04T10:00:00+00:00")],
            catalogue=[FakeYandexTrack("123", ["ENMY"], "Incomplete", 190_000)],
        )
        sp = FakeSpotify(catalogue=[sp_track("sp1", ["ENMY"], "Incomplete", 190_000)])

        main.sync_yandex_to_spotify(sp, ya, store)

        assert ya.tracks_calls == [["123"]]
        assert sp.added == ["sp1"]

    def test_unsaved_like_is_treated_as_a_failure(self, store):
        # Spotify accepted the write but the track is not in the library.
        ya = FakeYandex(
            liked=[FakeTrackShort("ya1", "2026-08-04T10:00:00+00:00")],
            catalogue=[FakeYandexTrack("ya1", ["ENMY"], "Incomplete", 190_000)],
        )
        sp = FakeSpotify(catalogue=[sp_track("sp1", ["ENMY"], "Incomplete", 190_000)],
                         contains_result=False)

        main.sync_yandex_to_spotify(sp, ya, store)

        assert store.get(YA_TO_SP, "ya1")["status"] == STATUS_ERROR
        assert store.should_attempt(YA_TO_SP, "ya1") is True

    def test_transient_spotify_error_is_retryable(self, store):
        ya = FakeYandex(
            liked=[FakeTrackShort("ya1", "2026-08-04T10:00:00+00:00")],
            catalogue=[FakeYandexTrack("ya1", ["ENMY"], "Incomplete")],
        )
        sp = FakeSpotify(search_error=RuntimeError("429 Too Many Requests"))

        main.sync_yandex_to_spotify(sp, ya, store)

        assert store.get(YA_TO_SP, "ya1")["status"] == STATUS_ERROR
        assert store.should_attempt(YA_TO_SP, "ya1") is True

    def test_records_the_mirror_to_suppress_echo(self, store):
        ya = FakeYandex(
            liked=[FakeTrackShort("ya1", "2026-08-04T10:00:00+00:00")],
            catalogue=[FakeYandexTrack("ya1", ["ENMY"], "Incomplete", 190_000)],
        )
        sp = FakeSpotify(catalogue=[sp_track("sp1", ["ENMY"], "Incomplete", 190_000)])

        main.sync_yandex_to_spotify(sp, ya, store)

        assert store.should_attempt(SP_TO_YA, "enmy|incomplete") is False


# ---------------------------------------------------------------------------
# Spotify write endpoints
# ---------------------------------------------------------------------------

class TestSpotifyLikeEndpoints:
    def test_uses_the_library_endpoint_first(self):
        sp = FakeSpotify()
        assert main.like_track_on_spotify(sp, "sp1") is True
        assert sp.put_calls == ["me/library?uris=spotify:track:sp1"]

    def test_falls_back_when_library_is_forbidden(self):
        sp = FakeSpotify(library_error=RuntimeError("http status: 403"))
        assert main.like_track_on_spotify(sp, "sp1") is True
        assert sp.added == ["sp1"]

    def test_fails_only_when_every_endpoint_fails(self):
        sp = FakeSpotify(library_error=RuntimeError("403"),
                         tracks_add_error=RuntimeError("403"))
        assert main.like_track_on_spotify(sp, "sp1") is False
        assert sp.added == []

    def test_forbidden_verification_does_not_fail_the_like(self):
        sp = FakeSpotify(contains_error=RuntimeError("http status: 403"))
        assert main.like_track_on_spotify(sp, "sp1") is True

    def test_verification_is_probed_once_then_dropped(self):
        # Retrying a forbidden read-back per track costs a request and logs an
        # error every time, for a save that actually worked.
        sp = FakeSpotify(contains_error=RuntimeError("http status: 403"))
        for track_id in ("sp1", "sp2", "sp3"):
            assert main.like_track_on_spotify(sp, track_id) is True
        assert sp.contains_calls == 1

    def test_verification_still_catches_a_missing_track(self):
        sp = FakeSpotify(contains_result=False)
        assert main.like_track_on_spotify(sp, "sp1") is False

    def test_reports_both_failures(self, caplog):
        sp = FakeSpotify(library_error=RuntimeError("library boom"),
                         tracks_add_error=RuntimeError("tracks boom"))
        with caplog.at_level("WARNING"):
            main.like_track_on_spotify(sp, "sp1")
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "library boom" in logged and "tracks boom" in logged


# ---------------------------------------------------------------------------
# Spotify paging
# ---------------------------------------------------------------------------

class TestSpotifyPaging:
    def test_reads_beyond_the_first_page(self, store):
        # 60 new likes used to lose the 10 that fell outside the first page.
        liked = [sp_track(f"sp{i}", ["Artist"], f"Song {i}") for i in range(60)]
        sp = FakeSpotify(liked=liked)

        fetched = main.get_spotify_liked_tracks(sp, store)

        assert len(fetched) == 60

    def test_stops_early_once_pages_are_all_known(self, store):
        liked = [sp_track(f"sp{i}", ["Artist"], f"Song {i}") for i in range(200)]
        for i in range(200):
            store.mark_synced(SP_TO_YA, main.normalize_key("Artist", f"Song {i}"))
        sp = FakeSpotify(liked=liked)

        fetched = main.get_spotify_liked_tracks(sp, store)

        # Two quiet pages are enough to stop; it must not walk all 200.
        assert len(fetched) == 100


# ---------------------------------------------------------------------------
# Yandex metadata fetching
# ---------------------------------------------------------------------------

class TestFetchYandexTracks:
    def test_falls_back_to_single_fetches(self, store):
        class FlakyYandex(FakeYandex):
            def tracks(self, ids):
                self.tracks_calls.append(list(ids))
                if len(ids) > 1:
                    raise RuntimeError("Artist.__init__() missing 1 required argument: 'id'")
                return super().tracks(ids)

        catalogue = [FakeYandexTrack(str(i), ["A"], f"S{i}") for i in range(3)]
        ya = FlakyYandex(catalogue=catalogue)

        fetched = main.fetch_yandex_tracks(ya, ["0", "1", "2"])

        assert {t.id for t in fetched} == {"0", "1", "2"}

    def test_reports_tracks_it_could_not_fetch(self, caplog, store):
        ya = FakeYandex(catalogue=[])

        with caplog.at_level("WARNING"):
            fetched = main.fetch_yandex_tracks(ya, ["missing"])

        assert fetched == []
        assert any("could not be fetched" in r.getMessage() for r in caplog.records)
