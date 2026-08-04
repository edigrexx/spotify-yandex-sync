"""Unit tests for the SQLite sync store."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import (  # noqa: E402
    NOT_FOUND_MAX_ATTEMPTS,
    NOT_FOUND_RETRY_DAYS,
    SP_TO_YA,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_SYNCED,
    YA_TO_SP,
    SyncStore,
)


@pytest.fixture()
def store(tmp_path):
    with SyncStore(str(tmp_path / "sync.db")) as s:
        yield s


def _age(store: SyncStore, direction: str, key: str, days: int) -> None:
    """Backdate a row's last attempt, to exercise the retry schedule."""
    when = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    store._db.execute(
        "UPDATE sync_state SET last_attempt = ? WHERE direction = ? AND key = ?",
        (when, direction, key),
    )
    store._db.commit()


class TestAttemptPolicy:
    def test_unknown_key_is_attempted(self, store):
        assert store.should_attempt(SP_TO_YA, "new|track") is True

    def test_synced_key_is_never_attempted_again(self, store):
        store.mark_synced(SP_TO_YA, "a|b", dst_id="123", score=0.95)
        assert store.should_attempt(SP_TO_YA, "a|b") is False

    def test_error_always_retries(self, store):
        # This is the bug that retired tracks on a transient 451.
        store.mark_error(SP_TO_YA, "a|b", "451 Unavailable For Legal Reasons")
        assert store.should_attempt(SP_TO_YA, "a|b") is True

    def test_not_found_does_not_retry_immediately(self, store):
        store.mark_not_found(SP_TO_YA, "a|b", best_score=0.4)
        assert store.should_attempt(SP_TO_YA, "a|b") is False

    def test_not_found_retries_after_the_wait(self, store):
        store.mark_not_found(SP_TO_YA, "a|b")
        _age(store, SP_TO_YA, "a|b", NOT_FOUND_RETRY_DAYS + 1)
        assert store.should_attempt(SP_TO_YA, "a|b") is True

    def test_not_found_gives_up_after_max_attempts(self, store):
        for _ in range(NOT_FOUND_MAX_ATTEMPTS):
            store.mark_not_found(SP_TO_YA, "a|b")
            _age(store, SP_TO_YA, "a|b", NOT_FOUND_RETRY_DAYS + 1)
        assert store.should_attempt(SP_TO_YA, "a|b") is False

    def test_corrupt_timestamp_retries(self, store):
        store.mark_not_found(SP_TO_YA, "a|b")
        store._db.execute("UPDATE sync_state SET last_attempt = 'garbage'")
        store._db.commit()
        assert store.should_attempt(SP_TO_YA, "a|b") is True


class TestDirections:
    def test_directions_are_independent(self, store):
        store.mark_synced(SP_TO_YA, "shared-key")
        assert store.should_attempt(SP_TO_YA, "shared-key") is False
        assert store.should_attempt(YA_TO_SP, "shared-key") is True


class TestRecording:
    def test_synced_row_keeps_both_ids_and_score(self, store):
        store.mark_synced(SP_TO_YA, "a|b", src_id="sp1", dst_id="ya1", score=0.91)
        row = store.get(SP_TO_YA, "a|b")
        assert row["status"] == STATUS_SYNCED
        assert row["src_id"] == "sp1"
        assert row["dst_id"] == "ya1"
        assert row["score"] == pytest.approx(0.91)

    def test_attempts_accumulate(self, store):
        store.mark_not_found(SP_TO_YA, "a|b")
        store.mark_not_found(SP_TO_YA, "a|b")
        assert store.get(SP_TO_YA, "a|b")["attempts"] == 2

    def test_error_then_success_upgrades_the_row(self, store):
        store.mark_error(SP_TO_YA, "a|b", "boom")
        assert store.get(SP_TO_YA, "a|b")["status"] == STATUS_ERROR
        store.mark_synced(SP_TO_YA, "a|b", dst_id="ya9")
        row = store.get(SP_TO_YA, "a|b")
        assert row["status"] == STATUS_SYNCED
        assert row["dst_id"] == "ya9"

    def test_later_rows_do_not_lose_earlier_ids(self, store):
        store.mark_synced(SP_TO_YA, "a|b", src_id="sp1", dst_id="ya1", score=0.9)
        store.mark_synced(SP_TO_YA, "a|b")
        row = store.get(SP_TO_YA, "a|b")
        assert row["src_id"] == "sp1" and row["dst_id"] == "ya1"

    def test_counts_by_status(self, store):
        store.mark_synced(SP_TO_YA, "one")
        store.mark_synced(SP_TO_YA, "two")
        store.mark_not_found(SP_TO_YA, "three")
        store.mark_error(SP_TO_YA, "four", "boom")
        assert store.counts(SP_TO_YA) == {
            STATUS_SYNCED: 2, STATUS_NOT_FOUND: 1, STATUS_ERROR: 1,
        }


class TestMigration:
    def test_imports_flat_cache(self, store, tmp_path):
        legacy = tmp_path / "synced_spotify_to_yandex.txt"
        legacy.write_text("artist|title\nother|song\n\n", encoding="utf-8")
        assert store.import_legacy_cache(SP_TO_YA, str(legacy)) == 2
        assert store.should_attempt(SP_TO_YA, "artist|title") is False
        assert store.should_attempt(SP_TO_YA, "other|song") is False

    def test_missing_file_is_not_an_error(self, store, tmp_path):
        assert store.import_legacy_cache(SP_TO_YA, str(tmp_path / "nope.txt")) == 0

    def test_import_does_not_clobber_richer_rows(self, store, tmp_path):
        store.mark_synced(SP_TO_YA, "artist|title", dst_id="ya1", score=0.99)
        legacy = tmp_path / "cache.txt"
        legacy.write_text("artist|title\n", encoding="utf-8")
        store.import_legacy_cache(SP_TO_YA, str(legacy))
        row = store.get(SP_TO_YA, "artist|title")
        assert row["dst_id"] == "ya1"
        assert row["score"] == pytest.approx(0.99)

    def test_import_is_idempotent(self, store, tmp_path):
        legacy = tmp_path / "cache.txt"
        legacy.write_text("a|b\nc|d\n", encoding="utf-8")
        store.import_legacy_cache(SP_TO_YA, str(legacy))
        store.import_legacy_cache(SP_TO_YA, str(legacy))
        assert store.counts(SP_TO_YA)[STATUS_SYNCED] == 2


class TestBaseline:
    def test_bulk_mark_and_is_empty(self, store):
        assert store.is_empty(YA_TO_SP) is True
        store.bulk_mark_synced(YA_TO_SP, ["1", "2", "3"])
        assert store.is_empty(YA_TO_SP) is False
        assert store.counts(YA_TO_SP)[STATUS_SYNCED] == 3

    def test_bulk_mark_is_empty_safe(self, store):
        assert store.bulk_mark_synced(YA_TO_SP, []) == 0


class TestPersistence:
    def test_state_survives_reopen(self, tmp_path):
        path = str(tmp_path / "sync.db")
        with SyncStore(path) as first:
            first.mark_synced(SP_TO_YA, "a|b", dst_id="ya1")
        with SyncStore(path) as second:
            assert second.should_attempt(SP_TO_YA, "a|b") is False

    def test_creates_missing_directory(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "sync.db")
        with SyncStore(path) as s:
            s.mark_synced(SP_TO_YA, "a|b")
            assert s.should_attempt(SP_TO_YA, "a|b") is False
