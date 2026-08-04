"""Unit tests for the track matching rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matching import (  # noqa: E402
    DEFAULT_THRESHOLD,
    TrackMeta,
    artist_similarity,
    clean,
    duration_similarity,
    normalize,
    pick_best,
    query_variants,
    score,
    similarity,
    split_artists,
    split_featured,
    strip_noise,
    transliterate,
)


def meta(artists, title, duration_ms=None, source_id=None) -> TrackMeta:
    if isinstance(artists, str):
        artists = [artists]
    return TrackMeta(artists=tuple(artists), title=title,
                     duration_ms=duration_ms, source_id=source_id)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalize:
    @pytest.mark.parametrize("raw,expected", [
        ("Beyoncé", "beyonce"),
        ("Ёлка", "елка"),
        ("Guns N' Roses", "guns n roses"),
        ("Guns N’ Roses", "guns n roses"),
        ("AC/DC", "ac dc"),
        ("Simon & Garfunkel", "simon and garfunkel"),
        ("  spaced   out  ", "spaced out"),
        ("", ""),
    ])
    def test_normalize(self, raw, expected):
        assert normalize(raw) == expected


class TestTransliterate:
    @pytest.mark.parametrize("cyrillic,latin", [
        ("Баста", "Basta"),
        ("Моя Мишель", "Moya Mishel"),
        ("Ленинград", "Leningrad"),
        ("Щербаков", "Scherbakov"),
    ])
    def test_matches_latin_spelling(self, cyrillic, latin):
        assert normalize(transliterate(cyrillic)) == normalize(latin)

    def test_latin_passes_through(self):
        assert transliterate("Papa Roach") == "Papa Roach"


# ---------------------------------------------------------------------------
# Title cleanup
# ---------------------------------------------------------------------------

class TestStripNoise:
    @pytest.mark.parametrize("raw,expected", [
        ("Song - Remastered 2011", "Song"),
        ("Song (2011 Remaster)", "Song"),
        ("Song (Deluxe Edition)", "Song"),
        ("Song - Radio Edit", "Song"),
        ("Song (Bonus Track)", "Song"),
        ("Song (Album Version)", "Song"),
        ("Song [Explicit]", "Song"),
        ("Song (Deluxe) - Remastered 2011", "Song"),
        ("Song", "Song"),
    ])
    def test_strips(self, raw, expected):
        assert strip_noise(raw) == expected

    def test_keeps_meaningful_parentheses(self):
        assert strip_noise("Song (Reprise)") == "Song (Reprise)"

    def test_does_not_eat_whole_title(self):
        # A title that is nothing but a noise word must survive cleanup.
        assert clean(meta("X", "Remaster")).clean_title


class TestSplitFeatured:
    @pytest.mark.parametrize("raw,title,featured", [
        ("Song (feat. Drake)", "Song", ["Drake"]),
        ("Song (ft. Drake)", "Song", ["Drake"]),
        ("Song feat. Drake", "Song", ["Drake"]),
        ("Song (with Drake)", "Song", ["Drake"]),
        ("Song (feat. Drake & Future)", "Song", ["Drake", "Future"]),
        ("Song", "Song", []),
    ])
    def test_split(self, raw, title, featured):
        got_title, got_featured = split_featured(raw)
        assert got_title == title
        assert got_featured == featured


class TestSplitArtists:
    @pytest.mark.parametrize("raw,expected", [
        ("Basta, Moya Mishel", ["Basta", "Moya Mishel"]),
        ("Simon & Garfunkel", ["Simon", "Garfunkel"]),
        ("A vs. B", ["A", "B"]),
        ("Solo", ["Solo"]),
    ])
    def test_split(self, raw, expected):
        assert split_artists(raw) == expected


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_identical(self):
        assert similarity("Doomed", "Doomed") == 1.0

    def test_cross_script(self):
        # The exact case from the production log.
        assert similarity("Basta", "Баста") > 0.9
        assert similarity("Moya Mishel", "Моя Мишель") > 0.9

    def test_unrelated(self):
        assert similarity("Doomed", "Sandstorm") < 0.5

    def test_word_order(self):
        assert similarity("Kill The Noise", "The Noise Kill") > 0.8


class TestArtistSimilarity:
    def test_single_shared_artist_is_enough(self):
        # Spotify credits a collaboration, Yandex credits only the lead.
        assert artist_similarity(["Basta", "Moya Mishel"], ["Баста"]) > 0.9

    def test_no_overlap(self):
        assert artist_similarity(["Papa Roach"], ["Nickelback"]) < 0.6

    def test_empty(self):
        assert artist_similarity([], ["X"]) == 0.0


class TestDurationSimilarity:
    def test_near_identical(self):
        assert duration_similarity(210_000, 211_000) == 1.0

    def test_far_apart(self):
        assert duration_similarity(210_000, 400_000) == 0.0

    def test_unknown(self):
        assert duration_similarity(None, 210_000) is None
        assert duration_similarity(210_000, None) is None

    def test_tapers(self):
        mid = duration_similarity(210_000, 220_000)
        assert 0.0 < mid < 1.0


# ---------------------------------------------------------------------------
# Scoring and selection
# ---------------------------------------------------------------------------

class TestScore:
    def test_exact_match_scores_high(self):
        src = meta("Papa Roach", "Kill The Noise", 199_000)
        cand = meta("Papa Roach", "Kill The Noise", 199_500)
        assert score(src, cand) > 0.95

    def test_cross_script_match_clears_threshold(self):
        src = meta(["Basta", "Moya Mishel"], "Если я буду танцевать", 215_000)
        cand = meta(["Баста", "Моя Мишель"], "Если я буду танцевать", 215_000)
        assert score(src, cand) > DEFAULT_THRESHOLD

    def test_remaster_noise_does_not_break_match(self):
        src = meta("Nirvana", "Come As You Are - Remastered 2011", 219_000)
        cand = meta("Nirvana", "Come As You Are", 219_000)
        assert score(src, cand) > DEFAULT_THRESHOLD

    def test_featured_artist_placement_does_not_break_match(self):
        src = meta("Eminem", "Stan (feat. Dido)", 404_000)
        cand = meta(["Eminem", "Dido"], "Stan", 404_000)
        assert score(src, cand) > DEFAULT_THRESHOLD

    def test_same_title_different_artist_is_rejected(self):
        src = meta("Bring Me The Horizon", "Doomed", 285_000)
        cand = meta("Some Other Band", "Doomed", 285_000)
        assert score(src, cand) < DEFAULT_THRESHOLD

    def test_duration_mismatch_penalises(self):
        src = meta("Artist", "Song", 200_000)
        near = meta("Artist", "Song", 201_000)
        far = meta("Artist", "Song", 380_000)
        assert score(src, near) > score(src, far)

    def test_karaoke_is_rejected(self):
        src = meta("Papa Roach", "Last Resort", 200_000)
        cand = meta("Karaoke Band", "Last Resort (Karaoke Version)", 200_000)
        assert score(src, cand) < DEFAULT_THRESHOLD

    def test_live_version_scores_below_studio(self):
        src = meta("Muse", "Hysteria", 227_000)
        studio = meta("Muse", "Hysteria", 227_000)
        live = meta("Muse", "Hysteria (Live)", 227_000)
        assert score(src, live) < score(src, studio)


class TestPickBest:
    def test_picks_highest_scoring(self):
        src = meta("Papa Roach", "Kill The Noise", 199_000)
        candidates = [
            meta("Other Band", "Kill The Noise", 199_000, source_id="wrong"),
            meta("Papa Roach", "Kill The Noise", 199_000, source_id="right"),
            meta("Papa Roach", "Born For Greatness", 210_000, source_id="other"),
        ]
        best, value = pick_best(src, candidates)
        assert best is not None and best.source_id == "right"
        assert value > 0.9

    def test_rejects_when_nothing_clears_threshold(self):
        src = meta("Bring Me The Horizon", "Doomed", 285_000)
        candidates = [meta("Unrelated", "Completely Different", 100_000)]
        best, value = pick_best(src, candidates)
        assert best is None
        assert value < DEFAULT_THRESHOLD

    def test_empty_candidates(self):
        best, value = pick_best(meta("A", "B"), [])
        assert best is None and value == 0.0

    def test_reports_near_miss_score(self):
        # A rejected match still reports how close it came, for tuning.
        src = meta("Artist", "Song", 200_000)
        candidates = [meta("Artist", "Different Song Entirely", 200_000)]
        best, value = pick_best(src, candidates)
        assert best is None
        assert 0.0 < value < DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

class TestQueryVariants:
    def test_most_specific_first(self):
        variants = query_variants(meta(["Basta", "Moya Mishel"], "Если я буду танцевать"))
        assert variants[0] == "Basta Moya Mishel Если я буду танцевать"

    def test_drops_noise_from_leading_query(self):
        variants = query_variants(meta("Nirvana", "Come As You Are - Remastered 2011"))
        assert variants[0] == "Nirvana Come As You Are"

    def test_keeps_raw_title_as_a_later_fallback(self):
        # Stripping is heuristic, so the untouched title stays available.
        variants = query_variants(meta("Nirvana", "Come As You Are - Remastered 2011"))
        assert any("Remastered" in v for v in variants[1:])

    def test_includes_title_only_fallback(self):
        variants = query_variants(meta("Artist", "Unique Title"))
        assert "Unique Title" in variants

    def test_no_duplicates(self):
        variants = query_variants(meta("Artist", "Song"))
        assert len(variants) == len({normalize(v) for v in variants})

    def test_adds_transliterated_variant_for_cyrillic(self):
        variants = query_variants(meta("Баста", "Медлячок"))
        assert any("basta" in v.lower() for v in variants)
