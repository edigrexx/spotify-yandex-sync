"""
matching.py
-----------
Track matching between Spotify and Yandex Music.

The naive approach — search for "Artist - Title" and take the first result —
fails in both directions.  It silently accepts whatever the search engine
returned first (a cover, a karaoke version, a different song with the same
name), and it misses correct matches because the two catalogues spell things
differently: Spotify writes Russian acts in Latin ("Basta", "Moya Mishel")
while Yandex writes them in Cyrillic ("Баста", "Моя Мишель"), and each service
decorates titles with its own "- Remastered 2011" / "(Deluxe)" noise.

This module provides pure functions with no I/O, so the matching rules can be
unit-tested without touching either API:

    query_variants(meta)          -> search strings to try, best first
    pick_best(src, candidates)    -> (candidate, score) or (None, best_score)

Scoring blends title similarity, artist similarity and — when both sides report
it — track duration, which is the single strongest disambiguator available.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence

# Accept a candidate only at or above this blended score.
DEFAULT_THRESHOLD = 0.68

# Below these similarities the axis is treated as a disagreement, not a weak
# signal, and the candidate is pushed decisively below the threshold.
ARTIST_GATE = 0.45
TITLE_GATE = 0.60

# Durations within this many milliseconds count as a perfect duration match.
DURATION_EXACT_MS = 3_000
# Beyond this difference the duration signal contributes nothing.
DURATION_MAX_MS = 20_000

# Cyrillic -> Latin. Lets "Моя Мишель" compare against Spotify's "Moya Mishel".
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    # Ukrainian / Belarusian letters that show up in the same catalogues.
    "і": "i", "ї": "yi", "є": "ye", "ґ": "g", "ў": "u",
}

# Edition/mastering noise: meaningful to a label, meaningless for matching.
_NOISE_WORDS = (
    r"remaster(?:ed)?(?:\s+version)?",
    r"re-?master(?:ed)?",
    r"deluxe(?:\s+edition)?",
    r"expanded(?:\s+edition)?",
    r"anniversary(?:\s+edition)?",
    r"bonus\s+track",
    r"single\s+version",
    r"album\s+version",
    r"radio\s+edit",
    r"radio\s+version",
    r"explicit(?:\s+version)?",
    r"clean(?:\s+version)?",
    r"mono(?:\s+version)?",
    r"stereo(?:\s+version)?",
    r"original\s+mix",
    r"digital\s+remaster(?:ed)?",
    r"\d{4}\s+remaster(?:ed)?",
    r"remaster(?:ed)?\s+\d{4}",
    r"bonus",
)

# Parenthesised or dash-suffixed noise, e.g. "Song (2011 Remaster)" or
# "Song - Remastered 2011".
_NOISE_BRACKET_RE = re.compile(
    r"[\(\[\{]\s*[^\)\]\}]*\b(?:" + "|".join(_NOISE_WORDS) + r")\b[^\)\]\}]*[\)\]\}]",
    re.IGNORECASE,
)
_NOISE_DASH_RE = re.compile(
    r"\s[-–—]\s[^-–—]*\b(?:" + "|".join(_NOISE_WORDS) + r")\b[^-–—]*$",
    re.IGNORECASE,
)

# "feat." and friends, either bracketed or trailing.
_FEAT_RE = re.compile(
    r"[\(\[]\s*(?:feat|ft|featuring|with)\.?\s+([^\)\]]+)[\)\]]"
    r"|\s(?:feat|ft|featuring)\.?\s+(.+)$",
    re.IGNORECASE,
)

# Markers that must agree between the two sides: a studio track and its live
# rendition are different recordings, and a karaoke version is not the song.
_LIVE_RE = re.compile(r"\b(live|unplugged|acoustic|concert|концерт|живо[йе])\b", re.IGNORECASE)
_COVER_RE = re.compile(
    r"\b(karaoke|tribute|cover|instrumental|made\s+famous|минус|караоке|кавер)\b",
    re.IGNORECASE,
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ARTIST_SPLIT_RE = re.compile(
    r"\s*(?:,|;|&|/|\bfeat\b\.?|\bft\b\.?|\bvs\b\.?|\bx\b|\band\b|\bи\b)\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrackMeta:
    """A track as seen by either service, reduced to what matching needs."""

    artists: tuple[str, ...]
    title: str
    duration_ms: Optional[int] = None
    source_id: Optional[str] = None
    # Populated by clean(): the title with edition noise and feat. removed.
    clean_title: str = ""
    all_artists: tuple[str, ...] = field(default_factory=tuple)
    is_live: bool = False
    is_cover: bool = False

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""

    def label(self) -> str:
        return f"{', '.join(self.artists)} - {self.title}"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Drop combining marks so "Beyoncé" and "Beyonce" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def transliterate(text: str) -> str:
    """Render Cyrillic as Latin. Text already in Latin passes through."""
    return "".join(_TRANSLIT.get(ch, _TRANSLIT.get(ch.lower(), ch)) if ch.lower() in _TRANSLIT else ch
                   for ch in text)


def normalize(text: str) -> str:
    """Lowercase, de-accent, drop punctuation and collapse whitespace."""
    if not text:
        return ""
    text = text.lower().replace("ё", "е").replace("&", " and ")
    # Unify the various quote and dash characters catalogues disagree on.
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'",
                                         "“": '"', "”": '"', "–": "-", "—": "-"}))
    text = strip_accents(text)
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def split_featured(title: str) -> tuple[str, list[str]]:
    """Split "Song (feat. X)" into ("Song", ["X"])."""
    featured: list[str] = []

    def _collect(match: re.Match) -> str:
        blob = match.group(1) or match.group(2) or ""
        featured.extend(a for a in split_artists(blob) if a)
        return " "

    return _SPACE_RE.sub(" ", _FEAT_RE.sub(_collect, title)).strip(), featured


def split_artists(blob: str) -> list[str]:
    """Split a combined artist string into individual names."""
    if not blob:
        return []
    return [part.strip() for part in _ARTIST_SPLIT_RE.split(blob) if part.strip()]


def strip_noise(title: str) -> str:
    """Remove remaster/deluxe/edition decorations from a title."""
    previous = None
    while previous != title:
        previous = title
        title = _NOISE_BRACKET_RE.sub(" ", title)
        title = _NOISE_DASH_RE.sub("", title)
    # An empty leftover bracket pair adds nothing.
    title = re.sub(r"[\(\[\{]\s*[\)\]\}]", " ", title)
    return _SPACE_RE.sub(" ", title).strip(" -–—")


def clean(meta: TrackMeta) -> TrackMeta:
    """
    Fill in the derived fields used for scoring.

    Featured artists are moved out of the title and into the artist list,
    because the two services disagree about where they belong.
    """
    title_no_feat, featured = split_featured(meta.title)
    cleaned = strip_noise(title_no_feat)
    # Keep the original when stripping ate everything (e.g. title == "Live").
    if not normalize(cleaned):
        cleaned = title_no_feat or meta.title

    artists: list[str] = []
    for name in list(meta.artists) + featured:
        for part in split_artists(name) or [name]:
            if part and part not in artists:
                artists.append(part)

    haystack = f"{meta.title} {' '.join(meta.artists)}"
    return TrackMeta(
        artists=meta.artists,
        title=meta.title,
        duration_ms=meta.duration_ms,
        source_id=meta.source_id,
        clean_title=cleaned,
        all_artists=tuple(artists),
        is_live=bool(_LIVE_RE.search(haystack)),
        is_cover=bool(_COVER_RE.search(haystack)),
    )


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    base = SequenceMatcher(None, a, b).ratio()
    # Reward containment, but only in proportion to how much of the longer
    # string the shorter one covers. A flat bonus would rate "Song" inside
    # "Different Song Entirely" as highly as inside "Song (Extended)".
    if a in b or b in a:
        coverage = min(len(a), len(b)) / max(len(a), len(b))
        base = max(base, 0.5 + 0.4 * coverage)
    # Compare as token sets too, so word order differences do not sink a match.
    ta, tb = set(a.split()), set(b.split())
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        base = max(base, jaccard)
        # Two multi-word names that share no word at all are different names.
        # Character-level similarity between such names is coincidence — plain
        # English band names routinely score ~0.45 on shared letters alone.
        if jaccard == 0.0 and len(ta) > 1 and len(tb) > 1:
            base = min(base, 0.40)
    return base


def similarity(a: str, b: str) -> float:
    """
    Similarity of two names, tolerant of script differences.

    Compares the normalised strings directly and again after transliterating
    both, taking the better of the two — that is what lets Spotify's "Basta"
    match Yandex's "Баста".
    """
    na, nb = normalize(a), normalize(b)
    direct = _ratio(na, nb)
    if direct >= 0.999:
        return direct
    ta, tb = normalize(transliterate(a)), normalize(transliterate(b))
    return max(direct, _ratio(ta, tb))


def artist_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """
    Best-pair similarity between two artist lists.

    Collaborations are credited inconsistently, so a single shared artist is
    enough; averaging over all of them would punish correct matches.
    """
    if not left or not right:
        return 0.0
    best = max(similarity(a, b) for a in left for b in right)
    # A joint credit on both sides is extra confirmation.
    joined = similarity(" ".join(left), " ".join(right))
    return max(best, joined)


def duration_similarity(a: Optional[int], b: Optional[int]) -> Optional[float]:
    """1.0 for near-identical lengths, tapering to 0.0; None when unknown."""
    if not a or not b:
        return None
    delta = abs(a - b)
    if delta <= DURATION_EXACT_MS:
        return 1.0
    if delta >= DURATION_MAX_MS:
        return 0.0
    return 1.0 - (delta - DURATION_EXACT_MS) / (DURATION_MAX_MS - DURATION_EXACT_MS)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(src: TrackMeta, cand: TrackMeta) -> float:
    """
    Blended confidence that `cand` is the same recording as `src`.

    Title and artist always contribute; duration joins in when both sides
    report it, and the weights are renormalised when it does not.
    """
    if not src.clean_title:
        src = clean(src)
    if not cand.clean_title:
        cand = clean(cand)

    title = similarity(src.clean_title, cand.clean_title)
    artist = artist_similarity(src.all_artists or src.artists,
                               cand.all_artists or cand.artists)
    duration = duration_similarity(src.duration_ms, cand.duration_ms)

    if duration is None:
        total = title * 0.625 + artist * 0.375
    else:
        total = title * 0.5 + artist * 0.3 + duration * 0.2

    # Weights alone are not enough: an identical title and duration would drag
    # a wrong artist over the line on their own. Treat a clear disagreement on
    # either axis as disqualifying rather than merely expensive.
    if artist < ARTIST_GATE:
        total -= 0.35
    if title < TITLE_GATE:
        total -= 0.30

    # A studio track and a live take are different recordings; karaoke and
    # tribute versions are not the song at all.
    if src.is_live != cand.is_live:
        total -= 0.25
    if src.is_cover != cand.is_cover:
        total -= 0.40

    return max(0.0, min(1.0, total))


def pick_best(
    src: TrackMeta,
    candidates: Iterable[TrackMeta],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[Optional[TrackMeta], float]:
    """
    Choose the best candidate above `threshold`.

    Returns (None, best_score) when nothing clears the bar, so the caller can
    log how close the near-miss was and tune the threshold from real data.
    """
    src = clean(src)
    best: Optional[TrackMeta] = None
    best_score = 0.0
    for cand in candidates:
        value = score(src, clean(cand))
        if value > best_score:
            best, best_score = cand, value
    if best is not None and best_score >= threshold:
        return best, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

def query_variants(meta: TrackMeta) -> list[str]:
    """
    Search strings to try in order, most specific first.

    Later variants drop information deliberately: a track credited to three
    artists on one service and one on the other will only be found once the
    extra names are out of the query.
    """
    meta = clean(meta)
    title = meta.clean_title or meta.title
    all_artists = " ".join(meta.artists)
    primary = meta.primary_artist

    variants = [
        f"{all_artists} {title}",
        f"{title} {all_artists}",
        f"{primary} {title}",
        f"{primary} {meta.title}",
        title,
    ]

    # Transliterated forms help when the catalogues disagree on script.
    translit_primary, translit_title = transliterate(primary), transliterate(title)
    if (translit_primary, translit_title) != (primary, title):
        variants.append(f"{translit_primary} {translit_title}")

    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        collapsed = _SPACE_RE.sub(" ", variant).strip()
        key = normalize(collapsed)
        if collapsed and key and key not in seen:
            seen.add(key)
            ordered.append(collapsed)
    return ordered
