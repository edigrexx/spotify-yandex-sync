"""
get_yandex_token.py
-------------------
Interactive helper to obtain (and verify) a YANDEX_TOKEN for Yandex Music.

Yandex OAuth tokens are bound to the application they were issued for: a token
minted for some other Yandex app (Books, Disk, …) carries that app's scopes and
is silently NOT accepted by api.music.yandex.net.  The rejection does not look
like a 401 — the Music API just treats the request as anonymous, so
`/account/status` returns 200 without a `uid` and every later call fails with
`ownerOtherwiseUserBindingError`.  This script therefore always verifies the
token against the Music API before printing it.

Three modes:

  paste (default)  Print the authorization URL, then accept whatever you manage
                   to copy out of the browser — the full redirect URL, a bare
                   token, or a line with junk around it.

  serve            Run a local callback server and capture the token fully
                   automatically.  Requires your OWN Yandex app registered with
                   redirect URI http://127.0.0.1:8888/callback (see --help).

  check            Verify a token you already have (from $YANDEX_TOKEN or --token)
                   without going through the browser again.

Usage:
    python get_yandex_token.py                    # paste mode
    python get_yandex_token.py --serve --client-id <your_app_client_id>
    python get_yandex_token.py --check            # verifies $YANDEX_TOKEN
"""

from __future__ import annotations

import argparse
import http.server
import os
import re
import sys
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timedelta, timezone
from typing import Optional

# client_id of the official Yandex Music mobile application.  Tokens issued for
# it are accepted by api.music.yandex.net; tokens from self-registered apps
# usually are not, because Yandex does not expose music scopes to third parties.
MUSIC_CLIENT_ID = "23cabbbdc6cd418abb4b39c32c41195d"

AUTHORIZE_URL = "https://oauth.yandex.ru/authorize"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8888
CALLBACK_PATH = "/callback"

# Yandex tokens look like `y0_...` / `y0__...` (current) or a 39-char legacy blob.
TOKEN_RE = re.compile(r"\by0_[A-Za-z0-9_\-]{20,}\b|\bAQAA[A-Za-z0-9_\-]{20,}\b")


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def build_authorize_url(client_id: str, redirect_uri: Optional[str] = None,
                        force_confirm: bool = False) -> str:
    """Build an OAuth implicit-grant authorization URL."""
    params = {"response_type": "token", "client_id": client_id}
    if redirect_uri:
        params["redirect_uri"] = redirect_uri
    if force_confirm:
        params["force_confirm"] = "yes"
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def extract_token(raw: str) -> tuple[Optional[str], Optional[int]]:
    """
    Pull an access token out of whatever the user pasted.

    Handles, in order of preference:
      * a redirect URL carrying the token in the fragment (#access_token=…),
        which is where the implicit grant puts it;
      * a URL carrying it in the query string (?access_token=…);
      * a bare token, or a token embedded in surrounding junk.

    Returns (token, expires_in_seconds) — expires_in is None when unknown.
    """
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return None, None

    if "://" in raw or raw.startswith("#") or "access_token=" in raw:
        parsed = urllib.parse.urlparse(raw)
        # The fragment is where the implicit grant delivers the token; fall back
        # to the query string, and finally to the raw text after a lone '#'.
        for blob in (parsed.fragment, parsed.query, raw.lstrip("#")):
            if not blob:
                continue
            fields = urllib.parse.parse_qs(blob)
            token = (fields.get("access_token") or [None])[0]
            if token:
                expires_raw = (fields.get("expires_in") or [None])[0]
                expires = int(expires_raw) if expires_raw and expires_raw.isdigit() else None
                return token.strip(), expires

    match = TOKEN_RE.search(raw)
    if match:
        return match.group(0), None

    # Last resort: a single opaque word is probably the token itself.
    if len(raw.split()) == 1 and len(raw) >= 20:
        return raw, None

    return None, None


def normalize_token(token: str) -> str:
    """Strip whitespace and an accidentally copied `OAuth ` / `Bearer ` prefix."""
    token = token.strip().strip('"').strip("'")
    for prefix in ("OAuth ", "Bearer ", "oauth ", "bearer "):
        if token.startswith(prefix):
            token = token[len(prefix):].strip()
    return token


def mask(token: str) -> str:
    """Render a token safely for logs: keep only the head and tail."""
    if len(token) <= 14:
        return "*" * len(token)
    return f"{token[:8]}…{token[-4:]}"


# ---------------------------------------------------------------------------
# Verification against the Music API
# ---------------------------------------------------------------------------

def verify_token(token: str) -> Optional[bool]:
    """
    Confirm the token is actually accepted by the Yandex *Music* API.

    Returns True when the API reports a real, identified account, False when the
    token is refused, and None when verification could not run at all.  A token
    that is valid for some other Yandex service authenticates nothing here and
    yields an anonymous status object with no uid — which is exactly the failure
    this script exists to catch.
    """
    try:
        from yandex_music import Client
        from yandex_music.exceptions import UnauthorizedError, YandexMusicError
    except ImportError:
        print("\n⚠️  yandex-music is not installed — cannot verify the token.")
        print("    Run `pip install yandex-music` and re-run with --check.")
        return None

    print(f"\nVerifying token {mask(token)} against api.music.yandex.net …")
    try:
        client = Client(token=token).init()
    except UnauthorizedError:
        print("❌  Rejected: the token is expired or revoked. Get a fresh one.")
        return False
    except YandexMusicError as exc:
        print(f"❌  Music API error: {exc}")
        if "451" in str(exc):
            print("    451 means the API is geo-blocked from this machine, not")
            print("    that the token is bad — retry from a different network,")
            print("    and set YANDEX_PROXY_URL for the daemon's host.")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"❌  Unexpected error: {exc}")
        return False

    account = getattr(client.me, "account", None)
    uid = getattr(account, "uid", None)
    if not uid:
        print("❌  The Music API answered, but anonymously — no account uid.")
        print("    The token is valid for Yandex, but NOT for Yandex Music.")
        print("    This is what a token minted for another Yandex app looks")
        print(f"    like. Re-authorize using client_id={MUSIC_CLIENT_ID}.")
        return False

    print(f"✅  Authenticated as uid={uid} login={account.login!r} region={account.region}")

    plus = getattr(client.me, "plus", None)
    if plus is not None:
        print(f"    Yandex Plus: {'yes' if plus.has_plus else 'no'}")

    try:
        liked = client.users_likes_tracks()
        print(f"    Liked tracks visible to the API: {len(liked) if liked else 0}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Could not read liked tracks: {exc}")
        print("    Authentication works, but the sync daemon will still fail.")
        return False

    return True


# ---------------------------------------------------------------------------
# Local callback server (serve mode)
# ---------------------------------------------------------------------------

# The implicit grant returns the token in the URL *fragment*, which browsers
# never send to the server.  So the callback page is a tiny JS shim: it reads
# location.hash in the browser and POSTs the value back to us.
_CALLBACK_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Yandex Music token</title>
<style>
  body { font: 16px/1.5 system-ui, sans-serif; margin: 15vh auto; max-width: 32rem;
         padding: 0 1rem; text-align: center; }
  code { background: #f4f4f5; padding: .15em .4em; border-radius: .25em; }
</style>
<h2 id="s">Capturing token…</h2>
<p id="d"></p>
<script>
  const hash = location.hash.startsWith('#') ? location.hash.slice(1) : location.hash;
  const token = new URLSearchParams(hash).get('access_token');
  const status = document.getElementById('s'), detail = document.getElementById('d');
  if (!token) {
    status.textContent = 'No token in the URL';
    detail.textContent = 'Authorization was denied, or the redirect dropped the fragment.';
  } else {
    fetch('/token', { method: 'POST', body: hash })
      .then(() => { status.textContent = '✅ Token captured';
                    detail.innerHTML = 'You can close this tab and return to the terminal.'; })
      .catch(e => { status.textContent = 'Could not reach the local script';
                    detail.textContent = String(e); });
  }
</script>
"""


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Serves the shim page and receives the fragment the shim posts back."""

    captured: Optional[str] = None
    done = threading.Event()

    def _reply(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != CALLBACK_PATH:
            self._reply(404, b"not found", "text/plain; charset=utf-8")
            return
        self._reply(200, _CALLBACK_PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        _CallbackHandler.captured = self.rfile.read(length).decode("utf-8", "replace")
        self._reply(200, b'{"ok":true}', "application/json")
        _CallbackHandler.done.set()

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        """Silence the default per-request stderr logging."""


def serve_mode(client_id: str, timeout: int = 300) -> tuple[Optional[str], Optional[int]]:
    """Capture a token via a local redirect URI, with no copy-pasting at all."""
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
    url = build_authorize_url(client_id, redirect_uri=redirect_uri)

    server = http.server.HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _CallbackHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Listening on {redirect_uri}")
    print("Opening the browser. If nothing happens, open this URL manually:\n")
    print(f"    {url}\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass

    print(f"Waiting up to {timeout}s for the redirect …")
    got = _CallbackHandler.done.wait(timeout)
    server.shutdown()

    if not got or not _CallbackHandler.captured:
        print("❌  Timed out — no callback received.")
        return None, None
    return extract_token(_CallbackHandler.captured)


# ---------------------------------------------------------------------------
# Paste mode
# ---------------------------------------------------------------------------

def paste_mode(client_id: str) -> tuple[Optional[str], Optional[int]]:
    """Print the authorization URL and read whatever the user copied back."""
    plain = build_authorize_url(client_id)
    confirm = build_authorize_url(client_id, force_confirm=True)
    verification = build_authorize_url(
        client_id, redirect_uri="https://oauth.yandex.ru/verification_code"
    )

    print("=" * 68)
    print("Step 1 — open this URL and sign in:\n")
    print(f"    {plain}\n")
    print("Step 2 — copy the ENTIRE URL from the address bar of the page you")
    print("         land on and paste it below. The token lives in the part")
    print("         after '#', so copy the whole thing, not just the visible")
    print("         start of it.")
    print()
    print("If you get bounced straight into Yandex Music and the address bar no")
    print("longer shows '#access_token=' — the page rewrote its own URL. Then:")
    print()
    print("  a) open your browser history (Ctrl+H / Cmd+Y) and search for")
    print("     'access_token' — the original redirect is recorded there;")
    print("  b) or try this variant, which lands on a plain Yandex page that")
    print("     shows the token as text:\n")
    print(f"       {verification}\n")
    print("  c) or force the consent screen so the redirect does not happen")
    print("     silently:\n")
    print(f"       {confirm}\n")
    print("A bare token works too, if you already have one.")
    print("=" * 68)

    try:
        raw = input("\nPaste URL or token here: ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return None, None
    return extract_token(raw)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Obtain and verify a YANDEX_TOKEN for Yandex Music.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "serve mode needs your own app registered at https://oauth.yandex.ru/client/new\n"
            f"with redirect URI http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}.\n"
            "Note that Yandex does not offer music scopes to third-party apps, so a\n"
            "self-registered client_id will authorize fine yet still be refused by the\n"
            "Music API — this script will tell you when that happens.\n"
            "For the Music API, prefer paste mode with the default client_id."
        ),
    )
    parser.add_argument("--serve", action="store_true",
                        help="capture the token via a local callback server")
    parser.add_argument("--check", action="store_true",
                        help="only verify an existing token (from --token or $YANDEX_TOKEN)")
    parser.add_argument("--token", help="token to verify with --check")
    parser.add_argument("--client-id", default=MUSIC_CLIENT_ID,
                        help=f"OAuth client_id (default: Yandex Music app {MUSIC_CLIENT_ID})")
    parser.add_argument("--timeout", type=int, default=300,
                        help="serve mode: seconds to wait for the redirect (default: 300)")
    args = parser.parse_args()

    expires_in: Optional[int] = None

    if args.check:
        token = args.token or os.environ.get("YANDEX_TOKEN")
        if not token:
            print("ERROR: pass --token or set YANDEX_TOKEN.")
            return 2
    elif args.serve:
        token, expires_in = serve_mode(args.client_id, args.timeout)
    else:
        token, expires_in = paste_mode(args.client_id)

    if not token:
        print("\n❌  No token found in the input.")
        return 1

    token = normalize_token(token)

    verified = verify_token(token)
    if verified is False:
        return 1

    print("\n" + "=" * 68)
    if verified:
        print("✅  Token verified against the Yandex Music API.")
    else:
        print("⚠️  Token extracted but NOT verified — check it before deploying.")
    print("=" * 68)
    if expires_in:
        expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        days = expires_in // 86400
        print(f"\nExpires in {days} days (around {expiry:%Y-%m-%d}). Implicit-grant")
        print("tokens cannot be refreshed — re-run this script when it lapses.")
    print("\nSet this in Coolify:\n")
    print(f"YANDEX_TOKEN={token}\n")
    print("Keep it secret: it grants full access to your Yandex Music account.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
