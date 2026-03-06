"""
get_spotify_token.py
--------------------
Run this script ONCE on your local machine to obtain a SPOTIFY_REFRESH_TOKEN.
The browser will open for Spotify login. After authorization the refresh token
will be printed — copy it into your Coolify environment variables.

Usage:
    pip install spotipy
    SPOTIFY_CLIENT_ID=xxx SPOTIFY_CLIENT_SECRET=yyy python get_spotify_token.py
"""

import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth

SCOPE = "user-library-read user-library-modify"
REDIRECT_URI = "http://127.0.0.1:8888/callback"

client_id = os.environ.get("SPOTIFY_CLIENT_ID")
client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

if not client_id or not client_secret:
    raise SystemExit(
        "ERROR: Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET "
        "environment variables before running this script."
    )

auth_manager = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    open_browser=True,
)

# This triggers the browser-based OAuth flow
token_info = auth_manager.get_access_token(as_dict=True)

refresh_token: str = token_info["refresh_token"]

print("\n" + "=" * 60)
print("✅  Authorization successful!")
print("=" * 60)
print(f"\nSPOTIFY_REFRESH_TOKEN={refresh_token}\n")
print("Copy the value above and add it as an environment variable in Coolify.")
print("=" * 60 + "\n")
