"""
convert_tracklist.py
--------------------
One-time helper: converts a Yandex Music export TXT file
(format: "Artist - Title", one track per line) into synced_ids.txt
with Yandex track IDs — one ID per line.

Run this ONCE locally, then commit synced_ids.txt to the repo.

Usage:
    pip3 install yandex-music
    YANDEX_TOKEN=your_token python3 convert_tracklist.py \
        --input "Мне нравится_7991456116.txt" \
        --output synced_ids.txt
"""

import argparse
import os
import sys
import time

REQUEST_DELAY = 1.5  # seconds between Yandex API calls


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Yandex TXT tracklist to synced_ids.txt")
    parser.add_argument("--input", required=True, help="Path to the input TXT file (Artist - Title)")
    parser.add_argument("--output", default="synced_ids.txt", help="Output file path (default: synced_ids.txt)")
    args = parser.parse_args()

    token = os.environ.get("YANDEX_TOKEN")
    if not token:
        sys.exit("ERROR: Set YANDEX_TOKEN environment variable before running.")

    # Import here so the script gives a clear error if not installed
    try:
        from yandex_music import Client
    except ImportError:
        sys.exit("ERROR: Run `pip3 install yandex-music` first.")

    print("Connecting to Yandex Music…")
    client = Client(token=token)
    client.init()
    print("Connected.\n")

    # Read input file
    with open(args.input, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]

    total = len(lines)
    print(f"Found {total} tracks in '{args.input}'. Starting lookup…\n")

    found_ids: list[str] = []
    not_found: list[str] = []

    for i, query in enumerate(lines, start=1):
        print(f"[{i}/{total}] Searching: {query}", end=" … ", flush=True)
        time.sleep(REQUEST_DELAY)
        try:
            result = client.search(query, type_="track")
            if result and result.tracks and result.tracks.results:
                track = result.tracks.results[0]
                found_ids.append(str(track.id))
                print(f"✅  id={track.id}")
            else:
                not_found.append(query)
                print("❌  not found")
        except Exception as exc:
            not_found.append(query)
            print(f"⚠️  error: {exc}")

    # Write output
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(found_ids) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! Found IDs: {len(found_ids)} / {total}")
    print(f"Not found:       {len(not_found)}")
    print(f"Output saved to: {args.output}")
    print(f"{'='*60}")

    if not_found:
        print("\nTracks not found on Yandex Music:")
        for t in not_found:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
