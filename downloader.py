#!/usr/bin/env python3
import argparse
import hashlib
import re
import zipfile
from pathlib import Path
from typing import Optional

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="File downloader with SHA-256 verification (supports Google Drive)."
    )
    parser.add_argument("--url", required=True, help="Download URL")
    parser.add_argument("--dst", required=True, help="Destination file path (e.g., data/dataset.zip)")
    parser.add_argument("--sha256", required=False, help="Expected SHA-256 hash (optional)")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Download chunk size (bytes)")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the ZIP file after successful hash verification.",
    )
    return parser.parse_args()


# -------------------- Google Drive helpers -------------------- #

def is_google_drive_url(url: str) -> bool:
    return ("drive.google.com" in url) or ("drive.usercontent.google.com" in url)


def extract_gdrive_file_id(url: str) -> Optional[str]:
    """
    Extract a Google Drive file ID from a variety of URL formats.
    Supports:
      - https://drive.google.com/file/d/<FILE_ID>/view
      - https://drive.google.com/open?id=<FILE_ID>
      - https://drive.google.com/uc?export=download&id=<FILE_ID>
      - https://drive.usercontent.google.com/download?id=<FILE_ID>&export=download
    """
    # Pattern: /file/d/<id>/
    m = re.search(r"/file/d/([^/]+)/", url)
    if m:
        return m.group(1)

    # Pattern: ?id=<id>
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)

    return None


def extract_confirm_token_from_html(html: str) -> Optional[str]:
    """
    Parse the Google Drive virus-scan warning HTML page and extract the confirm token.
    """
    m = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    if m:
        return m.group(1)
    return None


def resolve_gdrive_download_url(url: str, session: requests.Session) -> str:
    """
    Resolve the real downloadable URL for Google Drive files.
    1) Normalize to uc?export=download.
    2) If the response is an HTML warning page, extract the confirm token
       and construct the corresponding drive.usercontent.google.com download URL.
    """
    file_id = extract_gdrive_file_id(url)
    if not file_id:
        return url

    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = session.get(base_url, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").lower()

    # If the file is directly downloadable, return the URL
    if "text/html" not in ctype:
        return base_url

    # Otherwise, extract confirm token from HTML
    token = extract_confirm_token_from_html(r.text)
    if not token:
        return base_url

    # Google Drive's actual large-file download endpoint
    dl_url = (
        "https://drive.usercontent.google.com/download"
        f"?export=download&id={file_id}&confirm={token}"
    )
    return dl_url


# -------------------- Download & hash -------------------- #

def download_file(url: str, dst_path: str, chunk_size: int = 8192) -> Path:
    """
    Download the file at 'url' and save it to 'dst_path'.
    Automatically handles Google Drive confirm-token logic.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CKLD-downloader/1.0)"})

    if is_google_drive_url(url):
        print("[*] Google Drive URL detected. Resolving download URL...")
        url = resolve_gdrive_download_url(url, session)
        print(f"    Resolved URL: {url}")

    with session.get(url, stream=True) as r:
        r.raise_for_status()

        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    return dst


def compute_sha256(path: str) -> str:
    """
    Compute the SHA-256 digest of a file.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def extract_zip(zip_path: str, out_dir: str = "."):
    """
    Extract a ZIP file into out_dir (default: current working directory).
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)

    print(f"[+] Extracting {zip_path} to {out_dir.resolve()}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print("[✓] Extraction complete")


def main():
    args = parse_args()

    url = args.url
    dst_path = args.dst
    expected_sha256 = args.sha256
    chunk_size = args.chunk_size
    do_extract = args.extract

    print(f"[+] Downloading from: {url}")
    try:
        file_path = download_file(url, dst_path, chunk_size=chunk_size)
    except Exception as e:
        print(f"[✗] Download failed: {e}")
        raise SystemExit(1)

    print(f"[+] Saved to: {file_path}")

    # SHA-256 verification
    if expected_sha256:
        print("[+] Computing SHA-256...")
        actual_sha256 = compute_sha256(str(file_path))
        print(f"    Actual:   {actual_sha256}")
        print(f"    Expected: {expected_sha256}")

        if actual_sha256.lower() == expected_sha256.lower():
            print("[✓] Hash verification PASSED")
            if do_extract:
                extract_zip(str(file_path), ".")
        else:
            print("[✗] Hash verification FAILED")
            if do_extract:
                print("[!] Extraction skipped because hash verification failed.")
    else:
        print("[i] No expected SHA-256 provided; skipping verification.")
        if do_extract:
            print("[!] Extraction disabled when hash is not provided (safety measure).")


if __name__ == "__main__":
    main()
