# SPDX-License-Identifier: GPL-3.0-only
# Derived from Thatcher Effect Dataset Generator (https://github.com/Erfaniaa/thatcher-effect-dataset-generator)
"""Download the 140 public FFHQ face images required for Thatcher stimulus generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from main_ffhq import DEFAULT_MANIFEST, read_sources

METADATA_URL = "https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA"
METADATA_BYTES = 267793842
METADATA_MD5 = "425ae20f06a4da1d4dc0f46d40ba5fd6"
METADATA_SHA256 = "1be5c2d1a78196d45a9600558c04b46dffefa217b7c664d5be38ed28a4d9a0ee"
ALLOWED_HOSTS = {"drive.google.com", "drive.usercontent.google.com", "raw.githubusercontent.com", "dlib.net"}
USER_AGENT = "Thatcher-stimulus-acquisition/1.0"


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def allowed_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password:
        raise ValueError("Unexpected download host or scheme")
    return url


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        allowed_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.inside = True
        if self.inside and tag == "input" and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


def download(url, target, size, expected_md5=None, expected_sha256=None):
    allowed_url(url)
    if target.is_symlink():
        raise ValueError("Refusing an existing symlink")
    if target.exists():
        md5, sha = hashes(target)
        if target.stat().st_size == size and (not expected_md5 or md5 == expected_md5) and (not expected_sha256 or sha == expected_sha256):
            return {"bytes": size, "md5": md5, "sha256": sha, "status": "existing_verified"}
        raise ValueError(f"Existing file differs from expected checksum: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()
    opener = urllib.request.build_opener(SafeRedirect())
    original_url = url
    for attempt in range(2):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with opener.open(request, timeout=45) as response:
            first = response.read(65536)
            kind = response.headers.get("Content-Type", "")
            if "text/html" in kind or first.lstrip().lower().startswith((b"<!doctype html", b"<html")):
                body = (first + response.read(1048576)).decode("utf-8", errors="replace")
                form = ConfirmationForm()
                form.feed(body)
                original_id = urllib.parse.parse_qs(urllib.parse.urlsplit(original_url).query).get("id", [None])[0]
                if (attempt == 0 and "can't scan this file for viruses" in body and
                        form.action == "https://drive.usercontent.google.com/download" and
                        original_id is not None and form.fields.get("id") == original_id and
                        set(form.fields).issubset({"id", "confirm", "uuid", "export"}) and
                        not any(x in body.lower() for x in ("quota exceeded", "recaptcha/api.js", "sign in to continue"))):
                    url = form.action + "?" + urllib.parse.urlencode(form.fields)
                    continue
                raise ValueError("Download returned an access/quota/challenge page. Open the official URL in a browser.")
            count = len(first)
            md5 = hashlib.md5(first)
            sha = hashlib.sha256(first)
            if count > size:
                raise ValueError("Response exceeds expected size")
            with part.open("xb") as f:
                f.write(first)
                while True:
                    block = response.read(1048576)
                    if not block:
                        break
                    count += len(block)
                    if count > size:
                        raise ValueError("Response exceeds expected size")
                    f.write(block)
                    md5.update(block)
                    sha.update(block)
            if (count != size or (expected_md5 and md5.hexdigest() != expected_md5) or
                    (expected_sha256 and sha.hexdigest() != expected_sha256)):
                raise ValueError("Downloaded file checksum mismatch")
            part.rename(target)
            return {"bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest(),
                    "status": "downloaded_verified"}
    raise ValueError("Unresolved download confirmation")


def iter_object(path):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as f:
        buffer = ""
        position = 0
        eof = False

        def refill():
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            block = f.read(65536)
            if not block:
                eof = True
            buffer += block

        def skip():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        def take(expected):
            nonlocal position
            skip()
            if position >= len(buffer) or buffer[position] != expected:
                raise ValueError("Invalid top-level JSON structure")
            position += 1

        def value():
            nonlocal position
            skip()
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    position = end
                    return item
                except json.JSONDecodeError:
                    if eof:
                        raise
                    if len(buffer) - position > 2 * 1024 * 1024:
                        raise ValueError("Oversize metadata record")
                    refill()

        take("{")
        skip()
        if position < len(buffer) and buffer[position] == "}":
            return
        while True:
            key = value()
            take(":")
            item = value()
            yield key, item
            skip()
            if position < len(buffer) and buffer[position] == "}":
                position += 1
                skip()
                return
            take(",")


def select_metadata(path, source_manifest):
    rows = read_sources(source_manifest)
    needed = {str(int(Path(r["img_path"]).stem)): r for r in rows}
    selected = {}
    for key, item in iter_object(path):
        if key not in needed:
            continue
        image = item.get("image", {})
        meta = item.get("metadata", {})
        expected = "images1024x1024/" + needed[key]["img_path"]
        if image.get("file_path") != expected:
            raise ValueError("FFHQ metadata path differs from selected source")
        if "public domain" not in str(meta.get("license", "")).lower():
            raise ValueError("Selected metadata is not public domain")
        if image.get("pixel_size") != [1024, 1024]:
            raise ValueError("Selected source is not an aligned 1024x1024 image")
        allowed_url(image["file_url"])
        selected[key] = {
            "image": {k: image[k] for k in ("file_url", "file_path", "file_size", "file_md5", "pixel_size", "pixel_md5")},
            "metadata": {k: meta.get(k, "") for k in ("license", "license_url", "author", "photo_url")}
        }
    if set(selected) != set(needed):
        raise ValueError("Missing selected FFHQ images in metadata")
    return rows, selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True, help="Directory for downloaded FFHQ images")
    p.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--workers", type=int, default=4, help="Download concurrency (1-4)")
    args = p.parse_args()

    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)

    print("Downloading FFHQ dataset metadata (~268 MB)...")
    download(METADATA_URL, root / "ffhq-dataset-v2.json", METADATA_BYTES, METADATA_MD5, METADATA_SHA256)

    print("Extracting metadata for 140 selected scenes...")
    rows, selected = select_metadata(root / "ffhq-dataset-v2.json", args.source_manifest)
    (root / "selected_metadata.json").write_text(json.dumps(selected, indent=2) + "\n")

    image_bytes = sum(v["image"]["file_size"] for v in selected.values())
    print(f"Downloading 140 images ({image_bytes / (1024*1024):.1f} MB)...")

    entries = list(selected.items())

    def fetch(entry):
        key, item = entry
        image = item["image"]
        target = (root / image["file_path"]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        download(image["file_url"], target, image["file_size"], image["file_md5"])
        return key

    errors = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, entry): entry[0] for entry in entries}
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(entries):
                    print(f"Downloaded {completed}/{len(entries)} images")
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")

    if errors:
        p.exit(1, f"Failed to download {len(errors)} images:\n" + "\n".join(errors) + "\n")
    print("All 140 FFHQ images successfully acquired.")


if __name__ == "__main__":
    main()
