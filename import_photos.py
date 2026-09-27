"""Import image links pasted into a text file into the local meme gallery."""

from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


APP_FOLDER = Path(__file__).resolve().parent
PHOTO_FOLDER = APP_FOLDER / "memes" / "photos"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PAGE_BYTES = 1 * 1024 * 1024
IMAGE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
USER_AGENT = "FaceMemeStudio/1.0 (local personal image importer)"


class ImagePreviewParser(HTMLParser):
    """Find a page's Open Graph or Twitter preview image."""

    def __init__(self):
        super().__init__()
        self.image_url = None

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        values = dict(attrs)
        key = (values.get("property") or values.get("name") or "").lower()
        if key in {"og:image", "twitter:image"} and values.get("content"):
            self.image_url = values["content"]


def extract_urls(text: str) -> list[str]:
    """Find HTTPS links, including links pasted together without spaces."""
    return re.findall(r"https://[^\s]*?(?=https://|\s|$)", text)


def read_url(url: str, depth: int = 0):
    """Fetch one image or follow one page preview, with size and scheme limits."""
    if depth > 1:
        return None, None, "too many preview redirects"
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return None, None, "only HTTPS links are accepted"
    if parsed.hostname == "itunes.apple.com":
        return None, None, "App Store link is not a meme photo"
    if parsed.path.lower().endswith(".gif"):
        return None, None, "GIF is not supported by the current gallery"

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        final_url = response.geturl()
        if urlparse(final_url).scheme != "https":
            return None, None, "redirected to a non-HTTPS address"
        content_type = response.headers.get_content_type().lower()
        if content_type in IMAGE_SUFFIXES:
            content = response.read(MAX_IMAGE_BYTES + 1)
            if len(content) > MAX_IMAGE_BYTES:
                return None, None, "image exceeds the 8 MB limit"
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                return None, None, "download did not contain a readable image"
            return content, IMAGE_SUFFIXES[content_type], None

        if content_type in {"text/html", "application/xhtml+xml"}:
            page = response.read(MAX_PAGE_BYTES + 1)
            if len(page) > MAX_PAGE_BYTES:
                return None, None, "source page exceeds the 1 MB preview limit"
            parser = ImagePreviewParser()
            parser.feed(page.decode("utf-8", errors="replace"))
            if parser.image_url:
                preview_url = urljoin(final_url, parser.image_url)
                return read_url(preview_url, depth + 1)
            return None, None, "page has no image preview"

        return None, None, f"unsupported content type: {content_type}"


def source_was_imported(url: str, folder: Path = PHOTO_FOLDER) -> bool:
    """Check prior imports by URL hash so reordering links cannot duplicate files."""
    url_key = sha256(url.encode("utf-8")).hexdigest()[:12]
    return any(
        path.stem.endswith(f"_{url_key}") or path.stem == f"meme_{url_key}"
        for path in folder.iterdir()
    ) if folder.is_dir() else False


def import_links(source_file: Path) -> tuple[int, int]:
    """Download valid linked photos into the project gallery without overwriting."""
    if not source_file.is_file():
        raise FileNotFoundError(f"Could not find the link list: {source_file}")

    PHOTO_FOLDER.mkdir(parents=True, exist_ok=True)
    urls = extract_urls(source_file.read_text(errors="replace"))
    imported = 0
    skipped = 0

    for index, url in enumerate(urls, start=1):
        try:
            content, suffix, reason = read_url(url)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            content, suffix, reason = None, None, str(error)

        if content is None:
            skipped += 1
            print(f"[{index}/{len(urls)}] Skipped: {url} ({reason})")
            continue

        url_key = sha256(url.encode("utf-8")).hexdigest()[:12]
        if source_was_imported(url):
            print(f"[{index}/{len(urls)}] Already imported: {url}")
            continue

        destination = PHOTO_FOLDER / f"meme_{url_key}{suffix}"
        destination.write_bytes(content)
        imported += 1
        print(f"[{index}/{len(urls)}] Saved {destination.name} ({len(content) // 1024} KB)")

    print(f"Finished: imported {imported}; skipped {skipped}; found {len(urls)} links.")
    return imported, skipped


if __name__ == "__main__":
    links_file = Path(__file__).resolve().parent / "memes" / "photo_sources.txt"
    import_links(links_file)