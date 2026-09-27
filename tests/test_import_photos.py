import sys
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from import_photos import ImagePreviewParser, extract_urls, source_was_imported


class PhotoImportTests(unittest.TestCase):
    def test_extracts_links_pasted_without_spaces(self):
        text = "https://example.com/one.jpghttps://example.com/two.png"
        self.assertEqual(extract_urls(text), ["https://example.com/one.jpg", "https://example.com/two.png"])

    def test_ignores_non_url_text(self):
        self.assertEqual(extract_urls("paste links here, not file names"), [])

    def test_reads_open_graph_image_url(self):
        parser = ImagePreviewParser()
        parser.feed('<meta property="og:image" content="https://cdn.example.com/meme.jpg">')
        self.assertEqual(parser.image_url, "https://cdn.example.com/meme.jpg")

    def test_import_deduplication_survives_source_reordering(self):
        with TemporaryDirectory() as temporary_folder:
            folder = Path(temporary_folder)
            url = "https://example.com/meme.jpg"
            key = sha256(url.encode()).hexdigest()[:12]
            (folder / f"meme_04_{key}.jpg").touch()
            self.assertTrue(source_was_imported(url, folder))
            self.assertFalse(source_was_imported("https://example.com/other.jpg", folder))


if __name__ == "__main__":
    unittest.main()
