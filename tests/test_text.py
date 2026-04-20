from __future__ import annotations

import unittest

from tts_qwen_local.text import chunk_text, normalize_text, preprocess_text


class TextTests(unittest.TestCase):
    def test_normalize_text(self):
        text = "Hello   world\n\n\nThis is\t\ta test."
        normalized = normalize_text(text)
        self.assertEqual(normalized, "Hello world\n\nThis is a test.")

    def test_preprocess_expands_abbreviation(self):
        text = "Dr. Smith vs. Mr. Jones."
        processed = preprocess_text(text)
        self.assertIn("Doctor Smith", processed)
        self.assertIn("versus", processed)

    def test_chunk_text_short(self):
        chunks = chunk_text("Hello world.", 80)
        self.assertEqual(chunks, ["Hello world."])

    def test_chunk_text_long(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the English alphabet. "
            "It is commonly used for typing tests and display samples."
        )
        chunks = chunk_text(text, 60)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 60)

    def test_chunk_text_merges_tiny_tail(self):
        text = "A very long sentence that should mostly fit in one chunk but leave a short tail. End."
        chunks = chunk_text(text, 75)
        self.assertLessEqual(len(chunks), 2)
