"""Miso speaks English only: /synthesise and /synthesise/stream refuse any other
language instead of mispronouncing it, and accept region variants of `en`."""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

import miso_server as srv

PATHS = ("/synthesise", "/synthesise/stream")


class PrimaryLanguageTest(unittest.TestCase):
    def test_region_and_case_are_dropped(self):
        for value in ("en", "EN", "en-US", "en_GB", " en-au "):
            self.assertEqual(srv._primary_language(value), "en", value)

    def test_empty_means_english(self):
        for value in ("", None, "  "):
            self.assertEqual(srv._primary_language(value), "en")

    def test_other_languages_keep_their_code(self):
        self.assertEqual(srv._primary_language("de-DE"), "de")
        self.assertEqual(srv._primary_language("zh_Hans"), "zh")


class SynthesiseLanguageTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(srv.app)

    def test_non_english_is_422_before_the_model_loads(self):
        ensure_ready = mock.AsyncMock(side_effect=AssertionError("must not load the model"))
        with mock.patch.object(srv._mgr, "ensure_ready", ensure_ready):
            for path in PATHS:
                for language in ("de", "fr-FR", "zh_CN", "english"):
                    with self.subTest(path=path, language=language):
                        r = self.client.post(path, json={"text": "hallo", "language": language})
                        self.assertEqual(r.status_code, 422, r.text)
                        body = r.json()
                        self.assertIn("English only", body["error"])
                        self.assertIn(repr(language), body["error"])
                        self.assertEqual(body["supported_languages"], ["en"])
        ensure_ready.assert_not_called()

    def test_english_variants_get_past_the_language_check(self):
        # A malformed context is refused later (by _prepare_generation): reaching that
        # 422 instead of the language one shows the language was accepted.
        bad_context = [{"text": "a", "audio": "%%%"}]
        with mock.patch.object(srv._mgr, "ensure_ready", mock.AsyncMock()):
            for path in PATHS:
                for language in ("en", "en-US", "en_GB", "EN"):
                    with self.subTest(path=path, language=language):
                        r = self.client.post(path, json={"text": "hi", "language": language, "context": bad_context})
                        self.assertEqual(r.status_code, 422, r.text)
                        self.assertNotIn("English only", r.json()["error"])
                        self.assertNotIn("supported_languages", r.json())


if __name__ == "__main__":
    unittest.main()
