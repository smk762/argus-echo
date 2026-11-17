"""Sampling overrides for /synthesise: request > clone meta.json > env defaults."""

import json
import threading
import unittest
from unittest import mock

import numpy as np
from pydantic import ValidationError

import miso_server as srv


class _Audio:
    def detach(self):
        return self

    def cpu(self):
        return np.zeros(4, dtype=np.float32)


class _Generator:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return _Audio()

    def generate_stream(self, **kwargs):
        self.calls.append(("generate_stream", kwargs))
        yield _Audio()


class SamplingParamsTest(unittest.TestCase):
    def test_request_overrides_meta_and_clamps(self):
        req = srv.SynthesiseRequest(text="hello", temperature=5, topk=0)
        temperature, topk = srv._sampling_params(req, {"temperature": 0.4, "topk": 20})
        self.assertEqual(temperature, srv.TEMPERATURE_MAX)
        self.assertEqual(topk, srv.TOPK_MIN)

    def test_meta_overrides_env_when_request_omits_fields(self):
        req = srv.SynthesiseRequest(text="hello")
        temperature, topk = srv._sampling_params(req, {"temperature": "0.35", "topk": 12.2})
        self.assertEqual(temperature, 0.35)
        self.assertEqual(topk, 12)

    def test_env_defaults_when_meta_is_junk(self):
        req = srv.SynthesiseRequest(text="hello")
        temperature, topk = srv._sampling_params(req, {"temperature": "lively", "topk": True})
        self.assertEqual(temperature, srv._coerce_temperature(srv.TEMPERATURE))
        self.assertEqual(topk, srv._coerce_topk(srv.TOPK))

    def test_string_form_fields_coerce(self):
        req = srv.SynthesiseRequest(text="hello", temperature="0.4", topk="40")
        self.assertEqual(req.temperature, 0.4)
        self.assertEqual(req.topk, 40)

    def test_non_numeric_temperature_rejected(self):
        with self.assertRaises(ValidationError):
            srv.SynthesiseRequest(text="hello", temperature="warm")

    def test_non_finite_temperature_rejected(self):
        for bad in ("nan", "inf", float("nan")):
            with self.assertRaises(ValidationError):
                srv.SynthesiseRequest(text="hello", temperature=bad)

    def test_import_keeps_valid_sampling_fields_only(self):
        self.assertEqual(
            srv._sampling_meta_fields({"temperature": 0.05, "topk": 999, "name": "inara"}),
            {"temperature": srv.TEMPERATURE_MIN, "topk": srv.TOPK_MAX},
        )
        self.assertEqual(srv._sampling_meta_fields({"temperature": "nope"}), {})

    def test_clone_meta_key(self):
        self.assertEqual(srv._clone_meta_key("inara"), "voice_clones/inara/meta.json")
        self.assertEqual(
            srv._clone_meta_key("voice_clones/inara/prompt.pt"),
            "voice_clones/inara/meta.json",
        )

    def test_load_clone_meta_reads_sidecar(self):
        payload = json.dumps({"temperature": 0.4, "topk": 15}).encode()
        storage = mock.Mock()
        storage.exists.return_value = True
        storage.get.return_value = payload
        with mock.patch.object(srv, "_storage", storage):
            meta = srv._load_clone_meta("inara")
        self.assertEqual(meta["temperature"], 0.4)
        storage.exists.assert_called_with("voice_clones/inara/meta.json")

    def test_resolve_context_returns_meta_for_clone(self):
        import torch

        prompt = {
            "ref_audio": torch.zeros(8),
            "ref_text": "reference",
            "speaker": 2,
        }
        req = srv.SynthesiseRequest(text="hello", voice_clone_id="inara", temperature=0.4)
        with mock.patch.object(srv, "_load_clone", return_value=prompt), mock.patch.object(
            srv, "_load_clone_meta", return_value={"temperature": 1.1, "topk": 30}
        ) as load_meta:
            speaker, context, meta = srv._resolve_context(req)
        load_meta.assert_called_once_with("inara")
        self.assertEqual(speaker, 2)
        self.assertEqual(context[0].text, "reference")
        self.assertEqual(meta["topk"], 30)
        temperature, topk = srv._sampling_params(req, meta)
        self.assertEqual(temperature, 0.4)
        self.assertEqual(topk, 30)

    def test_generate_paths_forward_resolved_sampling(self):
        gen = _Generator()
        req = srv.SynthesiseRequest(text="hello", temperature=0.4, topk=15)
        with mock.patch.object(srv._mgr, "_generator", gen), mock.patch.object(
            srv, "_resolve_context", return_value=(0, [], {"temperature": 1.2, "topk": 80})
        ):
            srv._generate_audio(req)
            chunks = list(srv._generate_stream_chunks(req, threading.Event()))
        self.assertEqual(len(chunks), 1)
        for kind, kwargs in gen.calls:
            self.assertIn(kind, ("generate", "generate_stream"))
            self.assertEqual(kwargs["temperature"], 0.4)
            self.assertEqual(kwargs["topk"], 15)


if __name__ == "__main__":
    unittest.main()
