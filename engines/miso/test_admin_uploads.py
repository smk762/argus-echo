"""/admin/* bearer auth (ARGUS_ADMIN_TOKEN) and the upload caps on /clone-voice,
/analyze-clips and /clones/import. The model manager is mocked; no model loads."""

import importlib.util
import io
import tempfile
import unittest
import zipfile
from unittest import mock

from fastapi.testclient import TestClient

import miso_server as srv

HAS_TORCH = importlib.util.find_spec("torch") is not None


class AdminAuthTest(unittest.TestCase):
    def setUp(self):
        for name in ("demote_to_cpu", "unload", "ensure_ready"):
            p = mock.patch.object(srv._mgr, name, mock.AsyncMock())
            p.start()
            self.addCleanup(p.stop)
        # No `with`: the lifespan (preload, watchdog) does not run.
        self.client = TestClient(srv.app)

    def test_closed_when_token_unset(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "", "ARGUS_ADMIN_INSECURE": ""}):
            for route in ("/admin/evict", "/admin/unload", "/admin/wake"):
                self.assertEqual(self.client.post(route).status_code, 503, route)

    def test_insecure_opens_when_token_unset(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "", "ARGUS_ADMIN_INSECURE": "1"}):
            for route in ("/admin/evict", "/admin/unload", "/admin/wake"):
                self.assertEqual(self.client.post(route).status_code, 200, route)

    def test_token_wins_over_insecure(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "s3cret", "ARGUS_ADMIN_INSECURE": "1"}):
            self.assertEqual(self.client.post("/admin/wake").status_code, 401)

    def test_token_required_when_set(self):
        with mock.patch.dict("os.environ", {"ARGUS_ADMIN_TOKEN": "s3cret"}):
            for route in ("/admin/evict", "/admin/unload", "/admin/wake"):
                self.assertEqual(self.client.post(route).status_code, 401, route)
                bad = {"Authorization": "Bearer wrong"}
                self.assertEqual(self.client.post(route, headers=bad).status_code, 401, route)
                ok = {"Authorization": "Bearer s3cret"}
                self.assertEqual(self.client.post(route, headers=ok).status_code, 200, route)
        srv._mgr.unload.assert_awaited_once()
        srv._mgr.demote_to_cpu.assert_awaited_once()
        srv._mgr.ensure_ready.assert_awaited_once()


class UploadCapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for p in (
            mock.patch.object(srv, "_storage", srv.LocalStorage(self.tmp.name, "http://x")),
            mock.patch.object(srv, "MAX_CLIP_BYTES", 1024),
            mock.patch.object(srv, "MAX_ARCHIVE_BYTES", 1024),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(srv.app)

    def test_oversized_archive_is_413(self):
        r = self.client.post("/clones/import", files={"archive": ("v.zip", b"x" * 2048, "application/zip")})
        self.assertEqual(r.status_code, 413, r.text)

    def test_oversized_clip_to_analyze_is_413(self):
        r = self.client.post("/analyze-clips", files=[("audio", ("a.wav", b"x" * 2048, "audio/wav"))])
        self.assertEqual(r.status_code, 413, r.text)

    @unittest.skipUnless(HAS_TORCH, "clone-voice imports torch")
    def test_oversized_clip_to_clone_is_413(self):
        r = self.client.post(
            "/clone-voice",
            files=[("audio", ("a.wav", b"x" * 2048, "audio/wav"))],
            data={"companion_id": "alice"},
        )
        self.assertEqual(r.status_code, 413, r.text)
        self.assertEqual(srv._storage.list_keys("voice_clones/"), [])

    @unittest.skipUnless(HAS_TORCH, "_import_clone imports torch")
    def test_oversized_meta_in_archive_is_refused(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("prompt.safetensors", b"")
            zf.writestr("meta.json", b" " * (srv._MAX_META_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "meta.json is unexpectedly large"):
            srv._import_clone(buf.getvalue(), "", "")


if __name__ == "__main__":
    unittest.main()


class UploadBodyCapTest(unittest.TestCase):
    """smk762/voluptas#30: an oversized multipart body is refused before Starlette's
    parser spools it to disk."""

    def setUp(self):
        for p in (
            mock.patch.object(srv, "MAX_CLIP_BYTES", 1024),
            mock.patch.object(srv, "MAX_ARCHIVE_BYTES", 1024),
            mock.patch.object(srv, "MAX_CLIPS", 2),
            mock.patch.object(srv, "_FORM_SLACK", 512),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(srv.app)

    def _no_parse(self):
        from starlette.formparsers import MultiPartParser

        return mock.patch.object(MultiPartParser, "parse", side_effect=AssertionError("parser reached"))

    def test_declared_oversized_body_never_reaches_the_parser(self):
        # Each clip is under MAX_CLIP_BYTES; together they are over the body cap.
        files = [("audio", (f"c{i}.wav", b"x" * 1000, "audio/wav")) for i in range(3)]
        for route, data in (
            ("/clone-voice", {"companion_id": "abc"}),
            ("/analyze-clips", {}),
        ):
            with self._no_parse() as parse:
                r = self.client.post(route, files=files, data=data)
            self.assertEqual(r.status_code, 413, route)
            parse.assert_not_called()
        with self._no_parse() as parse:
            r = self.client.post("/clones/import", files={"archive": ("v.zip", b"x" * 2048, "application/zip")})
        self.assertEqual(r.status_code, 413)
        parse.assert_not_called()

    def test_chunked_oversized_body_is_413(self):
        boundary = "b0undary"
        head = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"archive\"; filename=\"v.zip\"\r\n"
            "Content-Type: application/zip\r\n\r\n"
        ).encode()

        def body():  # no Content-Length: sent chunked
            yield head
            for _ in range(8):
                yield b"x" * 512
            yield f"\r\n--{boundary}--\r\n".encode()

        from starlette.datastructures import UploadFile

        spooled = 0
        real_write = UploadFile.write

        async def counting_write(self, data):
            nonlocal spooled
            spooled += len(data)
            return await real_write(self, data)

        with mock.patch.object(UploadFile, "write", counting_write), \
             mock.patch.object(srv, "read_upload", side_effect=AssertionError("handler ran")) as handler:
            r = self.client.post(
                "/clones/import",
                content=body(),
                headers={"content-type": f"multipart/form-data; boundary={boundary}"},
            )
        self.assertEqual(r.status_code, 413, r.text)
        handler.assert_not_called()
        # The body is 4 KiB; the parser stopped being fed at the 1.5 KiB cap.
        self.assertLessEqual(spooled, srv.MAX_ARCHIVE_BYTES + srv._FORM_SLACK)

    def test_malformed_content_length_is_400(self):
        r = self.client.post(
            "/clones/import",
            content=b"x",
            headers={"content-type": "multipart/form-data; boundary=b", "content-length": "1e9"},
        )
        self.assertEqual(r.status_code, 400)

    def test_too_many_clips_is_422(self):
        files = [("audio", (f"c{i}.wav", b"x" * 10, "audio/wav")) for i in range(3)]
        r = self.client.post("/analyze-clips", files=files)
        self.assertEqual(r.status_code, 422, r.text)

    def test_parts_past_max_clips_are_never_spooled(self):
        from starlette.datastructures import UploadFile

        spooled = set()
        real_write = UploadFile.write

        async def counting_write(self, data):
            spooled.add(id(self))
            return await real_write(self, data)

        files = [("audio", (f"c{i}.wav", b"x" * 10, "audio/wav")) for i in range(5)]
        with mock.patch.object(UploadFile, "write", counting_write):
            r = self.client.post("/analyze-clips", files=files)
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn("At most 2 clips", r.json()["error"])
        self.assertLessEqual(len(spooled), srv.MAX_CLIPS)

    def test_other_routes_are_untouched(self):
        self.assertEqual(self.client.get("/clones").status_code, 200)
