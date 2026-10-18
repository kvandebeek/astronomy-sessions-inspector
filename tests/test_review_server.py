"""The review server: dataset state, and its HTTP endpoints on a live local server."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

import fits_light_stats as st
import fits_review_server as rv


@pytest.fixture
def scanned(project):
    dataset = st.build_dataset(project, darks_dir=project / "darks", workers=1, progress=False)
    cache = project / "c.json"
    st.write_json_atomic(cache, dataset)
    return project, cache


@pytest.fixture
def server(scanned):
    root, cache = scanned
    rv.Handler.holder = rv.DatasetHolder(cache, root)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rv.Handler)
    port = srv.server_address[1]
    rv.Handler.allowed_hosts = frozenset({f"127.0.0.1:{port}"})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, cache, port
    srv.shutdown()
    srv.server_close()


def request(port, method, path, body=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    res = conn.getresponse()
    payload = res.read()
    conn.close()
    return res.status, payload


def test_page_and_renders(server):
    _root, _cache, port = server
    status, page = request(port, "GET", "/")
    assert status == 200 and b"REVIEW_DATA" in page
    status, png = request(port, "GET", "/api/preview?id=0&stretch=auto")
    assert status == 200 and png[:4] == b"\x89PNG"


def test_unknown_host_is_refused(server):
    _root, _cache, port = server
    status, _ = request(port, "GET", "/api/version", host="evil.example:80")
    assert status == 403


def test_delete_restore_and_stale_version(server):
    root, _cache, port = server
    ds = rv.Handler.holder.get()
    path = rv.Path(ds.files[0]["path"])
    status, _ = request(port, "POST", "/api/delete", {"ids": [0], "version": "stale"})
    assert status == 409 and path.exists()
    status, _ = request(port, "POST", "/api/delete", {"ids": [0], "version": ds.version})
    assert status == 200 and not path.exists()
    assert (root / "_deleted").exists()
    status, _ = request(port, "POST", "/api/restore", {"ids": [0], "version": ds.version})
    assert status == 200 and path.exists()


def test_purge_needs_the_exact_count(server):
    _root, _cache, port = server
    ds = rv.Handler.holder.get()
    request(port, "POST", "/api/delete", {"ids": [0, 1], "version": ds.version})
    status, _ = request(port, "POST", "/api/purge", {"confirm": 1, "version": ds.version})
    assert status == 409
    status, body = request(port, "POST", "/api/purge", {"confirm": 2, "version": ds.version})
    assert status == 200 and json.loads(body)["removed"] == 2
    assert all(ds.files[i].get("missing") for i in (0, 1))


def test_restore_under_a_new_name_survives_a_restart(scanned):
    root, cache = scanned
    ds = rv.Dataset(cache, root)
    original = rv.Path(ds.files[0]["path"])
    ds.delete_many([0])
    original.write_bytes(b"something else took the name")
    ds.restore_many([0])
    renamed = ds.files[0]["path"]
    assert renamed != str(original)
    again = rv.Dataset(cache, root)
    assert again.files[0]["path"] == renamed


def test_dataset_reloads_when_the_json_changes(scanned):
    root, cache = scanned
    holder = rv.DatasetHolder(cache, root)
    first = holder.get()
    data = json.loads(cache.read_text())
    data[0]["files"] = data[0]["files"][:1]
    st.write_json_atomic(cache, data)
    holder._checked = 0.0
    assert holder.get() is not first and len(holder.get().files) < len(first.files)


def test_unreadable_frames_are_listed(scanned):
    root, cache = scanned
    data = json.loads(cache.read_text())
    data[0]["files"].append({"path": str(root / "x.fits"), "name": "x.fits", "error": "boom"})
    st.write_json_atomic(cache, data)
    ds = rv.Dataset(cache, root)
    assert ds.errors and ds.errors[0]["name"] == "x.fits"
    assert "Unreadable frames (1)" in rv.build_page(ds)
