"""Sign in with ChatGPT (utils.chatgpt_auth) and plan inference (utils.chatgpt_backend)."""
import base64
import hashlib
import json
import os
import stat
import threading
import time
import urllib.parse
import urllib.request
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from utils import chatgpt_auth as auth
from utils import chatgpt_backend as backend
from utils import llm


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_CHATGPT_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(backend, "_notice_shown", True)
    yield


# --- PKCE / authorize URL ----------------------------------------------------


def test_pkce_challenge_is_s256_of_verifier():
    verifier, challenge = auth.new_pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected and "=" not in challenge


def test_host_id_is_opaque_uuid_and_persisted():
    host = auth.load_host()
    assert host["ext_agent_host_id"].startswith("urn:uuid:")
    assert auth.load_host()["ext_agent_host_id"] == host["ext_agent_host_id"]
    mode = stat.S_IMODE(os.stat(auth._host_path()).st_mode)
    assert mode == 0o600


def test_authorize_url_first_registration_vs_reauth():
    host = {"ext_agent_host_id": "urn:uuid:abc"}
    url = auth.build_authorize_url(host, "http://127.0.0.1:1455/auth/callback", "st", "no", "ch")
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    assert url.startswith("https://auth.openai.com/api/accounts/authorize?")
    assert q["client_id"] == "dynamic_agent_client"
    assert q["agent_name_hint"] == "PaperOrchestra"
    assert q["scope"] == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    assert q["resource"] == "https://api.openai.com/v1"
    assert q["code_challenge_method"] == "S256" and q["response_type"] == "code"
    assert q["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"

    host["client_id"] = "oaiapp_123"
    url = auth.build_authorize_url(host, "http://127.0.0.1:1455/auth/callback", "st", "no", "ch",
                                   id_token_hint="old.jwt", login_hint="a@b.c")
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    assert q["client_id"] == "oaiapp_123"
    assert "agent_name_hint" not in q
    assert q["id_token_hint"] == "old.jwt" and q["login_hint"] == "a@b.c"


# --- callback validation -------------------------------------------------------


def test_callback_validation():
    new_host = {"ext_agent_host_id": "h"}
    assert auth.process_callback({"state": "s", "code": "c", "client_id": "oaiapp_1"}, "s", new_host) == ("c", "oaiapp_1")
    with pytest.raises(auth.ChatGPTAuthError, match="state mismatch"):
        auth.process_callback({"state": "x", "code": "c", "client_id": "oaiapp_1"}, "s", new_host)
    with pytest.raises(auth.ChatGPTAuthError, match="declined"):
        auth.process_callback({"state": "s", "error": "access_denied"}, "s", new_host)
    with pytest.raises(auth.ChatGPTAuthError, match="Registration incomplete"):
        auth.process_callback({"state": "s", "code": "c"}, "s", new_host)

    saved = {"ext_agent_host_id": "h", "client_id": "oaiapp_1"}
    assert auth.process_callback({"state": "s", "code": "c"}, "s", saved) == ("c", "oaiapp_1")
    with pytest.raises(auth.ChatGPTAuthError, match="unexpected client"):
        auth.process_callback({"state": "s", "code": "c", "client_id": "oaiapp_2"}, "s", saved)


# --- ID token validation -------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJWKS:
    def __init__(self, key):
        self.key = key

    def get_signing_key_from_jwt(self, token):
        return SimpleNamespace(key=self.key.public_key())


def make_id_token(key, **overrides):
    now = int(time.time())
    claims = {"iss": "https://auth.openai.com", "aud": "oaiapp_1", "sub": "user-1",
              "iat": now, "exp": now + 600, "nonce": "n1", "email": "me@example.com",
              "email_verified": True}
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


def test_validate_id_token(rsa_key):
    jwks = FakeJWKS(rsa_key)
    claims = auth.validate_id_token(make_id_token(rsa_key), "oaiapp_1", "n1", jwks)
    assert claims["sub"] == "user-1"
    with pytest.raises(jwt.InvalidAudienceError):
        auth.validate_id_token(make_id_token(rsa_key, aud="oaiapp_other"), "oaiapp_1", "n1", jwks)
    with pytest.raises(jwt.InvalidIssuerError):
        auth.validate_id_token(make_id_token(rsa_key, iss="https://evil"), "oaiapp_1", "n1", jwks)
    with pytest.raises(auth.ChatGPTAuthError, match="nonce"):
        auth.validate_id_token(make_id_token(rsa_key), "oaiapp_1", "other", jwks)
    with pytest.raises(jwt.ExpiredSignatureError):
        auth.validate_id_token(make_id_token(rsa_key, exp=int(time.time()) - 60), "oaiapp_1", "n1", jwks)
    with pytest.raises(jwt.InvalidAlgorithmError):
        hs = jwt.encode({"iss": "https://auth.openai.com", "aud": "oaiapp_1", "sub": "x",
                         "iat": 1, "exp": 2**31, "nonce": "n1"}, "s" * 32, algorithm="HS256")
        auth.validate_id_token(hs, "oaiapp_1", "n1", SimpleNamespace(
            get_signing_key_from_jwt=lambda t: SimpleNamespace(key="s" * 32)))


# --- full login flow (loopback server + mocked OAuth endpoints) -----------------


def test_login_end_to_end(rsa_key, monkeypatch):
    posted = []
    captured = {}

    def fake_browser(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        captured.update(q)
        redirect = q["redirect_uri"] + "?" + urllib.parse.urlencode(
            {"code": "authcode", "state": q["state"], "client_id": "oaiapp_1",
             "scope": "chatgpt.tokens.use.direct email offline_access openid profile resource.invoke"})
        threading.Thread(target=lambda: urllib.request.urlopen(redirect, timeout=5).read(), daemon=True).start()
        return True

    def fake_post(url, data=None, timeout=None):
        posted.append((url, data))
        return SimpleNamespace(status_code=200, json=lambda: {
            "access_token": "at1", "refresh_token": "rt1", "token_type": "Bearer",
            "expires_in": 3600, "id_token": make_id_token(rsa_key, nonce=captured["nonce"]),
            "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
        })

    monkeypatch.setattr(auth, "DEFAULT_PORT", 0)
    monkeypatch.setattr(auth.webbrowser, "open", fake_browser)
    monkeypatch.setattr(auth.requests, "post", fake_post)
    monkeypatch.setattr(auth, "_jwks_client", lambda: FakeJWKS(rsa_key))
    monkeypatch.setattr(auth.sys.stdin, "isatty", lambda: False)

    session = auth.login(timeout=10)
    assert session["access_token"] == "at1" and session["email"] == "me@example.com"
    assert auth.plan_usage_enabled(session)
    url, data = posted[0]
    assert url == "https://auth.openai.com/api/accounts/oauth/token"
    assert data["grant_type"] == "authorization_code" and data["client_id"] == "oaiapp_1"
    assert data["redirect_uri"] == captured["redirect_uri"]
    assert data["redirect_uri"].startswith("http://127.0.0.1:") and data["redirect_uri"].endswith("/auth/callback")
    assert hashlib.sha256(data["code_verifier"].encode()).digest() == base64.urlsafe_b64decode(
        captured["code_challenge"] + "==")
    assert auth.load_host()["client_id"] == "oaiapp_1"
    assert auth.load_host()["plan_notice_shown"] is True
    assert stat.S_IMODE(os.stat(auth._session_path()).st_mode) == 0o600


# --- access tokens / refresh -------------------------------------------------


def _session(**kw):
    s = {"client_id": "oaiapp_1", "access_token": "at1", "refresh_token": "rt1", "expires_in": 3600,
         "saved_at_epoch": time.time(), "scopes": ["chatgpt.tokens.use.direct", "openid"]}
    s.update(kw)
    auth.save_session(s)
    return s


def test_access_token_requires_login_and_scope():
    with pytest.raises(auth.ChatGPTLoginRequired):
        auth.get_access_token()
    _session(scopes=["openid"])
    with pytest.raises(auth.ChatGPTPlanUsageDisabled):
        auth.get_access_token()


def test_refresh_rotates_and_persists(monkeypatch):
    _session(saved_at_epoch=time.time() - 4000)
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append(data)
        return SimpleNamespace(status_code=200, json=lambda: {
            "access_token": "at2", "refresh_token": "rt2", "expires_in": 3600, "token_type": "Bearer"})

    monkeypatch.setattr(auth.requests, "post", fake_post)
    assert auth.get_access_token() == "at2"
    assert calls[0] == {"grant_type": "refresh_token", "client_id": "oaiapp_1",
                        "refresh_token": "rt1", "resource": "https://api.openai.com/v1"}
    assert auth.load_session()["refresh_token"] == "rt2"
    assert auth.get_access_token() == "at2" and len(calls) == 1  # fresh now


def test_concurrent_refresh_happens_once(monkeypatch):
    _session(saved_at_epoch=time.time() - 4000)
    calls = []

    def fake_post(url, data=None, timeout=None):
        calls.append(data)
        time.sleep(0.05)
        return SimpleNamespace(status_code=200, json=lambda: {
            "access_token": "at2", "refresh_token": "rt2", "expires_in": 3600})

    monkeypatch.setattr(auth.requests, "post", fake_post)
    results = []
    threads = [threading.Thread(target=lambda: results.append(auth.get_access_token())) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == ["at2"] * 6 and len(calls) == 1


def test_terminal_refresh_error_requires_login(monkeypatch):
    _session(saved_at_epoch=time.time() - 4000)
    monkeypatch.setattr(auth.requests, "post", lambda *a, **k: SimpleNamespace(
        status_code=400, json=lambda: {"error": "refresh_token_reused"}))
    with pytest.raises(auth.ChatGPTLoginRequired, match="refresh_token_reused"):
        auth.get_access_token()
    assert auth.load_session() is None


def test_transient_refresh_error_keeps_session(monkeypatch):
    _session(saved_at_epoch=time.time() - 4000)
    monkeypatch.setattr(auth.requests, "post", lambda *a, **k: SimpleNamespace(
        status_code=503, json=lambda: {"error": "temporarily_unavailable"}))
    with pytest.raises(auth.ChatGPTAuthError):
        auth.get_access_token()
    assert auth.load_session() is not None


# --- inference ---------------------------------------------------------------


def test_messages_to_responses_input():
    messages = llm.build_messages(
        ["Describe", llm.image(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")),
         llm.Part(kind="pdf", data=b"%PDF-1.4", mime_type="application/pdf", path="/x/paper.pdf")],
        "chatgpt/gpt-6-luna", system_prompt="Be terse.")
    instructions, items = backend.to_responses_input(messages)
    assert instructions == "Be terse."
    assert all(i["role"] != "system" for i in items)
    kinds = [p["type"] for p in items[0]["content"]]
    assert kinds == ["input_text", "input_image", "input_file"]
    assert items[0]["content"][2]["filename"] == "paper.pdf"
    assert items[0]["content"][2]["file_data"].startswith("data:application/pdf;base64,")


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *a):
        return False


def ev(t, **kw):
    return SimpleNamespace(type=t, **kw)


def install_client(monkeypatch, behaviours, seen):
    class Responses:
        def create(self, **kwargs):
            seen.append(kwargs)
            b = behaviours.pop(0)
            if isinstance(b, Exception):
                raise b
            return FakeStream(b)

    monkeypatch.setattr(backend, "_client", lambda token: SimpleNamespace(responses=Responses(), token=token))


def test_complete_streams_text_with_required_flags(monkeypatch):
    _session()
    seen = []
    install_client(monkeypatch, [[ev("response.output_text.delta", delta="Hel"),
                                  ev("response.output_text.delta", delta="lo"),
                                  ev("response.completed")]], seen)
    out = backend.complete("chatgpt/gpt-6-luna", [{"role": "system", "content": "sys"},
                                                  {"role": "user", "content": "hi"}])
    assert out == "Hello"
    assert seen[0]["store"] is False and seen[0]["stream"] is True
    assert seen[0]["model"] == "gpt-6-luna" and seen[0]["instructions"] == "sys"
    assert "temperature" not in seen[0] and "max_output_tokens" not in seen[0]


def test_complete_requires_completed_event(monkeypatch):
    _session()
    install_client(monkeypatch, [[ev("response.output_text.delta", delta="partial")]], [])
    with pytest.raises(backend.ChatGPTTemporaryError):
        backend.complete("chatgpt/m", [{"role": "user", "content": "hi"}])


def test_usage_limit_mid_stream_is_not_retried(monkeypatch):
    _session()
    failed = ev("response.failed", response=SimpleNamespace(
        error=SimpleNamespace(code="subscription_sharing_usage_limit_exceeded")))
    install_client(monkeypatch, [[failed], [ev("response.completed")]], [])
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(backend.ChatGPTUsageLimitError, match="settings/usage"):
        llm.call_llm(["hi"], "chatgpt/m", result_parsing_func=None)


def _status_error(status, body):
    import httpx
    from openai import APIStatusError
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIStatusError("err", response=httpx.Response(status, request=req), body=body)


def test_401_refreshes_once_then_succeeds(monkeypatch):
    _session()
    refreshed = []
    monkeypatch.setattr(auth, "get_access_token",
                        lambda force_refresh=False: refreshed.append(force_refresh) or ("new" if force_refresh else "old"))
    seen = []
    install_client(monkeypatch, [_status_error(401, {"detail": "expired"}),
                                 [ev("response.output_text.delta", delta="ok"), ev("response.completed")]], seen)
    assert backend.complete("chatgpt/m", [{"role": "user", "content": "hi"}]) == "ok"
    assert refreshed == [False, True]


def test_unavailable_is_retried_by_call_llm(monkeypatch):
    _session()
    install_client(monkeypatch, [
        _status_error(503, {"error": {"code": "subscription_sharing_usage_unavailable"}}),
        [ev("response.output_text.delta", delta='{"a": 1}'), ev("response.completed")],
    ], [])
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    out = llm.call_llm(["hi"], "chatgpt/m")
    assert out["parsed_response"] == {"a": 1}


def test_not_eligible_is_not_retried(monkeypatch):
    _session()
    seen = []
    install_client(monkeypatch, [
        _status_error(403, {"error": {"code": "subscription_sharing_user_not_eligible"}}),
        [ev("response.completed")],
    ], seen)
    with pytest.raises(backend.ChatGPTPlanError, match="not eligible"):
        llm.call_llm(["hi"], "chatgpt/m")
    assert len(seen) == 1


def test_call_llm_routes_chatgpt_away_from_litellm(monkeypatch):
    _session()
    monkeypatch.setattr(llm.litellm, "completion", lambda **k: pytest.fail("LiteLLM must not be used"))
    install_client(monkeypatch, [[ev("response.output_text.delta", delta="<think>x</think>plain"),
                                  ev("response.completed")]], [])
    out = llm.call_llm(["hi"], "chatgpt/m", result_parsing_func=None, temperature=0.3)
    assert out["raw_response"] == "plain"


def test_image_generation_unavailable_on_plan(monkeypatch):
    from utils import image_gen
    monkeypatch.setattr(image_gen.litellm, "image_generation", lambda **k: pytest.fail("no"))
    assert image_gen.generate_image("chatgpt/gpt-6-luna", "diagram") is None
