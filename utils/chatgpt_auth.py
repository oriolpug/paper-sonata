# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sign in with ChatGPT: use your ChatGPT plan (Plus/Pro) instead of an API key.

Implements OpenAI's official "Sign in with ChatGPT" flow for open-source,
locally run apps (https://developers.openai.com/siwc/token-sharing-open-source):
OAuth authorization code + PKCE with a loopback redirect on 127.0.0.1,
dynamic client registration, ID-token validation, and rotating refresh tokens.
Credentials never leave this machine except to auth.openai.com / api.openai.com.

Usage:
    uv run python -m utils.chatgpt_auth login     # "Continue with ChatGPT"
    uv run python -m utils.chatgpt_auth status
    uv run python -m utils.chatgpt_auth models    # models available to your plan
    uv run python -m utils.chatgpt_auth logout

Then use model names like `chatgpt/<model-slug>`.
"""

import argparse
import base64
import contextlib
import datetime
import hashlib
import http.server
import json
import os
import secrets
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
import webbrowser
from typing import Any, Dict, List, Optional

import jwt  # PyJWT
import requests

try:  # cross-process lock for refresh-token rotation (Unix)
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

APP_NAME = "PaperOrchestra"
AUTH_BASE = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE}/api/accounts/authorize"
TOKEN_URL = f"{AUTH_BASE}/api/accounts/oauth/token"
DISCOVERY_URL = f"{AUTH_BASE}/.well-known/openid-configuration"
DEFAULT_JWKS_URL = f"{AUTH_BASE}/.well-known/jwks.json"
ISSUER = AUTH_BASE
API_RESOURCE = "https://api.openai.com/v1"
REGISTRATION_CLIENT_ID = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
CALLBACK_PATH = "/auth/callback"
DEFAULT_PORT = 1455
USAGE_SETTINGS_URL = "https://chatgpt.com/settings/usage"
# Refresh this many seconds before the access token expires.
REFRESH_MARGIN = 300
# Asymmetric algorithms only; never accept "none" or HMAC for ID tokens.
ID_TOKEN_ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384"]

TERMINAL_REFRESH_ERRORS = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
}


class ChatGPTAuthError(Exception):
    """Base class for Sign in with ChatGPT errors."""


class ChatGPTLoginRequired(ChatGPTAuthError):
    """No usable session: the user must run `python -m utils.chatgpt_auth login`."""


class ChatGPTPlanUsageDisabled(ChatGPTAuthError):
    """Signed in, but ChatGPT plan usage was not granted (scope missing)."""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def config_dir() -> str:
    override = os.environ.get("PAPER_CHATGPT_CONFIG_DIR")
    if override:
        return override
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "paper-orchestra")


def _host_path() -> str:
    return os.path.join(config_dir(), "chatgpt_host.json")


def _session_path() -> str:
    return os.path.join(config_dir(), "chatgpt_session.json")


def _lock_path() -> str:
    return os.path.join(config_dir(), "chatgpt_session.lock")


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _write_json_private(path: str, data: Dict[str, Any]):
    """Atomic write with owner-only permissions (0600)."""
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def load_host() -> Dict[str, Any]:
    """Per-host registration data: ext_agent_host_id and the issued client_id.

    The host ID is an opaque UUIDv4 URN generated once per machine; it is not
    derived from any user-identifying value.
    """
    host = _read_json(_host_path()) or {}
    if not host.get("ext_agent_host_id"):
        host["ext_agent_host_id"] = f"urn:uuid:{uuid.uuid4()}"
        _write_json_private(_host_path(), host)
    return host


def save_host(host: Dict[str, Any]):
    _write_json_private(_host_path(), host)


def load_session() -> Optional[Dict[str, Any]]:
    return _read_json(_session_path())


def save_session(session: Dict[str, Any]):
    _write_json_private(_session_path(), session)


def clear_session():
    with contextlib.suppress(FileNotFoundError):
        os.remove(_session_path())


# ---------------------------------------------------------------------------
# PKCE / authorize URL
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def new_pkce_pair() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(
    host: Dict[str, Any],
    redirect_uri: str,
    state: str,
    nonce: str,
    code_challenge: str,
    id_token_hint: Optional[str] = None,
    login_hint: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    client_id = host.get("client_id")
    params = {
        "client_id": client_id or REGISTRATION_CLIENT_ID,
        "ext_agent_host_id": host["ext_agent_host_id"],
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": API_RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if not client_id:
        # First-time registration only; display metadata, not identity.
        params["agent_name_hint"] = APP_NAME
    else:
        if id_token_hint:
            params["id_token_hint"] = id_token_hint
        if login_hint:
            params["login_hint"] = login_hint
    if prompt:
        params["prompt"] = prompt
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"


# ---------------------------------------------------------------------------
# Loopback callback server
# ---------------------------------------------------------------------------

_SUCCESS_HTML = b"""<!doctype html><html><body style="font-family:sans-serif">
<h3>Signed in to PaperOrchestra with ChatGPT.</h3><p>You can close this tab.</p>
</body></html>"""
_FAIL_HTML = b"""<!doctype html><html><body style="font-family:sans-serif">
<h3>Sign-in did not complete.</h3><p>Return to the terminal for details.</p>
</body></html>"""


class _CallbackServer(http.server.HTTPServer):
    result: Optional[Dict[str, str]] = None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        self.server.result = params
        ok = "code" in params and "error" not in params
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SUCCESS_HTML if ok else _FAIL_HTML)

    def log_message(self, *args):  # keep codes/tokens out of logs
        pass


def _start_callback_server() -> _CallbackServer:
    try:
        return _CallbackServer(("127.0.0.1", DEFAULT_PORT), _CallbackHandler)
    except OSError:
        return _CallbackServer(("127.0.0.1", 0), _CallbackHandler)  # only the port may vary


def _wait_for_callback(server: _CallbackServer, timeout: float) -> Dict[str, str]:
    server.timeout = 1.0
    deadline = time.monotonic() + timeout
    while server.result is None and time.monotonic() < deadline:
        server.handle_request()
    if server.result is None:
        raise ChatGPTAuthError("Timed out waiting for the browser sign-in to complete.")
    return server.result


# ---------------------------------------------------------------------------
# Callback / token handling
# ---------------------------------------------------------------------------


def process_callback(params: Dict[str, str], expected_state: str, host: Dict[str, Any]) -> tuple[str, str]:
    """Validates the redirect and returns (authorization code, issued client_id)."""
    if not secrets.compare_digest(params.get("state", ""), expected_state):
        raise ChatGPTAuthError("Sign-in rejected: state mismatch.")
    if params.get("error"):
        if params["error"] == "access_denied":
            raise ChatGPTAuthError("Sign-in cancelled: consent was declined.")
        raise ChatGPTAuthError(f"Sign-in failed: {params['error']}")
    code = params.get("code")
    if not code:
        raise ChatGPTAuthError("Sign-in failed: no authorization code returned.")

    saved_id = host.get("client_id")
    returned_id = params.get("client_id")
    if saved_id:
        if returned_id and returned_id != saved_id:
            raise ChatGPTAuthError("Sign-in rejected: unexpected client ID returned.")
        return code, saved_id
    if not returned_id:
        raise ChatGPTAuthError("Registration incomplete: no client ID was issued.")
    return code, returned_id


def _token_request(data: Dict[str, str]) -> Dict[str, Any]:
    rsp = requests.post(TOKEN_URL, data=data, timeout=30)
    try:
        body = rsp.json()
    except ValueError:
        body = {"error": f"http_{rsp.status_code}"}
    if rsp.status_code != 200:
        body = body if isinstance(body, dict) else {}
        error = body.get("error")
        description = body.get("error_description") or ""
        if isinstance(error, dict):
            description = description or error.get("message") or ""
            error = error.get("code") or error.get("type")
        raise _TokenEndpointError(
            str(error or f"http_{rsp.status_code}"), rsp.status_code, str(description)
        )
    return body


class _TokenEndpointError(ChatGPTAuthError):
    def __init__(self, code: str, status: int, description: str = ""):
        message = f"token endpoint error: {code} (HTTP {status})"
        if description:
            message += f": {description}"
        super().__init__(message)
        self.code = code
        self.status = status
        self.description = description


def exchange_code(code: str, client_id: str, verifier: str, redirect_uri: str) -> Dict[str, Any]:
    return _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": API_RESOURCE,
        }
    )


_jwks_clients: Dict[str, jwt.PyJWKClient] = {}


def _jwks_client() -> jwt.PyJWKClient:
    url = DEFAULT_JWKS_URL
    try:
        url = requests.get(DISCOVERY_URL, timeout=15).json().get("jwks_uri") or url
    except Exception:
        pass
    if url not in _jwks_clients:
        _jwks_clients[url] = jwt.PyJWKClient(url, cache_keys=True)
    return _jwks_clients[url]


def validate_id_token(id_token: str, client_id: str, nonce: str, jwks_client=None) -> Dict[str, Any]:
    client = jwks_client or _jwks_client()
    signing_key = client.get_signing_key_from_jwt(id_token)  # refetches on unknown kid
    claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=ID_TOKEN_ALGORITHMS,
        audience=client_id,
        issuer=ISSUER,
        leeway=5,
        options={"require": ["exp", "iat", "sub", "iss", "aud"]},
    )
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise ChatGPTAuthError("Sign-in rejected: ID token nonce mismatch.")
    if not claims.get("sub"):
        raise ChatGPTAuthError("Sign-in rejected: ID token has no subject.")
    return claims


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _session_from_tokens(
    tokens: Dict[str, Any], client_id: str, host_id: str, claims: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "email": claims.get("email") if claims.get("email_verified") is True else None,
        "issuer": ISSUER,
        "subject": claims["sub"],
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "id_token": tokens.get("id_token"),
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "token_type": tokens.get("token_type", "Bearer"),
        "expires_in": int(tokens.get("expires_in", 3600)),
        "scopes": sorted((tokens.get("scope") or "").split()),
        "saved_at": _now_iso(),
        "saved_at_epoch": time.time(),
    }


def plan_usage_enabled(session: Optional[Dict[str, Any]]) -> bool:
    return bool(session) and PLAN_SCOPE in (session.get("scopes") or [])


def _show_plan_notice(host: Dict[str, Any]):
    """One-time confirmation required by the SIWC UX guidelines."""
    if host.get("plan_notice_shown"):
        return
    print("\n  You're using your ChatGPT plan")
    print("  Eligible usage in this app uses your ChatGPT plan. Manage usage in your ChatGPT settings:")
    print(f"  {USAGE_SETTINGS_URL}\n")
    if sys.stdin.isatty():
        with contextlib.suppress(EOFError):
            input("  [Got it] Press Enter to continue ")
    host["plan_notice_shown"] = True
    save_host(host)


def login(
    open_browser: bool = True,
    timeout: float = 300.0,
    prompt: Optional[str] = None,
    _retried: bool = False,
) -> Dict[str, Any]:
    """Runs the browser sign-in ("Continue with ChatGPT") and stores the session."""
    host = load_host()
    previous = load_session()
    verifier, challenge = new_pkce_pair()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)

    server = _start_callback_server()
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}{CALLBACK_PATH}"
    url = build_authorize_url(
        host,
        redirect_uri,
        state,
        nonce,
        challenge,
        id_token_hint=(previous or {}).get("id_token"),
        login_hint=(previous or {}).get("email"),
        prompt=prompt,
    )
    print("Continue with ChatGPT: complete sign-in in your browser.")
    print(f"If the browser does not open, visit:\n  {url}\n")
    if open_browser:
        webbrowser.open(url)
    try:
        params = _wait_for_callback(server, timeout)
    finally:
        server.server_close()

    code, client_id = process_callback(params, state, host)
    if host.get("client_id") != client_id:
        host["client_id"] = client_id
        save_host(host)

    try:
        tokens = exchange_code(code, client_id, verifier, redirect_uri)
    except _TokenEndpointError as e:
        if e.code != "invalid_grant" or _retried:
            raise
        # Documented recovery: discard the code and start a fresh
        # authorization with the retained issued client ID. This is expected
        # right after first-time registration.
        print("Finishing sign-in with the newly registered app; your browser will open again.")
        return login(open_browser=open_browser, timeout=timeout, prompt=prompt, _retried=True)
    claims = validate_id_token(tokens["id_token"], client_id, nonce)
    session = _session_from_tokens(tokens, client_id, host["ext_agent_host_id"], claims)
    save_session(session)

    if not plan_usage_enabled(session):
        raise ChatGPTPlanUsageDisabled(
            "Signed in, but ChatGPT plan usage was not granted. Run login again and "
            "allow plan usage, or use an API key model instead."
        )
    _show_plan_notice(host)
    return session


# ---------------------------------------------------------------------------
# Access tokens for inference (thread- and process-safe refresh)
# ---------------------------------------------------------------------------

_refresh_lock = threading.Lock()


@contextlib.contextmanager
def _process_lock():
    os.makedirs(config_dir(), mode=0o700, exist_ok=True)
    with open(_lock_path(), "a") as fh:
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _expired(session: Dict[str, Any], margin: float = REFRESH_MARGIN) -> bool:
    saved = float(session.get("saved_at_epoch") or 0)
    return time.time() >= saved + int(session.get("expires_in", 3600)) - margin


def refresh_session(session: Dict[str, Any]) -> Dict[str, Any]:
    try:
        tokens = _token_request(
            {
                "grant_type": "refresh_token",
                "client_id": session["client_id"],
                "refresh_token": session["refresh_token"],
                "resource": API_RESOURCE,
            }
        )
    except _TokenEndpointError as e:
        if e.code in TERMINAL_REFRESH_ERRORS:
            clear_session()
            raise ChatGPTLoginRequired(
                f"ChatGPT session ended ({e.code}). Sign in again: "
                "uv run python -m utils.chatgpt_auth login"
            ) from e
        if e.code == "invalid_client":
            raise ChatGPTAuthError("ChatGPT client configuration is invalid (invalid_client).") from e
        raise
    updated = dict(session)
    updated.update(
        {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token") or session["refresh_token"],
            "expires_in": int(tokens.get("expires_in", 3600)),
            "token_type": tokens.get("token_type", "Bearer"),
            "saved_at": _now_iso(),
            "saved_at_epoch": time.time(),
        }
    )
    if tokens.get("scope"):
        updated["scopes"] = sorted(tokens["scope"].split())
    if tokens.get("id_token"):
        updated["id_token"] = tokens["id_token"]
    save_session(updated)
    return updated


def get_access_token(force_refresh: bool = False) -> str:
    """Returns a valid access token, refreshing (and rotating) when needed."""
    session = load_session()
    if not session or not session.get("access_token"):
        raise ChatGPTLoginRequired(
            "Not signed in with ChatGPT. Run: uv run python -m utils.chatgpt_auth login"
        )
    if not plan_usage_enabled(session):
        raise ChatGPTPlanUsageDisabled(
            "ChatGPT plan usage is not enabled for this sign-in. Run login again and allow it."
        )
    if not force_refresh and not _expired(session):
        return session["access_token"]

    # Serialize refreshes across threads and processes: refresh tokens rotate.
    with _refresh_lock, _process_lock():
        current = load_session()
        if not current:
            raise ChatGPTLoginRequired("ChatGPT session was removed. Sign in again.")
        if current.get("refresh_token") != session.get("refresh_token") and not _expired(current):
            return current["access_token"]  # another worker already refreshed
        if not force_refresh and not _expired(current):
            return current["access_token"]
        return refresh_session(current)["access_token"]


def list_models(access_token: Optional[str] = None) -> List[Dict[str, str]]:
    token = access_token or get_access_token()
    rsp = requests.get(
        f"{API_RESOURCE}/models", headers={"Authorization": f"Bearer {token}"}, timeout=30
    )
    rsp.raise_for_status()
    body = rsp.json()
    entries = body.get("models") or body.get("data") or []
    return [
        {"slug": m.get("slug") or m.get("id"), "display_name": m.get("display_name") or m.get("id")}
        for m in entries
        if m.get("visibility", "list") == "list"
    ]


def logout():
    """Revokes the refresh token (best effort) and deletes the local session."""
    session = load_session()
    if session and session.get("refresh_token"):
        try:
            revocation = requests.get(DISCOVERY_URL, timeout=15).json().get("revocation_endpoint")
            if revocation:
                requests.post(
                    revocation,
                    data={
                        "token": session["refresh_token"],
                        "token_type_hint": "refresh_token",
                        "client_id": session["client_id"],
                    },
                    timeout=15,
                )
        except Exception as e:
            print(f"Warning: could not revoke token remotely ({type(e).__name__}).")
    clear_session()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _status() -> int:
    session = load_session()
    if not session:
        print("Not signed in with ChatGPT.")
        return 1
    expires = float(session.get("saved_at_epoch", 0)) + int(session.get("expires_in", 0))
    print(f"Signed in with ChatGPT{(' as ' + session['email']) if session.get('email') else ''}.")
    print(f"Plan usage: {'enabled' if plan_usage_enabled(session) else 'NOT granted'}")
    print(f"Access token expires: {datetime.datetime.fromtimestamp(expires).isoformat(timespec='seconds')}")
    print(f"Manage usage: {USAGE_SETTINGS_URL}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Use your ChatGPT plan with PaperOrchestra.")
    sub = parser.add_subparsers(dest="command", required=True)
    login_p = sub.add_parser("login", help="Continue with ChatGPT (browser sign-in).")
    login_p.add_argument("--no-browser", action="store_true", help="Print the URL instead of opening it.")
    login_p.add_argument(
        "--reconsent", action="store_true", help="Ask for consent again (e.g. to enable plan usage)."
    )
    sub.add_parser("status", help="Show sign-in status.")
    sub.add_parser("models", help="List models available to your ChatGPT plan.")
    sub.add_parser("logout", help="Sign out and delete local credentials.")
    args = parser.parse_args(argv)

    try:
        if args.command == "login":
            login(open_browser=not args.no_browser, prompt="consent" if args.reconsent else None)
            print("Signed in. Use models as chatgpt/<model>; list them with: "
                  "uv run python -m utils.chatgpt_auth models")
            return 0
        if args.command == "status":
            return _status()
        if args.command == "models":
            for m in list_models():
                print(f"chatgpt/{m['slug']:40s} {m['display_name']}")
            return 0
        if args.command == "logout":
            logout()
            print("Signed out.")
            return 0
    except ChatGPTAuthError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
