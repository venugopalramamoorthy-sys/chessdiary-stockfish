"""
Firebase ID token verification for this Flask service, and a minimal
per-IP rate limiter -- added after a security review (2026-10-03) found
/analyze and /gemini completely open: anyone with the URL (trivially
discoverable, it's a constant in the public Flutter web JS bundle)
could burn the GEMINI_API_KEY's quota/billing or run the Stockfish
subprocess for free, no chessdiary account required.

Verifies the token's signature against Google's public certs directly
(PyJWT + its JWKS client) rather than pulling in the full firebase-admin
SDK -- that needs a service-account credential to initialize, which
this Render service has no good way to hold; plain signature/issuer/
audience verification needs only the public project ID (already public
-- it's in firebase_options.dart) and Google's own public keys.
"""

import threading
import time
from collections import defaultdict
from functools import wraps

import jwt
from flask import jsonify, request
from jwt import PyJWKClient

FIREBASE_PROJECT_ID = "chessdiary-7f1e3"
_JWKS_URL = "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"
_jwk_client = PyJWKClient(_JWKS_URL)


def _verify_firebase_token(token: str) -> dict:
    signing_key = _jwk_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=FIREBASE_PROJECT_ID,
        issuer=f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}",
    )


def require_firebase_auth(fn):
    """Route decorator: any signed-in chessdiary user is accepted (this
    app is multi-user, unlike trading-terminal's single allowed email)
    -- the point is requiring a real account, not an allowlist."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing bearer token."}), 401
        token = header.removeprefix("Bearer ").strip()
        try:
            _verify_firebase_token(token)
        except Exception as e:
            return jsonify({"error": f"Invalid or expired token: {e}"}), 401
        return fn(*args, **kwargs)

    return wrapper


_lock = threading.Lock()
_hits: dict[str, list[float]] = defaultdict(list)


def rate_limited(max_per_minute: int, bucket: str):
    """Same minimal in-process-per-IP approach used for trading-
    terminal's rate_limit.py -- a free Render instance has no Redis
    either, and losing counters on a redeploy is harmless here."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.remote_addr or "unknown")
            key = f"{bucket}:{ip}"
            now = time.time()
            cutoff = now - 60
            with _lock:
                recent = [t for t in _hits[key] if t > cutoff]
                if len(recent) >= max_per_minute:
                    _hits[key] = recent
                    return jsonify({"error": "Too many requests -- slow down."}), 429
                recent.append(now)
                _hits[key] = recent
            return fn(*args, **kwargs)

        return wrapper

    return decorator
