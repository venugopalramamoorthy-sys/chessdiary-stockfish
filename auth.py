"""
Firebase ID token verification for this Flask service, and a minimal
per-user rate limiter -- added after a security review (2026-10-03) found
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

import logging
import threading
import time
from collections import defaultdict
from functools import wraps

import jwt
from flask import g, jsonify, request
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
            claims = _verify_firebase_token(token)
        except Exception as e:
            # Detail goes to the server log only -- don't hand verification
            # internals back to the caller
            logging.warning("Rejected Firebase token: %s", e)
            return jsonify({"error": "Invalid or expired token."}), 401
        g.uid = claims["sub"]
        return fn(*args, **kwargs)

    return wrapper


_lock = threading.Lock()
_hits: dict[str, list[float]] = defaultdict(list)
_daily: dict[str, tuple[int, int]] = {}  # key -> (day number, count)


def rate_limited(max_per_minute: int, bucket: str, max_per_day: int | None = None):
    """Per-user limits, keyed on the verified Firebase uid -- so it must sit
    below @require_firebase_auth. Keying on the uid rather than the client
    IP matters: the IP had to come from X-Forwarded-For, whose leftmost
    value the client controls, so sending a fresh value on every request
    skipped the limit entirely.

    In-process counters: a free Render instance has no Redis, so they reset
    on a redeploy/restart -- acceptable for bounding abuse, and the app
    runs a single gunicorn worker so every thread shares the same counts."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{bucket}:{g.uid}"
            now = time.time()
            cutoff = now - 60
            today = int(now // 86400)
            with _lock:
                recent = [t for t in _hits[key] if t > cutoff]
                if len(recent) >= max_per_minute:
                    _hits[key] = recent
                    return jsonify({"error": "Too many requests -- slow down."}), 429
                day, count = _daily.get(key, (today, 0))
                if day != today:
                    count = 0
                if max_per_day is not None and count >= max_per_day:
                    return jsonify({"error": "Daily limit reached -- try again tomorrow."}), 429
                recent.append(now)
                _hits[key] = recent
                _daily[key] = (today, count + 1)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
