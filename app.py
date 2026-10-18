from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess.engine
import chess.pgn
import io
import logging
import os
import requests as req

from auth import rate_limited, require_firebase_auth

app = Flask(__name__)
# Scoresheet photos/PDFs arrive base64-encoded inside the /gemini JSON body,
# which Gemini itself caps at 20 MB per request
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Local dev origins (e.g. "http://localhost:8080") come from the environment
# so production doesn't accept them
CORS(app, origins=[
    "https://chessdiary.app",
    "https://chessdiary-7f1e3.web.app",
    *filter(None, os.environ.get("CORS_DEV_ORIGINS", "").split(",")),
])

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
ANALYSIS_DEPTH = 12

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# The client used to choose the model, and the name went straight into the
# upstream URL -- letting any signed-in user pick pricier models or reach
# other Gemini API endpoints with our key. Only models the app uses pass.
GEMINI_ALLOWED_MODELS = {"gemini-2.5-flash"}
GEMINI_ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
    "application/pdf", "text/plain",
}
MAX_PROMPT_CHARS = 30_000

# A real game is well under this; it bounds how long one request can keep
# Stockfish busy
MAX_PGN_CHARS = 20_000
MAX_PLIES = 400


def classify_move(cp_loss):
    if cp_loss is None:
        return "good"
    if cp_loss <= 10:
        return "best"
    elif cp_loss <= 40:
        return "good"
    elif cp_loss <= 90:
        return "inaccuracy"
    elif cp_loss <= 200:
        return "mistake"
    else:
        return "blunder"


# A forced mate is flattened to a single magic number (10000) by the old
# is_mate()/mate() handling below, which throws away the actual mate
# distance. Two
# problems follow: (1) a mate-in-1 and a mate-in-10 both render as the
# identical "+100.00", which doesn't reflect what the engine actually
# found, and (2) subtracting that flat sentinel from a real centipawn
# score (e.g. 10000 - 250) produces a nonsensical "centipawn loss" like
# 9750 -- a sentinel isn't a real centipawn count, so arithmetic against
# one is meaningless. mate_eval() keeps the distance (so two different
# mate depths are two different numbers, display can say "Mate in N"),
# and cp_loss_for_move() below special-cases every transition across a
# mate boundary instead of ever subtracting through a sentinel.
MATE_CP_CEILING = 10000


def mate_eval(mate_in):
    """White-POV pseudo-centipawn value for a mate score, monotonic in
    distance (closer mate = larger magnitude) but always far outside any
    real centipawn range, so it still caps to +-2000 in eval_curve like
    the old flat sentinel did."""
    magnitude = max(MATE_CP_CEILING - abs(mate_in) * 10, 9000)
    return magnitude if mate_in > 0 else -magnitude


def cp_loss_for_move(prev_eval, prev_mate, current_eval, current_mate, is_white_move):
    """Centipawn loss for the side that just moved, aware of forced-mate
    transitions on either side of the move:

    - A move that keeps or delivers a forced mate for the mover is never
      penalized (cp_loss 0) -- it's as good as a move can be, regardless
      of whether a faster mate existed.
    - A move made while already facing a forced mate, that still faces
      one afterwards, isn't penalized further -- every legal move there
      loses by force, so there's no meaningful "loss" to attribute to
      this particular one.
    - A move that escapes a forced mate against the mover is an
      improvement, never a loss.
    - A move that walks the mover INTO a now-forced mate against
      themselves is graded on a fixed, sensible ceiling (still clearly a
      blunder) instead of a raw sentinel-minus-real-centipawns figure.
    - Anything not touching a mate score on either side is untouched:
      the normal real-centipawn subtraction, exactly as before.
    """
    sign = 1 if is_white_move else -1
    mover_mate_before = sign * prev_mate if prev_mate is not None else None
    mover_mate_after = sign * current_mate if current_mate is not None else None

    if mover_mate_after is not None and mover_mate_after > 0:
        return 0  # keeps or delivers a forced mate for the mover
    if mover_mate_before is not None and mover_mate_before < 0:
        if mover_mate_after is not None and mover_mate_after < 0:
            return 0  # already lost to forced mate, still is -- no fresh penalty
        if mover_mate_after is None:
            return 0  # escaped a forced mate against the mover
    if mover_mate_after is not None and mover_mate_after < 0:
        return 1000  # this move is what delivers the mover into a forced mate

    mover_before = sign * prev_eval
    mover_after = sign * current_eval
    return max(0, mover_before - mover_after)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "stockfish"})


@app.route("/analyze", methods=["POST"])
@require_firebase_auth
@rate_limited(10, "analyze", max_per_day=150)
def analyze():
    """
    Request: { "pgn": "1. e4 e5 ..." }

    Response:
    {
        "analysis": [flagged moves with motif placeholder],
        "evalCurve": [centipawn per half-move, white's perspective, capped ±2000]
    }
    """
    data = request.get_json(silent=True)
    pgn_text = data.get("pgn", "") if isinstance(data, dict) else ""

    if not isinstance(pgn_text, str) or not pgn_text.strip():
        return jsonify({"error": "No PGN provided"}), 400
    if len(pgn_text) > MAX_PGN_CHARS:
        return jsonify({"error": "PGN too long"}), 413

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return jsonify({"error": "Could not parse PGN"}), 400
        if sum(1 for _ in game.mainline_moves()) > MAX_PLIES:
            return jsonify({"error": f"Game too long (max {MAX_PLIES // 2} moves)"}), 413

        board = game.board()
        results = []
        eval_curve = []  # full centipawn curve, one value per half-move

        mate_entry = None  # the move that actually delivers checkmate, if any -- kept
        # separate from `results` so it can never be dropped by the top-20-by-
        # significance cut below (see its centipawnLoss of 0, which would
        # otherwise sort it to the bottom in a game with several real blunders).

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            move_number = 1
            prev_eval = 0
            prev_mate = None  # white-POV signed mate distance, or None if not mate

            for move in game.mainline_moves():
                san = board.san(move)
                is_white_move = board.turn == chess.WHITE

                board.push(move)

                if board.is_checkmate():
                    # The move just played ends the game -- the side to move
                    # now has zero legal moves, so there is nothing left for
                    # the engine to search. Calling engine.analyse() on a
                    # position like this previously produced a nonsensical
                    # score (e.g. -100.00 "blunder" on the actual mating
                    # move) because a terminal position has no well-defined
                    # search result. A move that delivers checkmate is by
                    # definition the best move on the board, in either
                    # player's favor depending on who just moved -- it can
                    # never be an inaccuracy/mistake/blunder, so it's handled
                    # here directly instead of going through classify_move().
                    current_mate = 1 if is_white_move else -1
                    current_eval = mate_eval(current_mate)
                    eval_curve.append(max(-2000, min(2000, current_eval)))
                    mate_entry = {
                        "moveNumber": move_number,
                        "move": san,
                        "quality": "checkmate",
                        "comment": f"Checkmate — {'White' if is_white_move else 'Black'} wins",
                        "centipawnLoss": 0,
                        "evalAfter": current_eval,
                        "isWhiteMove": is_white_move,
                    }
                    prev_eval = current_eval
                    prev_mate = current_mate
                    if not is_white_move:
                        move_number += 1
                    continue

                info = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
                score = info["score"].white()

                if score.is_mate():
                    current_mate = score.mate()
                    current_eval = mate_eval(current_mate)
                else:
                    current_mate = None
                    current_eval = score.score()

                # Full eval curve (capped ±2000 for storage efficiency)
                eval_curve.append(max(-2000, min(2000, current_eval)))

                cp_loss = cp_loss_for_move(prev_eval, prev_mate, current_eval, current_mate, is_white_move)
                quality = classify_move(cp_loss)

                if quality != "best" or move_number <= 10:
                    if current_mate is not None:
                        eval_display = f"Mate in {abs(current_mate)} for {'White' if current_mate > 0 else 'Black'}"
                    else:
                        eval_display = f"Engine evaluation: {current_eval / 100.0:+.2f}"
                    results.append({
                        "moveNumber": move_number,
                        "move": san,
                        "quality": quality,
                        "comment": eval_display,
                        "centipawnLoss": cp_loss,
                        "evalAfter": current_eval,
                        "isWhiteMove": is_white_move,
                    })

                prev_eval = current_eval
                prev_mate = current_mate
                if not is_white_move:
                    move_number += 1

        # Top 20 most significant moves, plus the checkmating move (if any)
        # unconditionally -- it's the single most important move of the
        # game regardless of where its centipawnLoss of 0 would otherwise
        # place it in a significance-ordered cut.
        results.sort(key=lambda x: -x["centipawnLoss"])
        significant = results[:20]
        if mate_entry is not None:
            significant.append(mate_entry)
        significant.sort(key=lambda x: x["moveNumber"])

        return jsonify({
            "analysis": significant,
            "evalCurve": eval_curve,
        })

    except Exception:
        logging.exception("Analysis failed")
        return jsonify({"error": "Analysis failed"}), 500


def _valid_gemini_contents(contents) -> bool:
    """Accept only the shape the app sends: one user turn holding a text
    prompt and at most one inline image/PDF."""
    if not isinstance(contents, list) or len(contents) != 1:
        return False
    turn = contents[0]
    if not isinstance(turn, dict) or set(turn) - {"parts", "role"}:
        return False
    parts = turn.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 2:
        return False
    texts = inline = 0
    for part in parts:
        if not isinstance(part, dict) or len(part) != 1:
            return False
        if "text" in part:
            if not isinstance(part["text"], str) or len(part["text"]) > MAX_PROMPT_CHARS:
                return False
            texts += 1
        elif "inlineData" in part:
            data = part["inlineData"]
            if (not isinstance(data, dict) or set(data) != {"mimeType", "data"}
                    or data["mimeType"] not in GEMINI_ALLOWED_MIME_TYPES
                    or not isinstance(data["data"], str)):
                return False
            inline += 1
        else:
            return False
    return texts == 1 and inline <= 1


@app.route("/gemini", methods=["POST"])
@require_firebase_auth
@rate_limited(10, "gemini", max_per_day=150)
def gemini_proxy():
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set")
        return jsonify({"error": "AI service unavailable"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid request"}), 400
    model = payload.get("model", "gemini-2.5-flash")
    if model not in GEMINI_ALLOWED_MODELS:
        return jsonify({"error": "Model not allowed"}), 400
    contents = payload.get("contents")
    if not _valid_gemini_contents(contents):
        return jsonify({"error": "Invalid request"}), 400

    try:
        resp = req.post(
            f"{GEMINI_BASE_URL}/{model}:generateContent",
            params={"key": GEMINI_API_KEY},
            json={"contents": contents},
            timeout=60,
        )
        body = resp.json()
    except Exception:
        logging.exception("Gemini request failed")
        return jsonify({"error": "AI service error"}), 502

    if resp.status_code != 200:
        # Pass on only Google's status code name (e.g. RESOURCE_EXHAUSTED,
        # which the app shows as "at capacity"), not the full error body
        status = (body.get("error") or {}).get("status", "UNKNOWN") if isinstance(body, dict) else "UNKNOWN"
        logging.warning("Gemini returned %s: %s", resp.status_code, status)
        return jsonify({"error": status}), resp.status_code
    return jsonify(body), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
