from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import chess.engine
import chess.pgn
import io
import os
import requests as req

from auth import rate_limited, require_firebase_auth

app = Flask(__name__)
CORS(app, origins=[
    "https://chessdiary.app",
    "https://chessdiary-7f1e3.web.app",
    "http://localhost:8080",
    "http://localhost:5000",
])

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
ANALYSIS_DEPTH = 12

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


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


@app.route("/debug-timing", methods=["GET"])
@require_firebase_auth
def debug_timing():
    # Temporary diagnostic route -- measures real wall-clock search time on
    # THIS production box at a given depth or time limit, for an explicit
    # depth/cost tradeoff discussion. Does not touch /analyze. Remove once
    # the numbers are reported.
    import time
    fen = request.args.get("fen", chess.STARTING_FEN)
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return jsonify({"error": f"bad fen: {e}"}), 400
    depth = request.args.get("depth")
    time_limit = request.args.get("time")
    if depth is None and time_limit is None:
        return jsonify({"error": "pass depth or time"}), 400
    limit_kwargs = {}
    if depth is not None:
        limit_kwargs["depth"] = int(depth)
    if time_limit is not None:
        limit_kwargs["time"] = float(time_limit)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        t0 = time.monotonic()
        info = engine.analyse(board, chess.engine.Limit(**limit_kwargs))
        elapsed = time.monotonic() - t0
        return jsonify({
            "fen": fen,
            "limit": limit_kwargs,
            "elapsed_seconds": round(elapsed, 3),
            "score": str(info["score"]),
            "depth_reached": info.get("depth"),
        })


@app.route("/analyze", methods=["POST"])
@require_firebase_auth
@rate_limited(20, "analyze")
def analyze():
    """
    Request: { "pgn": "1. e4 e5 ..." }

    Response:
    {
        "analysis": [flagged moves with motif placeholder],
        "evalCurve": [centipawn per half-move, white's perspective, capped ±2000]
    }
    """
    data = request.get_json()
    pgn_text = data.get("pgn", "")

    if not pgn_text.strip():
        return jsonify({"error": "No PGN provided"}), 400

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return jsonify({"error": "Could not parse PGN"}), 400

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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gemini", methods=["POST"])
@require_firebase_auth
@rate_limited(20, "gemini")
def gemini_proxy():
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set on server"}), 500

    payload = request.get_json()
    model = payload.get("model", "gemini-2.5-flash")
    contents = payload.get("contents", [])

    resp = req.post(
        f"{GEMINI_BASE_URL}/{model}:generateContent",
        params={"key": GEMINI_API_KEY},
        json={"contents": contents},
        timeout=60,
    )
    return jsonify(resp.json()), resp.status_code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
