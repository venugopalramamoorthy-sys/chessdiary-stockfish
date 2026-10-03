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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": "stockfish"})


@app.route("/engine-info", methods=["GET"])
def engine_info():
    # Temporary diagnostic route -- reports what Stockfish binary/options
    # this deployment is actually running, to root-cause an eval
    # discrepancy vs. a local reference engine. Remove once resolved.
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        board = chess.Board()
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        return jsonify({
            "id": engine.id,
            "options": {k: str(v) for k, v in engine.options.items() if k in (
                "Hash", "Threads", "UCI_LimitStrength", "UCI_Elo", "Skill Level", "Use NNUE"
            )},
            "startpos_depth12_score": str(info["score"]),
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
                    current_eval = 10000 if is_white_move else -10000
                    eval_curve.append(max(-2000, min(2000, current_eval)))
                    mate_entry = {
                        "moveNumber": move_number,
                        "move": san,
                        "quality": "checkmate",
                        "comment": f"Checkmate — engine evaluation: {current_eval / 100.0:+.2f}",
                        "centipawnLoss": 0,
                        "evalAfter": current_eval,
                        "isWhiteMove": is_white_move,
                    }
                    prev_eval = current_eval
                    if not is_white_move:
                        move_number += 1
                    continue

                info = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
                score = info["score"].white()

                if score.is_mate():
                    current_eval = 10000 if score.mate() > 0 else -10000
                else:
                    current_eval = score.score()

                # Full eval curve (capped ±2000 for storage efficiency)
                eval_curve.append(max(-2000, min(2000, current_eval)))

                if is_white_move:
                    cp_loss = max(0, prev_eval - current_eval)
                else:
                    cp_loss = max(0, current_eval - prev_eval)

                quality = classify_move(cp_loss)

                if quality != "best" or move_number <= 10:
                    eval_display = current_eval / 100.0
                    results.append({
                        "moveNumber": move_number,
                        "move": san,
                        "quality": quality,
                        "comment": f"Engine evaluation: {eval_display:+.2f}",
                        "centipawnLoss": cp_loss,
                        "evalAfter": current_eval,
                        "isWhiteMove": is_white_move,
                    })

                prev_eval = current_eval
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
