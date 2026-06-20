# stockfish_server/app.py
#
# A tiny Flask server that runs the Stockfish chess engine.
# Your Flutter app sends a PGN/move list here, and gets back
# real engine-level analysis: centipawn evaluation, best move,
# and move quality (best/good/inaccuracy/mistake/blunder).
#
# Deploy this FREE on Render.com (instructions in README).

from flask import Flask, request, jsonify
import chess
import chess.engine
import chess.pgn
import io
import os

app = Flask(__name__)

# Path to the Stockfish binary — set by the buildpack on Render
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")

# How deep the engine should think. Higher = more accurate but slower.
ANALYSIS_DEPTH = 12


def classify_move(cp_loss):
    """Classify a move's quality based on centipawn loss compared to best move."""
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


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Request body:
    {
        "pgn": "1. e4 e5 2. Nf3 Nc6 ..."
    }

    Response:
    {
        "analysis": [
            {
                "moveNumber": 1,
                "move": "e4",
                "quality": "best",
                "comment": "Engine eval: +0.3",
                "centipawnLoss": 0
            },
            ...
        ]
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

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            move_number = 1
            prev_eval = 0  # from white's perspective, in centipawns

            for move in game.mainline_moves():
                san = board.san(move)
                is_white_move = board.turn == chess.WHITE

                board.push(move)

                # Evaluate position AFTER the move
                info = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
                score = info["score"].white()

                if score.is_mate():
                    current_eval = 10000 if score.mate() > 0 else -10000
                else:
                    current_eval = score.score()

                # Centipawn loss = how much the position got worse
                # for the side that just moved
                if is_white_move:
                    cp_loss = max(0, prev_eval - current_eval)
                else:
                    cp_loss = max(0, current_eval - prev_eval)

                quality = classify_move(cp_loss)

                # Only include moves worth reporting (skip routine "best" moves
                # in the opening to keep response concise)
                if quality != "best" or move_number <= 10:
                    eval_display = current_eval / 100.0
                    results.append({
                        "moveNumber": move_number,
                        "move": san,
                        "quality": quality,
                        "comment": f"Engine evaluation: {eval_display:+.2f}",
                        "centipawnLoss": cp_loss
                    })

                prev_eval = current_eval
                if not is_white_move:
                    move_number += 1

        # Limit to the most significant 20 moves (favor mistakes/blunders)
        results.sort(key=lambda x: -x["centipawnLoss"])
        significant = results[:20]
        significant.sort(key=lambda x: x["moveNumber"])

        return jsonify({"analysis": significant})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
