from flask import Flask, request, jsonify
import chess
import chess.engine
import chess.pgn
import io
import os

app = Flask(__name__)

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
ANALYSIS_DEPTH = 12


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


@app.route("/analyze", methods=["POST"])
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

        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            move_number = 1
            prev_eval = 0

            for move in game.mainline_moves():
                san = board.san(move)
                is_white_move = board.turn == chess.WHITE

                board.push(move)

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

        # Top 20 most significant moves
        results.sort(key=lambda x: -x["centipawnLoss"])
        significant = results[:20]
        significant.sort(key=lambda x: x["moveNumber"])

        return jsonify({
            "analysis": significant,
            "evalCurve": eval_curve,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
