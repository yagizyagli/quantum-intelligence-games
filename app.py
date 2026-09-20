import os
from flask import Flask, jsonify, request
from src.chess.board import QuantumChessBoard
from src.chess.engine import QuantumChessEngine
from src.qml.reinforcement import QuantumReinforcementAgent

app = Flask(__name__, static_folder=".", static_url_path="")

# Initialize global game states for the live demo session
chess_board = QuantumChessBoard()
chess_engine = QuantumChessEngine(chess_board)
qml_agent = QuantumReinforcementAgent(num_actions=4)

@app.route("/")
def serve_index():
    """Serves the single-file UI dashboard from the root directory."""
    return app.send_static_file("index.html")

@app.route("/api/chess/state", methods=["GET"])
def get_chess_state():
    """Returns current quantum chess board configurations and probability mappings."""
    probabilities = chess_board.quantum_state.calculate_probabilities().tolist()
    grid_state = [[str(cell) if cell is not None else "" for cell in row] for row in chess_board.grid]
    
    return jsonify({
        "grid": grid_state,
        "probabilities": probabilities
    })

@app.route("/api/chess/split", methods=["POST"])
def chess_split_move():
    """Executes a quantum split move requested by the UI frontend."""
    data = request.json
    piece = data["piece"]
    from_c = tuple(data["from"])
    to_a = tuple(data["to_a"])
    to_b = tuple(data["to_b"])
    
    try:
        chess_board.execute_split_move(piece, from_c, to_a, to_b)
        return jsonify({"status": "success", "message": f"Split move executed for {piece}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/qml/predict", methods=["POST"])
def qml_predict():
    """Triggers the parameterized quantum neural network to evaluate current board values."""
    data = request.json
    state_features = data["features"]
    legal_moves = data["legal_moves"]
    
    try:
        chosen_action = qml_agent.select_action(state_features, legal_moves)
        return jsonify({"status": "success", "action": chosen_action, "epsilon": qml_agent.epsilon})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # Run server locally on standard port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
