# 🌌 Quantum Intelligence Games Suite

An industry-grade, ultra-lightweight, and zero-dependency simulation ecosystem fusing **Quantum Mechanics** (superposition, multi-state entanglement, wave function collapse) with **Quantum Machine Learning (QML)** algorithms. This suite reimagines classic strategy games—**Chess, Sudoku, and Minesweeper**—as probabilistic matrix state spaces optimized for hybrid classical-quantum reinforcement agents.

---

## 🌟 Key Highlights & Engineering Features

*   **🎰 Pure Client-Side Simulation (Serverless Architecture):** Bypasses cross-origin network latency and server locks by embedding the entire quantum matrix simulation engine and a hardware-accelerated neon cyberpunk HTML5 Canvas renderer inside a single decoupled file.
*   **🧠 Parameterized Quantum Circuits (PQC):** Utilizes differentiable programming pipelines with PennyLane to map discrete multi-dimensional board game features onto quantum state registers using optimized **Amplitude Embedding**.
*   **🔗 Cascading Decoherence Systems:** Employs an explicit graph-linked entanglement manager to orchestrate bidirectional cascade collapses across shared wave functions simultaneously.
*   **🛡️ Fully Automated Unit Testing Suite:** Backed by high-coverage unit tests guarding core probability matrices, constraint satisfaction paths, and parameter-shift gradient optimizations.

---

## 🗂️ Project Directory Tree

```text
quantum-intelligence-games/
├── .gitignore
├── LICENSE
├── README.md                 # Technical specification, API references & QML blueprints
├── requirements.txt          # Production environment locked versions
├── pyproject.toml            # PEP 621 compliant modern package metadata configuration
├── index.html                # Autonomous Interactive Web Dashboard & HTML5 Canvas Engine
├── app.py                    # Lightweight Python Microserver Controller
├── src/
│   ├── __init__.py
│   ├── core/                 # Abstract quantum physics logic core
│   │   ├── __init__.py
│   │   ├── state.py          # Superposition management & state vectors
│   │   └── entanglement.py   # Bidirectional entanglement registries & cascades
│   ├── chess/                # Quantum Chess implementation
│   │   ├── __init__.py
│   │   ├── board.py          # Wave function board grids, split/merge moves
│   │   └── engine.py         # Hybrid rule validation & capture resolutions
│   ├── sudoku/               # Quantum Sudoku implementation
│   │   ├── __init__.py
│   │   ├── grid.py           # 3D amplitude tensor modeling
│   │   └── solver.py         # Minimum entropy restriction heuristics (MRV)
│   ├── minesweeper/          # Quantum Minesweeper implementation
│   │   ├── __init__.py
│   │   └── field.py          # Probabilistic layout setups & field observations
│   ├── qml/                  # Hybrid Deep Quantum Learning AI
│   │   ├── __init__.py
│   │   ├── circuits.py       # PennyLane ansatz configuration loops
│   │   ├── models.py         # Nesterov-driven hybrid deep value networks
│   │   └── reinforcement.py  # Online Temporal Difference (TD) learning agents
│   └── utils/                # Tensor and linear algebra optimization math
│       ├── __init__.py
│       └── helpers.py        # L2-normalization pipelines & feature scaling
└── tests/                    # Automated testing pipelines
    ├── test_core.py          # Validates base simulation amplitudes & collapse operations
    ├── test_chess.py         # Validates split path probabilities & capture boundaries
    ├── test_sudoku.py        # Validates 3D amplitude cell entropy distributions
    └── test_qml.py           # Validates parameter-shift autograd optimization flows
```

---

## 🛠️ Quick Installation & Setup

1. **Clone the repository space:**
   ```bash
   git clone https://github.com/yagizyagli/quantum-intelligence-games
   cd quantum-intelligence-games
   ```

2. **Configure your localized virtual pipeline environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install pinned dependencies safely:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Boot up the interactive system ecosystem:**
   *   **Option A (Completely Serverless Showcase - Recommended):** Simply double-click or drag `index.html` directly into any modern web browser to interact with the hardware-accelerated local simulation fields.
   *   **Option B (Hybrid Python API Mode):** Launch the micro-orchestrator layout:
       ```bash
       python app.py
       ```

---

## 🧪 Comprehensive Testing Suite Coverage

Verify the mathematical precision and logic arrays of the quantum cores by running the automated unit testing suites:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📖 SDK Developer API Reference Guide

Developers can easily embed our abstract quantum physics modules into their own custom games.

### Initializing a Superposition Matrix Field
```python
from src.core.state import QuantumState
import numpy as np

# Initialize a standard 8x8 chessboard spatial matrix
quantum_register = QuantumState(size=8)

# Distribute equal 50/50 probability amplitude onto two distinct paths (|ψ| = 1/sqrt(2))
amplitude = 1.0 / np.sqrt(2)
quantum_register.apply_superposition(0, 2, amplitude + 0j)
quantum_register.apply_superposition(0, 4, amplitude + 0j)

# Extract real-time probability density layout maps
probability_distribution = quantum_register.calculate_probabilities()
print(f"Path 1 Probability: {probability_distribution}") # Outputs: 0.5
```

### Forcing a Localized State Vector Collapse
```python
# Observe the system, forcing the complex wave vector to drop into a single classical reality
observed_x, observed_y = quantum_register.measure_and_collapse()
print(f"Wave function collapsed deterministically onto tile: ({observed_x}, {observed_y})")
```

---

## 🧠 QML Mapped Architectural Specifications

The AI subsystem operates as a **Hybrid Classical-Quantum Policy-Value Network (HQNN)** utilizing an online reinforcement learning framework with Temporal Difference (TD) updates.

          [Classical Board State Input]
                        │ (L2 Normalization)
                        ▼
           [Amplitude Embedding Layers]
                        │
                        ▼ (Parametric Weights)
     ┌──────────────────────────────────────┐
     │ Layer 1: Parametric RX, RY, RZ Gates │
     ├──────────────────────────────────────┤
     │ Linear Qubit Entanglement (CNOT)     │
     ├──────────────────────────────────────┤
     │ Layer 2: Parametric RX, RY, RZ Gates │
     └──────────────────────────────────────┘
                        │
                        ▼
         [PauliZ Expectation Values]
                        │ (Range: [-1.0, 1.0])
                        ▼
         [Classical Bias Adjustments]
                        │
                        ▼
       [Optimized Q-Value Move Selection]


1. **Feature Space Constraints:** Game positions are collected as raw flat tensors and pipeline-converted into continuous probabilities using `QuantumTransformationHelpers.normalize_to_quantum_amplitudes` via **L2 Normalization**.
2. **Trainable Circuit Ansatz:** The parameter state registry shapes map angles into RX, RY, RZ gates applied dynamically over all computational threads, bound with a cyclical chain of CNOT gates to capture non-local matrix dependencies.
3. **Analytical Optimization Backprop:** Gradient evaluation handles step updates via **Parameter-Shift Rules** calculated through PennyLane's automated differentiation frameworks, utilizing a **Nesterov Momentum Optimizer** to actively eliminate localized training plateaus.

---

## 🧑‍💻 Author & Developer
*   **Yağız Yağlı:** [@yagizyagli](https://github.com/yagizyagli)
*   **Live Demo:** [@quantum-intelligence-games](https://yagizyagli.github.io/quantum-intelligence-games/)

---

## ⭐ Support and Open Source Contributions

If you find this hybrid quantum computing engineering architecture innovative, educational, or highly compelling for portfolio evaluations, please **give this repository a Star!** Contributions, code audits, or issue tickets regarding additional game engines are welcome via pull request workflows.

