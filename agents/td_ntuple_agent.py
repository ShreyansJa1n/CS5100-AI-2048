"""TD-learning agent with n-tuple networks for 2048.

Implements the afterstate TD(0) learning framework from:
    Jaskowski (2018), "Mastering 2048 with Delayed Temporal Coherence Learning,
    Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping."

Architecture:
    - 4 six-tuples (the 4x2-3x3 configuration) with symmetric sampling (8 symmetries)
    - Tile encoding: empty=0, v -> log2(v)  (c=16 possible values)
    - LUT size per tuple: 16^6 = 16,777,216 entries (float32)
    - Total parameters: 4 x 16^6 = 67,108,864

Usage (plug into Game2048GUI):
    from agents.td_ntuple_agent import td_ntuple_agent
    Game2048GUI(ai_move_provider=td_ntuple_agent).run()
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from agents.expectimax_agent import _apply_move

Direction = str
Board = List[List[int]]

DIRECTIONS: Tuple[Direction, ...] = ("up", "down", "left", "right")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_VALUES = 16          # tile indices 0..15  (empty, 2, 4, ..., 32768)
TUPLE_SIZE = 6
LUT_SIZE = NUM_VALUES ** TUPLE_SIZE  # 16^6 = 16,777,216
NUM_TUPLES = 4
NUM_SYMMETRIES = 8
NUM_ACTIVE = NUM_TUPLES * NUM_SYMMETRIES  # 32

# ---------------------------------------------------------------------------
# Precomputed position tables  (the key optimization)
#
# Instead of transforming the entire board 8 times per evaluate/update call,
# we precompute the 8 transformed position-lists for each n-tuple at import
# time.  Evaluation reads the *original* encoded board at the transformed
# positions — no board copies needed.
# ---------------------------------------------------------------------------

# The 4 base six-tuple shapes (row, col) on a 4x4 board.
_BASE_TUPLES: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],  # horizontal 4+2
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # vertical 2x3
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],  # 2x3 block
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],  # diagonal L
]


def _transform_pos(r: int, c: int, sym_id: int) -> Tuple[int, int]:
    """Apply one of the 8 symmetry transforms to a (row, col) position.

    sym_id  transform
    ------  ---------
    0       identity
    1       rotate 90 CW
    2       rotate 180
    3       rotate 270 CW
    4       reflect horizontal
    5       reflect + rotate 90
    6       reflect + rotate 180
    7       reflect + rotate 270
    """
    if sym_id == 0:
        return (r, c)
    if sym_id == 1:
        return (c, 3 - r)
    if sym_id == 2:
        return (3 - r, 3 - c)
    if sym_id == 3:
        return (3 - c, r)
    if sym_id == 4:
        return (r, 3 - c)
    if sym_id == 5:
        return (3 - c, 3 - r)
    if sym_id == 6:
        return (3 - r, c)
    # sym_id == 7
    return (c, r)


# ALL_POSITIONS[t][s] = list of 6 (row, col) for tuple t, symmetry s
ALL_POSITIONS: List[List[List[Tuple[int, int]]]] = []
for _base in _BASE_TUPLES:
    _sym_list = []
    for _sid in range(NUM_SYMMETRIES):
        _sym_list.append([_transform_pos(r, c, _sid) for (r, c) in _base])
    ALL_POSITIONS.append(_sym_list)

# Pre-compute 16^j powers for index calculation
_POWERS = tuple(NUM_VALUES ** j for j in range(TUPLE_SIZE))

# Precomputed log2 lookup:  tile_value -> encoded index.
# Supports tiles up to 2^15 = 32768.  Index 0 = empty.
_LOG2 = [0] * (32769)
for _p in range(1, 16):
    _LOG2[1 << _p] = _p


# ---------------------------------------------------------------------------
# Fast helpers  (inlined as much as possible for the hot path)
# ---------------------------------------------------------------------------

def _encode_flat(board: Board) -> List[int]:
    """Encode a 4x4 board into a flat length-16 list of tile indices."""
    log2 = _LOG2
    return [log2[board[0][0]], log2[board[0][1]], log2[board[0][2]], log2[board[0][3]],
            log2[board[1][0]], log2[board[1][1]], log2[board[1][2]], log2[board[1][3]],
            log2[board[2][0]], log2[board[2][1]], log2[board[2][2]], log2[board[2][3]],
            log2[board[3][0]], log2[board[3][1]], log2[board[3][2]], log2[board[3][3]]]


def _index_from_flat(flat: List[int], positions: List[Tuple[int, int]]) -> int:
    """Compute LUT index from a flat-encoded board and a position list."""
    # positions are (row, col); flat index = row*4 + col
    p = _POWERS
    return (flat[positions[0][0] * 4 + positions[0][1]] * p[0]
          + flat[positions[1][0] * 4 + positions[1][1]] * p[1]
          + flat[positions[2][0] * 4 + positions[2][1]] * p[2]
          + flat[positions[3][0] * 4 + positions[3][1]] * p[3]
          + flat[positions[4][0] * 4 + positions[4][1]] * p[4]
          + flat[positions[5][0] * 4 + positions[5][1]] * p[5])


# ---------------------------------------------------------------------------
# NTupleNetwork class
# ---------------------------------------------------------------------------

class NTupleNetwork:
    """N-tuple network value-function approximator for 2048 afterstates."""

    def __init__(self) -> None:
        self.luts: List[np.ndarray] = [
            np.zeros(LUT_SIZE, dtype=np.float32) for _ in range(NUM_TUPLES)
        ]
        # TC(0) accumulators — lazily allocated
        self._E: Optional[List[np.ndarray]] = None
        self._A: Optional[List[np.ndarray]] = None

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, board: Board) -> float:
        """Return V(board) = sum over all tuples and all 8 symmetries."""
        flat = _encode_flat(board)
        total = 0.0
        for t_idx in range(NUM_TUPLES):
            lut = self.luts[t_idx]
            sym_positions = ALL_POSITIONS[t_idx]
            for positions in sym_positions:
                total += float(lut[_index_from_flat(flat, positions)])
        return total

    # ── Updates ───────────────────────────────────────────────────────────

    def update(self, board: Board, delta: float, alpha: float) -> None:
        """TD(0) weight update for all LUT entries touched by *board*."""
        step = alpha * delta / NUM_ACTIVE
        flat = _encode_flat(board)
        for t_idx in range(NUM_TUPLES):
            lut = self.luts[t_idx]
            for positions in ALL_POSITIONS[t_idx]:
                lut[_index_from_flat(flat, positions)] += step

    def _ensure_tc_arrays(self) -> None:
        if self._E is None:
            self._E = [np.zeros(LUT_SIZE, dtype=np.float32) for _ in range(NUM_TUPLES)]
            self._A = [np.zeros(LUT_SIZE, dtype=np.float32) for _ in range(NUM_TUPLES)]

    def update_tc(self, board: Board, delta: float) -> None:
        """TC(0) update with per-weight adaptive learning rates.

        alpha_i = |E_i| / A_i  (or 1.0 when A_i == 0).
        """
        self._ensure_tc_arrays()
        flat = _encode_flat(board)
        abs_delta = abs(delta)
        inv_m = 1.0 / NUM_ACTIVE
        for t_idx in range(NUM_TUPLES):
            lut = self.luts[t_idx]
            E = self._E[t_idx]
            A = self._A[t_idx]
            for positions in ALL_POSITIONS[t_idx]:
                idx = _index_from_flat(flat, positions)
                a_val = A[idx]
                alpha_i = abs(E[idx]) / a_val if a_val != 0.0 else 1.0
                lut[idx] += alpha_i * delta * inv_m
                E[idx] += delta
                A[idx] += abs_delta

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save weights (and optional TC accumulators) to a .npz file."""
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        data = {f"lut{i}": self.luts[i] for i in range(NUM_TUPLES)}
        if self._E is not None:
            for i in range(NUM_TUPLES):
                data[f"E{i}"] = self._E[i]
                data[f"A{i}"] = self._A[i]
        np.savez_compressed(filepath, **data)

    def load(self, filepath: str) -> None:
        """Load weights (and optional TC accumulators) from a .npz file."""
        data = np.load(filepath)
        self.luts = [data[f"lut{i}"].astype(np.float32) for i in range(NUM_TUPLES)]
        if "E0" in data:
            self._E = [data[f"E{i}"].astype(np.float32) for i in range(NUM_TUPLES)]
            self._A = [data[f"A{i}"].astype(np.float32) for i in range(NUM_TUPLES)]


# ---------------------------------------------------------------------------
# Agent function (runtime — loads trained weights, picks best move)
# ---------------------------------------------------------------------------

_network: Optional[NTupleNetwork] = None
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "td_ntuple.npz")


def _get_network(weights_path: Optional[str] = None) -> NTupleNetwork:
    """Lazy-load the trained network on first call."""
    global _network
    if _network is None:
        path = weights_path or _DEFAULT_WEIGHTS
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TD n-tuple weights not found at {path}. "
                f"Run 'python train_td.py' first to train the agent."
            )
        _network = NTupleNetwork()
        _network.load(path)
    return _network


def td_ntuple_agent(state: dict, weights_path: Optional[str] = None) -> Optional[Direction]:
    """Return the best direction for the current board using the trained n-tuple network.

    Args:
        state:        dict returned by Game2048.get_state()
        weights_path: optional path to .npz weights file

    Returns:
        A direction string ("up"/"down"/"left"/"right"), or None if no move.
    """
    board: Board = state["board"]
    network = _get_network(weights_path)

    best_move: Optional[Direction] = None
    best_score = float("-inf")

    for direction in DIRECTIONS:
        afterstate, reward, moved = _apply_move(board, direction)
        if not moved:
            continue
        score = reward + network.evaluate(afterstate)
        if score > best_score:
            best_score = score
            best_move = direction

    return best_move
