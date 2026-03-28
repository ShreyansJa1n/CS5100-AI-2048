"""Expectimax search agent for 2048.

The agent models the game as a two-player tree:
  - MAX nodes  : agent picks the best direction
  - CHANCE nodes: tile spawns at a random empty cell (P(2)=0.9, P(4)=0.1)

At leaf nodes the board is scored by a hand-crafted heuristic combining
monotonicity, smoothness, empty-cell count, and a corner bonus.

Usage (plug straight into Game2048GUI):
    from agents.expectimax_agent import expectimax_agent
    Game2048GUI(ai_move_provider=expectimax_agent).run()

To adjust search depth (default 3):
    import functools
    Game2048GUI(ai_move_provider=functools.partial(expectimax_agent, depth=4)).run()
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

Direction = str
Board = List[List[int]]

DIRECTIONS: Tuple[Direction, ...] = ("up", "down", "left", "right")

# When a chance node has more empty cells than this, cap the branches evaluated.
# With many empty cells the board is easy and the exact spawn position matters less.
# Lowering this is the single biggest lever for speed:
#   2 cells → 4 chance branches  → depth-3 needs ~1,024 leaf evals/move  (fast)
#   6 cells → 12 chance branches → depth-3 needs ~9,216 leaf evals/move  (slow)
_MAX_CHANCE_CELLS = 2


# ── Pure board simulation (no Game2048 dependency, needed for speed) ──────────

def _merge_line(line: List[int]) -> Tuple[List[int], int, bool]:
    """Compact and merge a single row/column; return (result, reward, did_move)."""
    size = len(line)
    filtered = [x for x in line if x != 0]
    merged: List[int] = []
    reward = 0
    skip = False
    for i in range(len(filtered)):
        if skip:
            skip = False
            continue
        if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
            val = filtered[i] * 2
            merged.append(val)
            reward += val
            skip = True
        else:
            merged.append(filtered[i])
    merged.extend([0] * (size - len(merged)))
    return merged, reward, merged != line


def _apply_move(board: Board, direction: Direction) -> Tuple[Board, int, bool]:
    """Return (new_board, reward, moved) without mutating the input."""
    size = len(board)
    new = [row[:] for row in board]
    total_reward = 0
    moved = False

    if direction == "left":
        for r in range(size):
            line, reward, did = _merge_line(new[r])
            if did:
                new[r] = line
                total_reward += reward
                moved = True

    elif direction == "right":
        for r in range(size):
            line, reward, did = _merge_line(list(reversed(new[r])))
            if did:
                new[r] = list(reversed(line))
                total_reward += reward
                moved = True

    elif direction == "up":
        for c in range(size):
            col = [new[r][c] for r in range(size)]
            line, reward, did = _merge_line(col)
            if did:
                for r in range(size):
                    new[r][c] = line[r]
                total_reward += reward
                moved = True

    elif direction == "down":
        for c in range(size):
            col = [new[r][c] for r in reversed(range(size))]
            line, reward, did = _merge_line(col)
            if did:
                for r, val in zip(reversed(range(size)), line):
                    new[r][c] = val
                total_reward += reward
                moved = True

    return new, total_reward, moved


def _empty_cells(board: Board) -> List[Tuple[int, int]]:
    return [(r, c) for r, row in enumerate(board) for c, v in enumerate(row) if v == 0]


def _can_move(board: Board) -> bool:
    size = len(board)
    for r in range(size):
        for c in range(size):
            if board[r][c] == 0:
                return True
            if r + 1 < size and board[r][c] == board[r + 1][c]:
                return True
            if c + 1 < size and board[r][c] == board[r][c + 1]:
                return True
    return False


# ── Heuristic ──────────────────────────────────────────────────────────────────

def _heuristic(board: Board) -> float:
    """Score a board state without lookahead."""
    size = len(board)

    # Empty cell bonus — more free cells = more flexibility
    empty = sum(1 for r in range(size) for c in range(size) if board[r][c] == 0)
    empty_score = math.log(empty + 1) * 2.7

    mono = _monotonicity(board)
    smooth = _smoothness(board)

    # Reward keeping the max tile in any corner
    max_val = max(board[r][c] for r in range(size) for c in range(size))
    corners = (board[0][0], board[0][size - 1], board[size - 1][0], board[size - 1][size - 1])
    corner_bonus = math.log2(max_val) if max_val > 0 and max_val in corners else 0.0

    return empty_score + mono + smooth + corner_bonus


def _monotonicity(board: Board) -> float:
    """Reward boards where tile values decrease monotonically along rows/columns."""
    size = len(board)
    totals = [0.0, 0.0, 0.0, 0.0]  # up, down, left, right

    for r in range(size):
        for c in range(size - 1):
            a = math.log2(board[r][c]) if board[r][c] > 0 else 0.0
            b = math.log2(board[r][c + 1]) if board[r][c + 1] > 0 else 0.0
            if a > b:
                totals[2] += b - a
            elif b > a:
                totals[3] += a - b

    for c in range(size):
        for r in range(size - 1):
            a = math.log2(board[r][c]) if board[r][c] > 0 else 0.0
            b = math.log2(board[r + 1][c]) if board[r + 1][c] > 0 else 0.0
            if a > b:
                totals[0] += b - a
            elif b > a:
                totals[1] += a - b

    return max(totals[0], totals[1]) + max(totals[2], totals[3])


def _smoothness(board: Board) -> float:
    """Penalise large value differences between adjacent tiles."""
    size = len(board)
    smooth = 0.0
    for r in range(size):
        for c in range(size):
            if board[r][c] != 0:
                v = math.log2(board[r][c])
                if c + 1 < size and board[r][c + 1] != 0:
                    smooth -= abs(v - math.log2(board[r][c + 1]))
                if r + 1 < size and board[r + 1][c] != 0:
                    smooth -= abs(v - math.log2(board[r + 1][c]))
    return smooth


# ── Expectimax search ──────────────────────────────────────────────────────────

def _chance_node(board: Board, depth: int) -> float:
    """Expected value over all possible tile spawns."""
    empties = _empty_cells(board)
    if not empties:
        # Board is full after this move — go straight back to a MAX node
        return _max_node(board, depth)

    # Cap branching when the board is open (spawn position matters less then)
    cells = empties if len(empties) <= _MAX_CHANCE_CELLS else empties[:_MAX_CHANCE_CELLS]

    total = 0.0
    for (r, c) in cells:
        for tile, prob in ((2, 0.9), (4, 0.1)):
            child = [row[:] for row in board]
            child[r][c] = tile
            total += prob * _max_node(child, depth)
    return total / len(cells)


def _max_node(board: Board, depth: int) -> float:
    """Pick the best move; return its expected value."""
    if depth == 0 or not _can_move(board):
        return _heuristic(board)

    best = float("-inf")
    for direction in DIRECTIONS:
        new_board, reward, moved = _apply_move(board, direction)
        if not moved:
            continue
        score = reward + _chance_node(new_board, depth - 1)
        if score > best:
            best = score

    return best if best != float("-inf") else _heuristic(board)


# ── Public agent function ──────────────────────────────────────────────────────

def expectimax_agent(state: dict, depth: int = 3) -> Optional[Direction]:
    """Return the best direction for the current board state.

    Args:
        state:  dict returned by Game2048.get_state()
        depth:  how many MAX-node levels to search (default 3).
                Depth 4+ is noticeably slower; depth 2 is very fast but weaker.

    Returns:
        A direction string ("up"/"down"/"left"/"right"), or None if no move exists.
    """
    board = state["board"]
    best_move: Optional[Direction] = None
    best_score = float("-inf")

    for direction in DIRECTIONS:
        new_board, reward, moved = _apply_move(board, direction)
        if not moved:
            continue
        score = reward + _chance_node(new_board, depth - 1)
        if score > best_score:
            best_score = score
            best_move = direction

    return best_move
