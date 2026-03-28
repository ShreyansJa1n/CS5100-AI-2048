# CS 5100 — AI Agent for 2048: Technical Progress Report

**Team:** Rida Sameer & Shreyans Jain
**Course:** CS 5100 Foundations of Artificial Intelligence — Northeastern University
**Date:** March 2026

---

## 1. Project Overview

The goal of this project is to build and compare two AI agents that autonomously play the game 2048:

1. **Expectimax Search Agent** — a classical search-based agent using hand-crafted heuristics (baseline)
2. **TD Learning with N-Tuple Networks** — a reinforcement learning agent based on Jaśkowski (2018) (primary contribution)

The final deliverable compares both agents on metrics like average score, win rate, and max tile distribution.

---

## 2. The Game Engine (`game2048.py`)

### What it does

A fully self-contained 2048 game engine with two classes:

- `Game2048` — the headless game logic
- `Game2048GUI` — a Tkinter visualization wrapper

### The API

```python
game = Game2048(size=4, seed=42)
state = game.get_state()           # returns a state dict
moved, reward = game.move("left")  # executes a move
game.can_move()                    # False = game over
game.reset()                       # restart
```

The `get_state()` dict contains:

```python
{
    "board":           [[0, 2, 4, 0], ...],  # 4×4 grid, 0 = empty
    "score":           1234,
    "moves_made":      42,
    "game_over":       False,
    "won":             False,
    "available_moves": ["up", "left", "right"]
}
```

### The AI hook

`Game2048GUI` accepts an `ai_move_provider` callable:

```python
def my_agent(state: dict) -> str:   # returns "up"/"down"/"left"/"right"
    ...

Game2048GUI(ai_move_provider=my_agent, ai_delay_ms=100).run()
```

Any agent — expectimax, TD, random — only needs to implement this one function signature. This decouples the agent from the GUI completely.

### Why it's modeled this way

2048 is a **fully observable, nondiscounted discrete Markov Decision Process (MDP)**:

| MDP component | 2048 equivalent |
|---|---|
| State | Board configuration (4×4 grid of tile values) |
| Action | {up, down, left, right} |
| Transition | Stochastic — a 2 (90%) or 4 (10%) tile spawns at a random empty cell |
| Reward | Sum of merged tile values from the move |
| Terminal | No legal moves remain |

The state space is approximately **4.7×10²¹** states — far too large for tabular methods. This is why both agents use function approximation or search rather than storing values for every state.

---

## 3. The Expectimax Agent (`agents/expectimax_agent.py`)

### What it is

A **tree search agent** that looks ahead several moves before deciding. No training, no learned weights — purely algorithmic. Serves as the baseline to compare against the RL agent.

### How the tree works

The game alternates between two kinds of events, which map to two kinds of tree nodes:

```
MAX node   — agent picks the best direction (up/down/left/right)
CHANCE node — tile spawns at a random empty cell (P(2)=0.9, P(4)=0.1)
```

At depth 3, the search tree looks like this:

```
Current board
    │
    ├── try UP    → [CHANCE: avg over tile spawns] → try UP/DOWN/LEFT/RIGHT → ...
    ├── try DOWN  → [CHANCE: avg over tile spawns] → try UP/DOWN/LEFT/RIGHT → ...
    ├── try LEFT  → [CHANCE: avg over tile spawns] → ...
    └── try RIGHT → [CHANCE: avg over tile spawns] → ...
                                    │
                             evaluate leaf board
                             with heuristic function
```

The agent picks the direction with the highest expected score at the root.

### The heuristic function

At leaf nodes, the board is scored by a weighted combination of four signals:

#### 1. Empty cell bonus
```python
empty_score = log(empty_count + 1) * 2.7
```
More free cells = more flexibility for future moves. The log weighting prevents early-game boards (14 empty cells) from dominating the score.

#### 2. Monotonicity
Rewards boards where tile values decrease monotonically along rows and columns — this encourages the "snake" or "corner" strategy where the highest tile sits in a corner and values decrease outward.

```
Good (monotone)     Bad (chaotic)
512  256  128  64   512   4   64  128
 32   16    8   4    32 256    8   16
  2    .    .   .     2   .  128    .
  .    .    .   .     .   .    .    .
```

Computed in both horizontal and vertical directions; the best monotone direction is taken.

#### 3. Smoothness
Penalises large differences between adjacent tiles (using log₂ values). Smooth boards are easier to merge.

```python
smooth -= abs(log2(tile) - log2(neighbor))  # for each adjacent pair
```

#### 4. Corner bonus
Awards `log₂(max_tile)` points if the highest-value tile is in any corner. Keeping the max tile cornered is a well-known strategy in 2048 as it frees up merging lanes.

### The chance node and branching factor

**The problem:** With 16 cells on the board, up to 16 empty cells × 2 tile values = 32 branches per chance node. At depth 3, that's 4 × 32 × 4 × 32 × 4 = **65,536 leaf evaluations per move** in the worst case — too slow for pure Python.

**The solution — `_MAX_CHANCE_CELLS`:** Cap the number of empty cells the agent considers in each chance node. Set to 2 by default:

```
depth=3 with 2 chance cells: 4 × 4 × 4 × 4 × 4 = 1,024 leaf evaluations/move
depth=3 with 6 chance cells: 4 × 12 × 4 × 12 × 4 = 9,216 leaf evaluations/move
```

This is a known approximation — the agent misses some spawn positions but runs ~9x faster. Since the agent re-evaluates the real board state every move, the error doesn't accumulate.

**Current limitation:** The 2 cells are selected in top-left to bottom-right reading order, which biases the agent to ignore the bottom-right quadrant. This is acceptable for a baseline.

### Pure board simulation

The agent does **not** use `Game2048` internally for the search. Instead it has its own pure-function board simulation (`_apply_move`, `_merge_line`). This is important because:

- `Game2048.move()` is stateful — it mutates the board and spawns a real tile
- The search needs to simulate thousands of hypothetical moves without side effects
- Pure functions are faster (no object overhead) and easier to reason about

### What depth means concretely

Depth counts how many **agent moves** are simulated before scoring. Between each pair of agent moves sits one chance node (tile spawn).

```
depth=1 — agent moves once, scores immediately (no lookahead)
  MY MOVE → score board

depth=2 — agent moves, tile spawns, agent moves again, then scores
  MY MOVE → tile spawns → MY MOVE → score board

depth=3 — agent moves, tile spawns, agent moves, tile spawns, agent moves, scores
  MY MOVE → tile spawns → MY MOVE → tile spawns → MY MOVE → score board
```

Think of it like a chess player who can only think 1, 2, or 3 moves ahead.

### Depth vs. speed tradeoff

With `--chance-cells 2`, each chance node branches into `2 cells × 2 tiles = 4` possibilities.
The total leaf evaluations per move decision multiply as:

```
depth=1:  4                               =       4  evals/move
depth=2:  4 × 4 × 4                       =      64  evals/move
depth=3:  4 × 4 × 4 × 4 × 4             =   1,024  evals/move
depth=4:  4 × 4 × 4 × 4 × 4 × 4 × 4   =  16,384  evals/move
```

| `--depth` | Evals/move (chance-cells=2) | Approx time/game |
|---|---|---|
| 1 | 4 | < 1s |
| 2 | 64 | ~5s |
| 3 | 1,024 | ~60-90s |
| 4 | 16,384 | ~15min+ |

Default is depth 3 with `--chance-cells 2`.

---

## 4. The Evaluation Harness (`evaluate.py`)

### What it does

Runs any agent for N games headlessly (no GUI) and reports statistics. Also supports a verbose terminal mode and a live GUI mode.

### CLI flags

| Flag | Description |
|---|---|
| `--agent` | `expectimax` or `random` |
| `--games` | Number of games to run |
| `--depth` | Expectimax search depth |
| `--chance-cells` | Chance node branching cap |
| `--verbose` | Print board + chosen move at every step |
| `--gui` | Watch the agent play in the Tkinter window |
| `--gui-delay` | Milliseconds between moves in GUI mode |

### Output

```
Agent : expectimax (depth=3, chance_cells=2)
Games : 100
──────────────────────────────────────────
100%|████████| 100/100 [12:34<00:00, avg_score=8,421, best_tile=512, wins=2/100]

── Summary ─────────────────────────────────────
  Total time   : 754.2s  (7.54s/game)
  Avg score    :      8,421
  Median score :      7,840
  Std dev      :      2,103
  Best score   :     14,220
  Avg moves    :        412
  Win rate     :      2.0%  (2/100 reached 2048)

  Max tile distribution:
   1024 :  12.0%  ██████
    512 :  58.0%  █████████████████████████████
    256 :  28.0%  ██████████████
    128 :   2.0%  █
```

### Verbose mode (`--verbose --games 1`)

Prints the full board state and the agent's chosen direction at every single move. Useful for debugging and understanding why the agent makes particular decisions.

### The random agent baseline

A random agent that picks uniformly from available legal moves. It serves as the floor — any meaningful AI should significantly outperform it.

---

## 5. What's Next — TD Learning with N-Tuple Networks

This is the primary research contribution, based on:

> Jaśkowski, W. (2018). "Mastering 2048 with delayed temporal coherence learning, multistage weight promotion, redundant encoding, and carousel shaping." *IEEE Transactions on Games*, 10(1), 3–14.

### Why expectimax isn't enough

Expectimax has fundamental limitations:

- The heuristic is **hand-crafted** — it encodes our assumptions about what a good board looks like, which may be wrong or incomplete
- It has **no memory** — it doesn't learn from past games
- It is **computationally expensive** at runtime — every move requires a full tree search

### The TD + N-Tuple approach

Instead of designing a heuristic, the agent **learns** what good board states look like by playing thousands of games and updating a value function.

#### Afterstate learning

A key insight from the paper: rather than learning `V(state)` (the value of a board including the randomly-placed tile), learn `V(afterstate)` — the board immediately **after the agent's move but before the random tile appears**.

```
Standard:   s → agent moves → tile spawns → s'    learn V(s)
Afterstate: s → agent moves → afterstate  → tile spawns → s'    learn V(afterstate)
```

This is better because:
- The afterstate is **deterministic** given the current state and action
- It removes one source of variance from the learning target
- The agent can directly compare afterstates to pick the best move

#### N-Tuple networks

A **value function approximator** — a way to estimate how good a board position is without a neural network.

An n-tuple is a pattern of n cells on the board. For each pattern, there's a lookup table indexed by the tile values at those cells. The total board value is the sum across all tuples:

```
Tuple 1: cells (0,0), (0,1), (0,2), (0,3)  → lookup table → value₁
Tuple 2: cells (1,0), (1,1), (1,2), (1,3)  → lookup table → value₂
Tuple 3: cells (0,0), (1,0), (2,0), (3,0)  → lookup table → value₃
...
V(board) = value₁ + value₂ + value₃ + ...
```

Each tuple captures local structural features of the board. By using multiple tuples covering different regions and orientations, the network captures global board quality.

#### TD(0) weight update

After each move, the weights are updated with:

```
V(afterstate) += α × (reward + V(next_afterstate) − V(afterstate))
```

- `α` — learning rate
- `reward` — tile merge reward from the move
- `V(next_afterstate)` — estimated value of the resulting position
- `V(afterstate) - V(afterstate)` — the TD error (how wrong our estimate was)

Over thousands of games, the weights converge to accurately reflect the true long-term value of board positions.

### Files to build

| File | Purpose |
|---|---|
| `agents/td_ntuple_agent.py` | N-tuple network definition + inference-time agent |
| `train_td.py` | Training loop — plays N games, updates weights, saves to disk |
| `weights/ntuple_weights.npz` | Saved weights (numpy format, not pickle) |

### The production pipeline

```
train_td.py  (run once, takes hours)
     │
     ▼
weights/ntuple_weights.npz   (saved to disk)
     │
     ▼
agents/td_ntuple_agent.py    (loads weights once at import)
     │
     ▼
evaluate.py / Game2048GUI    (calls agent(state) each move — instant lookup)
```

---

## 6. Comparison Plan

Once both agents are built, evaluate both under the same conditions:

| Metric | Description |
|---|---|
| Average score | Mean final score across 100+ games |
| Win rate | % of games reaching the 2048 tile |
| Max tile distribution | % reaching 512 / 1024 / 2048 |
| Moves per game | Proxy for how long the agent survives |
| Time per move | Runtime cost |

Also test expectimax at multiple depths (1, 2, 3) to see how search depth affects performance — Jaśkowski (2018) found that certain RL configurations overfit to shallow search and degrade at deeper levels.
