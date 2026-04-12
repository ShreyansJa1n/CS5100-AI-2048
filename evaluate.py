"""Headless agent evaluator.

Runs an agent for N games without any GUI and prints a statistics summary.

Usage:
    python evaluate.py                                     # expectimax, 100 games, depth 3
    python evaluate.py --agent expectimax --games 50 --depth 4
    python evaluate.py --agent random --games 200
    python evaluate.py --verbose --games 1                 # step-by-step move log
    python evaluate.py --gui                               # watch the agent play live
    python evaluate.py --gui --gui-delay 200               # slower animation (ms)
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import time
from collections import Counter
from datetime import datetime
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from game2048 import Game2048
from agents.expectimax_agent import expectimax_agent


# ── Agent registry ─────────────────────────────────────────────────────────────

def _random_agent(state: dict) -> Optional[str]:
    """Picks a legal move at random — useful as a baseline."""
    import random
    moves = state["available_moves"]
    return random.choice(moves) if moves else None


def _build_agent(name: str, depth: int, chance_cells: int, weights: str = None) -> Callable:
    if name == "expectimax":
        import agents.expectimax_agent as em
        em._MAX_CHANCE_CELLS = chance_cells
        return functools.partial(expectimax_agent, depth=depth)
    if name == "random":
        return _random_agent
    if name == "td":
        from agents.td_ntuple_agent import td_ntuple_agent
        if weights:
            return functools.partial(td_ntuple_agent, weights_path=weights)
        return td_ntuple_agent
    raise ValueError(f"Unknown agent: {name}")


AGENTS = ["expectimax", "random", "td"]

ARROW = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


# ── Board pretty-printer ───────────────────────────────────────────────────────

def _fmt_board(board: List[List[int]]) -> str:
    """Return a coloured, aligned board string."""
    # ANSI colours keyed by tile value
    COLOURS = {
        0:    "\033[90m",     # dark grey
        2:    "\033[97m",     # white
        4:    "\033[93m",     # yellow
        8:    "\033[33m",     # dark yellow
        16:   "\033[91m",     # light red
        32:   "\033[31m",     # red
        64:   "\033[95m",     # magenta
        128:  "\033[94m",     # blue
        256:  "\033[96m",     # cyan
        512:  "\033[92m",     # green
        1024: "\033[32m",     # dark green
        2048: "\033[1;93m",   # bold bright yellow
    }
    RESET = "\033[0m"
    SEP = "┼" + ("─" * 6 + "┼") * 4
    lines = ["┼" + "──────┼" * 4]
    for row in board:
        cells = []
        for v in row:
            colour = COLOURS.get(v, "\033[1;91m")
            text = str(v) if v else "·"
            cells.append(f"{colour}{text:^6}{RESET}")
        lines.append("│" + "│".join(cells) + "│")
        lines.append(SEP)
    return "\n".join(lines)


# ── Single-game runner ─────────────────────────────────────────────────────────

def run_game(
    agent_fn: Callable,
    seed: Optional[int] = None,
    verbose: bool = False,
    game_number: int = 1,
) -> Dict:
    """Play one complete game headlessly and return result stats."""
    game = Game2048(seed=seed)

    if verbose:
        print(f"\n{'═' * 43}")
        print(f"  GAME {game_number}  —  starting position")
        print(f"{'═' * 43}")

    while True:
        state = game.get_state()

        if verbose:
            print(f"\n  Move {state['moves_made'] + 1:>4}  │  Score: {state['score']:>8,}")
            print(_fmt_board(state["board"]))

        if state["game_over"]:
            if verbose:
                print(f"\n  ✗ Game over  │  Final score: {state['score']:,}")
            break

        move = agent_fn(state)
        if move is None:
            break

        if verbose:
            arrow = ARROW.get(move, move)
            print(f"\n  Agent chose: {arrow}  {move.upper()}")
            print(f"  {'─' * 39}")

        game.move(move)

    state = game.get_state()
    max_tile = max(cell for row in state["board"] for cell in row)

    if verbose and state["won"]:
        print(f"\n  ★ Reached 2048!  Score: {state['score']:,}")

    return {
        "score":    state["score"],
        "moves":    state["moves_made"],
        "max_tile": max_tile,
        "won":      state["won"],
    }


# ── GUI runner ─────────────────────────────────────────────────────────────────

def run_gui(agent_fn: Callable, delay_ms: int) -> None:
    """Launch the Tkinter GUI with the agent driving the game."""
    from game2048 import Game2048GUI
    print("\nLaunching GUI — close the window to exit.")
    print("Arrow keys still work; press R or click New Game to reset.\n")
    Game2048GUI(ai_move_provider=agent_fn, ai_delay_ms=delay_ms).run()


# ── Batch evaluation ───────────────────────────────────────────────────────────

def evaluate(
    agent_fn: Callable,
    n_games: int,
    agent_label: str,
    verbose: bool = False,
) -> None:
    results: List[Dict] = []
    start = time.time()

    print(f"\nAgent : {agent_label}")
    print(f"Games : {n_games}")
    print("─" * 42)

    if verbose:
        # Verbose mode: full per-move output, no tqdm bar (they'd clash)
        for i in range(n_games):
            result = run_game(agent_fn, verbose=True, game_number=i + 1)
            results.append(result)
    else:
        with tqdm(
            total=n_games,
            unit="game",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        ) as pbar:
            for i in range(n_games):
                result = run_game(agent_fn, game_number=i + 1)
                results.append(result)

                avg_score = statistics.mean(r["score"] for r in results)
                wins_so_far = sum(r["won"] for r in results)
                pbar.set_postfix(
                    avg_score=f"{avg_score:,.0f}",
                    best_tile=max(r["max_tile"] for r in results),
                    wins=f"{wins_so_far}/{len(results)}",
                )
                pbar.update(1)

    elapsed = time.time() - start
    _print_summary(results, n_games, elapsed)
    _dump_results(results, n_games, elapsed, agent_label)


def _build_summary_lines(
    results: List[Dict], n_games: int, elapsed: float,
) -> List[str]:
    """Build the summary block as a list of plain-text lines (no ANSI)."""
    scores    = [r["score"]    for r in results]
    moves     = [r["moves"]    for r in results]
    tile_dist = Counter(r["max_tile"] for r in results)
    wins      = sum(r["won"] for r in results)

    lines: List[str] = []
    lines.append("")
    lines.append("── Summary " + "─" * 31)
    lines.append(f"  Total time   : {elapsed:.1f}s  ({elapsed / n_games:.2f}s/game)")
    lines.append(f"  Avg score    : {statistics.mean(scores):>10,.0f}")
    lines.append(f"  Median score : {statistics.median(scores):>10,.0f}")
    lines.append(f"  Std dev      : {statistics.stdev(scores) if n_games > 1 else 0:>10,.0f}")
    lines.append(f"  Best score   : {max(scores):>10,}")
    lines.append(f"  Worst score  : {min(scores):>10,}")
    lines.append(f"  Avg moves    : {statistics.mean(moves):>10.0f}")
    lines.append(f"  Median moves : {statistics.median(moves):>10.0f}")
    lines.append(f"  Win rate     : {wins / n_games * 100:>9.1f}%  ({wins}/{n_games} reached 2048)")
    lines.append("")
    lines.append("  Max tile distribution:")
    for tile in sorted(tile_dist.keys(), reverse=True):
        pct = tile_dist[tile] / n_games * 100
        bar = "█" * int(pct / 2)
        lines.append(f"    {tile:>5} : {pct:5.1f}%  ({tile_dist[tile]:>4} games)  {bar}")
    lines.append("─" * 42)
    return lines


def _print_summary(results: List[Dict], n_games: int, elapsed: float) -> None:
    for line in _build_summary_lines(results, n_games, elapsed):
        print(line)
    print()


def _dump_results(
    results: List[Dict],
    n_games: int,
    elapsed: float,
    agent_label: str,
) -> None:
    """Write a detailed results file into results/ with a datetime-based name."""
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_short = agent_label.split()[0]  # e.g. "expectimax", "td", "random"
    filename = f"results/{ts}_{agent_short}_{n_games}games.txt"

    scores    = [r["score"]    for r in results]
    moves     = [r["moves"]    for r in results]
    max_tiles = [r["max_tile"] for r in results]

    with open(filename, "w") as f:
        f.write(f"2048 Agent Evaluation Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Agent      : {agent_label}\n")
        f.write(f"Games      : {n_games}\n")
        f.write(f"\n")

        # Summary block
        for line in _build_summary_lines(results, n_games, elapsed):
            f.write(line + "\n")

        # Score percentiles
        sorted_scores = sorted(scores)
        f.write(f"\n  Score percentiles:\n")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = min(int(n_games * p / 100), n_games - 1)
            f.write(f"    p{p:<2} : {sorted_scores[idx]:>10,}\n")

        # Per-game results table
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Per-Game Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"{'Game':>6}  {'Score':>10}  {'Moves':>6}  {'Max Tile':>9}  {'Won':>4}\n")
        f.write(f"{'-'*6}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*4}\n")
        for i, r in enumerate(results, 1):
            won_str = "yes" if r["won"] else ""
            f.write(f"{i:>6}  {r['score']:>10,}  {r['moves']:>6}  {r['max_tile']:>9,}  {won_str:>4}\n")

    print(f"Results saved to {filename}")
    return filename


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 2048 AI agent.")
    parser.add_argument(
        "--agent", choices=AGENTS, default="expectimax",
        help="Agent to evaluate (default: expectimax)",
    )
    parser.add_argument(
        "--games", type=int, default=100,
        help="Number of games to run (default: 100)",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Search depth for expectimax (default: 3, ignored for random)",
    )
    parser.add_argument(
        "--chance-cells", type=int, default=2,
        help="Max empty cells expanded per chance node (default: 2). "
             "Higher = more accurate but exponentially slower.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print the board and chosen move at every step. "
             "Best used with --games 1.",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to .npz weights file for the TD agent (default: weights/td_ntuple.npz).",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Watch the agent play in the Tkinter GUI instead of running headlessly.",
    )
    parser.add_argument(
        "--gui-delay", type=int, default=100,
        help="Milliseconds between moves in GUI mode (default: 100).",
    )
    args = parser.parse_args()

    label = f"{args.agent}"
    if args.agent == "expectimax":
        label += f" (depth={args.depth}, chance_cells={args.chance_cells})"
    elif args.agent == "td":
        label += f" (weights={args.weights or 'default'})"
    agent_fn = _build_agent(args.agent, args.depth, args.chance_cells, weights=args.weights)

    if args.gui:
        run_gui(agent_fn, delay_ms=args.gui_delay)
    else:
        evaluate(agent_fn, args.games, label, verbose=args.verbose)


if __name__ == "__main__":
    main()
