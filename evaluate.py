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
import statistics
import time
from collections import Counter
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


def _build_agent(name: str, depth: int, chance_cells: int) -> Callable:
    if name == "expectimax":
        import agents.expectimax_agent as em
        em._MAX_CHANCE_CELLS = chance_cells
        return functools.partial(expectimax_agent, depth=depth)
    if name == "random":
        return _random_agent
    raise ValueError(f"Unknown agent: {name}")


AGENTS = ["expectimax", "random"]

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


def _print_summary(results: List[Dict], n_games: int, elapsed: float) -> None:
    scores    = [r["score"]    for r in results]
    moves     = [r["moves"]    for r in results]
    tile_dist = Counter(r["max_tile"] for r in results)
    wins      = sum(r["won"] for r in results)

    print("\n── Summary " + "─" * 31)
    print(f"  Total time   : {elapsed:.1f}s  ({elapsed / n_games:.2f}s/game)")
    print(f"  Avg score    : {statistics.mean(scores):>10,.0f}")
    print(f"  Median score : {statistics.median(scores):>10,.0f}")
    print(f"  Std dev      : {statistics.stdev(scores) if n_games > 1 else 0:>10,.0f}")
    print(f"  Best score   : {max(scores):>10,}")
    print(f"  Avg moves    : {statistics.mean(moves):>10.0f}")
    print(f"  Win rate     : {wins / n_games * 100:>9.1f}%  ({wins}/{n_games} reached 2048)")

    print("\n  Max tile distribution:")
    for tile in sorted(tile_dist.keys(), reverse=True):
        pct = tile_dist[tile] / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"    {tile:>5} : {pct:5.1f}%  {bar}")

    print("─" * 42 + "\n")


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
        "--gui", action="store_true",
        help="Watch the agent play in the Tkinter GUI instead of running headlessly.",
    )
    parser.add_argument(
        "--gui-delay", type=int, default=100,
        help="Milliseconds between moves in GUI mode (default: 100).",
    )
    args = parser.parse_args()

    label = (
        f"{args.agent}"
        + (f" (depth={args.depth}, chance_cells={args.chance_cells})"
           if args.agent == "expectimax" else "")
    )
    agent_fn = _build_agent(args.agent, args.depth, args.chance_cells)

    if args.gui:
        run_gui(agent_fn, delay_ms=args.gui_delay)
    else:
        evaluate(agent_fn, args.games, label, verbose=args.verbose)


if __name__ == "__main__":
    main()
