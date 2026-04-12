"""Standalone training script for the TD n-tuple network agent.

Implements afterstate TD(0) learning (Algorithm 1 from Jaskowski 2018) with an
optional TC(0) adaptive learning-rate variant.

Usage:
    python train_td.py                                          # defaults: 100k episodes, TD(0)
    python train_td.py --episodes 200000 --alpha 0.001
    python train_td.py --episodes 100000 --tc                   # TC(0) adaptive rates
    python train_td.py --resume weights/td_ntuple.npz           # continue training
    python train_td.py --resume weights/td_ntuple.npz --episodes 50000
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from typing import List, Optional, Tuple

from tqdm import tqdm

from game2048 import Game2048
from agents.expectimax_agent import _apply_move
from agents.td_ntuple_agent import NTupleNetwork, DIRECTIONS, Board, Direction


# ---------------------------------------------------------------------------
# Move selection
# ---------------------------------------------------------------------------

def select_best_move(
    board: Board,
    network: NTupleNetwork,
) -> Tuple[Optional[Direction], Optional[Board], float, float]:
    """Evaluate all moves, return (direction, afterstate, reward, V(afterstate)).

    The cached V(afterstate) avoids a redundant evaluate() call in the TD update.
    """
    best_dir: Optional[Direction] = None
    best_after: Optional[Board] = None
    best_score = float("-inf")
    best_reward = 0.0
    best_value = 0.0

    for direction in DIRECTIONS:
        afterstate, reward, moved = _apply_move(board, direction)
        if not moved:
            continue
        value = network.evaluate(afterstate)
        score = reward + value
        if score > best_score:
            best_score = score
            best_dir = direction
            best_after = afterstate
            best_reward = reward
            best_value = value

    return best_dir, best_after, best_reward, best_value


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    num_episodes: int,
    alpha: float,
    output_path: str,
    resume_path: Optional[str],
    use_tc: bool,
    save_interval: int,
    log_interval: int,
) -> None:
    network = NTupleNetwork()
    if resume_path:
        network.load(resume_path)
        print(f"Resumed weights from {resume_path}")

    scores: List[int] = []
    max_tiles: List[int] = []

    print(f"\nTraining  : {'TC(0)' if use_tc else f'TD(0), alpha={alpha}'}")
    print(f"Episodes  : {num_episodes:,}")
    print(f"Output    : {output_path}")
    print("─" * 50)

    t0 = time.time()

    with tqdm(total=num_episodes, unit="ep", dynamic_ncols=True) as pbar:
        for ep in range(num_episodes):
            game = Game2048()
            board = game.get_state()["board"]

            # First move to get initial afterstate
            move, afterstate, reward, value = select_best_move(board, network)
            if move is None:
                continue
            game.move(move)
            prev_afterstate = afterstate
            prev_value = value

            while True:
                state = game.get_state()
                board = state["board"]

                if state["game_over"]:
                    # Terminal update: future value is 0
                    delta = -prev_value
                    if use_tc:
                        network.update_tc(prev_afterstate, delta)
                    else:
                        network.update(prev_afterstate, delta, alpha)
                    break

                move, afterstate, reward, value = select_best_move(board, network)
                if move is None:
                    delta = -prev_value
                    if use_tc:
                        network.update_tc(prev_afterstate, delta)
                    else:
                        network.update(prev_afterstate, delta, alpha)
                    break

                # TD(0) update using cached values
                delta = reward + value - prev_value
                if use_tc:
                    network.update_tc(prev_afterstate, delta)
                else:
                    network.update(prev_afterstate, delta, alpha)

                game.move(move)
                prev_afterstate = afterstate
                prev_value = value

            # Record stats
            final = game.get_state()
            scores.append(final["score"])
            max_tiles.append(max(c for row in final["board"] for c in row))

            # Progress bar update
            if len(scores) >= log_interval:
                recent = scores[-log_interval:]
                pbar.set_postfix(
                    avg=f"{statistics.mean(recent):,.0f}",
                    best=f"{max(recent):,}",
                )
            pbar.update(1)

            # Periodic log
            if (ep + 1) % log_interval == 0:
                recent_scores = scores[-log_interval:]
                recent_tiles = max_tiles[-log_interval:]
                tile_counts = Counter(recent_tiles)
                avg = statistics.mean(recent_scores)
                tiles_str = "  ".join(
                    f"{t}:{tile_counts[t]}" for t in sorted(tile_counts, reverse=True)
                )
                tqdm.write(
                    f"  ep {ep+1:>7,} | avg {avg:>10,.0f} | tiles: {tiles_str}"
                )

            # Periodic save
            if (ep + 1) % save_interval == 0:
                network.save(output_path)
                tqdm.write(f"  [saved weights to {output_path}]")

    elapsed = time.time() - t0
    network.save(output_path)

    print("─" * 50)
    print(f"Training complete in {elapsed:.1f}s ({elapsed/num_episodes:.3f}s/ep)")
    print(f"Final avg score (last {min(log_interval, len(scores))}): "
          f"{statistics.mean(scores[-log_interval:]):,.0f}")
    print(f"Weights saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the TD n-tuple network agent for 2048."
    )
    parser.add_argument(
        "--episodes", type=int, default=100_000,
        help="Number of training episodes (default: 100000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0025,
        help="Learning rate for TD(0) (default: 0.0025, ignored with --tc)",
    )
    parser.add_argument(
        "--tc", action="store_true",
        help="Use TC(0) adaptive learning rates instead of fixed alpha",
    )
    parser.add_argument(
        "--output", type=str, default="weights/td_ntuple.npz",
        help="Path to save weights (default: weights/td_ntuple.npz)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from an existing .npz weights file",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10_000,
        help="Save weights every N episodes (default: 10000)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1_000,
        help="Print stats every N episodes (default: 1000)",
    )
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        alpha=args.alpha,
        output_path=args.output,
        resume_path=args.resume,
        use_tc=args.tc,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
