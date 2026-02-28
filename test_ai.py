"""Demo script that mimics an AI by replaying a fixed sequence of moves with delays.

Run: python3 test_ai.py

Arrow keys still work; press R or click "New Game" to reset. The demo will loop
through the scripted moves list while the GUI is open.
"""

import itertools
from typing import List, Optional

from game2048 import Game2048GUI, Direction

# Define a scripted move sequence to test visualization.
SCRIPTED_MOVES: List[Direction] = [
    "up", "left", "left", "down", "right", "up", "up", "left", "down", "right",
    "right", "up", "left", "down", "left", "up", "right", "down", "down", "left",
]


def scripted_ai(move_iter: itertools.cycle) -> callable:
    """Return an AI callback that cycles through a fixed move sequence."""

    def ai(_state: dict) -> Optional[Direction]:
        return next(move_iter, None)

    return ai


def main() -> None:
    move_iter = itertools.cycle(SCRIPTED_MOVES)
    Game2048GUI(ai_move_provider=scripted_ai(move_iter), ai_delay_ms=250).run()


if __name__ == "__main__":
    main()
