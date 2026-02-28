"""Lightweight 2048 game engine with a callable API for AI control.

Usage:
    game = Game2048(seed=42)
    state = game.get_state()
    moved, reward = game.move("left")
    state_after = game.get_state()
"""

from __future__ import annotations

import random
import tkinter as tk
from typing import Callable, Dict, List, Optional, Tuple


Direction = str  # one of "up", "down", "left", "right"


class Game2048:
    def __init__(self, size: int = 4, seed: int | None = None) -> None:
        if size < 2:
            raise ValueError("Board size must be at least 2")
        self.size = size
        self.random = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.board: List[List[int]] = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score: int = 0
        self.moves_made: int = 0
        self.won: bool = False
        self._spawn_initial_tiles()

    def _spawn_initial_tiles(self) -> None:
        # Standard 2048 starts with two tiles.
        for _ in range(2):
            self._add_random_tile()

    def _add_random_tile(self) -> None:
        empties = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
        if not empties:
            return
        r, c = self.random.choice(empties)
        self.board[r][c] = 4 if self.random.random() < 0.1 else 2

    def move(self, direction: Direction) -> Tuple[bool, int]:
        direction = direction.lower()
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("Direction must be one of 'up', 'down', 'left', 'right'")

        moved = False
        total_reward = 0

        for index in range(self.size):
            line = self._get_line(index, direction)
            merged_line, reward, did_move = self._merge_line(line)
            if did_move:
                self._set_line(index, direction, merged_line)
                moved = True
                total_reward += reward

        if moved:
            self.moves_made += 1
            self.score += total_reward
            self._add_random_tile()
            if any(val >= 2048 for row in self.board for val in row):
                self.won = True

        return moved, total_reward

    def can_move(self) -> bool:
        if any(0 in row for row in self.board):
            return True
        # Check for adjacent equal tiles.
        for r in range(self.size):
            for c in range(self.size):
                val = self.board[r][c]
                if r + 1 < self.size and self.board[r + 1][c] == val:
                    return True
                if c + 1 < self.size and self.board[r][c + 1] == val:
                    return True
        return False

    def get_state(self) -> Dict[str, object]:
        return {
            "board": [row[:] for row in self.board],
            "score": self.score,
            "moves_made": self.moves_made,
            "game_over": not self.can_move(),
            "won": self.won,
            "available_moves": self.available_moves(),
        }

    def available_moves(self) -> List[Direction]:
        moves: List[Direction] = []
        for direction in ("up", "down", "left", "right"):
            if self._would_move(direction):
                moves.append(direction)
        return moves

    def _would_move(self, direction: Direction) -> bool:
        for index in range(self.size):
            line = self._get_line(index, direction)
            merged_line, _, did_move = self._merge_line(line, dry_run=True)
            if did_move and merged_line != line:
                return True
        return False

    def _get_line(self, index: int, direction: Direction) -> List[int]:
        if direction == "left":
            return self.board[index][:]
        if direction == "right":
            return list(reversed(self.board[index]))
        if direction == "up":
            return [self.board[r][index] for r in range(self.size)]
        if direction == "down":
            return [self.board[r][index] for r in reversed(range(self.size))]
        raise ValueError("Invalid direction")

    def _set_line(self, index: int, direction: Direction, new_line: List[int]) -> None:
        if direction == "left":
            self.board[index] = new_line
        elif direction == "right":
            self.board[index] = list(reversed(new_line))
        elif direction == "up":
            for r in range(self.size):
                self.board[r][index] = new_line[r]
        elif direction == "down":
            for r, val in zip(reversed(range(self.size)), new_line):
                self.board[r][index] = val
        else:
            raise ValueError("Invalid direction")

    def _merge_line(self, line: List[int], dry_run: bool = False) -> Tuple[List[int], int, bool]:
        # Compacts a single row/column and merges equal neighbors once.
        filtered = [x for x in line if x != 0]
        merged: List[int] = []
        reward = 0
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue
            if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                new_val = filtered[i] * 2
                merged.append(new_val)
                reward += new_val
                skip = True
            else:
                merged.append(filtered[i])
        merged.extend([0] * (self.size - len(merged)))
        did_move = merged != line
        return merged, reward, did_move

    def __str__(self) -> str:
        lines = []
        for row in self.board:
            lines.append("\t".join(f"{val:4d}" if val else "   ." for val in row))
        return "\n".join(lines)


class Game2048GUI:
    """Tkinter visualization wrapper for Game2048.

    You can pass an optional `ai_move_provider` callable that receives the current
    state dict and should return a direction string (or None to skip a turn).
    When provided, the AI will auto-play while the window is open.
    """

    def __init__(
        self,
        size: int = 4,
        seed: int | None = None,
        ai_move_provider: Optional[Callable[[Dict[str, object]], Optional[Direction]]] = None,
        ai_delay_ms: int = 80,
    ) -> None:
        self.game = Game2048(size=size, seed=seed)
        self.ai_move_provider = ai_move_provider
        self.ai_delay_ms = ai_delay_ms

        self.root = tk.Tk()
        self.root.title("2048")
        self.root.resizable(False, False)
        self.colors = {
            0: ("#cdc1b4", "#776e65"),
            2: ("#eee4da", "#776e65"),
            4: ("#ede0c8", "#776e65"),
            8: ("#f2b179", "#f9f6f2"),
            16: ("#f59563", "#f9f6f2"),
            32: ("#f67c5f", "#f9f6f2"),
            64: ("#f65e3b", "#f9f6f2"),
            128: ("#edcf72", "#f9f6f2"),
            256: ("#edcc61", "#f9f6f2"),
            512: ("#edc850", "#f9f6f2"),
            1024: ("#edc53f", "#f9f6f2"),
            2048: ("#edc22e", "#f9f6f2"),
        }
        self._build_ui()
        self._render()
        if self.ai_move_provider is not None:
            self._schedule_ai_step()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#bbada0", bd=10)
        top.grid(row=0, column=0, padx=10, pady=10)

        header = tk.Frame(top, bg="#bbada0")
        header.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        self.score_var = tk.StringVar()
        self.status_var = tk.StringVar()
        tk.Label(header, textvariable=self.score_var, font=("Helvetica", 16, "bold"), bg="#bbada0", fg="#f9f6f2").pack(side="left")
        tk.Label(header, textvariable=self.status_var, font=("Helvetica", 12), bg="#bbada0", fg="#f9f6f2").pack(side="left", padx=(10, 0))
        tk.Button(header, text="New Game", command=self._reset).pack(side="right")

        self.tiles: List[List[tk.Label]] = []
        grid = tk.Frame(top, bg="#bbada0")
        grid.grid(row=1, column=0)
        for r in range(self.game.size):
            row_labels: List[tk.Label] = []
            for c in range(self.game.size):
                lbl = tk.Label(
                    grid,
                    width=4,
                    height=2,
                    font=("Helvetica", 24, "bold"),
                    bd=4,
                    relief="ridge",
                    anchor="center",
                )
                lbl.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
                row_labels.append(lbl)
            self.tiles.append(row_labels)

        self.root.bind("<Up>", lambda _: self._do_move("up"))
        self.root.bind("<Down>", lambda _: self._do_move("down"))
        self.root.bind("<Left>", lambda _: self._do_move("left"))
        self.root.bind("<Right>", lambda _: self._do_move("right"))
        self.root.bind("<r>", lambda _: self._reset())

    def _reset(self) -> None:
        self.game.reset()
        self._render()

    def _do_move(self, direction: Direction) -> None:
        moved, _ = self.game.move(direction)
        if moved:
            self._render()

    def _render(self) -> None:
        for r, row in enumerate(self.game.board):
            for c, val in enumerate(row):
                bg, fg = self.colors.get(val, ("#3c3a32", "#f9f6f2"))
                lbl = self.tiles[r][c]
                lbl.config(text=str(val) if val else "", bg=bg, fg=fg)
        state = self.game.get_state()
        self.score_var.set(f"Score: {state['score']}")
        if state["game_over"]:
            self.status_var.set("Game over")
        elif state["won"]:
            self.status_var.set("2048!")
        else:
            self.status_var.set("")

    def _schedule_ai_step(self) -> None:
        if self.ai_move_provider is None:
            return
        if not self.game.can_move():
            return
        state = self.game.get_state()
        direction = self.ai_move_provider(state)
        if direction:
            self._do_move(direction)
        self.root.after(self.ai_delay_ms, self._schedule_ai_step)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    Game2048GUI().run()
