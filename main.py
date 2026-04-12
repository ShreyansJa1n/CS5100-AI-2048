"""2048 AI Playground — multi-screen GUI application.

Launch with:
    python main.py

Provides:
    - Main menu with navigation
    - Manual play with AI hints
    - AI auto-play with speed controls
    - About screens (project + agents)
    - Evaluation report viewer
"""

from __future__ import annotations

import functools
import os
import tkinter as tk
import tkinter.messagebox as messagebox
from typing import Callable, Optional

from game2048 import Game2048

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

BG = "#bbada0"
FG = "#f9f6f2"
FG_MUTED = "#eee4da"
BTN_BG = "#8f7a66"
BTN_ACTIVE = "#776e65"
TEXT_BG = "#faf8ef"
TEXT_FG = "#776e65"
HINT_COLOR = "#edc22e"

TILE_COLORS = {
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

DIR_SYMBOL = {"up": "\u2191", "down": "\u2193", "left": "\u2190", "right": "\u2192"}

BTN_STYLE = dict(
    font=("Helvetica", 14), width=25, pady=8,
    bg=BTN_BG, fg=FG, activebackground=BTN_ACTIVE,
    activeforeground=FG, relief="flat", cursor="hand2",
)

SMALL_BTN = dict(
    font=("Helvetica", 11), bg=BTN_BG, fg=FG,
    activebackground=BTN_ACTIVE, activeforeground=FG,
    relief="flat", cursor="hand2", padx=10, pady=4,
)

BACK_BTN = dict(
    font=("Helvetica", 11), bg=BTN_BG, fg=FG,
    relief="flat", cursor="hand2",
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# About-screen text content
# ---------------------------------------------------------------------------

PROJECT_TEXT = """\
CS 5100 - AI Agent for the 2048 Game

Team:  Rida Sameer & Shreyans Jain
Course:  CS 5100 Foundations of Artificial Intelligence
         Northeastern University

Goal
----
Build and compare two AI agents that autonomously play the
game 2048. The game is modeled as a fully observable,
nondiscounted discrete MDP with approximately 4.7 x 10^21
states, making tabular methods infeasible.

The project evaluates two fundamentally different approaches:
  1. Classical tree search  (Expectimax)
  2. Reinforcement learning (TD Learning with N-Tuple Networks)

Architecture
------------
  game2048.py      Core game engine + Tkinter GUI
  agents/          AI agent implementations
  evaluate.py      Headless batch evaluator with statistics
  train_td.py      Standalone TD agent training script
  weights/         Saved n-tuple network weights (.npz)
  results/         Evaluation result logs

Key Reference
-------------
Jaskowski, W. (2018). "Mastering 2048 with delayed temporal
coherence learning, multistage weight promotion, redundant
encoding, and carousel shaping."
IEEE Transactions on Games, 10(1), 3-14.

GitHub:  https://github.com/ShreyansJa1n/CS5100-AI-2048
"""

AGENTS_TEXT_PARTS = [
    ("heading", "1. Expectimax Search Agent\n"),
    ("body", """
A classical tree-search agent that looks several moves ahead
before deciding on the best direction. It requires no training
or learned weights -- it is purely algorithmic.

How it works:
  The game tree alternates between two kinds of nodes:
    MAX nodes   -- the agent picks the best direction
    CHANCE nodes -- a tile spawns randomly (P(2)=0.9, P(4)=0.1)

  At leaf nodes the board is scored by a heuristic combining:
    - Empty cell bonus   (more free cells = more flexibility)
    - Monotonicity       (tiles decrease along rows/columns)
    - Smoothness         (penalizes large adjacent differences)
    - Corner bonus       (reward for max tile in a corner)

  Default depth: 3   (~1,024 leaf evaluations per move)
  No training required. Serves as the baseline agent.

"""),
    ("heading", "2. TD Learning with N-Tuple Networks\n"),
    ("body", """
A reinforcement learning agent based on Jaskowski (2018).
Instead of searching a game tree, it learns a value function
from hundreds of thousands of self-play games.

Architecture:
  - 4 six-tuples  (the "4x2 - 3x3" configuration)
  - Each tuple applied to all 8 board symmetries (rotations
    and reflections) for a total of 32 LUT lookups per eval
  - 67,108,864 trainable parameters (float32 lookup tables)
  - Tile encoding: empty=0, tile v -> log2(v)

Learning algorithm:
  - Afterstate TD(0): learns value of board states AFTER the
    agent's move but BEFORE the random tile spawn
  - TC(0) adaptive learning rates: automatically tunes
    per-weight learning rates using temporal coherence
  - Update rule:  V(s) += alpha * (r + V(s') - V(s))

At play time the agent simply tries all four moves, computes
reward + V(afterstate) for each, and picks the highest.
No search tree -- just 32 table lookups per candidate move.

Trained via:  python train_td.py --episodes 100000 --tc
Weights saved to:  weights/td_ntuple.npz
"""),
]


# ---------------------------------------------------------------------------
# App controller
# ---------------------------------------------------------------------------

class App:
    """Application controller. Owns the root window and manages screens."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("2048 \u2014 AI Playground")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self.container = tk.Frame(self.root, bg=BG)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.screens: dict[str, tk.Frame] = {}
        for name, cls in [
            ("menu", MenuScreen),
            ("about_project", AboutProjectScreen),
            ("about_agents", AboutAgentsScreen),
            ("report_list", ReportListScreen),
            ("mode_select", ModeSelectScreen),
        ]:
            screen = cls(parent=self.container, app=self)
            screen.grid(row=0, column=0, sticky="nsew")
            self.screens[name] = screen

        self.show_screen("menu")

    def show_screen(self, name: str) -> None:
        frame = self.screens[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def start_game(self, mode: str, agent_name: str) -> None:
        # Build agent first so we can fail gracefully
        agent_fn = self._build_agent(agent_name)
        if agent_fn is None:
            return

        if "game" in self.screens:
            self.screens["game"].cleanup()
            self.screens["game"].destroy()

        screen = GameScreen(
            parent=self.container, app=self,
            mode=mode, agent_name=agent_name, agent_fn=agent_fn,
        )
        screen.grid(row=0, column=0, sticky="nsew")
        self.screens["game"] = screen
        screen.tkraise()

    def show_report(self, filename: str) -> None:
        if "report_viewer" in self.screens:
            self.screens["report_viewer"].destroy()
        screen = ReportViewerScreen(
            parent=self.container, app=self, filename=filename,
        )
        screen.grid(row=0, column=0, sticky="nsew")
        self.screens["report_viewer"] = screen
        screen.tkraise()

    def _build_agent(self, name: str) -> Optional[Callable]:
        if name == "expectimax":
            from agents.expectimax_agent import expectimax_agent
            return expectimax_agent
        if name == "td":
            try:
                from agents.td_ntuple_agent import td_ntuple_agent, _get_network
                _get_network()  # force weight loading to catch missing file
                return td_ntuple_agent
            except FileNotFoundError as e:
                messagebox.showerror(
                    "Weights Not Found",
                    "TD N-Tuple weights not found.\n\n"
                    "Run the following to train the agent first:\n"
                    "  python train_td.py --episodes 100000 --tc",
                )
                return None
        return None

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Helper: top bar with back button
# ---------------------------------------------------------------------------

def _make_top_bar(parent: tk.Frame, app: App, back_to: str) -> tk.Frame:
    bar = tk.Frame(parent, bg=BG)
    bar.pack(fill="x", padx=20, pady=(15, 5))
    tk.Button(bar, text="\u2190 Back", command=lambda: app.show_screen(back_to),
              **BACK_BTN).pack(side="left")
    return bar


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

class MenuScreen(tk.Frame):
    def __init__(self, parent: tk.Frame, app: App) -> None:
        super().__init__(parent, bg=BG)
        tk.Label(self, text="2048", font=("Helvetica", 52, "bold"),
                 bg=BG, fg=FG).pack(pady=(50, 2))
        tk.Label(self, text="AI Playground", font=("Helvetica", 18),
                 bg=BG, fg=FG_MUTED).pack(pady=(0, 35))
        tk.Button(self, text="Start Game",
                  command=lambda: app.show_screen("mode_select"),
                  **BTN_STYLE).pack(pady=8)
        tk.Button(self, text="About the Project",
                  command=lambda: app.show_screen("about_project"),
                  **BTN_STYLE).pack(pady=8)
        tk.Button(self, text="About the Agents",
                  command=lambda: app.show_screen("about_agents"),
                  **BTN_STYLE).pack(pady=8)
        tk.Button(self, text="Evaluation Reports",
                  command=lambda: app.show_screen("report_list"),
                  **BTN_STYLE).pack(pady=8)


class AboutProjectScreen(tk.Frame):
    def __init__(self, parent: tk.Frame, app: App) -> None:
        super().__init__(parent, bg=BG)
        _make_top_bar(self, app, "menu")
        tk.Label(self, text="About the Project", font=("Helvetica", 20, "bold"),
                 bg=BG, fg=FG).pack(pady=(5, 10))
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        txt = tk.Text(frame, wrap="word", font=("Helvetica", 12),
                      bg=TEXT_BG, fg=TEXT_FG, relief="flat",
                      padx=15, pady=15, yscrollcommand=sb.set)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", PROJECT_TEXT)
        txt.config(state="disabled")


class AboutAgentsScreen(tk.Frame):
    def __init__(self, parent: tk.Frame, app: App) -> None:
        super().__init__(parent, bg=BG)
        _make_top_bar(self, app, "menu")
        tk.Label(self, text="About the Agents", font=("Helvetica", 20, "bold"),
                 bg=BG, fg=FG).pack(pady=(5, 10))
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        txt = tk.Text(frame, wrap="word", font=("Helvetica", 12),
                      bg=TEXT_BG, fg=TEXT_FG, relief="flat",
                      padx=15, pady=15, yscrollcommand=sb.set)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.tag_configure("heading", font=("Helvetica", 14, "bold"), foreground=TEXT_FG)
        txt.tag_configure("body", font=("Helvetica", 12), foreground=TEXT_FG)
        for tag, content in AGENTS_TEXT_PARTS:
            txt.insert("end", content, tag)
        txt.config(state="disabled")


class ReportListScreen(tk.Frame):
    def __init__(self, parent: tk.Frame, app: App) -> None:
        super().__init__(parent, bg=BG)
        self.app = app
        _make_top_bar(self, app, "menu")
        tk.Label(self, text="Evaluation Reports", font=("Helvetica", 20, "bold"),
                 bg=BG, fg=FG).pack(pady=(5, 10))

        self.list_frame = tk.Frame(self, bg=BG)
        self.list_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        sb = tk.Scrollbar(self.list_frame)
        sb.pack(side="right", fill="y")
        self.listbox = tk.Listbox(
            self.list_frame, font=("Courier", 12),
            bg=TEXT_BG, fg=TEXT_FG, selectbackground=BTN_BG, selectforeground=FG,
            yscrollcommand=sb.set, relief="flat",
        )
        self.listbox.pack(fill="both", expand=True)
        sb.config(command=self.listbox.yview)
        self.listbox.bind("<Double-Button-1>", self._open_selected)

        self.empty_label = tk.Label(
            self, text="No evaluation reports found.\n\nRun evaluate.py to generate reports.",
            font=("Helvetica", 13), bg=BG, fg=FG_MUTED,
        )
        self.report_files: list[str] = []

    def on_show(self) -> None:
        self.listbox.delete(0, "end")
        self.report_files = []
        if os.path.isdir(RESULTS_DIR):
            self.report_files = sorted(
                [f for f in os.listdir(RESULTS_DIR) if f.endswith(".txt")],
                reverse=True,
            )
        if not self.report_files:
            self.list_frame.pack_forget()
            self.empty_label.pack(pady=40)
        else:
            self.empty_label.pack_forget()
            self.list_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
            for f in self.report_files:
                self.listbox.insert("end", f"  {f}")

    def _open_selected(self, _event: object = None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        self.app.show_report(self.report_files[sel[0]])


class ReportViewerScreen(tk.Frame):
    def __init__(self, parent: tk.Frame, app: App, filename: str) -> None:
        super().__init__(parent, bg=BG)
        _make_top_bar(self, app, "report_list")
        tk.Label(self, text=filename, font=("Helvetica", 12, "bold"),
                 bg=BG, fg=FG).pack(pady=(5, 10))
        frame = tk.Frame(self, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        txt = tk.Text(frame, wrap="none", font=("Courier", 11),
                      bg=TEXT_BG, fg=TEXT_FG, relief="flat",
                      padx=10, pady=10, yscrollcommand=sb.set)
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath) as f:
                txt.insert("1.0", f.read())
        except OSError:
            txt.insert("1.0", f"Error: could not read {filepath}")
        txt.config(state="disabled")


class ModeSelectScreen(tk.Frame):
    def __init__(self, parent: tk.Frame, app: App) -> None:
        super().__init__(parent, bg=BG)
        self.app = app
        _make_top_bar(self, app, "menu")
        tk.Label(self, text="Game Setup", font=("Helvetica", 24, "bold"),
                 bg=BG, fg=FG).pack(pady=(10, 25))

        # Mode selection
        tk.Label(self, text="Game Mode", font=("Helvetica", 16, "bold"),
                 bg=BG, fg=FG_MUTED).pack(pady=(0, 8))
        self.mode_var = tk.StringVar(value="manual_hints")
        radio_opts = dict(
            font=("Helvetica", 13), bg=BG, fg=FG,
            selectcolor=BTN_ACTIVE, activebackground=BG,
            activeforeground=FG, anchor="w",
        )
        tk.Radiobutton(self, text="  Manual Play with AI Hints",
                       variable=self.mode_var, value="manual_hints",
                       **radio_opts).pack(padx=80, anchor="w")
        tk.Radiobutton(self, text="  AI Auto-Play (Watch the Agent)",
                       variable=self.mode_var, value="ai_autoplay",
                       **radio_opts).pack(padx=80, anchor="w")

        # Agent selection
        tk.Label(self, text="AI Agent", font=("Helvetica", 16, "bold"),
                 bg=BG, fg=FG_MUTED).pack(pady=(25, 8))
        self.agent_var = tk.StringVar(value="expectimax")
        tk.Radiobutton(self, text="  Expectimax  (tree search, depth 3)",
                       variable=self.agent_var, value="expectimax",
                       **radio_opts).pack(padx=80, anchor="w")
        tk.Radiobutton(self, text="  TD N-Tuple Network  (learned, instant)",
                       variable=self.agent_var, value="td",
                       **radio_opts).pack(padx=80, anchor="w")

        # Start
        tk.Button(self, text="Start Game", font=("Helvetica", 16, "bold"),
                  bg=BTN_BG, fg=FG, activebackground=BTN_ACTIVE,
                  activeforeground=FG, relief="flat", cursor="hand2",
                  width=20, pady=8, command=self._start).pack(pady=(35, 20))

    def _start(self) -> None:
        self.app.start_game(
            mode=self.mode_var.get(),
            agent_name=self.agent_var.get(),
        )


# ---------------------------------------------------------------------------
# Game screen
# ---------------------------------------------------------------------------

class GameScreen(tk.Frame):
    def __init__(
        self, parent: tk.Frame, app: App,
        mode: str, agent_name: str, agent_fn: Callable,
    ) -> None:
        super().__init__(parent, bg=BG)
        self.app = app
        self.mode = mode
        self.agent_name = agent_name
        self.agent_fn = agent_fn
        self.game = Game2048()

        self._ai_running = False
        self._ai_delay_ms = 80
        self._ai_after_id: Optional[str] = None
        self._hint_clear_id: Optional[str] = None

        self._build_ui()
        self._render()
        if self.mode == "manual_hints":
            self._bind_keyboard()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=15, pady=(10, 5))
        tk.Button(header, text="\u2190 Menu", command=self._back_to_menu,
                  **BACK_BTN).pack(side="left")
        info = tk.Frame(header, bg=BG)
        info.pack(side="right")
        self.score_var = tk.StringVar(value="Score: 0")
        self.moves_var = tk.StringVar(value="Moves: 0")
        tk.Label(info, textvariable=self.score_var,
                 font=("Helvetica", 14, "bold"), bg=BG, fg=FG).pack(side="right", padx=(10, 0))
        tk.Label(info, textvariable=self.moves_var,
                 font=("Helvetica", 11), bg=BG, fg=FG_MUTED).pack(side="right", padx=(10, 0))

        # Tile grid
        grid_frame = tk.Frame(self, bg=BG)
        grid_frame.pack(pady=10)
        self.tiles: list[list[tk.Label]] = []
        for r in range(4):
            row_labels: list[tk.Label] = []
            for c in range(4):
                lbl = tk.Label(grid_frame, width=4, height=2,
                               font=("Helvetica", 24, "bold"),
                               bd=4, relief="ridge", anchor="center")
                lbl.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
                row_labels.append(lbl)
            self.tiles.append(row_labels)

        # Hint / status area
        self.hint_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.hint_var,
                 font=("Helvetica", 16, "bold"), bg=BG, fg=HINT_COLOR).pack(pady=(5, 0))
        self.status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status_var,
                 font=("Helvetica", 14), bg=BG, fg=FG).pack(pady=(2, 5))

        # Controls bar
        controls = tk.Frame(self, bg=BG)
        controls.pack(fill="x", padx=20, pady=(5, 15))
        tk.Button(controls, text="New Game", command=self._reset,
                  **SMALL_BTN).pack(side="left", padx=(0, 5))

        # Agent label
        agent_display = "Expectimax" if self.agent_name == "expectimax" else "TD N-Tuple"
        tk.Label(controls, text=f"Agent: {agent_display}",
                 font=("Helvetica", 10), bg=BG, fg=FG_MUTED).pack(side="left", padx=(10, 0))

        if self.mode == "manual_hints":
            tk.Button(controls, text="Hint (H)", command=self._show_hint,
                      **SMALL_BTN).pack(side="right", padx=(5, 0))
        else:
            self._build_ai_controls(controls)

    def _build_ai_controls(self, parent: tk.Frame) -> None:
        tk.Button(parent, text="Slower", command=self._slower,
                  **SMALL_BTN).pack(side="right", padx=(5, 0))
        tk.Button(parent, text="Faster", command=self._faster,
                  **SMALL_BTN).pack(side="right", padx=(5, 0))
        self.speed_var = tk.StringVar(value=f"{self._ai_delay_ms}ms")
        tk.Label(parent, textvariable=self.speed_var,
                 font=("Helvetica", 10), bg=BG, fg=FG_MUTED).pack(side="right", padx=5)
        self.play_pause_btn = tk.Button(parent, text="Play",
                                        command=self._toggle_ai, **SMALL_BTN)
        self.play_pause_btn.pack(side="right", padx=5)

    # ── Rendering ─────────────────────────────────────────────────────

    def _render(self) -> None:
        for r, row in enumerate(self.game.board):
            for c, val in enumerate(row):
                bg, fg = TILE_COLORS.get(val, ("#3c3a32", "#f9f6f2"))
                self.tiles[r][c].config(text=str(val) if val else "", bg=bg, fg=fg)
        state = self.game.get_state()
        self.score_var.set(f"Score: {state['score']:,}")
        self.moves_var.set(f"Moves: {state['moves_made']}")
        if state["game_over"]:
            self.status_var.set("Game Over!")
            self._stop_ai()
        elif state["won"]:
            self.status_var.set("You reached 2048!")
        else:
            self.status_var.set("")

    # ── Move handling ─────────────────────────────────────────────────

    def _do_move(self, direction: str) -> None:
        if not self.game.can_move():
            return
        moved, _ = self.game.move(direction)
        if moved:
            self._clear_hint()
            self._render()

    def _reset(self) -> None:
        self._stop_ai()
        self.game.reset()
        self._clear_hint()
        self._render()
        if self.mode == "ai_autoplay" and hasattr(self, "play_pause_btn"):
            self.play_pause_btn.config(text="Play")
            self._ai_running = False

    # ── Keyboard ──────────────────────────────────────────────────────

    def _bind_keyboard(self) -> None:
        root = self.winfo_toplevel()
        root.bind("<Up>", lambda _: self._do_move("up"))
        root.bind("<Down>", lambda _: self._do_move("down"))
        root.bind("<Left>", lambda _: self._do_move("left"))
        root.bind("<Right>", lambda _: self._do_move("right"))
        root.bind("<r>", lambda _: self._reset())
        root.bind("<h>", lambda _: self._show_hint())

    def _unbind_keyboard(self) -> None:
        root = self.winfo_toplevel()
        for key in ("<Up>", "<Down>", "<Left>", "<Right>", "<r>", "<h>"):
            root.unbind(key)

    # ── Hint ──────────────────────────────────────────────────────────

    def _show_hint(self) -> None:
        if not self.game.can_move():
            return
        state = self.game.get_state()
        direction = self.agent_fn(state)
        if direction is None:
            self.hint_var.set("No move available")
            return
        symbol = DIR_SYMBOL.get(direction, "")
        self.hint_var.set(f"Hint: {symbol} {direction.upper()}")
        self._highlight_direction(direction)

    def _highlight_direction(self, direction: str) -> None:
        self._clear_tile_highlights()
        if direction == "up":
            targets = [(0, c) for c in range(4)]
        elif direction == "down":
            targets = [(3, c) for c in range(4)]
        elif direction == "left":
            targets = [(r, 0) for r in range(4)]
        elif direction == "right":
            targets = [(r, 3) for r in range(4)]
        else:
            return
        for r, c in targets:
            self.tiles[r][c].config(
                highlightbackground=HINT_COLOR, highlightthickness=3,
            )
        # Auto-clear after 2 seconds
        if self._hint_clear_id is not None:
            self.winfo_toplevel().after_cancel(self._hint_clear_id)
        self._hint_clear_id = self.winfo_toplevel().after(2000, self._clear_hint)

    def _clear_tile_highlights(self) -> None:
        for r in range(4):
            for c in range(4):
                self.tiles[r][c].config(highlightthickness=0)

    def _clear_hint(self) -> None:
        self.hint_var.set("")
        self._clear_tile_highlights()
        self._hint_clear_id = None

    # ── AI auto-play ──────────────────────────────────────────────────

    def _toggle_ai(self) -> None:
        if self._ai_running:
            self._stop_ai()
            self.play_pause_btn.config(text="Play")
        else:
            self._start_ai()
            self.play_pause_btn.config(text="Pause")

    def _start_ai(self) -> None:
        if not self.game.can_move():
            return
        self._ai_running = True
        self._schedule_ai_step()

    def _stop_ai(self) -> None:
        self._ai_running = False
        if self._ai_after_id is not None:
            self.winfo_toplevel().after_cancel(self._ai_after_id)
            self._ai_after_id = None

    def _schedule_ai_step(self) -> None:
        if not self._ai_running or not self.game.can_move():
            self._ai_running = False
            if hasattr(self, "play_pause_btn"):
                self.play_pause_btn.config(text="Play")
            return
        state = self.game.get_state()
        direction = self.agent_fn(state)
        if direction:
            self._do_move(direction)
        self._ai_after_id = self.winfo_toplevel().after(
            self._ai_delay_ms, self._schedule_ai_step,
        )

    def _faster(self) -> None:
        self._ai_delay_ms = max(10, self._ai_delay_ms - 20)
        self.speed_var.set(f"{self._ai_delay_ms}ms")

    def _slower(self) -> None:
        self._ai_delay_ms = min(500, self._ai_delay_ms + 20)
        self.speed_var.set(f"{self._ai_delay_ms}ms")

    # ── Navigation ────────────────────────────────────────────────────

    def _back_to_menu(self) -> None:
        self.cleanup()
        self.app.show_screen("menu")

    def cleanup(self) -> None:
        self._stop_ai()
        if self._hint_clear_id is not None:
            self.winfo_toplevel().after_cancel(self._hint_clear_id)
            self._hint_clear_id = None
        self._unbind_keyboard()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    App().run()
