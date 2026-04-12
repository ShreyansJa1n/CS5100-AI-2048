# Usage Guide: 2048 AI Playground

---

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Launch the GUI application
python main.py
```

This opens the main menu where you can play the game, watch an AI agent play, read about the project and agents, or view evaluation reports.

---

## What's Been Built

### Previously existing (before this session)

| Component | File | Description |
|-----------|------|-------------|
| Game engine | `game2048.py` | Core 2048 logic + basic Tkinter GUI |
| Expectimax agent | `agents/expectimax_agent.py` | Tree-search agent with heuristics |
| Evaluation harness | `evaluate.py` | Headless batch evaluator |
| Demo script | `test_ai.py` | Scripted move sequence demo |

### Built in this session

| Component | File | Description |
|-----------|------|-------------|
| TD learning agent | `agents/td_ntuple_agent.py` | N-tuple network value function with afterstate TD(0) / TC(0) learning |
| Training script | `train_td.py` | Standalone training loop, saves weights to `.npz` |
| GUI application | `main.py` | Multi-screen menu with game modes, hints, info screens, report viewer |
| Results dump | `evaluate.py` (modified) | Now saves detailed `.txt` files to `results/` with per-game data |
| Documentation | `CONTEXT.md`, `USAGE.md` | Technical context and this usage guide |

### What changed in existing files

- **`evaluate.py`** — Added `"td"` agent to the registry, `--weights` CLI flag, and automatic results file dump to `results/<datetime>_<agent>_<N>games.txt`
- **`.gitignore`** — Added `weights/` and `results/` (large binary and generated files)

### What was NOT changed

- `game2048.py` — untouched
- `agents/expectimax_agent.py` — untouched
- `test_ai.py` — untouched

---

## Running the GUI (`main.py`)

```bash
python main.py
```

### Main Menu

Four buttons:

- **Start Game** — Choose a game mode and AI agent, then play
- **About the Project** — Project description, team, architecture
- **About the Agents** — How each agent works
- **Evaluation Reports** — Browse and read results from `evaluate.py` runs

### Game Modes

**Manual Play with AI Hints**
- Use **arrow keys** to play the game yourself
- Press **H** or click **Hint** to ask the AI for its recommended move
- A gold arrow and edge highlight show the suggestion (clears after 2 seconds)
- The AI does NOT auto-play — it only advises

**AI Auto-Play (Watch the Agent)**
- Click **Play** to start the agent, **Pause** to stop
- **Faster** / **Slower** buttons adjust the move delay (10ms–500ms)
- Click **New Game** to reset
- The agent plays fully autonomously

### Agent Selection

Both modes let you choose between:
- **Expectimax** — tree search, ~60-90ms per move (slight UI pause on hints)
- **TD N-Tuple** — learned, instant (requires trained weights)

---

## Training the TD Agent

The TD agent needs trained weights before it can play. Training plays thousands of games and updates the n-tuple network's lookup tables.

### Basic training (recommended)

```bash
python train_td.py --episodes 100000 --tc
```

This runs 100k self-play games using TC(0) adaptive learning rates. Takes roughly 60-90 minutes. Weights are saved to `weights/td_ntuple.npz`.

### Options

```bash
# Fixed learning rate instead of adaptive
python train_td.py --episodes 100000 --alpha 0.0025

# Shorter run for testing
python train_td.py --episodes 1000 --log-interval 100

# Resume training from existing weights
python train_td.py --resume weights/td_ntuple.npz --episodes 50000 --tc

# Custom output path
python train_td.py --episodes 100000 --tc --output weights/my_weights.npz
```

### What to expect during training

The script prints periodic stats showing average score and max tile distribution:

```
  ep   5,000 | avg     18,432 | tiles: 1024:312  512:421  256:198  128:69
  ep  10,000 | avg     24,891 | tiles: 2048:23  1024:487  512:389  256:101
  ...
```

Weights are auto-saved every 10,000 episodes (configurable with `--save-interval`).

---

## Evaluating Agents

Run an agent headlessly for N games and get statistics:

```bash
# Expectimax (default: 100 games, depth 3)
python evaluate.py

# TD agent
python evaluate.py --agent td --games 100

# Random baseline
python evaluate.py --agent random --games 200

# Deeper expectimax search
python evaluate.py --agent expectimax --games 50 --depth 4

# Watch a single game step-by-step
python evaluate.py --agent expectimax --verbose --games 1

# Watch the agent play in a GUI window
python evaluate.py --agent td --gui

# Use custom weights for TD agent
python evaluate.py --agent td --weights weights/my_weights.npz --games 100
```

### Results output

Every run (except GUI mode) automatically saves a detailed results file:

```
results/2026-04-12_14-30-45_expectimax_100games.txt
```

The file contains:
- Run configuration (timestamp, agent, game count)
- Summary statistics (avg, median, std dev, best, worst, win rate)
- Score percentiles (p10 through p99)
- Max tile distribution
- Per-game results table (game#, score, moves, max tile, won)

You can browse these files from the GUI via **Evaluation Reports** on the main menu.

---

## Running the Original GUI

The original standalone game GUI still works independently:

```bash
# Play manually
python game2048.py

# Or with an AI agent via evaluate.py
python evaluate.py --gui --agent expectimax
```

---

## File Reference

| File | Lines | Purpose |
|------|------:|---------|
| `main.py` | ~590 | GUI app: menu, game modes, hints, info screens, report viewer |
| `game2048.py` | 281 | Game engine (`Game2048`) + standalone GUI (`Game2048GUI`) |
| `agents/expectimax_agent.py` | 251 | Expectimax tree-search agent |
| `agents/td_ntuple_agent.py` | 272 | TD n-tuple network agent + `NTupleNetwork` class |
| `train_td.py` | 230 | Training script with TD(0) / TC(0) |
| `evaluate.py` | 345 | Headless batch evaluator + results dump |
| `test_ai.py` | 37 | Scripted move demo |
| `CONTEXT.md` | — | Full technical context dump |
| `USAGE.md` | — | This file |

---

## Troubleshooting

**"TD N-Tuple weights not found"**
Run `python train_td.py --episodes 100000 --tc` first. The TD agent needs trained weights at `weights/td_ntuple.npz`.

**Training is slow**
Normal. Early episodes are fast (~40 ep/s) but slow down as the agent improves and games last longer (~15-20 ep/s). 100k episodes takes 60-90 minutes.

**Hint takes a moment (Expectimax)**
The expectimax agent searches a game tree at depth 3, which takes ~60-90ms. The UI freezes briefly during this computation. The TD agent is instant.

**Missing numpy**
```bash
source .venv/bin/activate
pip install numpy
```

**`evaluate.py` results not showing in GUI**
Results are saved to the `results/` directory. Make sure you ran `evaluate.py` at least once (not in `--gui` mode). The report list refreshes automatically each time you open it.
