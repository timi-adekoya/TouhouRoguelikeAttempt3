# Touhou Roguelike

## Requirements

- Python 3.13 or newer

## Running with uv (recommended)

From the project folder:

```sh
uv run main.py
```

uv installs the dependencies from `uv.lock` into `.venv` on first run.

## Running with pip

From the project folder:

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install msgpack tcod
python main.py
```

Always launch from the project folder: the tileset
(`dejavu10x10_gs_tc.png`) is loaded relative to the current directory.
Saves are written to `saves/` in the project folder.
