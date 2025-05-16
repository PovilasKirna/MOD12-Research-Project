# 🧪 Research Project

This is a collaborative Python 3.11 project managed with [Poetry](https://python-poetry.org/). It includes strict formatting, linting, and pre-commit enforcement to ensure code quality and consistency across the team.

---

## 📦 Tech Stack

- **Python 3.11**
- [Poetry](https://python-poetry.org/) for dependency and environment management
- [Black](https://black.readthedocs.io/en/stable/) for code formatting
- [Ruff](https://docs.astral.sh/ruff/) for fast linting and import sorting
- [pre-commit](https://pre-commit.com/) to enforce standards on each commit

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/research-project.git
cd research-project
```

### 2. Install Poetry (if not installed)

We recommend installing via [pipx](https://pypa.github.io/pipx/):

```bash
pipx install poetry
```

Ensure it's on your path:

```bash
poetry --version
```

### 3. Set Python 3.11 as your interpreter

Make sure Python 3.11 is installed via Homebrew or `pyenv`, then link it:

```bash
poetry env use /opt/homebrew/opt/python@3.11/bin/python3.11
```

### 4. Install project dependencies

```bash
poetry install
```

### 5. Set up pre-commit hooks

```bash
pipx install pre-commit
pre-commit install
```

---

## 🛠 Usage

Run your code within the Poetry environment:

```bash
poetry run python your_project/script.py
```

Run formatters manually:

```bash
poetry run black .
poetry run ruff check .
```

---

## ✅ Pre-commit Hooks

These tools will auto-run before each commit:

- `black`: Formats Python code
- `ruff`: Lints and auto-fixes issues

To run them manually:

```bash
pre-commit run --all-files
```

---

## 👥 Team Conventions

- Use `poetry` for installing dependencies, not `pip`.
- All code must pass `black` and `ruff`.
- Never commit without letting `pre-commit` run (it runs automatically).
- Use `poetry shell` to enter the project environment if needed.

---

## 📁 Project Structure

```plaintext
research-project/
├── your_project/         # main source code
├── tests/                # test suite
├── .pre-commit-config.yaml
├── pyproject.toml        # project + tool config
└── README.md
```

---

## 📜 License

Idk yet
