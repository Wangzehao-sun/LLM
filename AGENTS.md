# Repository Guidelines

## Project Structure & Module Organization

This repository has two main areas. `Data/` contains standalone Python utilities and prompt/template assets for data processing, including scripts such as `add_token_split_points.py` and `prompt_rewrite.py`. `Myverl/` contains the main Python package, derived from `verl`: source code lives under `Myverl/verl/`, examples under `Myverl/examples/`, documentation under `Myverl/docs/`, scripts under `Myverl/scripts/`, and tests under `Myverl/tests/`.

Keep new data utilities in `Data/` unless they are part of the reusable `verl` package. Package code should stay inside `Myverl/verl/` with tests in the matching `Myverl/tests/` category.

## Build, Test, and Development Commands

Run commands from `Myverl/` when working on the package:

```bash
cd Myverl
pip install -e ".[test]"
pre-commit run --all-files
pytest tests/**/test_*_on_cpu.py
pytest tests/utils/test_config_on_cpu.py
```

`pip install -e ".[test]"` installs the package and test tools in editable mode. `pre-commit run --all-files` applies Ruff linting and formatting checks. The CPU pytest pattern is the safest default for local validation; GPU, distributed, NPU, and e2e suites may require specialized hardware or environments.

Run standalone data scripts from the repository root, for example:

```bash
python Data/prompt_rewrite.py
```

## Coding Style & Naming Conventions

Python code uses Ruff and Ruff format, configured in `Myverl/pyproject.toml`. Follow existing package style, keep imports sorted, and prefer clear module-level functions over ad hoc script blocks. Test files use `test_*.py`; CPU-only tests end with `_on_cpu.py`. Use snake_case for Python files, functions, and variables.

## Testing Guidelines

Add or update tests when changing behavior in `Myverl/verl/`. Place tests in the closest matching folder under `Myverl/tests/`, such as `tests/utils/`, `tests/models/`, or `tests/trainer/`. Use `special_` directories only for dedicated workflows such as distributed, e2e, NPU, or standalone environment tests.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries, for example `Add data processing scripts`. Keep commits focused and avoid mixing unrelated `Data/` and `Myverl/` changes. Pull requests should include a concise description, affected paths, validation commands run, linked issues when applicable, and screenshots or logs only when they clarify behavior.

## Security & Configuration Tips

Do not commit secrets, tokens, local datasets, checkpoints, or generated large artifacts. Keep environment-specific paths and credentials out of source files; use local environment variables or ignored config files instead.
