# Contributing to ConformalGuard

Thanks for helping improve ConformalGuard. Keep changes reproducible and small enough to review.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests examples
```

## Pull requests

For behavior changes, add or update tests. For new shift mechanisms or metrics, document the assumptions and include a deterministic seed in tests. Do not present results from one dataset as universal guarantees.
