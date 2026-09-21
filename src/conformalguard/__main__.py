"""Allow ``python -m conformalguard`` to invoke the CLI."""

from conformalguard.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
