#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv run pytest "$@"