#!/usr/bin/env bash
rm -rf venv __pycache__ .pytest_cache
find . -type f -name "*.pyc" -delete
