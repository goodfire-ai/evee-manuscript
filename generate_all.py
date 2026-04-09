#!/usr/bin/env python3
"""Legacy entry point — use `uv run evee-ms figures` instead."""
import subprocess
import sys

sys.exit(subprocess.call([sys.executable, "-m", "src.cli", "figures"]))
