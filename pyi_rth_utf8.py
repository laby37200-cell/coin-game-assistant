"""PyInstaller runtime hook: force UTF-8 stdout/stderr on Windows."""
import sys
import os

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
