"""Portable windowed launcher for Apex Highlights."""
import ctypes
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
try:
    directory = root / 'apps' / 'ApexHighlights'
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / 'launch.pyw'), run_name='__main__')
except Exception as error:
    ctypes.windll.user32.MessageBoxW(0, str(error) + '\nRun Setup.cmd first.', 'Launch error', 0x10)
