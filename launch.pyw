"""Portable windowed launcher for Apex Highlights."""
import ctypes
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
# The portable bundle ships its OCR models next to this file.
if (root / 'models').is_dir():
    os.environ.setdefault('EASYOCR_MODULE_PATH', str(root / 'models'))
try:
    directory = root / 'apps' / 'ApexHighlights'
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory / 'launch.pyw'), run_name='__main__')
except Exception as error:
    help_text = ('Extract the complete portable ZIP again. Check Windows Security protection history for missing files.'
                 if (root / 'runtime').is_dir() else 'Run: py -3.12 install.py')
    ctypes.windll.user32.MessageBoxW(0, str(error) + '\n' + help_text, 'Launch error', 0x10)
