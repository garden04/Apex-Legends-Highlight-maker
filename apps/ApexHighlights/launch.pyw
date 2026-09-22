"""Windowed entry point with a log for startup failures."""
import ctypes
import os
from pathlib import Path
import runpy
import sys
import traceback

root = Path(__file__).resolve().parent
try:
    os.chdir(root)
    (root / 'data').mkdir(exist_ok=True)
    log = (root / 'data' / 'app.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    runpy.run_path(str(root / 'app.py'), run_name='__main__')
except Exception:
    detail = traceback.format_exc()
    try:
        (root / 'data' / 'startup_error.txt').write_text(detail, encoding='utf-8')
    except OSError:
        pass
    ctypes.windll.user32.MessageBoxW(
        0, '앱을 실행하지 못했습니다. Setup.cmd로 실행 환경을 설치해 주세요.\n'
        '자세한 내용은 data/startup_error.txt에 기록됩니다.', 'Apex Highlights', 0x10)
