"""Create the local virtual environment and install dependencies.

Usage: py -3.12 install.py [--skip-models]
"""
import argparse
import subprocess
import sys
import venv
from pathlib import Path

root = Path(__file__).resolve().parent
venv_dir = root / '.venv'
venv_python = venv_dir / 'Scripts' / 'python.exe'


def run(*args):
    subprocess.run([str(venv_python), *args], check=True, cwd=root)


def main():
    parser = argparse.ArgumentParser(description='Install Apex Highlights.')
    parser.add_argument('--skip-models', action='store_true', help='skip the OCR model download')
    options = parser.parse_args()

    if sys.version_info[:2] != (3, 12):
        sys.exit('Run this script with 64-bit Python 3.12: py -3.12 install.py')
    if not venv_python.exists():
        print('Creating virtual environment in .venv ...')
        venv.create(venv_dir, with_pip=True)

    try:
        run('-m', 'pip', 'install', '-r', str(root / 'requirements.txt'))
    except subprocess.CalledProcessError:
        sys.exit('Dependency installation failed.')
    if not options.skip_models:
        try:
            run(str(root / 'prepare_models.py'))
        except subprocess.CalledProcessError:
            sys.exit('OCR model preparation failed. Check network access and retry.')

    print(r'Setup complete. Run .\.venv\Scripts\pythonw.exe launch.pyw to open Apex Highlights.')
    print('Damage OCR also requires Tesseract with the English language data installed separately.')


if __name__ == '__main__':
    main()
