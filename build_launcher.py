"""Build the native GUI launcher with MinGW-w64 g++ (no runtime installation).

Usage: py build_launcher.py [--output build/ApexHighlights/ApexHighlights.exe]
Requires MinGW-w64 g++ on PATH, for example from w64devkit.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def build_launcher(output):
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('Install MinGW-w64 (e.g. w64devkit) and add its bin directory to PATH.')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-municode', '-mwindows', '-static',
                    '-Wl,--dynamicbase,--nxcompat',
                    str(ROOT / 'launcher' / 'launcher.cpp'), '-o', str(output)], check=True)
    print(f'Launcher: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'build/ApexHighlights/ApexHighlights.exe')
    build_launcher(parser.parse_args().output)
