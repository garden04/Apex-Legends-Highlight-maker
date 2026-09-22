"""Build a portable Windows bundle: unzip and run, no Python install needed.

Usage (any Python 3.9+ with pip, on 64-bit Windows):
    py build_portable.py [--tesseract-dir DIR] [--no-zip]

Output: build/ApexHighlights/ (the bundle) and dist/ApexHighlights-portable.zip.
The bundle holds the official python.org embeddable runtime, the packages from
requirements.txt, the EasyOCR models and a copy of an installed Tesseract.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from build_launcher import build_launcher

PYTHON_VERSION = '3.12.10'
PYTHON_URL = f'https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip'
PYTHON_SHA256 = '4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3'

root = Path(__file__).resolve().parent
build = root / 'build'
bundle = build / 'ApexHighlights'
runtime = bundle / 'runtime'
site_packages = runtime / 'Lib' / 'site-packages'
dist = root / 'dist'

PTH = '''python312.zip
.
Lib\\site-packages
import site
'''
# Build-time Qt tools (designer, uic, ...) and Qt WebEngine are not used by the app.
PRUNE_GLOBS = ['**/__pycache__', 'bin', 'torch/include', 'torch/share', 'torch/lib/*.lib', 'PySide6/*.exe',
               'PySide6/Qt6WebEngine*', 'PySide6/QtWebEngine*', 'PySide6/resources',
               'PySide6/translations/qtwebengine_locales']
MSVC_RUNTIME = ['msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll']


def step(message):
    print(f'\n== {message}', flush=True)


def fetch_python():
    cache = build / 'cache' / Path(PYTHON_URL).name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading', PYTHON_URL)
        with urllib.request.urlopen(PYTHON_URL) as response, open(cache.with_suffix('.part'), 'wb') as out:
            shutil.copyfileobj(response, out)
        cache.with_suffix('.part').replace(cache)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != PYTHON_SHA256:
        cache.unlink()
        sys.exit('Python download failed the SHA-256 check. Run the build again.')
    with zipfile.ZipFile(cache) as archive:
        archive.extractall(runtime)
    (runtime / 'python312._pth').write_text(PTH, encoding='ascii')


def install_packages():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(site_packages),
                    '--platform', 'win_amd64', '--python-version', '3.12', '--implementation', 'cp',
                    '--abi', 'cp312', '--only-binary=:all:', '--no-compile', '--disable-pip-version-check',
                    '-r', str(root / 'requirements.txt')], check=True)
    for pattern in PRUNE_GLOBS:
        for path in site_packages.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    # torch needs the MSVC C++ runtime; ship it app-locally so a clean PC works.
    for name in MSVC_RUNTIME:
        source = next((p for p in (site_packages / 'PySide6' / name, Path(os.environ['SystemRoot'], 'System32', name))
                       if p.is_file()), None)
        if source and not (runtime / name).exists():
            shutil.copy2(source, runtime / name)


def copy_app():
    files = subprocess.run(['git', 'ls-files', 'apps', 'launch.pyw', 'README.md'], cwd=root, check=True,
                           capture_output=True, text=True, encoding='utf-8').stdout.splitlines()
    for name in files:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    build_launcher(bundle / 'ApexHighlights.exe')


def copy_tesseract(source):
    if not (source / 'tesseract.exe').is_file():
        sys.exit(f'tesseract.exe not found in {source}. Pass --tesseract-dir.')
    target = bundle / 'tools' / 'tesseract'
    (target / 'tessdata').mkdir(parents=True)
    for path in [source / 'tesseract.exe', *source.glob('*.dll')]:
        shutil.copy2(path, target)
    shutil.copy2(source / 'tessdata' / 'eng.traineddata', target / 'tessdata')
    for name in ('LICENSE', 'AUTHORS', 'README.md'):
        if (source / 'doc' / name).is_file():
            shutil.copy2(source / 'doc' / name, target / name)
    env = {k: v for k, v in os.environ.items() if k not in ('TESSDATA_PREFIX',)}
    languages = subprocess.run([str(target / 'tesseract.exe'), '--list-langs'], check=True,
                               capture_output=True, text=True, env=env).stdout
    if 'eng' not in languages.split():
        sys.exit('Bundled Tesseract cannot find its English data.')


def prepare_models_and_smoke_test():
    env = dict(os.environ, EASYOCR_MODULE_PATH=str(bundle / 'models'), PYTHONNOUSERSITE='1')
    python = str(runtime / 'python.exe')
    subprocess.run([python, str(root / 'prepare_models.py')], check=True, cwd=bundle, env=env)
    check = ('import sys; sys.path.insert(0, "apps/ApexHighlights"); '
             'import PySide6.QtMultimediaWidgets, cv2, torch, easyocr, imageio_ffmpeg, qt_app, damage_filter; '
             'print("tesseract:", damage_filter.tesseract_path()); print("ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe())')
    subprocess.run([python, '-c', check], check=True, cwd=bundle, env=env)


def make_zip():
    dist.mkdir(exist_ok=True)
    output = dist / 'ApexHighlights-portable.zip'
    partial = output.with_suffix('.part')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(build))
    partial.replace(output)
    hasher = hashlib.sha256()
    with output.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    output.with_suffix('.zip.sha256').write_text(
        f'{digest}  {output.name}\n', encoding='ascii')
    print(f'{output} ({output.stat().st_size / 2**20:.0f} MB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tesseract-dir', type=Path, default=Path('C:/Program Files/Tesseract-OCR'))
    parser.add_argument('--no-zip', action='store_true', help='build the folder only')
    options = parser.parse_args()

    if not shutil.which('g++'):
        sys.exit('MinGW-w64 g++ is required. Add its bin directory to PATH.')
    if bundle.resolve() != (root / 'build' / 'ApexHighlights').resolve():
        sys.exit('Unexpected bundle path; refusing to remove it.')
    if bundle.exists():
        shutil.rmtree(bundle)
    step(f'Python {PYTHON_VERSION} embeddable runtime')
    fetch_python()
    step('Packages')
    install_packages()
    step('App files')
    copy_app()
    step('Tesseract')
    copy_tesseract(options.tesseract_dir)
    step('OCR models and smoke test')
    prepare_models_and_smoke_test()
    # The smoke test runs the app modules; keep the bundle free of runtime output.
    shutil.rmtree(bundle / 'apps' / 'ApexHighlights' / 'data', ignore_errors=True)
    if not options.no_zip:
        step('Zip')
        make_zip()


if __name__ == '__main__':
    main()
