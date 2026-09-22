"""Download OCR weights at installation time; do not commit them to Git."""
import easyocr

if __name__ == '__main__':
    print('Preparing Korean/English OCR models in the current user EasyOCR cache...')
    easyocr.Reader(['ko', 'en'], gpu=False, download_enabled=True, verbose=True)
    easyocr.Reader(['en'], gpu=False, detector=False, download_enabled=True, verbose=True)
    print('OCR model preparation complete.')
