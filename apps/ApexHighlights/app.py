"""Launch the Qt interface; processing remains in core.py."""
import sys
from qt_app import App,QApplication

if __name__=='__main__':
    application=QApplication(sys.argv)
    application.setApplicationName('Apex Highlights')
    window=App()
    window.show()
    sys.exit(application.exec())
