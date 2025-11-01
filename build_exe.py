"""
Skrypt do budowania wav2mp4.exe z PyInstaller
Zawiera wszystkie wymagane biblioteki
"""

import PyInstaller.__main__
import os
import sys

# Ścieżka do głównego pliku
main_file = 'main.py'

# Argumenty dla PyInstaller
args = [
    main_file,
    '--onefile',  # Jeden plik exe
    '--name=wav2mp4',  # Nazwa exe
    '--icon=NONE',  # Brak ikony (można dodać później)
    '--clean',  # Wyczyść cache przed buildem
    '--noconfirm',  # Nie pytaj o nadpisanie
    
    # Ukryte importy - tylko potrzebne biblioteki
    '--hidden-import=numpy',
    '--hidden-import=scipy.io.wavfile',
    '--hidden-import=scipy.signal',
    '--hidden-import=PIL.Image',
    '--hidden-import=PIL.ImageDraw',
    '--hidden-import=PIL.ImageFont',
    '--hidden-import=moviepy.editor',
    '--hidden-import=sklearn.cluster',
    '--hidden-import=imageio_ffmpeg',
    
    # Exclude niepotrzebnych pakietów (zmniejsza rozmiar exe)
    '--exclude-module=torch',
    '--exclude-module=tensorflow',
    '--exclude-module=matplotlib',
    '--exclude-module=pytest',
    '--exclude-module=IPython',
    '--exclude-module=jupyter',
    '--exclude-module=notebook',
    '--exclude-module=pandas',
    '--exclude-module=sympy',
    
    # Dodatkowe pliki danych (jeśli są potrzebne)
    # '--add-data=fonts;fonts',  # Jeśli masz własne fonty
    
    # Katalog wyjściowy
    '--distpath=dist',
    '--workpath=build',
    '--specpath=.',
    
    # Opcje konsoli
    '--console',  # Z oknem konsoli (można zmienić na --noconsole)
    
    # Optymalizacje
    '--optimize=2',  # Maksymalna optymalizacja Pythona
]

print("=" * 70)
print("WAV2MP4 VISION - PyInstaller Build Script")
print("=" * 70)
print()
print("Tworzenie pliku exe z wszystkimi bibliotekami...")
print()

# Uruchom PyInstaller
PyInstaller.__main__.run(args)

print()
print("=" * 70)
print("Build zakończony!")
print("=" * 70)
print()
print("Plik exe znajduje się w: dist/wav2mp4.exe")
print()
print("Test:")
print("  dist\\wav2mp4.exe --help")
print()
