# WAV to MP4 Converter with Audio Visualizer

Aplikacja konwertuje pliki WAV do MP4 z wizualizacją audio - sinusoidy stereo lub equalizera.

## ✨ Funkcje

- 🌊 **Trzy sinusoidy stereo + wokal** - żółta (lewy), zielona (prawy), czerwona (wokal 300-3000Hz)
- 🎤 **Ekstrakcja wokalu** - automatyczne wyodrębnienie częstotliwości wokalnych z sygnału
- 📈 **Wysoka rozdzielczość** - domyślnie 500 punktów, grubsze linie (4px) dla lepszej jakości
- 💫 **Efekt reverb wizualny** - trailing/echo effect z zanikającym śladem
- 📝 **Tekst na wideo** - CAPS, prawy dolny róg, 2% margines, mniejszy font (Arial/Roboto)
- 🖼️ **Znak wodny** - dodaj logo/watermark z pozycjonowaniem w % (domyślnie 10%, 10%)
- 🖼️ **Tło z obrazka** - pojedynczy plik lub katalog z płynnym przechodzeniem (crossfade)
- ⚡ **Tryb testowy** - renderuj tylko X% pliku dla szybkich sprawdzeń
- 🎚️ **Regulowane opacity** - wizualizacja (0.9) i tekst (0.8) osobno
- 📦 **Tryb batch** - automatyczne przetwarzanie katalogów z zestawami WAV+obrazki
- 📋 **Zachowanie metadanych** - wszystkie metadane z WAV są zachowywane
- 🎨 **Niestandardowe kolory** - wybierz własne kolory dla każdego kanału
- 📊 **Dwa style** - sinusoidy (domyślnie) lub equalizera (styl 'bars')

## Instalacja

```bash
pip install -r requirements.txt
```

**Wymagane:** FFmpeg musi być zainstalowany w systemie
- Windows: Pobierz z https://ffmpeg.org/download.html i dodaj do PATH

## Użycie

### Podstawowe

```bash
# 3 sinusoidy: żółty (L), zielony (R), czerwony (wokal)
python main.py song.wav output.mp4

# Z tekstem (zawsze CAPS)
python main.py song.wav output.mp4 --text "My Song Title"

# Test pierwszych 10% (szybkie sprawdzenie)
python main.py song.wav test.mp4 --test-length 10
```

### Z tłem i znakiem wodnym

```bash
# Pojedynczy obrazek jako tło
python main.py song.wav output.mp4 --background photo.jpg

# Katalog z obrazkami + znak wodny
python main.py song.wav output.mp4 --background ./images/ --watermark logo.png

# Pozycjonowanie znaku wodnego (% od top-left)
python main.py song.wav output.mp4 --watermark logo.png --watermark-x 5 --watermark-y 5
```

### Zaawansowane

```bash
# 4K z niestandardowymi kolorami (pomarańczowy + niebieski)
python main.py song.wav output.mp4 --resolution 3840x2160 --left-color 255,100,0 --right-color 0,200,255

# 60 FPS z wysokim bitrate i niestandardową opacity
python main.py song.wav output.mp4 --fps 60 --audio-bitrate 320k --opacity 0.95

# Stary styl (equalizera) z wieloma paskami
python main.py song.wav output.mp4 --style bars --bars 128

# Test z katalogiem example-pic
python main.py song.wav output.mp4 --background example-pic
```

### Tryb Batch

Przetwarzaj wiele plików naraz! Struktura:

```
batch-folder/
├── song1/
│   ├── audio.wav
│   ├── photo1.jpg
│   └── photo2.jpg
├── song2/
│   ├── music.wav
│   └── background.png
└── song3/
    └── track.wav  (bez obrazków = czarne tło)
```

Uruchom batch:

```bash
python main.py batch-folder dummy.mp4 --batch
```

Aplikacja automatycznie:
- Znajdzie wszystkie podkatalogi
- Wyszuka plik WAV w każdym
- Użyje obrazków z katalogu jako tła (jeśli są)
- Zapisze MP4 obok pliku WAV

## Parametry

| Parametr | Opis | Domyślnie |
|----------|------|-----------|
| `input` | Plik WAV wejściowy | - |
| `output` | Plik MP4 wyjściowy | - |
| `--resolution` | Rozdzielczość (WIDTHxHEIGHT) | 1920x1080 |
| `--audio-bitrate` | Bitrate audio (192k, 256k, 320k) | 320k |
| `--fps` | Klatki na sekundę | 30 |
| `--bars` | Liczba punktów wizualizacji | 500 |
| `--text` | Tekst w prawym dolnym rogu (CAPS, 2% margines) | brak |
| `--text-opacity` | Przezroczystość tekstu (0.0-1.0) | 0.8 |
| `--watermark` | Ścieżka do pliku znaku wodnego (PNG/JPG) | brak |
| `--watermark-x` | Pozycja X znaku wodnego (% od lewej) | 10 |
| `--watermark-y` | Pozycja Y znaku wodnego (% od góry) | 10 |
| `--test-length` | Renderuj tylko X% pliku (test) | brak (100%) |
| `--background` | Ścieżka do obrazka/katalogu | brak (czarne tło) |
| `--style` | Styl: `waveform` lub `bars` | waveform |
| `--left-color` | Kolor lewego kanału (R,G,B) | 255,255,0 (żółty) |
| `--right-color` | Kolor prawego kanału (R,G,B) | 0,255,0 (zielony) |
| `--opacity` | Przezroczystość wizualizacji (0.0-1.0) | 0.9 |
| `--batch` | Tryb batch processing | wyłączony |

## Przykłady kolorów

- Czerwony: `255,0,0`
- Zielony: `0,255,0`
- Niebieski: `0,0,255`
- Żółty: `255,255,0`
- Cyjan: `0,255,255`
- Magenta: `255,0,255`
- Pomarańczowy: `255,165,0`
- Różowy: `255,105,180`
- Fioletowy: `128,0,128`
