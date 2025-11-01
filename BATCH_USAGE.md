# Instrukcja użycia trybu batch

## Struktura katalogów

Stwórz katalog główny (np. `my-batch`) z podkatalogami:

```
my-batch/
├── song1/
│   ├── audio.wav          ← plik WAV (wymagany)
│   ├── photo1.jpg         ← obrazki (opcjonalne)
│   └── photo2.png
├── song2/
│   ├── music.wav
│   └── background.jpg
└── song3/
    └── track.wav          ← bez obrazków = czarne tło
```

## Uruchomienie

```bash
python main.py my-batch dummy.mp4 --batch
```

Uwagi:
- Argument `dummy.mp4` jest ignorowany w trybie batch (wymagany przez parser)
- Każdy podkatalog musi mieć conajmniej 1 plik WAV
- Obrazki są opcjonalne (jeśli brak = czarne tło)
- Plik MP4 zostanie zapisany w tym samym podkatalogu co WAV
- Nazwa MP4 będzie taka sama jak WAV

## Dodatkowe parametry

Wszystkie parametry działają w trybie batch:

```bash
python main.py my-batch dummy.mp4 --batch --resolution 2560x1440 --fps 60 --opacity 0.85
```
