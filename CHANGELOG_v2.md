# 🎵 Nowe Funkcje v2.0

## ✨ Co nowego?

### 🎤 Ekstrakcja Wokalu (Czerwona Fala)
Automatyczna separacja częstotliwości wokalnych (300-3000Hz) i wyświetlanie jako trzeciej fali w kolorze czerwonym.

### 📈 Wyższa Rozdzielczość
- Zwiększono domyślną liczbę punktów z 200 → **500**
- Bardziej płynna i szczegółowa wizualizacja
- Możliwość ustawienia nawet 800+ punktów dla ultra jakości

### 📝 Tekst na Wideo
- Wyświetlanie tekstu w prawym dolnym rogu
- Font: Arial/Roboto (automatyczne wykrywanie)
- Margines: 1% od prawej i dołu
- Cień dla lepszej czytelności
- Regulowana przezroczystość (domyślnie 0.8)

## 🎨 Kolory Wizualizacji

```
🟡 ŻÓŁTY  = Lewy kanał (L)
🟢 ZIELONY = Prawy kanał (R)
🔴 CZERWONY = Wokal (300-3000Hz)
```

Wszystkie trzy fale są na środku ekranu, nałożone na siebie z efektem reverb/trailing.

## 📋 Przykłady Użycia

### Podstawowy
```bash
python main.py song.wav output.mp4
```
**Rezultat:** 3 sinusoidy na czarnym tle

### Z tekstem
```bash
python main.py song.wav output.mp4 --text "My Song - Artist Name"
```
**Rezultat:** Tekst w prawym dolnym rogu z opacity 0.8

### Pełna jakość z tłem
```bash
python main.py song.wav output.mp4 \
  --background my-images/ \
  --text "Track Title 2025" \
  --bars 800 \
  --fps 60 \
  --resolution 3840x2160 \
  --audio-bitrate 320k \
  --opacity 0.95 \
  --text-opacity 0.9
```

### Batch z tekstem
```bash
python main.py my-albums/ dummy.mp4 --batch --text "Album Name 2025"
```

## 🎛️ Nowe Parametry

| Parametr | Opis | Domyślnie |
|----------|------|-----------|
| `--bars` | Punkty wizualizacji | 500 (było 200) |
| `--text` | Tekst na wideo | brak |
| `--text-opacity` | Przezroczystość tekstu | 0.8 |

## 💡 Wskazówki

1. **Wysoka rozdzielczość**: Użyj `--bars 800` dla 4K
2. **Czytelny tekst**: Zwiększ `--text-opacity 1.0` na jasnych tłach
3. **Dyskretny tekst**: Zmniejsz `--text-opacity 0.5` na ciemnych tłach
4. **Bez wokalu**: Obecnie brak opcji wyłączenia (zawsze czerwona fala)

## 🎵 Jak działa ekstrakcja wokalu?

Filtr pasmowy Butterwortha 4. rzędu:
- **Pasmo**: 300Hz - 3000Hz
- **Źródło**: Uśredniony sygnał stereo (L+R)/2
- **Algorytm**: `scipy.signal.filtfilt` dla zerowej opóźnienia fazowego
- **Amplituda**: 80% normalnej dla lepszej proporcji wizualnej

## 🚀 Performance

- **500 punktów**: ~10-12 fps renderowania (zalecane)
- **800 punktów**: ~8-10 fps renderowania (4K)
- **200 punktów**: ~15-18 fps renderowania (szybsze, mniej szczegółów)

Czas renderowania: ~10-15 minut dla 3-minutowego utworu (Full HD, 30fps)
