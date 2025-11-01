#!/usr/bin/env python3
"""
WAV to MP4 Converter with Audio Visualizer
Konwertuje pliki WAV do MP4 z wizualnym equalizrem
"""

import argparse
import sys
import os
import glob
import numpy as np
from scipy import signal
from scipy.io import wavfile
from moviepy import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont
import wave
import tempfile
import shutil


class BackgroundManager:
    """Klasa do zarządzania tłem z obrazków"""
    
    def __init__(self, background_path, width, height, duration, crossfade_duration=2.0):
        """
        Inicjalizacja managera tła
        
        Args:
            background_path: Ścieżka do pliku obrazka lub katalogu z obrazkami
            width: Szerokość wideo
            height: Wysokość wideo
            duration: Całkowity czas trwania wideo
            crossfade_duration: Czas przejścia między obrazkami (sekundy)
        """
        self.width = width
        self.height = height
        self.duration = duration
        self.crossfade_duration = crossfade_duration
        self.images = []
        self.use_ken_burns = False  # Efekt Ken Burns dla pojedynczego obrazka
        
        if background_path and os.path.exists(background_path):
            if os.path.isfile(background_path):
                # Pojedynczy plik
                self.images = [self._load_and_resize(background_path)]
            elif os.path.isdir(background_path):
                # Katalog z obrazkami
                patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(os.path.join(background_path, pattern)))
                files.sort()
                
                if files:
                    self.images = [self._load_and_resize(f) for f in files]
        
        # Jeśli brak obrazków, użyj czarnego tła
        if not self.images:
            black = Image.new('RGB', (width, height), color=(0, 0, 0))
            self.images = [black]
        
        # Oblicz czas wyświetlania każdego obrazka
        if len(self.images) > 1:
            self.time_per_image = duration / len(self.images)
        else:
            self.time_per_image = duration
            # Dla pojedynczego obrazka ZAWSZE użyj efektu Ken Burns (domyślnie)
            if background_path and os.path.exists(background_path):
                self.use_ken_burns = True
                # Załaduj większy obrazek dla Ken Burns (+10% offset)
                if os.path.isfile(background_path):
                    self.ken_burns_img = self._load_ken_burns_image(background_path)
                elif os.path.isdir(background_path):
                    # Dla katalogu - znajdź pierwszy obrazek
                    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
                    for pattern in patterns:
                        files = glob.glob(os.path.join(background_path, pattern))
                        if files:
                            files.sort()
                            self.ken_burns_img = self._load_ken_burns_image(files[0])
                            break
    
    def _load_and_resize(self, path):
        """Wczytaj i przeskaluj obrazek do rozmiaru wideo"""
        img = Image.open(path).convert('RGB')
        
        # Zachowaj proporcje, wypełnij całe tło
        img_ratio = img.width / img.height
        target_ratio = self.width / self.height
        
        if img_ratio > target_ratio:
            # Obrazek szerszy - skaluj po wysokości
            new_height = self.height
            new_width = int(new_height * img_ratio)
        else:
            # Obrazek wyższy - skaluj po szerokości
            new_width = self.width
            new_height = int(new_width / img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Wytnij środek
        left = (new_width - self.width) // 2
        top = (new_height - self.height) // 2
        img = img.crop((left, top, left + self.width, top + self.height))
        
        return img
    
    def _load_ken_burns_image(self, path):
        """Wczytaj obrazek w większym rozmiarze dla efektu Ken Burns (zoom + pan)
        Używamy 2x upscale dla większej rozdzielczości = płynniejszy ruch bez skoków"""
        img = Image.open(path).convert('RGB')
        
        # KROK 1: Upscale do 200% dla większej rozdzielczości (więcej pikseli = płynniejszy ruch)
        upscaled_w = img.width * 2
        upscaled_h = img.height * 2
        img = img.resize((upscaled_w, upscaled_h), Image.Resampling.LANCZOS)
        
        # KROK 2: Skaluj aby wypełnić ekran w 2x + 10% offset dla płynnego przesuwania
        scale = 1.1
        target_width = int(self.width * 2 * scale)  # 2x bo obraz jest upscaled
        target_height = int(self.height * 2 * scale)
        
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        # Zawsze wypełnij cały ekran (cover, nie contain)
        if img_ratio > target_ratio:
            # Obraz szerszy - skaluj po wysokości
            new_height = target_height
            new_width = int(new_height * img_ratio)
        else:
            # Obraz wyższy - skaluj po szerokości
            new_width = target_width
            new_height = int(new_width / img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img
    
    def get_frame(self, t):
        """
        Pobierz klatkę tła dla czasu t z płynnym przejściem
        
        Args:
            t: Czas w sekundach
            
        Returns:
            PIL Image
        """
        if len(self.images) == 1:
            # Efekt Ken Burns dla pojedynczego obrazka
            if self.use_ken_burns:
                return self._apply_ken_burns(t)
            return self.images[0].copy()
        
        # Który obrazek powinien być wyświetlany
        image_index = int(t / self.time_per_image)
        image_index = min(image_index, len(self.images) - 1)
        
        # Czas lokalny w ramach aktualnego obrazka
        local_t = t - (image_index * self.time_per_image)
        
        current_img = self.images[image_index]
        
        # Crossfade na końcu obrazka
        if image_index < len(self.images) - 1:
            fade_start = self.time_per_image - self.crossfade_duration
            if local_t >= fade_start:
                # Oblicz alpha dla przejścia
                alpha = (local_t - fade_start) / self.crossfade_duration
                alpha = np.clip(alpha, 0, 1)
                
                next_img = self.images[image_index + 1]
                
                # Blend dwóch obrazków
                blended = Image.blend(current_img, next_img, alpha)
                return blended
        
        return current_img.copy()
    
    def _apply_ken_burns(self, t):
        """Zastosuj płynny efekt Ken Burns (zoom do wypełnienia +10% offset, potem płynne przesuwanie)
        Obraz jest w 2x rozdzielczości, więc cropujemy w 2x i skalujemy z powrotem"""
        if not hasattr(self, 'ken_burns_img'):
            return self.images[0].copy()
        
        # Progress od 0 do 1 przez cały czas trwania (z easing dla płynności)
        progress = t / self.duration if self.duration > 0 else 0
        progress = np.clip(progress, 0, 1)
        
        # Smooth easing function (ease-in-out) dla ultra płynnego ruchu
        # Używamy sinusoidalnej krzywej dla naturalnego przyspieszenia/zwalniania
        smooth_progress = (1 - np.cos(progress * np.pi)) / 2
        
        # Wymiary obrazka w 2x rozdzielczości (już przeskalowanego do 2x * 110%)
        img_width = self.ken_burns_img.width
        img_height = self.ken_burns_img.height
        
        # Rozmiary docelowe w 2x (bo obraz jest upscaled)
        target_w = self.width * 2
        target_h = self.height * 2
        
        # Maksymalny dostępny offset (10% z każdej strony w przestrzeni 2x)
        max_offset_x = img_width - target_w
        max_offset_y = img_height - target_h
        
        # Szybka animacja z płynnym ruchem dzięki 2x upscale
        # Używamy 70% dostępnego zakresu dla widocznego ruchu, ale upscale 2x daje płynność!
        # Przy 1920x1080 * 2 = 3840x2160 i 10% offset -> max_offset ≈ 384px -> 70% z tego = ~269px całkowity zakres
        # To da ~269 różnych pozycji pikseli = ultra płynny ruch bez skoków!
        # 480 klatek / 269px = ~1.8 klatki na pixel = oko nie widzi skoków, tylko płynny ruch
        offset_x = int(smooth_progress * max_offset_x * 0.70)
        offset_y = int(smooth_progress * max_offset_y * 0.70)
        
        # Ogranicz do maksymalnego dostępnego offsetu (zabezpieczenie)
        offset_x = min(offset_x, max_offset_x)
        offset_y = min(offset_y, max_offset_y)
        
        # Wytnij fragment obrazka w 2x rozdzielczości z płynnym przesunięciem
        left = offset_x
        top = offset_y
        right = left + target_w
        bottom = top + target_h
        
        # Zabezpieczenie przed wyjściem poza granice (nie powinno się zdarzyć)
        if right > img_width:
            right = img_width
            left = right - target_w
        if bottom > img_height:
            bottom = img_height
            top = bottom - target_h
        
        # Crop w 2x rozdzielczości
        cropped = self.ken_burns_img.crop((left, top, right, bottom))
        
        # Skaluj z powrotem do docelowej rozdzielczości z wysoką jakością (Lanczos)
        cropped = cropped.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Dodaj delikatny efekt rozgrzanego powietrza (heat distortion)
        cropped = self._apply_heat_distortion(cropped, t)
        
        return cropped
    
    def _apply_heat_distortion(self, img, t):
        """Zastosuj delikatny efekt rozgrzanego powietrza (heat haze) - płynna animacja"""
        # Konwertuj do numpy array
        img_array = np.array(img, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Parametry fali (bardzo subtelne) - płynna animacja co klatkę
        frequency = 0.4  # Częstotliwość fali (niższa = większe fale)
        amplitude = 0.5  # Amplituda przesunięcia (bardzo mała - jeszcze subtelniej)
        speed = 0.1  # Bardzo wolna animacja dla płynności (wolniej)
        
        # Utwórz siatkę współrzędnych
        x = np.arange(width)
        y = np.arange(height)
        
        # Fala sinusoidalna zmieniająca się w czasie (pionowo)
        # Różne częstotliwości dla x i y dla naturalnego efektu
        # Płynna animacja co klatkę dla subtelnego efektu
        wave_y = amplitude * np.sin(2 * np.pi * frequency * (y / height) + t * speed)
        wave_x = amplitude * 0.7 * np.sin(2 * np.pi * frequency * 1.3 * (x / width) + t * speed * 0.8)
        
        # Utwórz macierze przesunięć
        shift_y = np.tile(wave_y.reshape(-1, 1), (1, width))
        shift_x = np.tile(wave_x, (height, 1))
        
        # Nowe współrzędne z przesunięciem
        y_coords, x_coords = np.meshgrid(y, x, indexing='ij')
        y_new = np.clip(y_coords + shift_y, 0, height - 1).astype(np.int32)
        x_new = np.clip(x_coords + shift_x, 0, width - 1).astype(np.int32)
        
        # Zastosuj przesunięcie
        distorted = img_array[y_new, x_new]
        
        # Konwertuj z powrotem do uint8
        distorted = np.clip(distorted, 0, 255).astype(np.uint8)
        
        return Image.fromarray(distorted)


class AudioVisualizer:
    """Klasa do analizy audio i generowania wizualizacji fal dźwiękowych"""
    
    def __init__(self, wav_file, num_bars=64):
        """
        Inicjalizacja wizualizatora
        
        Args:
            wav_file: Ścieżka do pliku WAV
            num_bars: Liczba próbek dla fali (używane dla sinusoidy)
        """
        self.wav_file = wav_file
        self.num_bars = num_bars
        
        # Wczytaj plik audio
        self.sample_rate, self.audio_data = wavfile.read(wav_file)
        
        # Zapisz stereo/mono info
        self.is_stereo = len(self.audio_data.shape) > 1
        
        if self.is_stereo:
            # Rozdziel kanały
            self.left_channel = self.audio_data[:, 0]
            self.right_channel = self.audio_data[:, 1]
        else:
            # Mono - użyj tego samego dla obu kanałów
            self.left_channel = self.audio_data
            self.right_channel = self.audio_data
        
        # Normalizuj do float -1.0 do 1.0
        if self.left_channel.dtype == np.int16:
            self.left_channel = self.left_channel.astype(np.float32) / 32768.0
            self.right_channel = self.right_channel.astype(np.float32) / 32768.0
        elif self.left_channel.dtype == np.int32:
            self.left_channel = self.left_channel.astype(np.float32) / 2147483648.0
            self.right_channel = self.right_channel.astype(np.float32) / 2147483648.0
        
        self.duration = len(self.left_channel) / self.sample_rate
        
    def get_frequency_spectrum(self, start_time, duration=0.05):
        """
        Oblicz widmo częstotliwości dla danego momentu
        
        Args:
            start_time: Czas początkowy w sekundach
            duration: Długość okna analizy w sekundach
            
        Returns:
            Array z amplitudami dla każdego paska equalizera
        """
        start_sample = int(start_time * self.sample_rate)
        window_samples = int(duration * self.sample_rate)
        end_sample = min(start_sample + window_samples, len(self.audio_data))
        
        if start_sample >= len(self.audio_data):
            return np.zeros(self.num_bars)
        
        # Pobierz fragment audio
        audio_chunk = self.audio_data[start_sample:end_sample]
        
        if len(audio_chunk) == 0:
            return np.zeros(self.num_bars)
        
        # Zastosuj okno Hanninga dla lepszej analizy
        window = np.hanning(len(audio_chunk))
        audio_chunk = audio_chunk * window
        
        # FFT - Fast Fourier Transform
        fft = np.fft.fft(audio_chunk)
        fft_magnitude = np.abs(fft[:len(fft)//2])
        
        # Logarytmiczna skala częstotliwości (bardziej naturalna dla ucha)
        freqs = np.fft.fftfreq(len(audio_chunk), 1/self.sample_rate)
        freqs = freqs[:len(freqs)//2]
        
        # Podziel częstotliwości na pasy (logarytmicznie)
        min_freq = 20  # Hz
        max_freq = min(20000, self.sample_rate / 2)  # Hz
        
        freq_bands = np.logspace(np.log10(min_freq), np.log10(max_freq), self.num_bars + 1)
        
        bar_heights = np.zeros(self.num_bars)
        
        for i in range(self.num_bars):
            # Znajdź indeksy dla danego pasma
            mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i + 1])
            if np.any(mask):
                # Średnia amplituda w paśmie
                bar_heights[i] = np.mean(fft_magnitude[mask])
        
        # Normalizuj i zastosuj skalę logarytmiczną dla lepszego efektu wizualnego
        bar_heights = np.log10(bar_heights + 1)
        max_height = np.max(bar_heights) if np.max(bar_heights) > 0 else 1
        bar_heights = bar_heights / max_height
        
        # Wygładź (aby uniknąć zbyt gwałtownych zmian)
        bar_heights = np.clip(bar_heights, 0, 1)
        
        return bar_heights
    
    def get_waveform_data(self, start_time, window_duration=0.05, num_points=200):
        """
        Pobierz dane fali dźwiękowej dla obu kanałów
        
        Args:
            start_time: Czas początkowy w sekundach
            window_duration: Długość okna w sekundach
            num_points: Liczba punktów do zwrócenia
            
        Returns:
            Tuple (left_wave, right_wave) - arrays z amplitudami
        """
        start_sample = int(start_time * self.sample_rate)
        window_samples = int(window_duration * self.sample_rate)
        end_sample = min(start_sample + window_samples, len(self.left_channel))
        
        if start_sample >= len(self.left_channel):
            return np.zeros(num_points), np.zeros(num_points)
        
        # Pobierz fragmenty
        left_chunk = self.left_channel[start_sample:end_sample]
        right_chunk = self.right_channel[start_sample:end_sample]
        
        if len(left_chunk) == 0:
            return np.zeros(num_points), np.zeros(num_points)
        
        # Resample do num_points
        if len(left_chunk) > num_points:
            # Downsampling
            indices = np.linspace(0, len(left_chunk) - 1, num_points).astype(int)
            left_wave = left_chunk[indices]
            right_wave = right_chunk[indices]
        else:
            # Interpolacja jeśli za mało próbek
            x_old = np.arange(len(left_chunk))
            x_new = np.linspace(0, len(left_chunk) - 1, num_points)
            left_wave = np.interp(x_new, x_old, left_chunk)
            right_wave = np.interp(x_new, x_old, right_chunk)
        
        return left_wave, right_wave
    
    def extract_vocal_frequencies(self, start_time, window_duration=0.05, num_points=200):
        """
        Ekstraktuj częstotliwości wokalne (300Hz-3000Hz) z sygnału stereo
        
        Args:
            start_time: Czas początkowy w sekundach
            window_duration: Długość okna w sekundach
            num_points: Liczba punktów do zwrócenia
            
        Returns:
            Array z amplitudami pasma wokalnego
        """
        start_sample = int(start_time * self.sample_rate)
        window_samples = int(window_duration * self.sample_rate)
        end_sample = min(start_sample + window_samples, len(self.left_channel))
        
        if start_sample >= len(self.left_channel):
            return np.zeros(num_points)
        
        # Pobierz fragmenty i uśrednij stereo
        left_chunk = self.left_channel[start_sample:end_sample]
        right_chunk = self.right_channel[start_sample:end_sample]
        mono_chunk = (left_chunk + right_chunk) / 2
        
        if len(mono_chunk) == 0:
            return np.zeros(num_points)
        
        # Zastosuj filtr pasmowy dla częstotliwości wokalnych (300Hz-3000Hz)
        nyquist = self.sample_rate / 2
        low_freq = 300 / nyquist
        high_freq = min(3000 / nyquist, 0.99)
        
        # Projektuj filtr Butterwortha
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [low_freq, high_freq], btype='band')
        
        try:
            vocal_chunk = filtfilt(b, a, mono_chunk)
        except:
            # Jeśli filtrowanie się nie uda, użyj surowego sygnału
            vocal_chunk = mono_chunk
        
        # Resample do num_points
        if len(vocal_chunk) > num_points:
            indices = np.linspace(0, len(vocal_chunk) - 1, num_points).astype(int)
            vocal_wave = vocal_chunk[indices]
        else:
            x_old = np.arange(len(vocal_chunk))
            x_new = np.linspace(0, len(vocal_chunk) - 1, num_points)
            vocal_wave = np.interp(x_new, x_old, vocal_chunk)
        
        return vocal_wave


class VideoGenerator:
    """Klasa do generowania wideo z wizualizacją"""
    
    def __init__(self, width, height, fps=30, bars=64, waveform_style='waveform', 
                 left_color=(0, 100, 255), right_color=(0, 255, 100), opacity=0.9,
                 vocal_color=(255, 0, 100), text=None, text_opacity=0.8,
                 watermark=None, watermark_x=10, watermark_y=10, add_flares=True,
                 flare_duration=500, screen_flash_intensity=0.3):
        """
        Inicjalizacja generatora wideo
        
        Args:
            width: Szerokość wideo
            height: Wysokość wideo
            fps: Klatki na sekundę
            bars: Liczba pasków equalizera (lub punktów dla waveform)
            waveform_style: 'bars' dla equalizera, 'waveform' dla sinusoid
            left_color: Kolor lewego kanału (R, G, B)
            right_color: Kolor prawego kanału (R, G, B)
            vocal_color: Kolor wokalu (R, G, B)
            opacity: Przezroczystość wizualizacji (0.0-1.0)
            text: Tekst do wyświetlenia (None = brak)
            text_opacity: Przezroczystość tekstu (0.0-1.0)
            watermark: Ścieżka do pliku znaku wodnego (None = brak)
            watermark_x: Pozycja X znaku wodnego w % (0-100)
            watermark_y: Pozycja Y znaku wodnego w % (0-100)
            add_flares: Dodaj flary na szczytach amplitudy (True/False)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.bars = bars
        self.waveform_style = waveform_style
        self.left_color = left_color
        self.right_color = right_color
        self.vocal_color = vocal_color
        self.opacity = opacity
        self.text = text
        self.text_opacity = text_opacity
        self.watermark = watermark
        self.watermark_x = watermark_x
        self.watermark_y = watermark_y
        self.add_flares = add_flares
        
        # Historia dla efektu reverb (trailing)
        self.wave_history = []
        
        # Historia flar (aktywne flary z czasem życia)
        self.active_flares = []  # Lista: (x, y, color, birth_time, is_record)
        self.flare_lifetime = flare_duration / 1000.0  # Konwersja ms na sekundy
        self.current_time = 0  # Aktualny czas w sekundach
        
        # System rekordów amplitudy (resetuje się co kilka sekund)
        self.amplitude_record = 0.0  # Maksymalna amplituda od ostatniego resetu
        self.record_reset_interval = 3.0  # Reset co 3 sekundy
        self.last_record_reset = 0.0  # Czas ostatniego resetu
        
        # Absolutny rekord amplitudy (nigdy się nie resetuje)
        self.absolute_record = 0.0  # Najwyższa amplituda w całym utworze
        self.screen_flash_intensity = screen_flash_intensity  # Intensywność flasha (0.0-1.0)
        self.active_flashes = []  # Lista: (birth_time, intensity, x, y) - flashe rozchodzące się od punktu
        self.flash_duration = 0.3  # 300ms trwania flasha (dłużej dla efektu rozchodzenia)
        
        # Załaduj font
        self.font = self._load_font()
        
        # Załaduj watermark jeśli jest podany
        self.watermark_img = None
        if watermark and os.path.exists(watermark):
            self.watermark_img = self._load_watermark(watermark)
        
        # Kolory gradientu (od niebieskiego przez zielony do czerwonego)
        if waveform_style == 'bars':
            self.colors = self._generate_gradient_colors()
    
    def _load_font(self):
        """Załaduj font Arial lub Roboto"""
        font_size = int(self.height * 0.015)  # 1.5% wysokości ekranu (50% mniejszy)
        
        # Spróbuj różne fonty
        font_names = [
            'arial.ttf',
            'Arial.ttf',
            'roboto.ttf',
            'Roboto-Regular.ttf',
            'segoeui.ttf',  # Windows fallback
        ]
        
        for font_name in font_names:
            try:
                # Spróbuj załadować z systemowych czcionek Windows
                font_path = f"C:\\Windows\\Fonts\\{font_name}"
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, font_size)
            except:
                continue
        
        # Fallback do domyślnego fontu
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except:
            return ImageFont.load_default()
    
    def _load_watermark(self, watermark_path):
        """Załaduj i przeskaluj watermark"""
        try:
            watermark = Image.open(watermark_path).convert('RGBA')
            
            # Przeskaluj watermark do max 15% szerokości ekranu
            max_width = int(self.width * 0.15)
            ratio = max_width / watermark.width
            new_width = max_width
            new_height = int(watermark.height * ratio)
            
            watermark = watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return watermark
        except Exception as e:
            print(f"⚠️ Nie można załadować watermark: {e}")
            return None
    
    def _draw_watermark(self, img):
        """Rysuj watermark na obrazie"""
        if self.watermark_img is None:
            return
        
        # Oblicz pozycję w pikselach na podstawie % od lewej górnej
        x = int((self.watermark_x / 100) * self.width)
        y = int((self.watermark_y / 100) * self.height)
        
        # Nałóż watermark
        img.paste(self.watermark_img, (x, y), self.watermark_img)
        
    def _generate_gradient_colors(self):
        """Generuj gradient kolorów dla pasków"""
        colors = []
        for i in range(self.bars):
            # Gradient: niebieski -> cyan -> zielony -> żółty -> czerwony
            ratio = i / self.bars
            
            if ratio < 0.25:
                # Niebieski do cyan
                r = 0
                g = int((ratio / 0.25) * 255)
                b = 255
            elif ratio < 0.5:
                # Cyan do zielonego
                r = 0
                g = 255
                b = int((1 - (ratio - 0.25) / 0.25) * 255)
            elif ratio < 0.75:
                # Zielony do żółtego
                r = int(((ratio - 0.5) / 0.25) * 255)
                g = 255
                b = 0
            else:
                # Żółty do czerwonego
                r = 255
                g = int((1 - (ratio - 0.75) / 0.25) * 255)
                b = 0
            
            colors.append((r, g, b))
        
        return colors
    
    def create_frame(self, bar_heights=None, smoothed_heights=None, 
                    left_wave=None, right_wave=None, vocal_wave=None, background=None, time=0):
        """
        Utwórz pojedynczą klatkę z wizualizacją
        
        Args:
            bar_heights: Array z wysokościami pasków (0-1) - dla stylu 'bars'
            smoothed_heights: Poprzednie wysokości dla wygładzania
            left_wave: Array z amplitudami lewego kanału - dla stylu 'waveform'
            right_wave: Array z amplitudami prawego kanału - dla stylu 'waveform'
            vocal_wave: Array z amplitudami wokalu - dla stylu 'waveform'
            background: PIL Image z tłem (opcjonalne)
            time: Aktualny czas w sekundach (dla animacji flar)
            
        Returns:
            numpy array z wizualizacją
        """
        # Zapisz aktualny czas dla animacji flar
        self.current_time = time
        
        # Użyj tła lub utwórz czarne
        if background is not None:
            img = background.copy()
        else:
            img = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        
        # Utwórz warstwę z wizualizacją
        overlay = Image.new('RGBA', (self.width, self.height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        if self.waveform_style == 'waveform' and left_wave is not None and right_wave is not None:
            # Rysuj sinusoidy
            self._draw_waveforms(draw, left_wave, right_wave, vocal_wave)
        elif self.waveform_style == 'bars' and bar_heights is not None:
            # Rysuj equalizera (stary sposób)
            self._draw_bars(draw, bar_heights, smoothed_heights)
        
        # NIE stosuj blur - ostre linie dla lepszej jakości
        # overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Nałóż wizualizację na tło
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        
        # Dodaj flash na cały ekran dla absolutnych rekordów (przed tekstem)
        if self.screen_flash_intensity > 0:
            self._draw_screen_flash(img)
        
        # Dodaj tekst jeśli jest ustawiony
        if self.text:
            self._draw_text(img)
        
        # Dodaj watermark jeśli jest ustawiony
        if self.watermark:
            self._draw_watermark(img)
        
        img = img.convert('RGB')
        
        return np.array(img)
    
    def _draw_screen_flash(self, img):
        """Rysuj flash rozchodzący się od miejsca absolutnego rekordu amplitudy (jak ripple)"""
        # Usuń wygasłe flashe
        self.active_flashes = [
            flash for flash in self.active_flashes
            if (self.current_time - flash[0]) < self.flash_duration
        ]
        
        # Rysuj wszystkie aktywne flashe jako ripple wysokiej rozdzielczości
        draw = ImageDraw.Draw(img)
        for birth_time, intensity, x, y in self.active_flashes:
            age = self.current_time - birth_time
            progress = age / self.flash_duration  # 0 do 1
            
            # Flash rozchodzi się jak ripple - wiele cienkich koncentrycznych okręgów
            max_distance = max(self.width, self.height) * 1.5
            
            # Wysoka rozdzielczość - więcej cieńszych fal (15-20 fal)
            num_ripples = 18
            for i in range(num_ripples):
                # Każda fala zaczyna się z mniejszym opóźnieniem (bardziej gęste)
                wave_delay = i * 0.05  # 5% opóźnienia między falami (było 15%)
                wave_progress = (progress - wave_delay) / (1.0 - wave_delay)
                
                if wave_progress > 0 and wave_progress < 1:
                    # Promień dla tej fali
                    wave_radius = wave_progress * max_distance
                    
                    # Dynamiczne opacity - nieliniowe zanikanie (płynniejsze)
                    # Użyj krzywej ease-out dla naturalnego zanikania
                    fade_curve = 1 - (wave_progress ** 1.5)  # Szybsze zanikanie na końcu
                    
                    # Dodatkowy gradient dla odległych fal
                    distance_fade = 1.0 - (i / num_ripples) * 0.7  # Ostatnie fale: 30% intensywności
                    
                    # Połącz intensity, krzywą zanikania i gradient odległości
                    current_opacity = int(255 * intensity * fade_curve * distance_fade)
                    
                    if current_opacity > 8 and wave_radius > 3:
                        # Cienkie pierścienie (stała szerokość 2-3px dla subtelności)
                        ring_width = 2 if wave_progress > 0.3 else 3
                        
                        # Kolor biały z dynamicznym opacity
                        color_ring = (255, 255, 255, current_opacity)
                        
                        # Rysuj cienki pierścień (outline)
                        draw.ellipse(
                            [x - wave_radius, y - wave_radius,
                             x + wave_radius, y + wave_radius],
                            outline=color_ring,
                            width=ring_width
                        )
    
    def _draw_text(self, img):
        """Rysuj tekst w prawym dolnym rogu"""
        draw = ImageDraw.Draw(img)
        
        # Konwertuj tekst na wielkie litery (CAPS)
        text_caps = self.text.upper()
        
        # Oblicz pozycję (2% marginesu)
        margin_x = int(self.width * 0.02)
        margin_y = int(self.height * 0.02)
        
        # Pobierz rozmiar tekstu
        try:
            bbox = draw.textbbox((0, 0), text_caps, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback dla starszych wersji Pillow
            text_width, text_height = draw.textsize(text_caps, font=self.font)
        
        # Pozycja: prawy dolny róg z marginesem
        x = self.width - text_width - margin_x
        y = self.height - text_height - margin_y
        
        # Rysuj tekst z cieniem dla lepszej czytelności
        shadow_offset = 2
        shadow_color = (0, 0, 0, int(255 * self.text_opacity))
        text_color = (255, 255, 255, int(255 * self.text_opacity))
        
        # Cień
        draw.text((x + shadow_offset, y + shadow_offset), text_caps, 
                 font=self.font, fill=shadow_color)
        # Tekst
        draw.text((x, y), text_caps, font=self.font, fill=text_color)
    
    def _draw_waveforms(self, draw, left_wave, right_wave, vocal_wave=None):
        """
        Rysuj trzy sinusoidy (żółta dla lewego, zielona dla prawego, czerwona dla wokalu) na środku ekranu z efektem reverb
        
        Args:
            draw: ImageDraw object
            left_wave: Array z amplitudami lewego kanału
            right_wave: Array z amplitudami prawego kanału
            vocal_wave: Array z amplitudami wokalu (opcjonalne)
        """
        # Parametry
        center_y = self.height / 2
        amplitude_scale = self.height * 0.35  # 35% wysokości dla amplitudy
        line_width = 1  # Ultra cienka linia - neon
        vocal_line_width = 1  # Wokal też 1px (neon)
        glow_width = 8  # Szerokość glow (blur)
        
        # Lewy kanał - żółty (na środku)
        points_left = []
        for i, amp in enumerate(left_wave):
            x = (i / len(left_wave)) * self.width
            y = center_y + (amp * amplitude_scale)
            points_left.append((x, y))
        
        # Prawy kanał - zielony (na środku - na sobie)
        points_right = []
        for i, amp in enumerate(right_wave):
            x = (i / len(right_wave)) * self.width
            y = center_y + (amp * amplitude_scale)
            points_right.append((x, y))
        
        # Wokal - czerwony (na środku, jeśli jest)
        points_vocal = []
        if vocal_wave is not None:
            for i, amp in enumerate(vocal_wave):
                x = (i / len(vocal_wave)) * self.width
                y = center_y + (amp * amplitude_scale * 0.8)  # Trochę mniejsza amplituda
                points_vocal.append((x, y))
        
        # Dodaj do historii dla efektu reverb
        self.wave_history.append((points_left, points_right, points_vocal))
        
        # Zachowaj tylko ostatnie N klatek dla efektu trailing
        max_history = 5
        if len(self.wave_history) > max_history:
            self.wave_history.pop(0)
        
        # Rysuj trailing (starsze fale z mniejszą opacity)
        for idx, history_item in enumerate(self.wave_history[:-1]):
            # Rozpakowuj historię (może być 2 lub 3 elementy)
            if len(history_item) == 3:
                old_left, old_right, old_vocal = history_item
            else:
                old_left, old_right = history_item
                old_vocal = []
            
            # Oblicz opacity dla starszych klatek (efekt zanikania)
            age_factor = (idx + 1) / len(self.wave_history)
            trail_opacity = 0.4 * age_factor * 0.5  # Reverb z opacity 0.4
            
            # Rysuj trailing lewego kanału z glow
            if len(old_left) > 1:
                for i in range(len(old_left) - 1):
                    # Glow (blur effect) - rysuj grubszą linię z mniejszą opacity
                    glow_opacity = int(255 * trail_opacity * 0.3)
                    color_glow = self.left_color + (glow_opacity,)
                    draw.line([old_left[i], old_left[i + 1]], 
                             fill=color_glow, width=glow_width)
                    # Ostra linia 1px na wierzchu
                    color_with_alpha = self.left_color + (int(255 * trail_opacity),)
                    draw.line([old_left[i], old_left[i + 1]], 
                             fill=color_with_alpha, width=line_width)
            
            # Rysuj trailing prawego kanału z glow
            if len(old_right) > 1:
                for i in range(len(old_right) - 1):
                    # Glow (blur effect)
                    glow_opacity = int(255 * trail_opacity * 0.3)
                    color_glow = self.right_color + (glow_opacity,)
                    draw.line([old_right[i], old_right[i + 1]], 
                             fill=color_glow, width=glow_width)
                    # Ostra linia 1px na wierzchu
                    color_with_alpha = self.right_color + (int(255 * trail_opacity),)
                    draw.line([old_right[i], old_right[i + 1]], 
                             fill=color_with_alpha, width=line_width)
            
            # Rysuj trailing wokalu z glow
            if len(old_vocal) > 1:
                for i in range(len(old_vocal) - 1):
                    # Glow (blur effect)
                    glow_opacity = int(255 * trail_opacity * 0.3)
                    color_glow = self.vocal_color + (glow_opacity,)
                    draw.line([old_vocal[i], old_vocal[i + 1]], 
                             fill=color_glow, width=glow_width)
                    # Ostra linia 1px na wierzchu
                    color_with_alpha = self.vocal_color + (int(255 * trail_opacity),)
                    draw.line([old_vocal[i], old_vocal[i + 1]], 
                             fill=color_with_alpha, width=line_width)
        
        # Rysuj aktualną falę lewego kanału (pełna opacity) z glow
        if len(points_left) > 1:
            for i in range(len(points_left) - 1):
                # Glow (blur effect) - grubsza linia z mniejszą opacity
                glow_opacity = int(255 * self.opacity * 0.4)
                color_glow = self.left_color + (glow_opacity,)
                draw.line([points_left[i], points_left[i + 1]], 
                         fill=color_glow, width=glow_width)
                # Ostra linia 1px na wierzchu (neon)
                color_with_alpha = self.left_color + (int(255 * self.opacity),)
                draw.line([points_left[i], points_left[i + 1]], 
                         fill=color_with_alpha, width=line_width)
        
        # Rysuj aktualną falę prawego kanału z glow
        if len(points_right) > 1:
            for i in range(len(points_right) - 1):
                # Glow (blur effect)
                glow_opacity = int(255 * self.opacity * 0.4)
                color_glow = self.right_color + (glow_opacity,)
                draw.line([points_right[i], points_right[i + 1]], 
                         fill=color_glow, width=glow_width)
                # Ostra linia 1px na wierzchu (neon)
                color_with_alpha = self.right_color + (int(255 * self.opacity),)
                draw.line([points_right[i], points_right[i + 1]], 
                         fill=color_with_alpha, width=line_width)
        
        # Rysuj aktualną falę wokalu (na wierzchu) z glow - czerwony neon
        if len(points_vocal) > 1:
            for i in range(len(points_vocal) - 1):
                # Glow (blur effect) - mocniejszy dla wokalu
                glow_opacity = int(255 * self.opacity * 0.5)
                color_glow = self.vocal_color + (glow_opacity,)
                draw.line([points_vocal[i], points_vocal[i + 1]], 
                         fill=color_glow, width=glow_width)
                # Ostra linia 1px na wierzchu (czerwony neon)
                color_with_alpha = self.vocal_color + (int(255 * self.opacity),)
                draw.line([points_vocal[i], points_vocal[i + 1]], 
                         fill=color_with_alpha, width=vocal_line_width)
        
        # Rysuj flary na szczytach amplitudy (opcjonalnie)
        if self.add_flares:
            self._draw_flares(draw, points_left, points_right, points_vocal, center_y)
    
    def _draw_flares(self, draw, points_left, points_right, points_vocal, center_y):
        """Rysuj animowane flary (ripple effect) na szczytach amplitudy"""
        
        # Znajdź lokalne maksima i dodaj nowe flary
        self._detect_and_add_flares(points_left, center_y)
        self._detect_and_add_flares(points_right, center_y)
        if points_vocal:
            self._detect_and_add_flares(points_vocal, center_y, is_vocal=True)
        
        # Usuń wygasłe flary (starsze niż lifetime)
        self.active_flares = [
            flare for flare in self.active_flares 
            if (self.current_time - flare[3]) < self.flare_lifetime
        ]
        
        # Rysuj wszystkie aktywne flary z animacją ripple
        for flare_data in self.active_flares:
            # Rozpakowuj dane flary (obsługuj stary i nowy format)
            if len(flare_data) == 5:
                x, y, color, birth_time, is_record = flare_data
            else:
                x, y, color, birth_time = flare_data
                is_record = False
            
            age = self.current_time - birth_time
            progress = age / self.flare_lifetime  # 0 do 1
            
            # Parametry intensywności dla rekordów amplitudy
            if is_record:
                max_radius = 35  # Większy promień dla rekordów
                base_opacity = 250  # Bardziej intensywny
                line_width_base = 5  # Grubsza linia
                inner_glow_duration = 0.5  # Dłuższy blask
            else:
                max_radius = 20  # Normalny promień
                base_opacity = 200
                line_width_base = 3
                inner_glow_duration = 0.3
            
            # Efekt ripple: okrąg powiększa się i zanika
            current_radius = progress * max_radius
            
            # Opacity zanika liniowo
            current_opacity = int(base_opacity * (1 - progress))
            
            # Rysuj ripple jako rozszerzający się okrąg
            if current_radius > 1:
                # Grubsza linia na początku, cieńsza na końcu
                line_width = max(1, int(line_width_base * (1 - progress)))
                
                color_with_alpha = color + (current_opacity,)
                
                # Rysuj okrąg (outline)
                draw.ellipse(
                    [x - current_radius, y - current_radius, 
                     x + current_radius, y + current_radius],
                    outline=color_with_alpha,
                    width=line_width
                )
                
                # Dodaj wewnętrzny blask (mniejszy okrąg)
                if progress < inner_glow_duration:
                    inner_radius = current_radius * 0.5
                    inner_opacity = int(current_opacity * 0.6)
                    inner_color = color + (inner_opacity,)
                    draw.ellipse(
                        [x - inner_radius, y - inner_radius,
                         x + inner_radius, y + inner_radius],
                        fill=inner_color
                    )
                    
                    # Dla rekordów - dodaj jeszcze intensywniejszy środek
                    if is_record and progress < 0.2:
                        core_radius = current_radius * 0.2
                        core_opacity = int(255 * (1 - progress / 0.2))
                        # Biały środek dla maksymalnej jasności
                        core_color = (255, 255, 255, core_opacity)
                        draw.ellipse(
                            [x - core_radius, y - core_radius,
                             x + core_radius, y + core_radius],
                            fill=core_color
                        )
    
    def _detect_and_add_flares(self, points, center_y, is_vocal=False):
        """Wykryj lokalne maksima i dodaj nowe flary"""
        if not points or len(points) < 3:
            return
        
        # Reset rekordów amplitudy co kilka sekund
        if self.current_time - self.last_record_reset > self.record_reset_interval:
            self.amplitude_record = 0.0
            self.last_record_reset = self.current_time
        
        for i in range(1, len(points) - 1):
            x, y = points[i]
            y_prev = points[i-1][1]
            y_next = points[i+1][1]
            
            # Szukaj lokalnych maksimów (szczytów)
            distance_from_center = abs(y - center_y)
            threshold = 15 if is_vocal else 25
            
            if distance_from_center > threshold:
                if (y < y_prev and y < y_next) or (y > y_prev and y > y_next):
                    # Sprawdź czy flara już nie istnieje w tym miejscu (unikaj duplikatów)
                    exists = any(
                        abs(flare[0] - x) < 10 and abs(flare[1] - y) < 10
                        for flare in self.active_flares
                    )
                    
                    if not exists:
                        # Sprawdź czy to nowy rekord amplitudy w oknie 3s
                        is_record = distance_from_center > self.amplitude_record
                        if is_record:
                            self.amplitude_record = distance_from_center
                        
                        # Sprawdź czy to absolutny rekord (najwyższy ever)
                        if distance_from_center > self.absolute_record:
                            self.absolute_record = distance_from_center
                            # Dodaj flash rozchodzący się od miejsca rekordu
                            if self.screen_flash_intensity > 0:
                                # Intensywność zależy od tego jak duży jest skok
                                flash_intensity = self.screen_flash_intensity
                                # Zapisz pozycję (x, y) gdzie wystąpił rekord
                                self.active_flashes.append((self.current_time, flash_intensity, x, y))
                        
                        # Kolor flary zależy od pozycji (częstotliwości)
                        if is_vocal:
                            flare_color = (255, 100, 50)  # Wokal - czerwony/pomarańczowy
                        else:
                            ratio = i / len(points)
                            flare_color = self._get_flare_color(ratio)
                        
                        # Dodaj nową flarę z informacją czy to rekord
                        self.active_flares.append((x, y, flare_color, self.current_time, is_record))
    
    def _get_flare_color(self, ratio):
        """Pobierz kolor flary na podstawie pozycji (częstotliwości)"""
        # Gradient kolorów: niski -> średni -> wysoki
        # Niebieski (bas) -> Cyan -> Zielony -> Żółty -> Pomarańczowy -> Czerwony (wysoki)
        
        if ratio < 0.2:
            # Niskie częstotliwości - niebieski/cyan
            r = int(ratio / 0.2 * 100)
            g = int(ratio / 0.2 * 200)
            b = 255
        elif ratio < 0.4:
            # Średnie-niskie - cyan/zielony
            local_ratio = (ratio - 0.2) / 0.2
            r = 0
            g = 200 + int(local_ratio * 55)
            b = 255 - int(local_ratio * 155)
        elif ratio < 0.6:
            # Średnie - zielony/żółty
            local_ratio = (ratio - 0.4) / 0.2
            r = int(local_ratio * 255)
            g = 255
            b = 100 - int(local_ratio * 100)
        elif ratio < 0.8:
            # Średnie-wysokie - żółty/pomarańczowy
            local_ratio = (ratio - 0.6) / 0.2
            r = 255
            g = 255 - int(local_ratio * 100)
            b = 0
        else:
            # Wysokie częstotliwości - pomarańczowy/czerwony
            local_ratio = (ratio - 0.8) / 0.2
            r = 255
            g = 155 - int(local_ratio * 155)
            b = 0
        
        return (r, g, b)
    
    def _draw_bars(self, draw, bar_heights, smoothed_heights):
        """
        Rysuj equalizera (paski)
        
        Args:
            draw: ImageDraw object  
            bar_heights: Array z wysokościami pasków (0-1)
            smoothed_heights: Poprzednie wysokości dla wygładzania
        """
        # Wygładź przejścia między klatkami
        if smoothed_heights is not None:
            bar_heights = 0.7 * bar_heights + 0.3 * smoothed_heights
        
        # Parametry pasków
        bar_width = self.width / self.bars
        max_bar_height = self.height * 0.8
        base_y = self.height * 0.9
        
        # Rysuj paski
        for i, height in enumerate(bar_heights):
            x = i * bar_width
            bar_h = height * max_bar_height
            y = base_y - bar_h
            
            # Główny pasek z alpha
            color = self.colors[i]
            color_with_alpha = color + (int(255 * 0.6),)
            
            # Konwertuj do int dla rectangle
            x1, y1 = int(x + 1), int(y)
            x2, y2 = int(x + bar_width - 1), int(base_y)
            
            # Rysuj prostokąt
            for yi in range(y1, y2):
                draw.line([x1, yi, x2, yi], fill=color_with_alpha)


def process_batch(batch_dir, args):
    """
    Przetwarzanie wsadowe katalogów
    
    Args:
        batch_dir: Katalog zawierający podkatalogi z plikami WAV i obrazkami
        args: Argumenty z parsera
    """
    if not os.path.isdir(batch_dir):
        print(f"❌ {batch_dir} nie jest katalogiem")
        return
    
    # Parse kolorów
    left_color = tuple(map(int, args.left_color.split(',')))
    right_color = tuple(map(int, args.right_color.split(',')))
    
    print(f"🔄 Tryb batch: przetwarzam katalog {batch_dir}")
    print("=" * 70)
    
    # Szukaj podkatalogów
    subdirs = [d for d in os.listdir(batch_dir) 
               if os.path.isdir(os.path.join(batch_dir, d))]
    
    if not subdirs:
        print(f"❌ Brak podkatalogów w {batch_dir}")
        return
    
    total = len(subdirs)
    for idx, subdir in enumerate(subdirs, 1):
        subdir_path = os.path.join(batch_dir, subdir)
        print(f"\n[{idx}/{total}] 📁 Przetwarzam: {subdir}")
        print("-" * 70)
        
        # Znajdź plik WAV
        wav_files = glob.glob(os.path.join(subdir_path, "*.wav")) + \
                   glob.glob(os.path.join(subdir_path, "*.WAV"))
        
        if not wav_files:
            print(f"⚠️  Brak pliku WAV w {subdir}, pomijam...")
            continue
        
        wav_file = wav_files[0]  # Użyj pierwszego znalezionego
        
        # Sprawdź czy są obrazki w podkatalogu
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
        
        # Użyj katalogu jako tło jeśli są obrazki, w przeciwnym razie None
        background = subdir_path if image_files else None
        
        # Wygeneruj nazwę pliku wyjściowego
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        output_file = os.path.join(subdir_path, f"{base_name}.mp4")
        
        try:
            create_video_from_wav(
                wav_file,
                output_file,
                resolution=args.resolution,
                audio_bitrate=args.audio_bitrate,
                fps=args.fps,
                bars=args.bars,
                background=background,
                waveform_style=args.style,
                left_color=left_color,
                right_color=right_color,
                opacity=args.opacity,
                text=args.text,
                text_opacity=args.text_opacity,
                watermark=args.watermark,
                watermark_x=args.watermark_x,
                watermark_y=args.watermark_y,
                test_length=args.test_length,
                add_flares=not args.no_flares,
                flare_duration=args.flare_duration,
                screen_flash_intensity=args.screen_flash
            )
            print(f"✅ Ukończono: {output_file}")
        except Exception as e:
            print(f"❌ Błąd dla {subdir}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print(f"🎉 Batch processing zakończony! Przetworzono {total} katalogów.")


def create_video_from_wav(input_wav, output_mp4, resolution="1920x1080", 
                         audio_bitrate="320k", fps=30, bars=750,
                         background=None, waveform_style='waveform',
                         left_color=(255, 255, 0), right_color=(0, 255, 0),
                         opacity=0.9, text=None, text_opacity=0.8,
                         watermark=None, watermark_x=10, watermark_y=10,
                         test_length=None, add_flares=True, flare_duration=500,
                         screen_flash_intensity=0.0):
    """
    Główna funkcja konwertująca WAV do MP4 z wizualizacją
    
    Args:
        input_wav: Ścieżka do pliku WAV
        output_mp4: Ścieżka do pliku MP4 wyjściowego
        resolution: Rozdzielczość w formacie "WIDTHxHEIGHT"
        audio_bitrate: Bitrate audio (np. "320k", "192k")
        fps: Klatki na sekundę
        bars: Liczba pasków equalizera/punktów fali
        background: Ścieżka do obrazka/katalogu z tłem
        waveform_style: 'waveform' dla sinusoid, 'bars' dla equalizera
        left_color: Kolor lewego kanału (R, G, B)
        right_color: Kolor prawego kanału (R, G, B)
        opacity: Przezroczystość wizualizacji (0.0-1.0)
        text: Tekst do wyświetlenia (None = brak)
        text_opacity: Przezroczystość tekstu (0.0-1.0)
    """
    print(f"📁 Wczytuję plik: {input_wav}")
    
    # Jeśli output_mp4 nie ma ścieżki, zapisz w lokalizacji input
    if not os.path.dirname(output_mp4):
        input_dir = os.path.dirname(os.path.abspath(input_wav))
        output_mp4 = os.path.join(input_dir, output_mp4)
    
    # Parse resolution
    width, height = map(int, resolution.lower().split('x'))
    print(f"📺 Rozdzielczość: {width}x{height}")
    print(f"🎵 Bitrate audio: {audio_bitrate}")
    print(f"🎬 FPS: {fps}")
    print(f"📊 Styl wizualizacji: {waveform_style}")
    
    # Auto-wykrywanie obrazka z katalogu utworu (jeśli nie podano background)
    if not background:
        input_dir = os.path.dirname(os.path.abspath(input_wav))
        # Szukaj obrazków (png, jpg, jpeg) w katalogu utworu
        image_exts = ['.png', '.jpg', '.jpeg']
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in image_exts):
                background = os.path.join(input_dir, file)
                print(f"🖼️  Auto: Znaleziono tło {file} w katalogu utworu")
                break
    
    if background:
        print(f"🖼️  Tło: {background}")
    
    # Tryb testowy - obetnij plik WAV najpierw
    wav_to_process = input_wav
    temp_wav = None
    
    if test_length is not None:
        print(f"⚡ TRYB TESTOWY: Przycinam plik do {test_length}%")
        
        # Wczytaj WAV
        sample_rate, audio_data = wavfile.read(input_wav)
        original_duration = len(audio_data) / sample_rate
        target_duration = original_duration * (test_length / 100)
        target_samples = int(sample_rate * target_duration)
        
        # Obetnij audio
        cut_audio = audio_data[:target_samples]
        
        # Zapisz do tymczasowego pliku w katalogu tmp
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        
        temp_wav = os.path.join(tmp_dir, f"temp_test_{os.path.basename(input_wav)}")
        wavfile.write(temp_wav, sample_rate, cut_audio)
        
        wav_to_process = temp_wav
        print(f"⚡ TRYB TESTOWY: {test_length}% pliku ({target_duration:.2f}s z {original_duration:.2f}s)")
    
    # Inicjalizuj analizator audio
    visualizer = AudioVisualizer(wav_to_process, num_bars=bars)
    
    print(f"⏱️  Długość: {visualizer.duration:.2f} sekund")
    print(f"🔊 Format: {'Stereo' if visualizer.is_stereo else 'Mono'}")
    
    # Inicjalizuj generator wideo
    video_gen = VideoGenerator(width, height, fps, bars, waveform_style, 
                              left_color, right_color, opacity,
                              vocal_color=(255, 50, 50), text=text, text_opacity=text_opacity,
                              watermark=watermark, watermark_x=watermark_x, watermark_y=watermark_y,
                              add_flares=add_flares, flare_duration=flare_duration,
                              screen_flash_intensity=screen_flash_intensity)
    
    # Inicjalizuj manager tła
    bg_manager = BackgroundManager(background, width, height, visualizer.duration)
    
    # Stan dla wygładzania animacji
    previous_heights = np.zeros(bars)
    previous_left_wave = np.zeros(bars)
    previous_right_wave = np.zeros(bars)
    previous_vocal_wave = np.zeros(bars)
    
    def make_frame(t):
        """Funkcja generująca klatkę dla czasu t"""
        nonlocal previous_heights, previous_left_wave, previous_right_wave, previous_vocal_wave
        
        # Pobierz tło
        bg = bg_manager.get_frame(t)
        
        if waveform_style == 'waveform':
            # Pobierz dane fali dla obu kanałów
            left_wave, right_wave = visualizer.get_waveform_data(t, num_points=bars)
            
            # Wygładzanie
            left_wave = 0.7 * left_wave + 0.3 * previous_left_wave
            right_wave = 0.7 * right_wave + 0.3 * previous_right_wave
            
            # Ekstraktuj wokal
            vocal_wave = visualizer.extract_vocal_frequencies(t, num_points=bars)
            vocal_wave = 0.7 * vocal_wave + 0.3 * previous_vocal_wave
            
            # Utwórz klatkę
            frame = video_gen.create_frame(
                left_wave=left_wave,
                right_wave=right_wave,
                vocal_wave=vocal_wave,
                background=bg,
                time=t
            )
            
            # Zapamiętaj
            previous_left_wave = left_wave
            previous_right_wave = right_wave
            previous_vocal_wave = vocal_wave
        else:
            # Styl equalizera (bars)
            bar_heights = visualizer.get_frequency_spectrum(t)
            
            # Utwórz klatkę
            frame = video_gen.create_frame(
                bar_heights=bar_heights,
                smoothed_heights=previous_heights,
                background=bg,
                time=t
            )
            
            # Zapamiętaj
            previous_heights = bar_heights
        
        return frame
    
    print("🎨 Generuję wizualizację...")
    
    # Utwórz klip wideo
    video_clip = VideoClip(make_frame, duration=visualizer.duration)
    video_clip = video_clip.with_fps(fps)
    
    # Wczytaj audio
    audio_clip = AudioFileClip(wav_to_process)
    
    # Połącz wideo z audio
    final_clip = video_clip.with_audio(audio_clip)
    
    print(f"💾 Zapisuję do: {output_mp4}")
    
    # Zapisz jako MP4 z dobrą jakością
    # USUŃ WSZYSTKIE METADANE z pliku wejściowego (tylko treść audio/wideo)
    final_clip.write_videofile(
        output_mp4,
        codec='libx264',
        audio_codec='aac',
        audio_bitrate=audio_bitrate,
        fps=fps,
        preset='slow',  # Lepsza jakość, wolniejsze kodowanie
        bitrate='8000k',  # Wysokie bitrate wideo dla dobrej jakości
        # NIE kopiuj metadanych - usuń wszystkie metadane z pliku wejściowego
        ffmpeg_params=['-map_metadata', '-1']
    )
    
    print("✅ Gotowe!")
    print(f"📦 Plik zapisany: {output_mp4}")
    
    # Usuń tymczasowy plik WAV
    if temp_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
            print("🗑️  Usunięto tymczasowy plik")
        except:
            pass


def main():
    """Główna funkcja programu"""
    parser = argparse.ArgumentParser(
        description='Konwertuj WAV do MP4 z wizualną wizualizacją audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Podstawowe z tekstem
  python main.py song.wav output.mp4 --text "My Song Title"
  
  # Z tłem i znakiem wodnym
  python main.py song.wav output.mp4 --background photo.jpg --watermark logo.png
  
  # Test pierwszych 10%% (szybkie sprawdzenie)
  python main.py song.wav test.mp4 --test-length 10
  
  # Pełna konfiguracja
  python main.py song.wav output.mp4 --background ./images/ --text "Song 2025" --watermark logo.png --watermark-x 5 --watermark-y 5
  
  # Tryb batch
  python main.py batch-folder dummy.mp4 --batch
        """
    )
    
    parser.add_argument('input', help='Plik WAV wejściowy')
    parser.add_argument('output', help='Plik MP4 wyjściowy')
    parser.add_argument('--resolution', default='1920x1080',
                       help='Rozdzielczość wideo (domyślnie: 1920x1080)')
    parser.add_argument('--audio-bitrate', default='320k',
                       help='Bitrate audio (domyślnie: 320k)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Klatki na sekundę (domyślnie: 30)')
    parser.add_argument('--bars', type=int, default=750,
                       help='Liczba punktów wizualizacji (domyślnie: 750 dla wyższej rozdzielczości)')
    parser.add_argument('--background', default=None,
                       help='Ścieżka do obrazka lub katalogu z obrazkami dla tła')
    parser.add_argument('--style', default='waveform', choices=['waveform', 'bars'],
                       help='Styl wizualizacji: waveform (sinusoidy) lub bars (equalizera)')
    parser.add_argument('--left-color', default='255,255,0',
                       help='Kolor lewego kanału w formacie R,G,B (domyślnie: 255,255,0 - żółty)')
    parser.add_argument('--right-color', default='0,255,0',
                       help='Kolor prawego kanału w formacie R,G,B (domyślnie: 0,255,0 - zielony)')
    parser.add_argument('--opacity', type=float, default=0.9,
                       help='Przezroczystość wizualizacji 0.0-1.0 (domyślnie: 0.9)')
    parser.add_argument('--text', default=None,
                       help='Tekst do wyświetlenia w prawym dolnym rogu (zawsze CAPS)')
    parser.add_argument('--text-opacity', type=float, default=0.8,
                       help='Przezroczystość tekstu 0.0-1.0 (domyślnie: 0.8)')
    parser.add_argument('--watermark', default=None,
                       help='Ścieżka do pliku znaku wodnego (PNG/JPG z alpha channel)')
    parser.add_argument('--watermark-x', type=float, default=10,
                       help='Pozycja X znaku wodnego w %% od lewej (domyślnie: 10)')
    parser.add_argument('--watermark-y', type=float, default=10,
                       help='Pozycja Y znaku wodnego w %% od góry (domyślnie: 10)')
    parser.add_argument('--test-length', type=float, default=None,
                       help='Renderuj tylko X%% pliku dla szybkich testów (np. 10 = pierwsze 10%%)')
    parser.add_argument('--no-flares', action='store_true',
                       help='Wyłącz kolorowe flary na szczytach amplitudy (domyślnie: włączone)')
    parser.add_argument('--flare-duration', type=int, default=500,
                       help='Czas życia flary w milisekundach (domyślnie: 500ms)')
    parser.add_argument('--screen-flash', type=float, default=0.0,
                       help='Intensywność flasha rozchodzącego się od rekordów (0.06-0.9, 0=wyłączone, domyślnie: 0.0 - wyłączone)')
    parser.add_argument('--batch', action='store_true',
                       help='Tryb batch - przetwarzaj katalogi z podkatalogami zawierającymi WAV+obrazki')
    
    args = parser.parse_args()
    
    # Parse kolorów
    try:
        left_color = tuple(map(int, args.left_color.split(',')))
        right_color = tuple(map(int, args.right_color.split(',')))
        
        if len(left_color) != 3 or len(right_color) != 3:
            raise ValueError("Kolory muszą mieć 3 składowe (R,G,B)")
    except ValueError as e:
        print(f"❌ Błąd parsowania kolorów: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.batch:
            # Tryb batch processing
            process_batch(args.input, args)
        else:
            # Pojedynczy plik
            create_video_from_wav(
                args.input,
                args.output,
                resolution=args.resolution,
                audio_bitrate=args.audio_bitrate,
                fps=args.fps,
                bars=args.bars,
                background=args.background,
                waveform_style=args.style,
                left_color=left_color,
                right_color=right_color,
                opacity=args.opacity,
                text=args.text,
                text_opacity=args.text_opacity,
                watermark=args.watermark,
                watermark_x=args.watermark_x,
                watermark_y=args.watermark_y,
                test_length=args.test_length,
                add_flares=not args.no_flares,
                flare_duration=args.flare_duration,
                screen_flash_intensity=args.screen_flash
            )
    except Exception as e:
        print(f"❌ Błąd: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
