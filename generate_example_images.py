"""
Generator przykładowych obrazków dla example-pic
Tworzy kolorowe gradienty jako tła testowe
"""
from PIL import Image, ImageDraw
import os

def create_gradient_image(width, height, color1, color2, filename):
    """Utwórz gradient między dwoma kolorami"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    img.save(filename)
    print(f"✅ Utworzono: {filename}")

# Katalog docelowy
output_dir = "example-pic"
os.makedirs(output_dir, exist_ok=True)

# Rozmiar obrazków (Full HD)
width, height = 1920, 1080

# Generuj różne gradienty
gradients = [
    ("bg1_blue_purple.jpg", (0, 50, 100), (150, 50, 150)),
    ("bg2_purple_pink.jpg", (150, 50, 150), (255, 100, 150)),
    ("bg3_red_orange.jpg", (200, 50, 50), (255, 150, 50)),
    ("bg4_orange_yellow.jpg", (255, 150, 50), (255, 220, 100)),
    ("bg5_green_cyan.jpg", (50, 150, 100), (50, 200, 200)),
    ("bg6_cyan_blue.jpg", (50, 200, 200), (50, 100, 200)),
]

for filename, color1, color2 in gradients:
    filepath = os.path.join(output_dir, filename)
    create_gradient_image(width, height, color1, color2, filepath)

print(f"\n🎉 Utworzono {len(gradients)} przykładowych obrazków w katalogu '{output_dir}'")
