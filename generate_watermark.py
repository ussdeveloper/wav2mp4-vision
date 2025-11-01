"""
Generator przykładowego znaku wodnego
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_watermark(text="DEMO", width=300, height=100, output_file="watermark.png"):
    """Utwórz prosty znak wodny z tekstem"""
    # Utwórz obraz z przezroczystym tłem
    img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Spróbuj załadować font
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Oblicz pozycję tekstu (wyśrodkuj)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width, text_height = draw.textsize(text, font=font)
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Rysuj ramkę
    border_color = (255, 255, 255, 180)
    draw.rectangle([5, 5, width-5, height-5], outline=border_color, width=3)
    
    # Rysuj tekst z cieniem
    shadow_color = (0, 0, 0, 150)
    text_color = (255, 255, 255, 200)
    
    draw.text((x + 2, y + 2), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=text_color)
    
    # Zapisz
    img.save(output_file)
    print(f"✅ Utworzono znak wodny: {output_file}")

# Utwórz przykładowe znaki wodne
create_watermark("DEMO", output_file="watermark_demo.png")
create_watermark("TEST", width=250, height=80, output_file="watermark_test.png")

print("\n🎉 Utworzono przykładowe znaki wodne!")
print("Użycie: --watermark watermark_demo.png --watermark-x 10 --watermark-y 10")
