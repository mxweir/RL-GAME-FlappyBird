# generate_assets.py
from PIL import Image, ImageDraw
import pygame
import sys

def create_background(width, height, filename):
    """Erstellt eine einfache Himmelshintergrundgrafik mit Wolken."""
    image = Image.new('RGB', (width, height), (135, 206, 235))  # Himmelblau
    draw = ImageDraw.Draw(image)
    
    # Zeichne einige einfache Wolken
    draw.ellipse((50, 50, 150, 100), fill=(255, 255, 255))
    draw.ellipse((120, 80, 220, 130), fill=(255, 255, 255))
    draw.ellipse((300, 60, 400, 110), fill=(255, 255, 255))
    
    image.save(filename)
    print(f"Hintergrundbild gespeichert als {filename}")

def create_bird_frames(size, colors, filenames):
    """Erstellt einfache Vogel-Frames als farbige Kreise mit Flügeln."""
    for i, color in enumerate(colors):
        image = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Zeichne den Körper des Vogels
        draw.ellipse((10, 10, size[0]-10, size[1]-10), fill=color)
        
        # Zeichne einen einfachen Flügel
        if i == 1:
            draw.polygon([(20, 30), (40, 20), (40, 40)], fill=(255, 0, 0, 255))
        elif i == 2:
            draw.polygon([(30, 20), (50, 30), (30, 40)], fill=(255, 0, 0, 255))
        
        image.save(filenames[i])
        print(f"Bilder für Vogel-Frame {i+1} gespeichert als {filenames[i]}")

def create_pipe(width, height, filename):
    """Erstellt eine einfache Pipe-Grafik als grünes Rechteck."""
    image = Image.new('RGBA', (width, height), (0, 255, 0, 255))
    draw = ImageDraw.Draw(image)
    
    # Optional: Füge Text oder Muster hinzu
    # draw.rectangle([0, 0, width-1, height-1], outline=(0, 200, 0))
    
    image.save(filename)
    print(f"Pipe-Grafik gespeichert als {filename}")

def create_ground(width, height, filename):
    """Erstellt eine einfache Boden-Grafik als braunes Rechteck mit Gras."""
    image = Image.new('RGBA', (width, height), (222, 184, 135, 255))  # Holzfarbe
    draw = ImageDraw.Draw(image)
    
    # Zeichne Gras
    draw.rectangle([0, 0, width, height//2], fill=(34, 139, 34, 255))  # Grasgrün
    
    # Optional: Füge Textur oder Muster hinzu
    # draw.line([(0, height//2), (width, height//2)], fill=(0, 100, 0), width=2)
    
    image.save(filename)
    print(f"Boden-Grafik gespeichert als {filename}")

def main():
    # Parameter für die Grafiken
    screen_width = 400
    screen_height = 600
    
    # Vogel-Frames
    bird_size = (34, 24)  # Größe anpassen nach Bedarf
    bird_colors = [(255, 255, 0), (255, 215, 0), (255, 255, 224)]  # Verschiedene Gelbtöne
    bird_filenames = ['assets/bird1.png', 'assets/bird2.png', 'assets/bird3.png']
    create_bird_frames(bird_size, bird_colors, bird_filenames)
    
    # Pipe
    pipe_width = 70
    pipe_height = 500  # Hohe Pipes
    create_pipe(pipe_width, pipe_height, 'assets/pipe.png')
    
    # Boden
    ground_width = screen_width
    ground_height = 100  # Beispielhöhe, anpassen nach Bedarf
    create_ground(ground_width, ground_height, 'assets/ground.png')
    
    print("Alle Grafiken wurden erfolgreich erstellt.")

if __name__ == "__main__":
    # Stelle sicher, dass der assets-Ordner existiert
    import os
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    main()
