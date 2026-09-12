import os
import shutil
import colorsys
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

NUM_SPECIES = 100
# Entirely new species nomenclature and taxonomical IDs
SPECIES_NAMES = [f"novaspp_{i:03d}" for i in range(NUM_SPECIES)]

# Generate 100 distinct procedural color palettes using randomized prime-offset hue distribution
SPECIES_COLORS = []
ACCENT_COLORS = []
for i in range(NUM_SPECIES):
    # Base primary wing hue
    hue = (i * 0.618033988749895 + 0.382) % 1.0
    sat = 0.75 + 0.20 * ((i % 5) / 5.0)
    val = 0.75 + 0.20 * ((i % 4) / 4.0)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    SPECIES_COLORS.append((int(r * 255), int(g * 255), int(b * 255)))

    # High-contrast secondary accent hue (for eyespots and wing borders)
    accent_hue = (hue + 0.5) % 1.0
    ar, ag, ab = colorsys.hsv_to_rgb(accent_hue, 0.90, 0.90)
    ACCENT_COLORS.append((int(ar * 255), int(ag * 255), int(ab * 255)))

CAPTURE_SETTINGS = ["leaf", "flower", "museum", "sky"]
BG_COLORS = [
    (38, 145, 42),   # 0: Leaf (deep foliage green)
    (195, 35, 140),  # 1: Flower (rich magenta)
    (210, 185, 135), # 2: Museum (neutral collector parchment)
    (75, 170, 235),  # 3: Sky (open atmospheric cyan)
]

def draw_entirely_new_butterfly(species_id: int, setting_id: int, size=(224, 224)):
    bg_color = BG_COLORS[setting_id]
    img_np = np.full((size[1], size[0], 3), bg_color, dtype=np.uint8)
    # Add realistic environmental micro-texture
    noise = np.random.randint(-24, 24, img_np.shape, dtype=np.int16)
    img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)

    cx, cy = size[0] // 2, size[1] // 2
    wing_color = SPECIES_COLORS[species_id]
    accent_color = ACCENT_COLORS[species_id]

    # 6 distinct structural wing morphology archetypes
    archetype = species_id % 6
    span_x = 46 + (species_id % 7) * 4
    span_y = 40 + (species_id % 5) * 4

    if archetype == 0:  # Emerald Swallowtail (elongated hind tails)
        draw.polygon([(cx, cy), (cx - span_x, cy - span_y), (cx - span_x - 14, cy + 10), (cx - span_x + 8, cy + 38), (cx, cy + 18)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x, cy - span_y), (cx + span_x + 14, cy + 10), (cx + span_x - 8, cy + 38), (cx, cy + 18)], fill=wing_color, outline=(20, 20, 20))
    elif archetype == 1:  # Broad Rounded Brushfoot (Monarch style)
        draw.polygon([(cx, cy), (cx - span_x - 8, cy - span_y + 8), (cx - span_x, cy + 24), (cx, cy + 20)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x + 8, cy - span_y + 8), (cx + span_x, cy + 24), (cx, cy + 20)], fill=wing_color, outline=(20, 20, 20))
    elif archetype == 2:  # Pointed Hawk-Moth (aerodynamic angular wings)
        draw.polygon([(cx, cy), (cx - span_x - 20, cy - span_y - 12), (cx - span_x + 8, cy + 8), (cx, cy + 8)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x + 20, cy - span_y - 12), (cx + span_x - 8, cy + 8), (cx, cy + 8)], fill=wing_color, outline=(20, 20, 20))
    elif archetype == 3:  # Scalloped Anglewing (jagged outer margins)
        draw.polygon([(cx, cy), (cx - span_x, cy - span_y), (cx - span_x - 12, cy - 10), (cx - span_x - 4, cy + 5), (cx - span_x - 10, cy + 20), (cx, cy + 15)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x, cy - span_y), (cx + span_x + 12, cy - 10), (cx + span_x + 4, cy + 5), (cx + span_x + 10, cy + 20), (cx, cy + 15)], fill=wing_color, outline=(20, 20, 20))
    elif archetype == 4:  # Clearwing / Glasswing (fenestrated dual-panel)
        draw.polygon([(cx, cy), (cx - span_x, cy - span_y), (cx - span_x + 5, cy), (cx, cy)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx - span_x + 5, cy + 5), (cx - span_x - 6, cy + 26), (cx, cy + 16)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x, cy - span_y), (cx + span_x - 5, cy), (cx, cy)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x - 5, cy + 5), (cx + span_x + 6, cy + 26), (cx, cy + 16)], fill=wing_color, outline=(20, 20, 20))
    else:  # Caligo / Owl Eyespot (concentric submarginal ocelli)
        draw.polygon([(cx, cy), (cx - span_x - 5, cy - span_y + 5), (cx - span_x - 5, cy + 25), (cx, cy + 20)], fill=wing_color, outline=(20, 20, 20))
        draw.polygon([(cx, cy), (cx + span_x + 5, cy - span_y + 5), (cx + span_x + 5, cy + 25), (cx, cy + 20)], fill=wing_color, outline=(20, 20, 20))
        # Large distinguishing eyespots
        draw.ellipse([cx - span_x//2 - 10, cy - 8, cx - span_x//2 + 10, cy + 12], fill=accent_color, outline=(10, 10, 10))
        draw.ellipse([cx + span_x//2 - 10, cy - 8, cx + span_x//2 + 10, cy + 12], fill=accent_color, outline=(10, 10, 10))

    # Detailed wing venation architecture
    draw.line([(cx - 15, cy - 10), (cx - span_x + 12, cy - span_y + 12)], fill=(25, 25, 25), width=2)
    draw.line([(cx + 15, cy - 10), (cx + span_x - 12, cy - span_y + 12)], fill=(25, 25, 25), width=2)
    draw.line([(cx - 10, cy), (cx - span_x + 5, cy + 10)], fill=(30, 30, 30), width=1)
    draw.line([(cx + 10, cy), (cx + span_x - 5, cy + 10)], fill=(30, 30, 30), width=1)

    # Abdomen and antenna
    draw.ellipse([cx - 5, cy - 30, cx + 5, cy + 30], fill=(20, 20, 20))
    draw.line([(cx - 3, cy - 30), (cx - 14, cy - 45)], fill=(15, 15, 15), width=2)
    draw.line([(cx + 3, cy - 30), (cx + 14, cy - 45)], fill=(15, 15, 15), width=2)

    return img

def create_dataset():
    random.seed(42)
    np.random.seed(42)

    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    # 1. Generate Training Data (15 images per species = 1,500 total, 85% spurious bias)
    train_counts_per_species = 15
    for s_id, s_name in enumerate(SPECIES_NAMES):
        os.makedirs(f"data/train/{s_name}", exist_ok=True)
        primary_setting = s_id % len(CAPTURE_SETTINGS)
        for i in range(train_counts_per_species):
            setting_id = primary_setting if random.random() < 0.85 else random.choice([x for x in range(4) if x != primary_setting])
            img = draw_entirely_new_butterfly(s_id, setting_id)
            img.save(f"data/train/{s_name}/img_{i:04d}.png")

    # 2. Generate Validation Data (1 image per setting per species = 400 total across 400 groups)
    # Used strictly for DFR linear reweighting & tuning
    val_records = []
    val_per_group = 1
    for s_id, s_name in enumerate(SPECIES_NAMES):
        os.makedirs(f"data/val/{s_name}", exist_ok=True)
        for setting_id, setting_name in enumerate(CAPTURE_SETTINGS):
            for i in range(val_per_group):
                filename = f"{s_name}/val_{setting_name}_{i:03d}.png"
                img = draw_entirely_new_butterfly(s_id, setting_id)
                img.save(f"data/val/{filename}")
                val_records.append({
                    "file": filename,
                    "label": s_id,
                    "capture_setting": setting_name
                })

    df_val = pd.DataFrame(val_records)
    df_val.to_csv("data/val/groups.csv", index=False)

    # 3. Generate Held-Out Test Data (3 images per setting per species = 1,200 total across 400 groups)
    # Strictly held-out for unbiased out-of-distribution evaluation
    test_records = []
    test_per_group = 3
    for s_id, s_name in enumerate(SPECIES_NAMES):
        os.makedirs(f"data/test/{s_name}", exist_ok=True)
        for setting_id, setting_name in enumerate(CAPTURE_SETTINGS):
            for i in range(test_per_group):
                filename = f"{s_name}/test_{setting_name}_{i:03d}.png"
                img = draw_entirely_new_butterfly(s_id, setting_id)
                img.save(f"data/test/{filename}")
                test_records.append({
                    "file": filename,
                    "label": s_id,
                    "capture_setting": setting_name
                })

    df_test = pd.DataFrame(test_records)
    df_test.to_csv("data/test/groups.csv", index=False)

    print(f"100 Species 3-Split Dataset Created!")
    print(f"  Train: {NUM_SPECIES * train_counts_per_species} images (biased 85%)")
    print(f"  Val:   {len(df_val)} images across {NUM_SPECIES * len(CAPTURE_SETTINGS)} groups (1/group)")
    print(f"  Test:  {len(df_test)} images across {NUM_SPECIES * len(CAPTURE_SETTINGS)} groups (3/group, strictly held-out)")

if __name__ == "__main__":
    create_dataset()
