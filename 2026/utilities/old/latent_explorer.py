import evdev
import pygame
import sys
import threading
import numpy as np
import random
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
# !!! PASTE YOUR DEVICE NAMES HERE !!!
DEVICE_NAME_PEN = "Wacom Intuos BT M Pen"
DEVICE_NAME_PAD = "Wacom Intuos BT M Pad"

# Audio / Visual Settings
SAMPLE_RATE = 44100
MASTER_VOLUME = 0.2
DENSITY_RADIUS = 50  # Pixel radius to check for neighbor density
WIDTH, HEIGHT = 1400, 900 # Window size

# Pentatonic Scale (C Major Pentatonic) for "pleasant" harmony
SCALE_FREQS = [
    261.63, 293.66, 329.63, 392.00, 440.00,  # C4 - A4
    523.25, 587.33, 659.25, 783.99, 880.00,  # C5 - A5
    1046.50, 1174.66, 1318.51, 1567.98       # C6 - G6 (Sparkles)
]

# Apple-style Palette
COLORS = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (88, 86, 214), 
    (255, 204, 0), (255, 149, 0), (48, 176, 199), (242, 242, 247), 
    (142, 142, 147), (255, 45, 85)
]

# === GLOBAL STATE ===
state = {
    "x": 0, "y": 0, "pressure": 0,
    "in_proximity": False,
    "running": True,
    "selected_index": None,
    "selection_pending": False, # Triggered by button press
    "tab_w": 21600, "tab_h": 13500 # Default fallback
}

# === AUDIO ENGINE ===
class GranularSynthesizer:
    """
    Generates audio based on density.
    Low Density = Deep, rare notes.
    High Density = Shimmering, frequent high notes.
    """
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=16, buffer=1024)
        pygame.mixer.set_num_channels(32)
        self.sounds = []
        self._precompute_grains()
        self.last_density = 0.0

    def _precompute_grains(self):
        """Pre-renders sine waves with exponential decay (plucks)."""
        print("🎵 Audio: Synthesizing granular tones...")
        for freq in SCALE_FREQS:
            duration = 0.6
            t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
            wave = np.sin(2 * np.pi * freq * t) # Fundamental
            wave += 0.1 * np.sin(2 * np.pi * freq * 2 * t) # Harmonic warmth
            envelope = np.exp(-5 * t) # Pluck envelope
            wave = (wave * envelope * 32767 * 0.5).astype(np.int16)
            
            s = pygame.mixer.Sound(buffer=wave.tobytes())
            vol = 1.0 if freq > 500 else 0.7
            s.set_volume(vol * MASTER_VOLUME)
            self.sounds.append(s)

    def update(self, density_ratio):
        self.last_density += (density_ratio - self.last_density) * 0.1
        d = self.last_density
        
        # Chance to play increases with density
        chance = 0.02 + (d * 0.35)
        
        if random.random() < chance:
            # Higher density unlocks higher octaves
            max_index = int(len(self.sounds) * (0.4 + 0.6 * d))
            idx = random.randint(0, min(max_index, len(self.sounds)-1))
            self.sounds[idx].play()

    def play_select(self):
        """Distinct sound for selection."""
        self.sounds[0].play()
        self.sounds[4].play()

    def stop(self):
        pygame.mixer.quit()

# === DATA PROCESSING ===
def load_data(n_samples=2500):
    print("📥 MNIST: Loading dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Subsample
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sub, y_sub = X[idx], y[idx]
    
    print("📊 MNIST: Computing t-SNE (this can take a while)...")
    X_scaled = StandardScaler().fit_transform(X_sub)
    # Use a fixed random state for reproducibility
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate='auto', init='pca').fit_transform(X_scaled)
    
    return X_tsne, y_sub, X_sub

def create_digit_surface(image_data):
    """Converts 784-vector to pygame surface."""
    img = image_data.reshape(28, 28)
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    s = pygame.Surface((28, 28))
    for r in range(28):
        for c in range(28):
            v = img[r, c]
            s.set_at((c, r), (v, v, v))
    return pygame.transform.scale(s, (200, 200))

# === HARDWARE INPUT ===
def get_device(name):
    try:
        return next((evdev.InputDevice(p) for p in evdev.list_devices() 
                     if evdev.InputDevice(p).name == name), None)
    except: return None

def input_loop(pen_dev):
    """Threaded input loop using evdev."""
    global state
    for event in pen_dev.read_loop():
        if not state["running"]: break
        
        if event.type == evdev.ecodes.EV_ABS:
            if event.code == evdev.ecodes.ABS_X: state["x"] = event.value
            elif event.code == evdev.ecodes.ABS_Y: state["y"] = event.value
            elif event.code == evdev.ecodes.ABS_PRESSURE: state["pressure"] = event.value
            
        elif event.type == evdev.ecodes.EV_KEY:
            if event.code == evdev.ecodes.BTN_TOOL_PEN:
                state["in_proximity"] = (event.value == 1)
            elif event.code == evdev.ecodes.BTN_STYLUS2 and event.value == 1:
                state["selection_pending"] = True
                print("🖱️ Button 2 Pressed")

# === MAIN APPLICATION ===
def main():
    global state
    
    # 1. Setup Devices
    pen = get_device(DEVICE_NAME_PEN)
    if not pen:
        print(f"❌ Device '{DEVICE_NAME_PEN}' not found. Check naming.")
        return

    try:
        pen.grab()
        abs_info = pen.capabilities()[evdev.ecodes.EV_ABS]
        state["tab_w"] = abs_info[evdev.ecodes.ABS_X].max
        state["tab_h"] = abs_info[evdev.ecodes.ABS_Y].max
        print(f"🔒 Pen Grabbed. Tablet Area: {state['tab_w']}x{state['tab_h']}")
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    # 2. Calculate Screen Mapping (Letterboxing)
    # This logic ensures the scatter plot has the EXACT aspect ratio of the tablet
    ratio_tablet = state["tab_w"] / state["tab_h"]
    ratio_screen = WIDTH / HEIGHT

    if ratio_tablet > ratio_screen:
        scale = WIDTH / state["tab_w"]
        draw_w = WIDTH
        draw_h = int(state["tab_h"] * scale)
        offset_x = 0
        offset_y = (HEIGHT - draw_h) // 2
    else:
        scale = HEIGHT / state["tab_h"]
        draw_h = HEIGHT
        draw_w = int(state["tab_w"] * scale)
        offset_x = (WIDTH - draw_w) // 2
        offset_y = 0
    
    print(f"📐 Mapping: Scale {scale:.4f} | Offset ({offset_x}, {offset_y})")

    # 3. Load Data & Normalize
    X_tsne, labels, X_orig = load_data()
    
    # Normalize t-SNE coordinates strictly to the Tablet's Aspect Ratio Area
    # We normalize 0..1 then multiply by draw_w/draw_h
    x_min, x_max = X_tsne[:, 0].min(), X_tsne[:, 0].max()
    y_min, y_max = X_tsne[:, 1].min(), X_tsne[:, 1].max()
    
    # Add padding to data so points don't touch edges
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.05
    y_min -= y_range * 0.05
    x_max += x_range * 0.05
    y_max += y_range * 0.05

    # Pre-calculate screen coordinates for all points
    # Logic: Normalized_2D * Size + Offset
    screen_pts = np.zeros_like(X_tsne)
    screen_pts[:, 0] = offset_x + ((X_tsne[:, 0] - x_min) / (x_max - x_min)) * draw_w
    screen_pts[:, 1] = offset_y + ((X_tsne[:, 1] - y_min) / (y_max - y_min)) * draw_h
    screen_pts = screen_pts.astype(int)

    # 4. Start Engine
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Latent Space Explorer (t-SNE)")
    font_lg = pygame.font.SysFont("Arial", 40)
    font_sm = pygame.font.SysFont("Arial", 18)
    
    audio = GranularSynthesizer()
    threading.Thread(target=input_loop, args=(pen,), daemon=True).start()

    # Pre-render background scatter
    bg_surface = pygame.Surface((WIDTH, HEIGHT))
    bg_surface.fill((20, 20, 20)) # Dark Grey
    
    # Draw valid area border
    pygame.draw.rect(bg_surface, (40, 40, 40), (offset_x, offset_y, draw_w, draw_h), 2)
    
    for i, (sx, sy) in enumerate(screen_pts):
        pygame.draw.circle(bg_surface, COLORS[labels[i]], (sx, sy), 2)

    clock = pygame.time.Clock()
    sel_img = None
    sel_lbl = None

    # === DRAW LOOP ===
    while state["running"]:
        # Handle Window Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: state["running"] = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: state["running"] = False

        # 1. Map Pen to Screen
        cur_x = int(state["x"] * scale) + offset_x
        cur_y = int(state["y"] * scale) + offset_y

        # 2. Logic (Nearest Neighbor)
        # We search raw screen coordinates since we pre-calculated them
        dists = np.sqrt((screen_pts[:, 0] - cur_x)**2 + (screen_pts[:, 1] - cur_y)**2)
        hover_idx = np.argmin(dists)
        hover_dist = dists[hover_idx]
        
        # Audio Density Calculation
        nearby_count = np.sum(dists < DENSITY_RADIUS)
        density_val = min(1.0, nearby_count / 40.0)
        
        if state["in_proximity"]:
            audio.update(density_val)

        # Selection
        if state["selection_pending"]:
            state["selection_pending"] = False
            if hover_dist < 40: # Snap distance
                state["selected_index"] = hover_idx
                sel_lbl = labels[hover_idx]
                sel_img = create_digit_surface(X_orig[hover_idx])
                audio.play_select()

        # 3. Render
        screen.blit(bg_surface, (0, 0))

        # Draw Hover
        if hover_dist < 40:
            hx, hy = screen_pts[hover_idx]
            pygame.draw.circle(screen, (255, 255, 255), (hx, hy), 10, 1)

        # Draw Selection
        if state["selected_index"] is not None:
            sx, sy = screen_pts[state["selected_index"]]
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), 15, 2)
            pygame.draw.line(screen, (100, 100, 100), (sx, sy), (WIDTH-150, 150), 1)

        # Draw Cursor
        if state["in_proximity"]:
            # Ring expands with pressure
            r = 5 + int((state["pressure"] / 4096) * 10)
            pygame.draw.circle(screen, (255, 255, 255), (cur_x, cur_y), r, 1)

        # UI Overlays
        if sel_img:
            # Info Box
            pygame.draw.rect(screen, (30, 30, 30), (WIDTH-220, 50, 200, 260))
            pygame.draw.rect(screen, (60, 60, 60), (WIDTH-220, 50, 200, 260), 1)
            screen.blit(sel_img, (WIDTH-220, 50))
            lbl = font_lg.render(str(sel_lbl), True, (255, 255, 255))
            screen.blit(lbl, (WIDTH-140, 260))

        stats = font_sm.render(f"Density: {int(density_val*100)}%", True, (150, 150, 150))
        screen.blit(stats, (20, HEIGHT-40))

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    try: pen.ungrab()
    except: pass
    audio.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()