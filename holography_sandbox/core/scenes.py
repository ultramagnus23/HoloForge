"""
scenes.py
---------
Synthetic test scenes for holographic experiments.

All scenes return float32 arrays normalised to [0, 1].
Default resolution: 512 × 512 (change via `size` parameter).

Available scenes
----------------
  point_sources()     — isolated bright dots at different depths
  resolution_chart()  — USAF-style bar chart for resolving power
  checkerboard()      — spatial frequency reference
  depth_gradient()    — smooth depth ramp
  letters()           — text "HOLO" rendered in pixels
  multi_depth_scene() — layered objects at multiple z-planes
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _blank(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return arr
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Scenes
# ─────────────────────────────────────────────────────────────────────────────

def point_sources(size: int = 512, n_points: int = 5, seed: int = 7) -> np.ndarray:
    """Random bright point sources — good for testing PSF and speckle."""
    rng = np.random.default_rng(seed)
    img = _blank(size)
    xs = rng.integers(size // 8, 7 * size // 8, n_points)
    ys = rng.integers(size // 8, 7 * size // 8, n_points)
    for x, y in zip(xs, ys):
        img[y, x] = 1.0
    return img


def gaussian_spots(size: int = 512, n_spots: int = 5, sigma: float = 8.0, seed: int = 7) -> np.ndarray:
    """Gaussian blobs — softer version of point sources."""
    rng = np.random.default_rng(seed)
    img = _blank(size)
    Y, X = np.mgrid[:size, :size]
    for _ in range(n_spots):
        cx = rng.integers(size // 6, 5 * size // 6)
        cy = rng.integers(size // 6, 5 * size // 6)
        brightness = rng.uniform(0.5, 1.0)
        img += brightness * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return _normalise(img)


def resolution_chart(size: int = 512) -> np.ndarray:
    """
    Simplified USAF resolution chart — alternating bar groups at
    increasing spatial frequencies across the image.
    """
    img = _blank(size)
    freqs = [4, 8, 16, 32, 64]           # bar widths in pixels
    y_offset = size // 10
    bar_height = size // (len(freqs) + 1)

    for i, freq in enumerate(freqs):
        y_start = y_offset + i * bar_height
        y_end = y_start + bar_height - 4
        x = np.arange(size)
        stripe = ((x // freq) % 2).astype(np.float32)
        img[y_start:y_end, :] = stripe[np.newaxis, :]
    return img


def checkerboard(size: int = 512, tile: int = 32) -> np.ndarray:
    """Classic checkerboard — maximum spatial frequency reference."""
    Y, X = np.mgrid[:size, :size]
    return (((X // tile) + (Y // tile)) % 2).astype(np.float32)


def depth_gradient(size: int = 512) -> np.ndarray:
    """
    Smooth horizontal depth gradient.
    Useful as a depth-map input to multi-plane experiments.
    """
    ramp = np.linspace(0, 1, size, dtype=np.float32)
    return np.tile(ramp, (size, 1))


def circle_ring(size: int = 512, n_rings: int = 4) -> np.ndarray:
    """Concentric rings — good for testing radial symmetry in reconstruction."""
    img = _blank(size)
    cx, cy = size // 2, size // 2
    Y, X = np.mgrid[:size, :size]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = size * 0.45
    for i in range(n_rings):
        r_inner = max_r * i / n_rings
        r_outer = max_r * (i + 0.4) / n_rings
        img[(r >= r_inner) & (r < r_outer)] = 1.0 if i % 2 == 0 else 0.5
    return img


def letters(size: int = 512) -> np.ndarray:
    """
    Render the word 'HOLO' as a binary pixel image.
    Uses PIL for text rendering if available, otherwise falls back to
    a hand-coded bitmap.
    """
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        img_pil = PILImage.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img_pil)
        font_size = size // 5
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        text = "HOLO"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2
        y = (size - th) // 2
        draw.text((x, y), text, fill=255, font=font)
        return np.array(img_pil).astype(np.float32) / 255.0
    except Exception:
        # Fallback: simple rectangle
        img = _blank(size)
        m = size // 4
        img[m:3*m, m:3*m] = 1.0
        return img


def multi_depth_scene(
    size: int = 512,
    n_planes: int = 4,
) -> list:
    """
    Return a list of (amplitude_layer, z_weight) tuples representing
    objects at different depth layers.

    The caller is responsible for propagating each layer to the
    appropriate z-distance and summing the hologram contributions.

    Returns
    -------
    list of float32 arrays, one per depth plane
    """
    layers = [
        gaussian_spots(size, n_spots=3, sigma=12, seed=10),
        circle_ring(size, n_rings=2),
        resolution_chart(size),
        letters(size),
    ]
    # Trim or pad to n_planes
    while len(layers) < n_planes:
        layers.append(gaussian_spots(size, n_spots=2, sigma=6, seed=len(layers)))
    return layers[:n_planes]
