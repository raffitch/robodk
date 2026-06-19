"""Generate the Tasni desktop-app icon (tasni.ico + tasni.png).

On-brand with tasni.ai: a minimal teal robotic arm (the site's accent mark) on a
slate rounded-square tile. Brand colors sampled from the tasni.ai logo:
teal #58b0a0, slate #707888. Run:  py -3.10 assets/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
S = 1024  # master size

TEAL = (88, 176, 160)
TEAL_HI = (190, 233, 224)
BG_TOP = (60, 72, 88)
BG_BOT = (28, 34, 43)


def _gradient(size: int, top, bot) -> Image.Image:
    g = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        g.putpixel((0, y), tuple(round(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    return g.resize((size, size))


def _rounded_mask(size: int, radius: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=255)
    return m


def _link(draw: ImageDraw.ImageDraw, p0, p1, width, color):
    """A round-capped thick segment (line + end discs)."""
    draw.line([p0, p1], fill=color, width=width)
    r = width // 2
    for (x, y) in (p0, p1):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _dot(draw, c, r, color):
    draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=color)


def build_master() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = _gradient(S, BG_TOP, BG_BOT).convert("RGBA")
    img.paste(tile, (0, 0), _rounded_mask(S, int(S * 0.22)))

    d = ImageDraw.Draw(img)
    # Robotic arm reaching up-right (echoes the tasni.ai accent mark).
    base_c = (int(S * 0.34), int(S * 0.74))
    elbow = (int(S * 0.60), int(S * 0.49))
    end = (int(S * 0.70), int(S * 0.29))

    # base plinth
    d.rounded_rectangle([int(S * 0.24), int(S * 0.745), int(S * 0.46), int(S * 0.80)],
                        radius=int(S * 0.02), fill=TEAL)
    _link(d, base_c, elbow, int(S * 0.075), TEAL)   # upper arm
    _link(d, elbow, end, int(S * 0.066), TEAL)      # forearm
    _dot(d, base_c, int(S * 0.045), TEAL)           # shoulder joint
    _dot(d, elbow, int(S * 0.040), BG_BOT)          # elbow pivot (cutout look)
    _dot(d, elbow, int(S * 0.022), TEAL)
    _dot(d, end, int(S * 0.072), TEAL_HI)           # end-effector (bright)
    _dot(d, end, int(S * 0.034), TEAL)
    return img


def main() -> None:
    master = build_master()
    master.save(HERE / "tasni.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    master.save(HERE / "tasni.ico", format="ICO",
                sizes=[(s, s) for s in sizes])
    print("wrote", HERE / "tasni.ico", "and tasni.png")


if __name__ == "__main__":
    main()
