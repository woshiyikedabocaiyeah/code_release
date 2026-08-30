from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
# --- figure output redirected to _organized/figures/ -----------------------
OUT_DIR = ROOT.parents[1] / "figures" / "hddm_project" / "figma_exports"
OUT_PATH = OUT_DIR / "embodiment_spectrum_questions.png"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


FONT_REG = font("Arial.ttf", 24)
FONT_MED = font("Arial Bold.ttf", 30)
FONT_BIG = font("Arial Bold.ttf", 40)
FONT_SMALL = font("Arial.ttf", 22)
FONT_RESPONSE = font("Arial Bold.ttf", 26)


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def color_lerp(c1: str, c2: str, t: float) -> tuple[int, int, int]:
    a = ImageColor.getrgb(c1)
    b = ImageColor.getrgb(c2)
    return tuple(lerp(x, y, t) for x, y in zip(a, b))


def draw_gradient_background(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    top = ImageColor.getrgb("#F3F8FF")
    bottom = ImageColor.getrgb("#FFF3EA")
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(lerp(top[i], bottom[i], t) for i in range(3))
        draw.line((0, y, w, y), fill=c)


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(
        box,
        radius=32,
        fill=(255, 255, 255, 186),
        outline="#D8E2EF",
        width=2,
    )


def draw_multiline_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    top_y: int,
    max_width: int,
    font_obj: ImageFont.FreeTypeFont,
    fill: str = "#1C2533",
    line_gap: int = 10,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), proposal, font=font_obj)
        if bbox[2] - bbox[0] <= max_width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = top_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_obj)
        x = center_x - (bbox[2] - bbox[0]) / 2
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def draw_semantic_icon(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    blue = "#79A7D7"
    orange = "#D88A5D"
    grey = "#68758B"
    shapes = [
        ("rect", cx - 115, cy - 40),
        ("circle", cx - 55, cy - 40),
        ("circle", cx - 115, cy + 15),
        ("rect", cx - 55, cy + 15),
        ("circle", cx + 55, cy - 40),
        ("rect", cx + 115, cy - 40),
        ("triangle", cx + 55, cy + 18),
        ("triangle", cx + 115, cy + 18),
    ]
    for kind, x, y in shapes:
        if kind == "rect":
            draw.rounded_rectangle((x - 16, y - 16, x + 16, y + 16), radius=3, fill=blue if x < cx else orange)
        elif kind == "circle":
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=blue if x < cx else orange)
        else:
            draw.polygon([(x, y - 18), (x - 18, y + 16), (x + 18, y + 16)], fill=blue if x < cx else orange)

    draw.rectangle((cx - 88, cy - 72, cx - 18, cy + 68), outline=grey, width=3)
    draw.rectangle((cx + 18, cy - 48, cx + 92, cy + 42), outline=grey, width=3)
    draw.polygon([(cx - 8, cy), (cx + 6, cy - 8), (cx + 6, cy + 8)], fill=grey)
    draw.line((cx - 18, cy, cx - 8, cy), fill=grey, width=3)
    draw.line((cx + 6, cy, cx + 18, cy), fill=grey, width=3)
    for i, yy in enumerate([cy - 24, cy, cy + 24]):
        draw.line((cx + 34, yy, cx + 76, yy), fill="#AAB6C4", width=3)
        draw.ellipse((cx + 28, yy - 6, cx + 40, yy + 6), fill=[blue, grey, orange][i])


def draw_intuitive_icon(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    outline = "#6C7482"
    teal = "#8ED0C7"
    orange = "#D8A38A"
    blue = "#8CAFD6"
    for shift, col1, col2 in [(-80, blue, teal), (0, orange, blue), (80, orange, orange)]:
        x = cx + shift
        draw.rounded_rectangle((x - 55, cy - 40, x + 55, cy + 18), radius=28, outline=outline, width=2)
        draw.ellipse((x - 30, cy + 6, x - 16, cy + 20), outline=outline, width=2)
        draw.ellipse((x - 12, cy + 18, x - 4, cy + 26), outline=outline, width=2)
        draw.rounded_rectangle((x - 34, cy - 14, x - 8, cy + 12), radius=4, fill=col1)
        draw.polygon([(x + 5, cy + 8), (x + 24, cy - 14), (x + 40, cy + 8)], fill=col2)
        draw.line((x - 4, cy - 2, x + 2, cy - 2), fill=outline, width=2)
        draw.line((x + 8, cy - 2, x + 14, cy - 2), fill=outline, width=2)

    draw.ellipse((cx - 34, cy + 54, cx + 34, cy + 126), fill="#F0B6B6", outline="#B17272", width=2)
    draw.rectangle((cx - 8, cy + 114, cx + 8, cy + 138), fill="#F0B6B6", outline="#B17272", width=2)
    draw.arc((cx - 26, cy + 62, cx + 26, cy + 120), 10, 170, fill="#B17272", width=2)
    draw.arc((cx - 20, cy + 66, cx + 20, cy + 116), 190, 350, fill="#B17272", width=2)


def draw_action_icon(draw: ImageDraw.ImageDraw, cx: int, cy: int) -> None:
    blue = "#7BA6CF"
    orange = "#CC8D72"
    hand = "#F2C4A0"
    outline = "#6B7280"
    draw.polygon(
        [(cx - 110, cy - 42), (cx - 78, cy - 58), (cx - 48, cy - 40), (cx - 78, cy - 26)],
        fill=blue,
        outline="#5A7E9F",
    )
    draw.polygon(
        [(cx - 110, cy - 42), (cx - 110, cy + 6), (cx - 78, cy + 22), (cx - 78, cy - 26)],
        fill="#678FB5",
        outline="#5A7E9F",
    )
    draw.polygon(
        [(cx - 78, cy - 26), (cx - 48, cy - 40), (cx - 48, cy + 6), (cx - 78, cy + 22)],
        fill="#9BBBE0",
        outline="#5A7E9F",
    )

    draw.ellipse((cx + 68, cy - 58, cx + 116, cy - 10), fill="#B48772", outline="#7E5C50", width=2)
    draw.rounded_rectangle((cx + 52, cy + 18, cx + 90, cy + 68), radius=6, fill=orange, outline="#915E48", width=2)
    draw.ellipse((cx + 52, cy + 10, cx + 90, cy + 26), fill="#D79D82", outline="#915E48", width=2)
    draw.ellipse((cx + 52, cy + 60, cx + 90, cy + 76), fill="#C58166", outline="#915E48", width=2)

    hand_pts = [
        (cx - 26, cy + 70), (cx - 10, cy + 20), (cx + 6, cy - 10), (cx + 22, cy - 14),
        (cx + 30, cy + 0), (cx + 40, cy - 8), (cx + 54, cy - 6), (cx + 56, cy + 10),
        (cx + 44, cy + 28), (cx + 28, cy + 38), (cx + 12, cy + 52), (cx - 4, cy + 74),
    ]
    draw.polygon(hand_pts, fill=hand, outline="#B98567")
    draw.arc((cx - 10, cy - 44, cx + 88, cy + 42), 210, 335, fill=outline, width=2)
    draw.arc((cx - 78, cy - 28, cx + 22, cy + 52), 15, 120, fill=outline, width=2)
    draw.arc((cx - 10, cy + 6, cx + 96, cy + 98), 160, 260, fill=outline, width=2)
    draw.line((cx - 86, cy - 46, cx + 82, cy - 46), fill=outline, width=2)
    draw.text((cx - 10, cy - 66), "Distance", font=FONT_SMALL, fill="#1F2937", anchor="mm")


def draw_gradient_arrow(img: Image.Image, draw: ImageDraw.ImageDraw, y: int) -> None:
    x0, x1 = 92, 1508
    thickness = 18
    for x in range(x0, x1):
        t = (x - x0) / max(x1 - x0, 1)
        col = color_lerp("#6DA9DF", "#E07A45", t)
        draw.line((x, y, x + 1, y), fill=col, width=thickness)
    draw.polygon([(x0 - 34, y), (x0 + 8, y - 24), (x0 + 8, y + 24)], fill="#6DA9DF")
    draw.polygon([(x1 + 34, y), (x1 - 8, y - 24), (x1 - 8, y + 24)], fill="#E07A45")


def draw_response_card(draw: ImageDraw.ImageDraw, center_x: int, top_y: int, accent: str) -> None:
    x0, y0, x1, y1 = center_x - 130, top_y, center_x + 130, top_y + 116
    draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill="white", outline="#C7D4E1", width=2)
    draw.rounded_rectangle((x0, y0, x1, y0 + 26), radius=18, fill="#FFF7F2", outline="#C7D4E1", width=2)
    for i, dot in enumerate(["#F87171", "#FBBF24", "#34D399"]):
        draw.ellipse((x0 + 16 + i * 18, y0 + 8, x0 + 28 + i * 18, y0 + 20), fill=dot)
    draw.text((center_x, y0 + 64), "Yes / No", font=FONT_RESPONSE, fill=accent, anchor="mm")


def main() -> None:
    W, H = 1680, 980
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw_gradient_background(draw, W, H)
    draw_panel(draw, (26, 26, W - 26, H - 26))

    columns = [
        (
            300,
            "Semantic",
            'Does the event happening inside the box follow the concept of "permanence"?',
            draw_semantic_icon,
            "#3E7CC4",
        ),
        (
            840,
            "Intuitive",
            'Does the event happening inside the box feel physically "correct" and plausible?',
            draw_intuitive_icon,
            "#198D7B",
        ),
        (
            1380,
            "Action",
            "Is it physically possible for you to pick up a green cone from inside the box after the camera rolls up?",
            draw_action_icon,
            "#D85F39",
        ),
    ]

    icon_y = {"Semantic": 395, "Intuitive": 410, "Action": 425}
    question_y = {"Semantic": 150, "Intuitive": 150, "Action": 150}
    question_width = {"Semantic": 340, "Intuitive": 350, "Action": 380}

    for x, title, question, icon_fn, accent in columns:
        draw.text((x, 108), title, font=FONT_BIG, fill="#101827", anchor="mm")
        draw_multiline_centered(
            draw,
            question,
            x,
            question_y[title],
            question_width[title],
            FONT_REG,
            fill="#243244",
            line_gap=6,
        )
        icon_fn(draw, x, icon_y[title])

    draw_gradient_arrow(img, draw, 655)

    for x, _, _, _, accent in columns:
        draw.ellipse((x - 18, 637, x + 18, 673), fill="white", outline=accent, width=6)
        draw.line((x, 673, x, 732), fill="#9FB0C4", width=2)
        draw_response_card(draw, x, 734, accent)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH, quality=95)


if __name__ == "__main__":
    main()
