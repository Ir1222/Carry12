import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SRC = Path(r"D:\D_Downloads\summary (13).csv")
OUT_DIR = Path(r"D:\AAAProject\0806\PhysHSI\outputs\velocity_tracking_summary13")


AXES = [
    ("vx", "vx_actual_mean", "vx", "m/s", "#2b6cb0"),
    ("vy", "vy_actual_mean", "vy", "m/s", "#2f855a"),
    ("yaw_rate", "yaw_rate_actual_mean", "yaw", "rad/s", "#b7791f"),
]


def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
        ".title{font-size:20px;font-weight:700}",
        ".subtitle{font-size:12px;fill:#52606d}",
        ".axis{stroke:#52606d;stroke-width:1}",
        ".grid{stroke:#d9e2ec;stroke-width:1}",
        ".ideal{stroke:#7b8794;stroke-width:1.5;stroke-dasharray:5 5}",
        ".label{font-size:11px;fill:#334e68}",
        ".small{font-size:9px;fill:#52606d}",
        "</style>",
    ]


def svg_footer():
    return ["</svg>"]


def line(x1, y1, x2, y2, cls=None, stroke="#000", width=1, dash=None):
    attrs = [f'x1="{x1:.2f}"', f'y1="{y1:.2f}"', f'x2="{x2:.2f}"', f'y2="{y2:.2f}"']
    if cls:
        attrs.append(f'class="{cls}"')
    else:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{width}"')
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return "<line " + " ".join(attrs) + "/>"


def text(x, y, body, cls="label", anchor="middle"):
    safe = str(body).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}">{safe}</text>'


def circle(x, y, r, fill, stroke="#ffffff", width=1.5):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def rect(x, y, w, h, fill, stroke="none", width=1):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def data_range(vals, fallback=(-1.0, 1.0)):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return fallback
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-9:
        pad = max(0.1, abs(hi) * 0.2)
    else:
        pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, n - 1)
    mag = 10 ** math.floor(math.log10(abs(raw)))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(raw - m * mag)) * mag
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        ticks.append(round(v, 6))
        v += step
    return ticks


def fmt_tick(v):
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def draw_scatter_panel(parts, rows, xkey, ykey, title, x, y, w, h, color, include_labels=True):
    vals = []
    for r in rows:
        vals.extend([f(r, xkey), f(r, ykey)])
    lo, hi = data_range(vals)
    lo = min(lo, 0)
    hi = max(hi, 0)

    def sx(v):
        return x + (v - lo) / (hi - lo) * w

    def sy(v):
        return y + h - (v - lo) / (hi - lo) * h

    parts.append(text(x + w / 2, y - 10, title, "label"))
    for t in nice_ticks(lo, hi):
        px, py = sx(t), sy(t)
        parts.append(line(px, y, px, y + h, cls="grid"))
        parts.append(line(x, py, x + w, py, cls="grid"))
        parts.append(text(px, y + h + 14, fmt_tick(t), "small"))
        parts.append(text(x - 8, py + 3, fmt_tick(t), "small", "end"))
    parts.append(line(x, y + h, x + w, y + h, cls="axis"))
    parts.append(line(x, y, x, y + h, cls="axis"))
    parts.append(line(sx(lo), sy(lo), sx(hi), sy(hi), cls="ideal"))
    parts.append(text(x + w / 2, y + h + 31, "command", "small"))
    parts.append(text(x - 28, y + h / 2, "actual", "small", "middle"))

    for r in rows:
        cmd = f(r, xkey)
        actual = f(r, ykey)
        if math.isnan(cmd) or math.isnan(actual):
            continue
        completed = int(float(r["trial_completed"])) == 1
        fill = color if completed else "#d64545"
        stroke = "#263238" if completed else "#7f1d1d"
        parts.append(circle(sx(cmd), sy(actual), 4.8 if completed else 5.8, fill, stroke, 1.2))
        if include_labels:
            parts.append(text(sx(cmd) + 7, sy(actual) - 5, r["trial_id"], "small", "start"))


def write_scatter(rows):
    width, height = 1500, 950
    parts = svg_header(width, height)
    parts.append(text(28, 34, "Velocity command vs actual mean per trial", "title", "start"))
    parts.append(text(28, 56, "Dashed line is perfect tracking. Red points are failed trials. Labels are trial IDs.", "subtitle", "start"))
    parts.append(circle(1130, 32, 5, "#2b6cb0", "#263238"))
    parts.append(text(1142, 36, "completed", "small", "start"))
    parts.append(circle(1230, 32, 6, "#d64545", "#7f1d1d"))
    parts.append(text(1244, 36, "failed", "small", "start"))

    panel_w, panel_h = 390, 300
    xs = [90, 570, 1050]
    ys = [115, 560]
    pure_modes = {"vx": "vx", "vy": "vy", "yaw_rate": "yaw"}
    for i, (cmd, actual, label, unit, color) in enumerate(AXES):
        pure = [r for r in rows if r["mode"] == pure_modes[cmd]]
        mixed = [r for r in rows if r["mode"] == "mixed" and abs(f(r, cmd)) > 1e-12]
        draw_scatter_panel(parts, pure, cmd, actual, f"Pure {label} ({unit})", xs[i], ys[0], panel_w, panel_h, color)
        draw_scatter_panel(parts, mixed, cmd, actual, f"Mixed {label} ({unit})", xs[i], ys[1], panel_w, panel_h, color)

    (OUT_DIR / "velocity_cmd_actual_scatter.svg").write_text("\n".join(parts + svg_footer()), encoding="utf-8")


def draw_pairs_panel(parts, rows, cmd_key, actual_key, title, x, y, w, h, color):
    vals = []
    for r in rows:
        vals.extend([f(r, cmd_key), f(r, actual_key)])
    lo, hi = data_range(vals)
    lo = min(lo, 0)
    hi = max(hi, 0)

    def sy(v):
        return y + h - (v - lo) / (hi - lo) * h

    n = len(rows)
    step = w / max(1, n)

    parts.append(text(x + w / 2, y - 12, title, "label"))
    for t in nice_ticks(lo, hi):
        py = sy(t)
        parts.append(line(x, py, x + w, py, cls="grid"))
        parts.append(text(x - 8, py + 3, fmt_tick(t), "small", "end"))
    parts.append(line(x, y + h, x + w, y + h, cls="axis"))
    parts.append(line(x, y, x, y + h, cls="axis"))
    parts.append(line(x, sy(0), x + w, sy(0), stroke="#7b8794", width=1, dash="3 4"))

    for i, r in enumerate(rows):
        cx = x + step * (i + 0.5)
        cmd = f(r, cmd_key)
        actual = f(r, actual_key)
        completed = int(float(r["trial_completed"])) == 1
        fail_color = "#d64545"
        parts.append(line(cx, sy(cmd), cx, sy(actual), stroke="#94a3b8", width=1.2))
        parts.append(circle(cx, sy(cmd), 4.3, "#ffffff", "#111827", 1.4))
        parts.append(circle(cx, sy(actual), 5.0, color if completed else fail_color, "#263238", 1.0))
        label = f'{r["trial_id"]}:{r["mode"]}'
        parts.append(f'<text x="{cx:.2f}" y="{y+h+14:.2f}" class="small" text-anchor="end" transform="rotate(-55 {cx:.2f} {y+h+14:.2f})">{label}</text>')

    parts.append(circle(x + w - 130, y - 28, 4.2, "#ffffff", "#111827", 1.4))
    parts.append(text(x + w - 120, y - 24, "cmd", "small", "start"))
    parts.append(circle(x + w - 78, y - 28, 5.0, color, "#263238", 1.0))
    parts.append(text(x + w - 68, y - 24, "actual", "small", "start"))


def write_pairs(rows):
    width, height = 1550, 1160
    parts = svg_header(width, height)
    parts.append(text(28, 36, "Each velocity command paired with its actual mean", "title", "start"))
    parts.append(text(28, 58, "Open marker is command. Filled marker is actual velocity mean. Vertical line length is signed tracking error.", "subtitle", "start"))
    y0 = 110
    for i, (cmd, actual, label, unit, color) in enumerate(AXES):
        axis_rows = [r for r in rows if abs(f(r, cmd)) > 1e-12]
        axis_rows.sort(key=lambda r: (r["mode"] != label, f(r, cmd), r["trial_id"]))
        draw_pairs_panel(parts, axis_rows, cmd, actual, f"{label} command vs actual ({unit})", 90, y0 + i * 340, 1380, 235, color)

    (OUT_DIR / "velocity_cmd_actual_pairs.svg").write_text("\n".join(parts + svg_footer()), encoding="utf-8")


def write_table(rows):
    out = OUT_DIR / "velocity_cmd_actual_long.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial_id", "mode", "axis", "command", "actual_mean", "signed_error_actual_minus_cmd", "abs_error", "completed", "termination_reason"])
        for r in rows:
            for cmd, actual, label, _unit, _color in AXES:
                command = f(r, cmd)
                actual_v = f(r, actual)
                if abs(command) <= 1e-12 and r["mode"] != "stand":
                    continue
                writer.writerow([
                    r["trial_id"],
                    r["mode"],
                    label,
                    f"{command:.6f}",
                    f"{actual_v:.6f}",
                    f"{actual_v - command:.6f}",
                    f"{abs(actual_v - command):.6f}",
                    r["trial_completed"],
                    r["termination_reason"],
                ])


def load_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_SMALL = load_font(11)
FONT_LABEL = load_font(14)
FONT_TITLE = load_font(22, bold=True)
FONT_SUBTITLE = load_font(13)


def draw_text(draw, xy, body, font=FONT_LABEL, fill="#1f2933", anchor="mm"):
    draw.text(xy, str(body), font=font, fill=fill, anchor=anchor)


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def draw_png_scatter_panel(draw, rows, xkey, ykey, title, x, y, w, h, color):
    vals = []
    for r in rows:
        vals.extend([f(r, xkey), f(r, ykey)])
    lo, hi = data_range(vals)
    lo = min(lo, 0)
    hi = max(hi, 0)

    def sx(v):
        return x + (v - lo) / (hi - lo) * w

    def sy(v):
        return y + h - (v - lo) / (hi - lo) * h

    draw_text(draw, (x + w / 2, y - 18), title, FONT_LABEL)
    for t in nice_ticks(lo, hi):
        px, py = sx(t), sy(t)
        draw.line((px, y, px, y + h), fill="#d9e2ec", width=1)
        draw.line((x, py, x + w, py), fill="#d9e2ec", width=1)
        draw_text(draw, (px, y + h + 15), fmt_tick(t), FONT_SMALL, "#52606d")
        draw_text(draw, (x - 10, py), fmt_tick(t), FONT_SMALL, "#52606d", "rm")
    draw.rectangle((x, y, x + w, y + h), outline="#52606d", width=1)
    draw.line((sx(lo), sy(lo), sx(hi), sy(hi)), fill="#7b8794", width=2)
    draw_text(draw, (x + w / 2, y + h + 34), "command", FONT_SMALL, "#52606d")
    draw_text(draw, (x - 38, y + h / 2), "actual", FONT_SMALL, "#52606d")

    for r in rows:
        cmd = f(r, xkey)
        actual = f(r, ykey)
        if math.isnan(cmd) or math.isnan(actual):
            continue
        completed = int(float(r["trial_completed"])) == 1
        fill = color if completed else "#d64545"
        radius = 5 if completed else 6
        cx, cy = sx(cmd), sy(actual)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=hex_to_rgb(fill), outline="#263238", width=1)
        draw_text(draw, (cx + 18, cy - 9), r["trial_id"], FONT_SMALL, "#52606d", "mm")


def write_scatter_png(rows):
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (28, 34), "Velocity command vs actual mean per trial", FONT_TITLE, anchor="lm")
    draw_text(draw, (28, 58), "Dashed line is perfect tracking. Red points are failed trials. Labels are trial IDs.", FONT_SUBTITLE, "#52606d", "lm")
    draw.ellipse((1130, 27, 1140, 37), fill=hex_to_rgb("#2b6cb0"), outline="#263238")
    draw_text(draw, (1148, 32), "completed", FONT_SMALL, "#52606d", "lm")
    draw.ellipse((1230, 26, 1242, 38), fill=hex_to_rgb("#d64545"), outline="#7f1d1d")
    draw_text(draw, (1250, 32), "failed", FONT_SMALL, "#52606d", "lm")

    panel_w, panel_h = 390, 300
    xs = [90, 570, 1050]
    ys = [115, 560]
    pure_modes = {"vx": "vx", "vy": "vy", "yaw_rate": "yaw"}
    for i, (cmd, actual, label, unit, color) in enumerate(AXES):
        pure = [r for r in rows if r["mode"] == pure_modes[cmd]]
        mixed = [r for r in rows if r["mode"] == "mixed" and abs(f(r, cmd)) > 1e-12]
        draw_png_scatter_panel(draw, pure, cmd, actual, f"Pure {label} ({unit})", xs[i], ys[0], panel_w, panel_h, color)
        draw_png_scatter_panel(draw, mixed, cmd, actual, f"Mixed {label} ({unit})", xs[i], ys[1], panel_w, panel_h, color)
    img.save(OUT_DIR / "velocity_cmd_actual_scatter.png")


def paste_rotated_label(img, text_body, cx, cy):
    label = Image.new("RGBA", (125, 18), (255, 255, 255, 0))
    d = ImageDraw.Draw(label)
    d.text((0, 0), text_body, font=FONT_SMALL, fill="#52606d")
    rot = label.rotate(55, expand=True)
    img.paste(rot, (int(cx - 12), int(cy - 8)), rot)


def draw_png_pairs_panel(img, draw, rows, cmd_key, actual_key, title, x, y, w, h, color):
    vals = []
    for r in rows:
        vals.extend([f(r, cmd_key), f(r, actual_key)])
    lo, hi = data_range(vals)
    lo = min(lo, 0)
    hi = max(hi, 0)

    def sy(v):
        return y + h - (v - lo) / (hi - lo) * h

    n = len(rows)
    step = w / max(1, n)
    draw_text(draw, (x + w / 2, y - 16), title, FONT_LABEL)
    for t in nice_ticks(lo, hi):
        py = sy(t)
        draw.line((x, py, x + w, py), fill="#d9e2ec", width=1)
        draw_text(draw, (x - 10, py), fmt_tick(t), FONT_SMALL, "#52606d", "rm")
    draw.rectangle((x, y, x + w, y + h), outline="#52606d", width=1)
    draw.line((x, sy(0), x + w, sy(0)), fill="#7b8794", width=1)

    for i, r in enumerate(rows):
        cx = x + step * (i + 0.5)
        cmd = f(r, cmd_key)
        actual = f(r, actual_key)
        completed = int(float(r["trial_completed"])) == 1
        actual_color = color if completed else "#d64545"
        draw.line((cx, sy(cmd), cx, sy(actual)), fill="#94a3b8", width=2)
        draw.ellipse((cx - 4, sy(cmd) - 4, cx + 4, sy(cmd) + 4), fill="white", outline="#111827", width=2)
        draw.ellipse((cx - 5, sy(actual) - 5, cx + 5, sy(actual) + 5), fill=hex_to_rgb(actual_color), outline="#263238", width=1)
        paste_rotated_label(img, f'{r["trial_id"]}:{r["mode"]}', cx, y + h + 12)

    draw.ellipse((x + w - 132, y - 34, x + w - 124, y - 26), fill="white", outline="#111827", width=2)
    draw_text(draw, (x + w - 118, y - 30), "cmd", FONT_SMALL, "#52606d", "lm")
    draw.ellipse((x + w - 76, y - 35, x + w - 66, y - 25), fill=hex_to_rgb(color), outline="#263238")
    draw_text(draw, (x + w - 60, y - 30), "actual", FONT_SMALL, "#52606d", "lm")


def write_pairs_png(rows):
    width, height = 1550, 1160
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_text(draw, (28, 36), "Each velocity command paired with its actual mean", FONT_TITLE, anchor="lm")
    draw_text(draw, (28, 60), "Open marker is command. Filled marker is actual velocity mean. Vertical line length is signed tracking error.", FONT_SUBTITLE, "#52606d", "lm")
    y0 = 110
    for i, (cmd, actual, label, unit, color) in enumerate(AXES):
        axis_rows = [r for r in rows if abs(f(r, cmd)) > 1e-12]
        axis_rows.sort(key=lambda r: (r["mode"] != label, f(r, cmd), r["trial_id"]))
        draw_png_pairs_panel(img, draw, axis_rows, cmd, actual, f"{label} command vs actual ({unit})", 90, y0 + i * 340, 1380, 235, color)
    img.save(OUT_DIR / "velocity_cmd_actual_pairs.png")


def main():
    with SRC.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_scatter(rows)
    write_pairs(rows)
    write_scatter_png(rows)
    write_pairs_png(rows)
    write_table(rows)
    print(OUT_DIR / "velocity_cmd_actual_scatter.svg")
    print(OUT_DIR / "velocity_cmd_actual_pairs.svg")
    print(OUT_DIR / "velocity_cmd_actual_scatter.png")
    print(OUT_DIR / "velocity_cmd_actual_pairs.png")
    print(OUT_DIR / "velocity_cmd_actual_long.csv")


if __name__ == "__main__":
    main()
