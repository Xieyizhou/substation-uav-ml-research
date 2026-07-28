#!/usr/bin/env python3
"""Build a polished local GitHub release demo video from existing assets.

This is release-media tooling only. It does not run flight code, generate
experiment outputs, modify logs, or change UAV behavior.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SIZE = (1920, 1080)
WIDTH, HEIGHT = SIZE
FPS = 30
BLACK = "#000000"
WHITE = "#ffffff"
MUTED = "#b8b8b8"
OUTPUT = ROOT / "release_assets/uav_path_planning_demo_preview.mp4"

DEFAULT_ASSETS = {
    "grid": ROOT / "docs/assets/grid_path.png",
    "gazebo": ROOT / "release_assets/raw/gazebo_flight_6x.mp4",
    "results": ROOT / "release_assets/raw/results_summary.png",
    "github": ROOT / "release_assets/raw/github_home.png",
}

SECTION_DURATIONS = {
    "title": 2.5,
    "grid": 6.0,
    "gazebo_target": 14.0,
    "results": 7.5,
    "github": 6.0,
}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def ffmpeg(args: list[str]) -> None:
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args])


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def require_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"ERROR: missing required tool(s): {', '.join(missing)}")


def system_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_center(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.ImageFont,
    y: int,
    fill: str = WHITE,
) -> None:
    width, _ = text_size(draw, text, font)
    draw.text(((WIDTH - width) / 2, y), text, font=font, fill=fill)


def draw_caption(draw: ImageDraw.ImageDraw, caption: str) -> None:
    caption_font = system_font(38)
    width, height = text_size(draw, caption, caption_font)
    x = (WIDTH - width) / 2
    y = HEIGHT - 86
    draw.text((x, y), caption, font=caption_font, fill=WHITE)
    draw.line((x, y + height + 16, x + width, y + height + 16), fill="#333333", width=1)


def contain_box(
    source_size: tuple[int, int],
    max_size: tuple[int, int],
) -> tuple[int, int]:
    src_w, src_h = source_size
    max_w, max_h = max_size
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


def render_title(path: Path) -> None:
    image = Image.new("RGB", SIZE, BLACK)
    draw = ImageDraw.Draw(image)
    title_font = system_font(82, bold=True)
    subtitle_font = system_font(34)

    draw_center(draw, "UAV Path Planning Demo", font=title_font, y=438)
    draw_center(draw, "PX4 · Gazebo · MAVSDK · A*", font=subtitle_font, y=548, fill=MUTED)
    image.save(path)


def render_placeholder(path: Path, *, title: str, caption: str, detail: str = "") -> None:
    image = Image.new("RGB", SIZE, BLACK)
    draw = ImageDraw.Draw(image)
    draw_center(draw, title, font=system_font(58, bold=True), y=448)
    draw_center(draw, detail or "Source asset missing", font=system_font(30), y=530, fill=MUTED)
    draw_caption(draw, caption)
    image.save(path)


def render_image_section(path: Path, *, source: Path, caption: str) -> None:
    image = Image.new("RGB", SIZE, BLACK)
    source_image = Image.open(source).convert("RGB")
    target_size = contain_box(source_image.size, (int(WIDTH * 0.88), int(HEIGHT * 0.76)))
    content = source_image.resize(target_size, Image.Resampling.LANCZOS)
    x = (WIDTH - content.width) // 2
    y = (HEIGHT - content.height) // 2 - 24
    image.paste(content, (x, y))

    draw = ImageDraw.Draw(image)
    draw_caption(draw, caption)
    image.save(path)


def render_results_card(path: Path) -> None:
    image = Image.new("RGB", SIZE, BLACK)
    draw = ImageDraw.Draw(image)
    title_font = system_font(58, bold=True)
    row_font = system_font(34)
    note_font = system_font(26)

    draw.text((360, 246), "Curated Results", font=title_font, fill=WHITE)
    rows = [
        ("Static A*", "completed"),
        ("Perception response", "305 slow-down events"),
        ("Replan log-only", "4 local replan candidates"),
        ("Active replan prototype", "1 route replacement"),
    ]
    y = 380
    for label, value in rows:
        draw.text((360, y), label, font=row_font, fill=WHITE)
        draw.text((1040, y), value, font=row_font, fill=MUTED)
        y += 78

    draw.text(
        (360, 745),
        "Active replanning is prototype-level and reserved for deeper validation.",
        font=note_font,
        fill=MUTED,
    )
    draw_caption(draw, "Curated simulation results")
    image.save(path)


def render_video_overlay(path: Path) -> None:
    overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    caption = "PX4/Gazebo execution via MAVSDK waypoints"
    label = "6x speed"

    label_font = system_font(28)
    label_w, label_h = text_size(draw, label, label_font)
    padding_x = 18
    padding_y = 10
    box = (
        WIDTH - label_w - padding_x * 2 - 54,
        46,
        WIDTH - 54,
        46 + label_h + padding_y * 2,
    )
    draw.rounded_rectangle(box, radius=12, fill=(0, 0, 0, 178), outline=(120, 120, 120, 160), width=1)
    draw.text((box[0] + padding_x, box[1] + padding_y - 2), label, font=label_font, fill=WHITE)

    caption_font = system_font(38)
    cap_w, cap_h = text_size(draw, caption, caption_font)
    cap_x = (WIDTH - cap_w) / 2
    cap_y = HEIGHT - 86
    draw.rectangle((0, HEIGHT - 140, WIDTH, HEIGHT), fill=(0, 0, 0, 185))
    draw.text((cap_x, cap_y), caption, font=caption_font, fill=WHITE)
    draw.line((cap_x, cap_y + cap_h + 16, cap_x + cap_w, cap_y + cap_h + 16), fill=(80, 80, 80, 180), width=1)
    overlay.save(path)


def image_to_video(image: Path, output: Path, *, duration: float) -> None:
    ffmpeg(
        [
            "-loop",
            "1",
            "-i",
            image.as_posix(),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={FPS},format=yuv420p",
            "-an",
            output.as_posix(),
        ]
    )


def video_to_segment(source: Path, overlay: Path, output: Path) -> float:
    source_duration = ffprobe_duration(source)
    duration = 14.0 if source_duration > 14.5 else source_duration
    filters = (
        f"[0:v]scale={int(WIDTH * 0.9)}:{int(HEIGHT * 0.78)}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={FPS},setsar=1[base];"
        "[base][1:v]overlay=0:0:format=auto,format=yuv420p[out]"
    )
    ffmpeg(
        [
            "-i",
            source.as_posix(),
            "-loop",
            "1",
            "-i",
            overlay.as_posix(),
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            output.as_posix(),
        ]
    )
    return duration


def concat(segments: list[Path], output: Path, work_dir: Path) -> None:
    list_file = work_dir / "segments.txt"
    list_file.write_text(
        "".join(f"file '{segment.as_posix()}'\n" for segment in segments),
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file.as_posix(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output.as_posix(),
        ]
    )


def warn_missing(path: Path, replacement: str) -> None:
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    print(f"WARNING: missing {display}; using {replacement}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the polished local GitHub release demo video."
    )
    parser.add_argument("--grid", default=DEFAULT_ASSETS["grid"].as_posix())
    parser.add_argument("--gazebo", default=DEFAULT_ASSETS["gazebo"].as_posix())
    parser.add_argument("--results", default=DEFAULT_ASSETS["results"].as_posix())
    parser.add_argument("--github", default=DEFAULT_ASSETS["github"].as_posix())
    parser.add_argument("--output", default=OUTPUT.as_posix())
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> None:
    args = parse_args()
    require_tools()

    assets = {
        "grid": resolve(args.grid),
        "gazebo": resolve(args.gazebo),
        "results": resolve(args.results),
        "github": resolve(args.github),
    }
    output = resolve(args.output)

    with tempfile.TemporaryDirectory(prefix="make_demo_video_", dir=output.parent) as temp:
        work_dir = Path(temp)
        stills = {
            "title": work_dir / "title.png",
            "grid": work_dir / "grid.png",
            "gazebo_placeholder": work_dir / "gazebo_placeholder.png",
            "gazebo_overlay": work_dir / "gazebo_overlay.png",
            "results": work_dir / "results.png",
            "github": work_dir / "github.png",
        }
        segments = [
            work_dir / "01_title.mp4",
            work_dir / "02_grid.mp4",
            work_dir / "03_gazebo.mp4",
            work_dir / "04_results.mp4",
            work_dir / "05_github.mp4",
        ]

        render_title(stills["title"])
        image_to_video(stills["title"], segments[0], duration=SECTION_DURATIONS["title"])

        grid_caption = "A* route planning with obstacle inflation"
        if assets["grid"].exists():
            render_image_section(stills["grid"], source=assets["grid"], caption=grid_caption)
        else:
            warn_missing(assets["grid"], "black path-preview placeholder")
            render_placeholder(stills["grid"], title="A* path preview", caption=grid_caption)
        image_to_video(stills["grid"], segments[1], duration=SECTION_DURATIONS["grid"])

        if assets["gazebo"].exists():
            render_video_overlay(stills["gazebo_overlay"])
            gazebo_duration = video_to_segment(assets["gazebo"], stills["gazebo_overlay"], segments[2])
            print(f"Using Gazebo clip duration: {gazebo_duration:.2f}s")
        else:
            warn_missing(assets["gazebo"], "black flight-execution placeholder")
            render_placeholder(
                stills["gazebo_placeholder"],
                title="PX4/Gazebo flight execution",
                caption="PX4/Gazebo execution via MAVSDK waypoints",
                detail="Add release_assets/raw/gazebo_flight_6x.mp4",
            )
            image_to_video(
                stills["gazebo_placeholder"],
                segments[2],
                duration=SECTION_DURATIONS["gazebo_target"],
            )

        if assets["results"].exists():
            render_image_section(
                stills["results"],
                source=assets["results"],
                caption="Curated simulation results",
            )
        else:
            warn_missing(assets["results"], "black minimalist results card")
            render_results_card(stills["results"])
        image_to_video(stills["results"], segments[3], duration=SECTION_DURATIONS["results"])

        github_caption = "Code, sample outputs, and release assets available on GitHub"
        if assets["github"].exists():
            render_image_section(stills["github"], source=assets["github"], caption=github_caption)
        else:
            warn_missing(assets["github"], "black GitHub/release placeholder")
            render_placeholder(
                stills["github"],
                title="GitHub release",
                caption=github_caption,
                detail="v0.2.0",
            )
        image_to_video(stills["github"], segments[4], duration=SECTION_DURATIONS["github"])

        concat(segments, output, work_dir)

    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
