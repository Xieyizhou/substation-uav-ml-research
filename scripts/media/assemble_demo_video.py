#!/usr/bin/env python3
"""Assemble a short release-demo MP4 from existing local assets.

This script does not run experiments, generate plots, or change flight behavior.
It only composes release media from already-existing images, videos, and curated
sample output files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
WIDTH = 1920
HEIGHT = 1080
FPS = 30


def run_ffmpeg(args: list[str]) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", *args]
    subprocess.run(command, check=True)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to assemble the demo video.")


def rel(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def video_base_filter() -> str:
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x08111f,"
        f"fps={FPS},setsar=1,format=yuv420p"
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                line = candidate
            else:
                wrapped.append(line)
                line = word
        wrapped.append(line)
    return wrapped


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    *,
    font: ImageFont.ImageFont,
    y: int,
    fill: str,
    line_gap: int,
) -> None:
    current_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        draw.text(((WIDTH - width) / 2, current_y), line, font=font, fill=fill)
        current_y += height + line_gap


def render_card_image(output: Path, *, title: str, body: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#08111f")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 150), fill="#102a43")
    draw.rectangle((110, 185, WIDTH - 110, HEIGHT - 135), outline="#d9e2ec", width=3)

    title_font = load_font(74)
    body_font = load_font(44)
    body_lines = wrap_lines(draw, body, body_font, WIDTH - 360)

    draw_centered_lines(
        draw,
        [title],
        font=title_font,
        y=260,
        fill="#ffffff",
        line_gap=18,
    )
    draw_centered_lines(
        draw,
        body_lines,
        font=body_font,
        y=430,
        fill="#d9e2ec",
        line_gap=24,
    )
    image.save(output)


def render_captioned_image(output: Path, *, source: Path, title: str) -> None:
    base = Image.new("RGB", (WIDTH, HEIGHT), "#08111f")
    source_image = Image.open(source).convert("RGB")
    contained = ImageOps.contain(source_image, (WIDTH, HEIGHT - 150))
    x = (WIDTH - contained.width) // 2
    y = 150 + ((HEIGHT - 150 - contained.height) // 2)
    base.paste(contained, (x, y))

    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 0, WIDTH, 132), fill="#08111f")
    title_font = load_font(52)
    draw_centered_lines(
        draw,
        [title],
        font=title_font,
        y=38,
        fill="#ffffff",
        line_gap=0,
    )
    base.save(output)


def make_png_segment(output: Path, *, source: Path, duration: int) -> None:
    run_ffmpeg(
        [
            "-y",
            "-loop",
            "1",
            "-i",
            source.as_posix(),
            "-t",
            str(duration),
            "-vf",
            f"fps={FPS},format=yuv420p",
            "-an",
            output.as_posix(),
        ]
    )


def make_card(
    output: Path,
    build_dir: Path,
    *,
    duration: int,
    title: str,
    body: str,
) -> None:
    card_image = build_dir / f"{output.stem}.png"
    render_card_image(card_image, title=title, body=body)
    make_png_segment(output, source=card_image, duration=duration)


def make_image_segment(
    output: Path,
    build_dir: Path,
    *,
    source: Path,
    duration: int,
    title: str,
) -> None:
    captioned_image = build_dir / f"{output.stem}.png"
    render_captioned_image(captioned_image, source=source, title=title)
    make_png_segment(output, source=captioned_image, duration=duration)


def make_video_segment(
    output: Path,
    *,
    source: Path,
    duration: int,
) -> None:
    run_ffmpeg(
        [
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            source.as_posix(),
            "-t",
            str(duration),
            "-vf",
            video_base_filter(),
            "-an",
            output.as_posix(),
        ]
    )


def summarize_results(comparison_csv: Path) -> str:
    if not comparison_csv.exists():
        return (
            "Curated sample outputs\n"
            "comparison_summary.csv\n"
            "comparison_summary.md\n"
            "selected_runs.json"
        )

    with comparison_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pass_count = sum(1 for row in rows if row.get("final_status") == "PASS")
    violations = sum(int(float(row.get("safety_buffer_violation_count") or 0)) for row in rows)
    slow_down = sum(int(float(row.get("slow_down_event_count") or 0)) for row in rows)
    replacements = sum(int(float(row.get("active_route_replacement_count") or 0)) for row in rows)

    return (
        "Curated landmark comparison\n"
        f"{pass_count}/{len(rows)} stages passed current checks\n"
        f"{violations} safety-buffer violations\n"
        f"{slow_down} slow_down events\n"
        f"{replacements} active route replacement"
    )


def concat_segments(segments: list[Path], output: Path, build_dir: Path) -> None:
    concat_file = build_dir / "segments.txt"
    concat_file.write_text(
        "".join(f"file '{segment.as_posix()}'\n" for segment in segments),
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file.as_posix(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output.as_posix(),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a 35-45 second UAV portfolio demo video."
    )
    parser.add_argument(
        "--path-preview",
        default="docs/assets/grid_path.png",
        help="A* grid path preview image.",
    )
    parser.add_argument(
        "--gazebo-video",
        default="release_assets/source/gazebo_flight.mp4",
        help="Optional Gazebo flight recording.",
    )
    parser.add_argument(
        "--terminal-screenshot",
        default="release_assets/source/terminal_log.png",
        help="Optional terminal/log screenshot.",
    )
    parser.add_argument(
        "--result-screenshot",
        default="release_assets/source/result_summary.png",
        help="Optional result summary screenshot.",
    )
    parser.add_argument(
        "--github-screenshot",
        default="release_assets/source/github_release.png",
        help="Optional GitHub release screenshot.",
    )
    parser.add_argument(
        "--comparison-csv",
        default="data/sample_outputs/comparison_summary.csv",
        help="Curated comparison CSV used for the result card.",
    )
    parser.add_argument(
        "--output",
        default="release_assets/uav_path_planning_demo_preview.mp4",
        help="Output MP4 path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_ffmpeg()

    path_preview = rel(Path(args.path_preview))
    if not path_preview.exists():
        raise SystemExit(f"Missing required path preview image: {path_preview}")

    output = rel(Path(args.output))
    gazebo_video = rel(Path(args.gazebo_video))
    terminal_screenshot = rel(Path(args.terminal_screenshot))
    result_screenshot = rel(Path(args.result_screenshot))
    github_screenshot = rel(Path(args.github_screenshot))
    comparison_csv = rel(Path(args.comparison_csv))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="demo_video_", dir=output.parent) as temp_name:
        build_dir = Path(temp_name)
        segments = [
            build_dir / "01_title.mp4",
            build_dir / "02_path_preview.mp4",
            build_dir / "03_gazebo.mp4",
            build_dir / "04_results.mp4",
            build_dir / "05_release.mp4",
        ]

        make_card(
            segments[0],
            build_dir,
            duration=5,
            title="Autonomous UAV Path Planning",
            body=(
                "PX4/Gazebo simulation demo\n"
                "A* planning + MAVSDK execution\n"
                "Risk response and local replanning evaluation"
            ),
        )
        make_image_segment(
            segments[1],
            build_dir,
            source=path_preview,
            duration=10,
            title="A* grid path preview",
        )

        if gazebo_video.exists():
            make_video_segment(
                segments[2],
                source=gazebo_video,
                duration=10,
            )
        else:
            make_card(
                segments[2],
                build_dir,
                duration=10,
                title="PX4/Gazebo execution",
                body=(
                    "Optional source asset missing:\n"
                    f"{Path(args.gazebo_video).as_posix()}\n"
                    "Add a compact local recording to include flight footage."
                ),
            )

        result_source = result_screenshot if result_screenshot.exists() else terminal_screenshot
        if result_source.exists():
            make_image_segment(
                segments[3],
                build_dir,
                source=result_source,
                duration=10,
                title="Risk-response and result summary",
            )
        else:
            make_card(
                segments[3],
                build_dir,
                duration=10,
                title="Risk-response and result summary",
                body=summarize_results(comparison_csv),
            )

        if github_screenshot.exists():
            make_image_segment(
                segments[4],
                build_dir,
                source=github_screenshot,
                duration=6,
                title="GitHub release: v0.2.0",
            )
        else:
            make_card(
                segments[4],
                build_dir,
                duration=6,
                title="GitHub release",
                body=(
                    "v0.2.0\n"
                    "Attach the generated MP4 as a release asset\n"
                    "Keep raw logs and large media out of git"
                ),
            )

        concat_segments(segments, output, build_dir)

    print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
