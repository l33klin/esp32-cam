#!/usr/bin/env python3
"""Convert a bounded local clip into a broadly decodable H.264/AAC MP4."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--start", type=float, default=0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.start < 0:
        parser.error("seconds must be positive and start must be nonnegative")
    if not args.input.is_file() or args.output.exists():
        parser.error("Input must exist and output must be a new file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-ss", str(args.start),
        "-i", str(args.input), "-t", str(args.seconds), "-map", "0:v:0", "-map", "0:a?",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(args.output)], check=True)
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_name,width,height,r_frame_rate", "-of", "json", str(args.output)], text=True)
    print(json.dumps(json.loads(probe), indent=2))


if __name__ == "__main__":
    main()
