#!/usr/bin/env python3
"""
Transcribe a folder of audio files with WhisperX and store the results as JSON.

Directory layout expected:
    ./data/audio/      – source audio files (e.g. *.m4a, *.wav, *.mp3)
    ./data/transcripts/ – destination folder for *.json files
"""

import os
import json
import argparse
from pathlib import Path

# third‑party libraries ---------------------------------------------------------
import whisperx
import torch

# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using WhisperX."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("./data/audio"),
        help="Folder that contains the audio files to transcribe.",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("./data/transcripts"),
        help="Folder where the JSON transcript files will be written.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model"),
        help="Folder where WhisperX model files will be cached/downloaded.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16" if torch.cuda.is_available() else "int8",
        help="Precision used for inference (float16, bfloat16, int8, etc.).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the transcriber (larger → faster but more memory).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".m4a", ".wav", ".mp3", ".flac"],
        help="Audio file extensions to look for.",
    )
    return parser.parse_args()


def get_audio_files(audio_dir: Path, extensions: list[str]) -> list[Path]:
    """Return a sorted list of audio files that match the allowed extensions."""
    files = [
        p for p in audio_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    files.sort()
    return files


def transcribe_file(
    model,
    audio_path: Path,
    batch_size: int,
) -> dict:
    """
    Load an audio file, run the WhisperX model and return the full result dict.
    Only the `segments` key is later written to disk, but we keep the whole dict
    in case you need extra info later.
    """
    # WhisperX expects a numpy array (float32) sampled at 16 kHz.
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)
    return result


# def transcribe_file_cached(
#     model,
#     audio_path: Path,
#     batch_size: int,
#     transcript_path: Path,
# ) -> dict:
#     """
#     Load an audio file, run the WhisperX model and return the full result dict.
#     Only the `segments` key is later written to disk, but we keep the whole dict
#     in case you need extra info later.
#     """
#     if trans
#     transcribe_file(audio_path, batch_size)
#     return result


def write_transcript(
    transcript_path: Path,
    segments: list[dict],
) -> None:
    """
    Write the list of segment dictionaries to a JSON file.
    The structure matches WhisperX's default output.
    """
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with transcript_path.open("w", encoding="utf-8") as fp:
        json.dump({"segments": segments}, fp, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------------------
    # Load WhisperX model once – this can take several seconds/minutes
    # ----------------------------------------------------------------------
    print(f"[+] Loading WhisperX model `large-v3-turbo` onto {args.device} ...")
    model = whisperx.load_model(
        "large-v3-turbo",
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(args.model_dir),
    )
    print("[+] Model loaded.\n")

    # ----------------------------------------------------------------------
    # Gather files to process
    # ----------------------------------------------------------------------
    audio_files = get_audio_files(args.audio_dir, args.extensions)
    if not audio_files:
        print(f"[!] No audio files found in {args.audio_dir} with extensions {args.extensions}")
        return

    print(f"[+] Found {len(audio_files)} audio file(s) to transcribe.\n")

    # ----------------------------------------------------------------------
    # Process each file
    # ----------------------------------------------------------------------
    for audio_path in audio_files:
        # Output file will have the same stem but .json extension
        transcript_path = args.transcript_dir / f"{audio_path.stem}.json"

        # Skip already‑transcribed files (optional – comment out if you want to force re‑run)
        if transcript_path.is_file():
            print(f"[=] Skipping {audio_path.name} – transcript already exists.")
            continue

        print(f"[>] Transcribing {audio_path.name} ...")
        try:
            result = transcribe_file(model, audio_path, batch_size=args.batch_size)
        except Exception as exc:
            print(f"[!] Transcription failed for {audio_path.name}: {exc}")
            continue

        # Write only the segments (you can also store the whole result if you wish)
        write_transcript(transcript_path, result.get("segments", []))
        print(f"[+] Saved transcript to {transcript_path}\n")

    print("[*] All done!")


if __name__ == "__main__":
    main()
