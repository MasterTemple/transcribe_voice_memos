# Transcribe Voice Memos

Transcribe Voice Memos from my iPhone

### Setup

_This might not work on your computer_

Copy all voice memos to `./data/audio/*.m4a` (the output will be in `./data/transcripts/*.json`)

```bash
source ./.venv-gpu/bin/activate
uv run main.py
```

### Extra

This did help me set up whisperx: https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2554279029
