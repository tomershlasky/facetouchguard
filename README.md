# Face Touch Guard

A lightweight macOS utility that uses your webcam to detect when you touch your face and plays an alert sound to help you break the habit. Runs 100% locally — no cloud APIs, no data leaves your machine.

Built with OpenCV and MediaPipe for real-time face and hand tracking.

## How It Works

1. Captures frames from your webcam
2. Detects your face using MediaPipe Face Landmarker (478 landmarks)
3. Detects your hands using MediaPipe Hand Landmarker (21 landmarks per hand)
4. When a fingertip enters the face bounding box for several consecutive frames — plays an alert sound
5. Cooldown timer prevents alert spam

### False Positive Reduction

- **Consecutive frame requirement** — a single frame doesn't trigger an alert, only sustained contact
- **Cooldown period** — minimum 3 seconds between alerts
- **Tight face bounding box** — small margin around detected face landmarks so nearby gestures don't trigger

## Setup

**Requirements:** Python 3.9+, macOS (uses `afplay` for audio)

```bash
git clone https://github.com/tomershlasky/facetouchguard.git
cd facetouchguard
pip3 install -r requirements.txt
python3 main.py
```

On first run, two MediaPipe model files (~10MB total) are downloaded automatically to `models/`.

## Usage

```
python3 main.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `d` | Toggle debug overlay (face box + fingertip dots) |
| `SPACE` | Pause / resume detection |

### Shell Alias (Optional)

Add to your `~/.zshrc` for quick access:

```bash
alias faceguard="python3 ~/path/to/facetouchguard/main.py"
```

Then just run `faceguard` from any terminal.

## Configuration

Edit the constants at the top of `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `ALERT_SOUND` | `Sosumi.aiff` | macOS system sound to play |
| `COOLDOWN_SECONDS` | `3.0` | Minimum seconds between alerts |
| `CONSECUTIVE_FRAMES_NEEDED` | `5` | Frames of sustained touch before alerting |
| `FACE_BOX_MARGIN` | `20` | Pixels of padding around face bounding box |

Available macOS sounds: `Basso`, `Blow`, `Bottle`, `Frog`, `Funk`, `Glass`, `Hero`, `Morse`, `Ping`, `Pop`, `Purr`, `Sosumi`, `Submarine`, `Tink`

## Tech Stack

- **[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)** — Face Landmarker + Hand Landmarker (Tasks API)
- **[OpenCV](https://opencv.org/)** — Webcam capture and display
- **Python** — Single-file, ~170 lines

## Inspired By

[FaceGuard](https://github.com/timpratim/faceguard) by [@timpratim](https://github.com/timpratim)

## License

MIT
