# Gesture Webcam Audio

This package provides a webcam capture entry point and dependency setup for MediaPipe and audio output.

## Conda setup (recommended for PyAudio)

```bash
conda env create -f environment.yml
conda activate gesture-webcam-audio
```

## Build wheel

```bash
python -m pip install --upgrade pip build
python -m build --wheel
```

The wheel will be created in `dist/`.

## Run webcam preview

```bash
gesture-webcam
```

Press `q` to quit.
