# Face Meme Studio

A local webcam project that matches your live facial geometry to the closest face in a folder of meme images, then overlays that photo on your face. It uses MediaPipe landmarks and nearest-feature matching; it does not understand meme captions, identify people, or match the meaning of a joke.

## Setup

Use Python 3.11. In Terminal, from this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Allow camera access for Terminal if macOS asks. The camera runs only while the app is open. Press `q` to quit, `m` to toggle the meme overlay, `s` to browse photos, and `d` to toggle the face mesh.

## Add meme photos

Put JPG, JPEG, PNG, or WebP images directly in `memes/photos/`; each photo is a separate entry. Choose images you made or have permission to use. Face matching works best with a clearly visible face. Images without a detected face can still be browsed manually with `s`.

Images are ignored by Git and stay on your computer. To import image links, create `memes/photo_sources.txt` with one HTTPS URL per line, then run `python import_photos.py`. That source-list file is also ignored by Git.

## How matching works

At startup, MediaPipe indexes faces in the local photos. While the camera runs, the app compares eye, eyebrow, and mouth measurements against each indexed image. It expands the face crop to show more surrounding meme context. Terminal prints a filename and distance when the selected match changes; a lower distance means closer geometry, not a confidence percentage.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic landmarks and do not open the camera.
