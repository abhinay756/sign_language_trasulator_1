# HandSign: fresh real-time sign-language recognizer

This is a new project for recognizing these eight labels from a webcam:

`A`, `B`, `C`, `HI`, `BYE`, `THANK_YOU`, `YES`, and `NO`

It combines the useful ideas from the two supplied projects into one smaller pipeline:

1. MediaPipe extracts hand landmarks.
2. The collector saves landmark sequences, not raw video.
3. A PyTorch GRU learns both static poses (`A`, `B`, `C`) and motion gestures (`HI`, `BYE`, `THANK_YOU`).
4. One real-time program displays the current prediction and confidence.

The old projects and their checkpoints are not imported or required. You can either collect and train your own data or use the included training bundle.

## 1. Install

From this folder:

```powershell
uv sync
```

If you do not use `uv`, create a Python 3.10+ environment and install the dependencies from `pyproject.toml`.

## 2. Collect your own examples

```powershell
uv run collect-signs --samples 40 --camera 0
```

The window cycles through every label. For each sample:

- Hold `SPACE` to start recording the next 30-frame sequence.
- For `A`, `B`, and `C`, hold the pose in front of the camera.
- For `HI`, `BYE`, and `THANK_YOU`, perform the complete motion during the 30 frames.
- Press `Q` to stop.

The default output is `data/landmarks/<LABEL>/0000.npy`. Delete a label folder if you want to recollect that label.

Good starting data is 40–80 sequences per label, with different positions, distances, lighting, and speeds.

## 3. Train

```powershell
uv run train-signs --epochs 40
```

The best checkpoint is written to `artifacts/gesture_model.pt`.

## 4. Test the webcam

```powershell
uv run realtime-signs --checkpoint artifacts/gesture_model.pt --camera 0
```

Press `Q` to quit. The display includes the detected hand landmarks, smoothed prediction, confidence, and FPS.

## 5. Evaluate a saved model

```powershell
uv run evaluate-signs --checkpoint artifacts/gesture_model.pt
```

## 6. Run it on another laptop using the included dataset

The repository contains `artifacts/handsign_training_bundle.zip`. It includes 341 landmark sequences and the trained eight-class model. From the project root, extract it while preserving its folders:

```powershell
Expand-Archive artifacts/handsign_training_bundle.zip -DestinationPath . -Force
```

Then run the recognizer without collecting new samples or retraining:

```powershell
uv run realtime-signs --checkpoint artifacts/gesture_model.pt --camera 0
```

To retrain from the included dataset instead:

```powershell
uv run train-signs --epochs 40
```

The dataset contains hand landmark sequences rather than raw camera images. A friend can also collect a new dataset by following Section 2.

## Project layout

```text
config/classes.json       labels and capture settings
src/sign_language/        application code
data/landmarks/           your collected sequences
artifacts/                trained model checkpoints
```

### Extending the alphabet

Add more labels to `config/classes.json`, collect samples again, then retrain. The model output size is read from that file, so no Python code needs to change.

### Important expectation

The supplied A/B/C and hello/thank-you checkpoints cannot reliably recognize this combined six-label set. A new checkpoint must be trained from examples collected for these exact labels; this project is set up to do that.
