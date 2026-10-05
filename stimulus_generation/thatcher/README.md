# Thatcher Stimulus Generation

Scripts to generate Thatcher effect stimuli (140 scenes × 4 variants: upright/inverted original and modified) from FFHQ face images.

Derived from the [Thatcher Effect Dataset Generator](https://github.com/Erfaniaa/thatcher-effect-dataset-generator) under the GNU GPLv3 license (see [LICENSE](LICENSE)).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download FFHQ Images and Landmark Predictor
```bash
WORK_DIR="./work"

# Download the 140 public FFHQ face images (~186 MB)
python acquire_ffhq.py --destination "$WORK_DIR/ffhq"

# Download dlib 68-point landmark predictor (~64 MB)
python acquire_predictor.py --destination "$WORK_DIR/weights"
```

### 2. Generate Stimuli
```bash
python main_ffhq.py --mode main \
  --input-root "$WORK_DIR/ffhq/images1024x1024" \
  --metadata "$WORK_DIR/ffhq/selected_metadata.json" \
  --predictor "$WORK_DIR/weights/shape_predictor_68_face_landmarks.dat" \
  --output "$WORK_DIR/stimuli_thatcher"
```

The output directory contains:
- `info.csv`: Scene metadata
- `0/` ... `139/`: Each directory contains `original_str.png`, `original_inv.png`, `modified_str.png`, and `modified_inv.png`.
