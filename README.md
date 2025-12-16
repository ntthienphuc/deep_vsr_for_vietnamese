# Video-Only Lip Reading with Curriculum Learning

End-to-end pipeline that turns word-level annotated MP4 clips into training-ready visual features and trains a transformer-based lip-reading model with curriculum learning. Everything runs through `pipeline.py`, and the repo includes Docker support for reproducibility.

## How the Pipeline Runs
- Input: MP4 videos and matching word-level CSV labels (`Word, Start, End` in seconds) sharing the same basename under `data_3/processed_csv_final/`.
- Convert CSV labels to TXT transcripts in `data_3/processed_mp4/` while keeping the MP4s alongside the TXT files.
- Split long utterances (longer than `MAIN_REQ_INPUT_LENGTH` characters) into two halves, time-shift the second half to start at 0.0s, and save the paired MP4/TXT files to `data_3/split_output/`.
- Build `pretrain.txt`, `train.txt`, `val.txt`, and `test.txt` index lists, then move everything to `data_3/main/split_output/`.
- Extract ROI sequences and visual features with the pretrained visual frontend; save `.png` ROI mosaics and `.npy` feature tensors next to each sample. Generate `preval.txt` for pretrain/validation splitting.
- Train the VideoNet transformer with a curriculum word-count schedule `[1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37]`, auto-halving the batch size on OOM, and stopping early when validation WER plateaus and the learning rate reaches `FINAL_LR`. Checkpoints and plots are written under `checkpoints/`.
- Clean up `processed_csv_final/` and `processed_mp4/` when the run finishes.

## Requirements
- NVIDIA GPU with drivers matching CUDA 11.3 runtime.
- Docker + NVIDIA Container Toolkit **or** Python 3.8+ with FFmpeg available on PATH.
- Python deps: `pip install -r requirements.txt` (PyTorch, OpenCV, moviepy, matplotlib, etc.).

## Data and Model Preparation
- Default data root is `data_3/` (change `args["DATA_DIRECTORY"]` in `config.py` if needed).
- Place raw files in `data_3/processed_csv_final/`:
  - `sample.mp4`
  - `sample.csv` (columns: `Word, Start, End`; times in seconds; basename matches the MP4)
- The converter writes TXT labels to `data_3/processed_mp4/`. Ensure the MP4 files are also present there (copy them if they are only in `processed_csv_final/`) so later steps can find matching video/label pairs.
- Pretrained weights expected in `models/`:
  - `video-only.pt` (initial video model if you want to fine-tune)
  - `visual_frontend.pt` (required for feature extraction)
  - `language_model.pt` (optional for beam-search decoding)
  - `trained_model.pt` (your latest trained weights if resuming)

## Running with Docker Compose
```bash
docker compose up --build
```
- Mounts `data_3/` and `models/` into the container and runs `python3 pipeline.py`.
- Use `docker compose run --rm --build --gpus all lipreading_pipeline python3 pipeline.py` if you prefer an ad-hoc run.
- If `checkpoints/` exists, the trainer will ask whether to delete it before starting.

## Running Locally (without Docker)
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python pipeline.py
```

## Key Configuration (`config.py`)
- Paths: `DATA_DIRECTORY`, `DEMO_DIRECTORY`, `PRETRAINED_MODEL_FILE`, `TRAINED_MODEL_FILE`, `TRAINED_LM_FILE`, `TRAINED_FRONTEND_FILE`.
- Data: `MAIN_REQ_INPUT_LENGTH`, `PRETRAIN_VAL_SPLIT`, `NUM_WORKERS`, character set in `CHAR_TO_INDEX`.
- Training: `BATCH_SIZE`, `NUM_STEPS`, `SAVE_FREQUENCY`, `INIT_LR`→`FINAL_LR` scheduler, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA`.
- Model: transformer dims `TX_NUM_FEATURES`, `TX_ATTENTION_HEADS`, `TX_NUM_LAYERS`, `TX_FEEDFORWARD_DIM`, `TX_DROPOUT`, positional encoding `PE_MAX_LENGTH`, `NUM_CLASSES`.
- Decoding: `USE_LM`, `BEAM_WIDTH`, `LM_WEIGHT_ALPHA`, `LENGTH_PENALTY_BETA`, `THRESH_PROBABILITY`.
- Curriculum counts live in `ModelTrainer.train_model()`; edit the list there to change the schedule.

## Outputs and Artifacts
- Split lists: `data_3/pretrain.txt`, `train.txt`, `val.txt`, `test.txt`, `preval.txt`.
- Preprocessed samples: `data_3/main/split_output/<name>.mp4/.txt/.png/.npy`.
- Checkpoints: `checkpoints/models/wordcount_<k>_step_<n>_wer_<x>.pt` saved every `SAVE_FREQUENCY` steps (and at the end of each curriculum stage).
- Plots: `checkpoints/plots/wordcount_<k>_step_<n>_loss.png` and `_wer.png`.
- Archived artifacts in `final/` stay untouched by the pipeline.

## Troubleshooting
- GPU not visible in Docker: check `nvidia-smi` inside the container or run with `--gpus all`.
- Mismatched MP4/TXT counts: make sure every MP4 in `processed_mp4/` has a TXT file with the same basename; the pipeline zips sorted lists.
- OOM during training: the trainer halves the batch size automatically; you can also lower `BATCH_SIZE` or shorten the curriculum list.
- Checkpoint prompt: if you want to keep previous runs, answer `n` when asked about deleting `checkpoints/`.
