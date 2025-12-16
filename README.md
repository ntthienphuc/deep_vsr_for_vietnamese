# Video-Only Lip Reading with Curriculum Learning

End-to-end pipeline that turns word-level annotated MP4 clips into training-ready visual features and trains a transformer-based lip-reading model with curriculum learning. Everything still runs through `python pipeline.py`, now backed by a cleaner package layout.

## Project Layout
- `lipreading/` — source package  
  - `config.py` — all runtime paths and hyperparameters  
  - `pipeline.py` — orchestrates the full pipeline  
  - `data_processing/` — CSV→TXT conversion, label/video splitting, split file creation, preprocessing  
  - `datasets/` — dataset classes and collate helpers  
  - `models/` — model definitions (VideoNet, visual frontend, char LM)  
  - `training/` — curriculum trainer  
  - `utils/` — decoding, metrics, preprocessing helpers  
- `models/` — place pretrained/finetuned `.pt` weights here (visual frontend, LM, checkpoints)
- `data_3/` — your data root (processed CSV/MP4, split files, preprocessed features)
- `checkpoints/` — created during training (models + plots)
- `final/` — archived demo outputs (untouched by the pipeline)

## Quickstart (Local)
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python pipeline.py
```

## Quickstart (Docker Compose)
```bash
docker compose up --build
```
- Uses `CMD ["python3", "pipeline.py"]`, mounts `data_3/` and `models/`, and enables GPU via the compose file.
- For ad-hoc runs: `docker compose run --rm --build --gpus all lipreading_pipeline python3 pipeline.py`

## Data and Weights
- Default data root: `data_3/` (change in `lipreading/config.py`).
- Place raw pairs in `data_3/processed_csv_final/`:
  - `sample.mp4`
  - `sample.csv` (`Word, Start, End` in seconds; basename matches the MP4)
- Converter writes TXT labels to `data_3/processed_mp4/`; ensure the matching MP4s live there too.
- Put weights in top-level `models/`:
  - `visual_frontend.pt` (required for preprocessing)
  - `video-only.pt` (optional warm-start)
  - `language_model.pt` (optional for beam search)
  - `trained_model.pt` (latest checkpoint if resuming)

## Pipeline Stages
1) Convert CSV labels to TXT transcripts in `processed_mp4/`.  
2) Split long utterances (> `MAIN_REQ_INPUT_LENGTH`) into halves with adjusted timings.  
3) Create `pretrain/train/val/test` split lists and move data under `data_3/main/split_output/`.  
4) Extract ROI mosaics (`.png`) and visual features (`.npy`) with the pretrained frontend; generate `preval.txt`.  
5) Train VideoNet with curriculum word counts `[1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37]`, auto-halving batch size on OOM, early-stopping when WER plateaus at `FINAL_LR`.  
6) Save checkpoints/plots under `checkpoints/`, then clean `processed_csv_final/` and `processed_mp4/`.

## Key Configuration (`lipreading/config.py`)
- Paths: `DATA_DIRECTORY`, `DEMO_DIRECTORY`, `PRETRAINED_MODEL_FILE`, `TRAINED_MODEL_FILE`, `TRAINED_LM_FILE`, `TRAINED_FRONTEND_FILE`.
- Data: `MAIN_REQ_INPUT_LENGTH`, `PRETRAIN_VAL_SPLIT`, `NUM_WORKERS`, `CHAR_TO_INDEX`.
- Training: `BATCH_SIZE`, `NUM_STEPS`, `SAVE_FREQUENCY`, `INIT_LR`→`FINAL_LR`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA`.
- Model: `TX_NUM_FEATURES`, `TX_ATTENTION_HEADS`, `TX_NUM_LAYERS`, `TX_FEEDFORWARD_DIM`, `TX_DROPOUT`, `PE_MAX_LENGTH`, `NUM_CLASSES`.
- Decoding: `USE_LM`, `BEAM_WIDTH`, `LM_WEIGHT_ALPHA`, `LENGTH_PENALTY_BETA`, `THRESH_PROBABILITY`.
- Curriculum list lives in `ModelTrainer.train_model()` inside `lipreading/training/model_trainer.py`.

## Outputs
- Lists: `data_3/pretrain.txt`, `train.txt`, `val.txt`, `test.txt`, `preval.txt`.
- Preprocessed samples: `data_3/main/split_output/<name>.mp4/.txt/.png/.npy`.
- Checkpoints: `checkpoints/models/wordcount_<k>_step_<n>_wer_<x>.pt`.
- Plots: `checkpoints/plots/wordcount_<k>_step_<n>_loss.png` and `_wer.png`.

## Troubleshooting
- GPU missing in Docker: run `nvidia-smi` inside the container or add `--gpus all`.
- MP4/TXT mismatch: ensure every basename has both files in `processed_mp4/`; the pipeline zips sorted lists.
- OOM: batch auto-halves; you can also reduce `BATCH_SIZE` or shorten the curriculum.
- Checkpoint prompt: answer `n` if you want to keep an existing `checkpoints/`.
