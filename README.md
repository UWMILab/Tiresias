# Tiresias

Tiresias is a language-conditioned planning and world-modeling system for
navigation, driving, 3D gaming, and robot manipulation. It generates 128 action
candidates, retains eight using a Trajectory Ranker, and selects the final plan
based on world-model evidence.

## Installation

```bash
git clone https://github.com/F1y1113/Tiresias.git
cd Tiresias
conda env create -f environment.yml
conda activate tiresias
```

Download [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B):

```bash
mkdir -p pretrained_models
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir pretrained_models/Wan2.2-TI2V-5B
```

Clone the official [DINOv3](https://github.com/facebookresearch/dinov3)
repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git \
  pretrained_models/dinov3
```

Request access to **ViT-L/16 distilled, LVD-1689M** from
[Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/),
then download it using the URL provided by Meta:

```bash
mkdir -p pretrained_models/dinov3_weights
wget -c "<DINOv3-download-URL>" \
  -O pretrained_models/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m.pth
```

## Data

```bash
bash download_data.sh
```

The downloaded data has the following structure:

```text
data/
├── navigation/
│   ├── train/
│   │   ├── <trajectory>/
│   │   │   ├── <frame>.jpg
│   │   │   └── traj_data.pkl
│   │   └── ...
│   ├── val_seen/
│   └── val_unseen/
├── driving/
│   └── ...
├── gaming/
│   └── ...
└── robot/
    └── ...
```

Prepare all four domains:

```bash
python -m scripts.prepare_data --domain all
```

Use `navigation`, `driving`, `gaming`, or `robot` instead of `all` to prepare
one domain.

## Training

The examples below use `navigation`. Set `--domain` to `driving`, `gaming`, or
`robot` to use another domain. Set `--domain all` to process all four domains;
outputs are written to separate domain directories.

Training defaults are defined in `scripts/train_*.sh` and the corresponding
Python entry points: `ldp/train.py`, `lcwm/main.py`,
`rankers/tr_ranker/train.py`, and `rankers/v_ranker/train.py`.
All four training commands accept `--data_ratio`; append `--data_ratio 0.1` to
train a module on 10% of the training set.

### 1. Language diffusion policy

Prepare the language cache:

```bash
python -m ldp.cache_language --domain navigation
```

This creates the language caches for `train`, `val_seen`, and `val_unseen`.

Train the policy:

```bash
bash scripts/train_ldp.sh --domain navigation
```

Checkpoint: `weights/diffusion_policy/navigation/proposal/LDP.pt`.

### 2. World model

Prepare the DINOv3 features:

```bash
python -m lcwm.datasets.export_features --domain navigation
```

Train the world model:

```bash
bash scripts/train_wm.sh --domain navigation
```

For single-GPU training, append `--num-processes 1`.

Checkpoint: `weights/world_model/navigation/model.safetensors`.

### 3. Trajectory Ranker

Prepare the candidates and labels:

```bash
bash scripts/prepare_tr_ranker.sh --domain navigation
```

Train the ranker:

```bash
bash scripts/train_tr_ranker.sh --domain navigation
```

Checkpoint: `weights/rankers/navigation/128_to_8/model.pt`.

### 4. V-Ranker

Prepare the world-model evidence and labels:

```bash
bash scripts/prepare_v_ranker.sh --domain navigation
```

Train the ranker:

```bash
bash scripts/train_v_ranker.sh --domain navigation
```

Checkpoint: `weights/rankers/navigation/8_to_1/model.pt`.

## Inference

Run a prepared split through the full LDP → Trajectory Ranker (128→8) → World
Model → V-Ranker and score fusion (8→1) pipeline:

```bash
python tiresias_inference.py \
  --domain navigation \
  --split val_seen \
  --output-root outputs
```

Set `--domain` to a different domain to load only its four checkpoints, or set
`--domain all` to process all four domains. Outputs are written to
`outputs/<domain>/<split>/`.

To run inference on custom trajectories, set `--input` to one trajectory
directory or to a directory containing trajectory directories. Each trajectory
directory follows the format shown in the Data section. For example, with
`--domain navigation`, the input can contain Navigation trajectories from
Tiresias or other trajectories in the same format. Replace
`YOUR_INPUT_DIRECTORY` below with the input path:

```bash
python tiresias_inference.py \
  --domain navigation \
  --input YOUR_INPUT_DIRECTORY \
  --output-root outputs
```

`--split` and `--input` are two alternative modes:

- `--split` processes a complete prepared split and writes its results to
  `outputs/<domain>/<split>/`.
- `--input` prepares the supplied trajectories, caches their instructions, and
  writes the results to `outputs/<domain>/input/`.

Both modes produce the following pipeline artifacts. `RUN_DIRECTORY` is the
output directory for the selected mode shown above.

```text
RUN_DIRECTORY/
├── candidates_n128.pt        # LDP candidates
├── tr_top8_selection.pt      # Trajectory Ranker scores
├── top8.json                 # Top-8 trajectories
├── initial_observations.json
├── wm_config.json
├── wm_evidence.pt            # World-model evidence
├── fused_top1.json           # Fused Top-1 selection
├── selected_actions.jsonl    # Final action sequence
├── wm_input/
└── videos/                   # Generated videos
```

Input mode also creates the following files in the same directory:

```text
input_queries.json        # LDP inputs created from the custom trajectories
input_planning.jsonl      # Metadata for initial observations and action export
input_wm.jsonl            # Metadata for the final world-model rollout
input_language.pt         # Cached instruction embeddings
input_frames/             # Canonically named context frames
```
