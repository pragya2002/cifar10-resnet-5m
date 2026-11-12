# CIFAR-10 ResNet under 5M parameters

NYU Tandon Deep Learning, Fall 2026, Project 1. Kaggle team **Bonnie & Clyde**: Aditya Jha (aj4955), Pragya Awasthi (pa2755).

We train a pre activation Wide ResNet (WRN-22, widths 72/144/264, **4,806,954 trainable parameters**) from scratch on the CIFAR-10 training images, using knowledge distillation from larger teachers that were also trained from scratch on the same data. No pretrained weights or external data are used.

## Results

| Submission | Model | Teachers | Train data | Val (5k, TTA) | Public | Private |
| --- | --- | --- | --- | --- | --- | --- |
| kd1p_sam_600 (final) | WRN-22, 4,806,954 | PyramidNet-272 | 45k | 97.90% | 0.897 | 0.872 |
| kd5_sam_full600 (final) | WRN-22, 4,806,954 | 3 x WRN-28-10, Shake-Shake 26 2x96d, PyramidNet-272 | 50k | 98.12% (45k run) | 0.897 | 0.870 |

Final private leaderboard: 0.872, 34th of 81 teams. The full list of runs is in the report.

## Repository layout

| Path | What it is |
| --- | --- |
| `train.py` | Training, distillation, TTA inference and `submission.csv` writer. Identical to the `%%writefile train.py` cell in the notebooks |
| `notebooks/explore_session1.ipynb` | Runs `kd1p_sam_600` (final submission) and `kd5_lightreg_600` |
| `notebooks/explore_session2.ipynb` | Runs `pyr164_kd5_600` (PyramidNet-164 ShakeDrop student, 4,924,834 params) |
| `requirements.txt` | Python packages (Colab already has them) |

## Setup

Colab with a GPU (we used an A100). Download the competition data:

```
pip install -q -U kagglehub
python -c "import kagglehub; print(kagglehub.competition_download('cifar-10-res-net-architecture-challenge-dl'))"
```

Point `--data_root` at the folder that contains `train_images/`, `test_images/`, `train_labels.csv` and `test.csv`. The first run builds a cached `data_cache.npz`; pass it with `--cache` afterwards.

## Reproducing the final submissions

Every run prints the trainable parameter count and asserts it is at most 5,000,000, logs validation accuracy, and writes `submission.csv`, `final_ema.pt`, `config.json` and `summary.json` into `OUT/<run_name>/`.

1. Train the teachers (45k split; teachers are over the limit and are used only for distillation, never submitted as finals):

```
# WRN-28-10, about 36.5M params, trained three times with different seeds (run names wrn_teacher, wrn_teacher_s1, wrn_teacher_s2)
python train.py --mode teacher --run_name wrn_teacher --arch wrn --depth 28 --widths 160,320,640 --dropout 0.3 --out_dir OUT --cache OUT/data_cache.npz
```

```
# Shake-Shake 26 2x96d and PyramidNet-272 (alpha 200) with ShakeDrop, about 26.2M params each
python train.py --mode teacher --run_name shake_teacher   --arch shake   --depth 26  --widths 96  --out_dir OUT --cache OUT/data_cache.npz
python train.py --mode teacher --run_name pyramid_teacher --arch pyramid --depth 272 --widths 200 --shakedrop 0.5 --out_dir OUT --cache OUT/data_cache.npz
```

The WRN-28-10 teachers used 200 epochs (the default). Each run's exact settings, including epochs and SAM for the Shake-Shake and PyramidNet teachers, are saved in its `config.json`.

2. Distill the WRN-22 students (WRN-22 72/144/264 is the default architecture):

```
# kd1p_sam_600: PyramidNet-272 teacher only, 45k split
python train.py --mode kd --sam --epochs 600 --run_name kd1p_sam_600 \
  --teachers OUT/pyramid_teacher/final_ema.pt \
  --out_dir OUT --cache OUT/data_cache.npz --data_root DATA

# kd5_sam_full600: five teachers, all 50k training images
python train.py --mode kd --sam --epochs 600 --full_train --run_name kd5_sam_full600 \
  --teachers OUT/wrn_teacher/final_ema.pt,OUT/wrn_teacher_s1/final_ema.pt,OUT/wrn_teacher_s2/final_ema.pt,OUT/shake_teacher/final_ema.pt,OUT/pyramid_teacher/final_ema.pt \
  --out_dir OUT --cache OUT/data_cache.npz --data_root DATA
```

## Training recipe (defaults in `train.py`)

| Setting | Value |
| --- | --- |
| Optimizer | SGD, Nesterov momentum 0.9, batch size 128 |
| Learning rate | 0.1 peak, 5 epoch linear warmup, cosine decay to 0 |
| Weight decay | 5e-4 on conv and linear weights, none on BN and biases |
| SAM | rho 0.05 |
| Augmentation | reflect pad crop + flip, TrivialAugment, random erasing p 0.5; CutMix (alpha 1.0) or MixUp (alpha 0.2) on half the batches, off for the last 10% of epochs |
| Label smoothing | 0.1 |
| EMA | decay 0.999, BN buffers included; the EMA model is evaluated and submitted |
| Distillation | loss = 0.8 x KL(student, mean of teachers) at T = 4 (scaled by T^2) + 0.2 x CE; teachers run online on the same augmented batch |
| TTA | original + horizontal flip + reflect padded shifts, shift chosen on validation |
| Seeds | seed 42, split seed 42, 5,000 image validation split |

## Rules compliance

* All models are trained from scratch on the provided 50,000 training images only.
* Test images are never used for training, fine tuning or model selection. The `--unlabeled_test` option in `train.py` was never used for any submission.
* Teachers over 5M parameters are used only to produce soft targets during training.

## References

He et al., Deep Residual Learning (2016) and Identity Mappings (2016); Zagoruyko and Komodakis, Wide Residual Networks (2016); Hinton et al., Distilling the Knowledge in a Neural Network (2015); Foret et al., SAM (2021); Han et al., PyramidNet (2017); Yamada et al., ShakeDrop (2019); Gastaldi, Shake-Shake (2017); Muller and Hutter, TrivialAugment (2021).
