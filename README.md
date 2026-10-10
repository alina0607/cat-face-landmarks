# Cat Face Landmarks

> A Vision Transformer that finds nine landmarks on a cat's face (both eyes, the mouth, and three points
> on each ear), pretrained with self-supervision on unlabeled cat photos and then fine-tuned on a small
> hand-labeled set. Built from scratch in PyTorch and trained on a single Apple M-series GPU.

**Facial landmark detection** places a fixed set of named points on a face. For cats the points carry
most of the face's geometry: the eyes and mouth fix its position, scale and tilt; the ears fix its outline.
This model serves [Sell Anything](https://github.com/alina0607/sell-anything), a game whose cats are drawn by
a generative model. Every training face for that model has to be aligned with the same nine points, and
the largest clean source of cat faces comes without them.

## Results

Test split of the CAT dataset (1,000 faces, evaluated once per run). Both runs share every fine-tuning setting
and were trained for 100 epochs on the same GPU type (a Kaggle T4); only the starting weights differ.

| Encoder initialisation | NME | Failure rate (NME > 0.1) |
|---|---|---|
| Random (from scratch) | 0.0780 | 21.4% |
| MAE pretraining on 13,061 unlabeled cat faces | **0.0670** | **12.2%** |

Pretraining lowers the error on every one of the nine landmarks; the largest gains are on the mouth and the ear
tips, the smallest on the ear bases, which remain the hardest points. Learning curves, per-landmark errors and the
worst faces of each run are in [`notebooks/results.ipynb`](notebooks/results.ipynb).

## The problem: labels in one dataset, quality in another

| | [CAT dataset](https://archive.org/details/CAT_DATASET) | [AFHQ v2](https://github.com/clovaai/stargan-v2), cats |
|---|---|---|
| Images | 9,996 photos | 5,558 face crops (5,065 train, 493 test) |
| Landmarks | 9 per face, hand-placed | none |
| Framing | whole scenes; the median face spans 226 px | aligned faces filling a 512×512 frame |
| License | research use only (per the authors) | CC BY-NC 4.0 |

The CAT dataset has the labels but small, loosely framed faces. AFHQ has large, sharp, consistently framed
faces but no labels. The goal is a model that learns the landmarks from the first and places them
reliably on the second, where it has never seen a single label.

## Approach

### 1. Self-supervised pretraining on unlabeled cats

A Vision Transformer is first pretrained as a **masked autoencoder** (MAE; He et al., 2022): each image is
cut into patches, three quarters of them are hidden, and the network learns to reconstruct the missing
pixels from the visible ones. No labels are needed, so pretraining uses every cat face available,
**including the unlabeled AFHQ faces the model will later be applied to**. The encoder learns what cat
faces look like in both datasets before it learns a single landmark, which narrows the gap between them.

### 2. Fine-tuning for landmarks

The pretrained encoder is then fine-tuned on the CAT dataset's labeled faces with a lightweight decoder in
the style of **ViTPose** (Xu et al., 2022): the patch features are upsampled into one heat map per
landmark. Each landmark's position is the heat map's expected value (**integral regression**, or
soft-argmax; Sun et al., 2018) rather than its peak, so predictions are not tied to the heat map's grid
and the loss is the distance to the true point itself.

The difference in framing between the datasets is handled with augmentation: each training face is cut
out at a random zoom between tight on the face and twice its size, rotated, shifted, mirrored (with the
left and right landmarks swapped) and recolored.

### 3. Evaluation

- **Normalized mean error (NME):** the distance from each predicted landmark to the true one, divided by
  the distance between the eyes, reported per landmark on a held-out test set that is evaluated once.
- **Ablation:** the same Vision Transformer trained from scratch, without pretraining, to measure what
  self-supervision contributes.
- **Transfer to AFHQ (planned):** AFHQ has no landmarks, so a sample will be labeled by hand and used only for
  testing.

## References

- K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.
- Y. Xu, J. Zhang, Q. Zhang, D. Tao. *ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation.* NeurIPS 2022.
- X. Sun, B. Xiao, F. Wei, S. Liang, Y. Wei. *Integral Human Pose Regression.* ECCV 2018.
- W. Zhang, J. Sun, X. Tang. *Cat Head Detection: How to Effectively Exploit Shape and Texture Features.* ECCV 2008.
- Y. Choi, Y. Uh, J. Yoo, J.-W. Ha. *StarGAN v2: Diverse Image Synthesis for Multiple Domains.* CVPR 2020.

## Data and licenses

Neither dataset is redistributed here. The CAT dataset's authors state it is for research purposes only,
and AFHQ is released under CC BY-NC 4.0; both are downloaded by the user from their original sources.
