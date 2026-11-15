---
title: Homework 2 -- Object Detection on Drone Videos
author: Pavlunyshyn Bohdan
course: Artificial Intelligence · UCU
date: 2026-10-12
---

## Introduction

The core task of this homework is to develop a deep learning model capable of accurately identifying objects across the video. The objects are split into 8 different classes: person, bicycle, car, van, truck, tricycle, bus, motor. For detection 4 algorithms were chosen: One-stage Anchor-based, Two-stage Anchor-based, One-stage Anchor-free and a Query-based. Also a custom pipeline was developed, which will be explained more broadly in other section.

The models were trained mainly on a free Tesla T4 GPU in Google Colab and on NVIDIA GeForce GTX 1660 SUPER (local PC). 

All models and helper functions were written in `hw2_detection.py` and `hw2_detection.ipynb` is used for data visualization and models run.

Metric mAP@[.5:.95] was used to compare different models.

As a result, submission from models ensemble predictions gave **0.282** on the Kaggle public leaderboard, which beats the baseline of **0.222**.

## Method

### Data analysis and preprocessing

The data is represented as clips and each clip is made of frames. In total there are **56** train clips and **24,201** frames, **48** test clips with **3,811** frames and **7** validation clips with **2,846** frames. Resolution varies from clip to clip (from **680x382** to **1920x1080**).

After resizing each frame to 800px the median object size (√area) is equal to ~32px. 50% of objects are <32px and only 5% <14px (Figure 1). Also, 80% of boxes have h/w in 0.48–2.28, so default anchor ratios (0.5, 1, 2) were kept.

In conclusion, the smallest objects become the biggest problems. To adress this, we monitored *AP_small*, tuned anchors and tried P2 level.

![Object sizes](images/object_sizes.png)

*Figure 1. Object size (√area) per class in original pixels (left) and for all objects before and after resizing to 800 px (right); dashed lines mark 16 / 32 / 96 px.*

Object density varies strongly between different clips (from **0** to **180** objects per frame), the median is **43** objects per frame and 99% of frames have <= 142 objects. If we compare frames from two distinct clips, the difference in density is much bigger than in frames from the same clip. As a result, the data is split by clips to prevent leakage of data. Also, due to high density there are up to 500 detections per image and 500-query RT-DETR variant

![Scene density](images/density_per_clip.png)

*Figure 2. Objects per frame in train and val (left) and per clip, sorted by median density (right; orange = validation clips).*

The classes are not distributed equally. Classes `person` and `car` are the most frequent, while `bus` is the rarest.

![Class distribution](images/class_distribution.png)

*Figure 3. Number of objects per class in the training and validation sets (log scale).*

**Ignored regions:** in VisDrone clips, there are areas where objects aren't annotated due to their small size or their large number (crowds of people). Also class `others` fits here. If this problem is ignored, the model will see a huge group of unlabelled but relevant objects and learn to treat them as background.

In order to prevent this:
- **during training:** such areas are filled with grey, so the model doesn't learn something wrong. Also GT boxes mostly inside these areas are dropped.
- **during validation:** detections that lie more than halfway within an ignore region are also ignored before calculating mAP. The only exception is detection which coincides with the annotated box: we keep it.

**Conversion check:** the conversion has been verified in two ways and it is correct.
- **visually:** we drew the boxes around objects in frames and they land perfectly on the objects.
- **automatically:** compared with annotations in COCO format on 200 random frames. The amount of boxes and ignored regions match and IoU ≈ 1.

**Split by clip:** the official train/val split was used. All clips were checked on repeatability: none appears in both.

**Frame subsampling:** in order to make training faster and easier only every 10th frame was used. After 1 frame **95%** of boxes stay in place, compared to only **21%** after 10 frames. In validation data every 5th frame was used. This leaves ~2,400 training and ~570 validation frames.

**Resize and augmentation:** every frame is resized to 800px (short side) and <= 1333px (long side), proportions remain the same. Big frames shrink, while the small ones become bigger. For data augmentation only horizontal flip with probability of 0.5 was used.

**Normalization:** ImageNet mean/std inside torchvision models, 0–1 input for RT-DETR.

### Training protocol

Every detector have used the same protocol: COCO-pretrained weights, the same validation and train frames, 8 epochs, an effective batch of 4 and seed 42. CNNs were trained with SGD (lr 0.005, cosine schedule, mixed precision), RT-DETR with AdamW (lr 1e-4, backbone ×0.1). After every epoch the model was evaluated on validation data and the best epoch was kept. Changes in mAP smaller than ±0.01 are treated as noise. All CNNs were trained on a Colab T4, while RT-DETR was trained locally on a GTX 1660 SUPER in fp32 (fp16 gave NaN losses) with 90-minute budget per every run.

### Detectors

#### Two-stage anchor-based: Faster R-CNN

Torchvision `fasterrcnn_resnet50_fpn_v2` was used and COCO-pretrained, with a new classifier for 8 classes. The RPN generates candidate boxes while the second stage classifies each proposal and improves its box. Two changes were tested in turn: more RPN proposals (from 2000 to 3000 in training and from 1000 to 2000 during testing) and a GIoU box loss in the second stage instead of smooth-L1.

#### One-stage anchor-based: RetinaNet

`retinanet_resnet50_fpn_v2` was used. It uses focal loss against the large number of background anchors. Anchor sizes were chosen from an anchor-coverage check. With anchors from **32px** only 25% of small and none of the tiny objects could reach IoU >= 0.5. However, with anchors from **16px**, the numbers grew to **95%** and **55%**. For this task anchor sizes were tuned from 32–512 to 16–256 px and focal loss gamma was changed (2.0 -> 1.5).

#### One-stage anchor-free: FCOS

In FCOS a feature-map location is positive for some box if it lies inside of it and is close to the centre of the box. The box size must fit the FPN level. Simulation of this rule (with default levels of P3-P7) showed that 18.6% of tiny objects get no positive location, but an extra P2 level drops this number to 1.5%. The P2 level (off/on) and the box loss (GIoU -> CIoU) were tuned.

#### Query-based: RT-DETR

`PekingU/rtdetr_r50vd` from Hugging Face was used. Hungarian (bipartite) matching assigns each object to one query during training, so the model learns not to produce duplicates and needs no NMS. As frames may contain up to 180 objects, 300 queries give only ~2 queries per object in the densest frames, so 500 queries were tested.

### Pipeline

The pipeline is based on the best required CNN, `frcnn_props_giou`, and the same validation frames are used for measuring steps. 
- **Part A:** changes only for inference. A lower score threshold (0.001) with up to 500 detections per frame (A1), the NMS threshold 0.5 / 0.6 / 0.7 (A2) and horizontal flip augmentation combined with NMS (A3).
- **Part B:** changes the model itself. Smaller RPN anchors (16-256px) with the same protocol (B1). Also, an ensemble of Faster R-CNN (with A1) and RT-DETR q500, merged with class-wise NMS (B2).

## Results

### Required detectors

**Table 1.** Validation results of all required detectors (same frames, same metric; the best epoch by val mAP is kept).

| Model | Family | mAP | AP50 | AP_small | AP_large | AR | Best epoch |
|---|---|---|---|---|---|---|---|
| frcnn_base | two-stage | 0.246 | 0.478 | 0.105 | 0.451 | 0.350 | 6 |
| frcnn_props | two-stage | 0.242 | 0.494 | 0.137 | 0.422 | 0.390 | 3 |
| frcnn_props_giou | two-stage | 0.250 | 0.494 | 0.121 | 0.435 | 0.399 | 4 |
| retina_base | one-stage, anchors | 0.193 | 0.369 | 0.078 | 0.355 | 0.357 | 8 |
| retina_anchors | one-stage, anchors | 0.171 | 0.367 | 0.070 | 0.301 | 0.383 | 8 |
| retina_anchors_focal | one-stage, anchors | 0.179 | 0.381 | 0.069 | 0.324 | 0.379 | 8 |
| fcos_base | anchor-free | 0.190 | 0.375 | 0.094 | 0.344 | 0.406 | 4 |
| fcos_p2 | anchor-free | 0.196 | 0.390 | 0.089 | 0.365 | 0.428 | 6 |
| fcos_p2_ciou | anchor-free | 0.194 | 0.388 | 0.094 | 0.361 | 0.433 | 6 |
| rtdetr_base | query-based | 0.248 | 0.489 | 0.128 | 0.417 | 0.489 | 5* |
| rtdetr_q500 | query-based | 0.253 | 0.501 | 0.130 | 0.418 | 0.480 | 4* |

\* RT-DETR runs were stopped by the 90-minute training budget after 4–5 of 8 epochs.

#### Faster R-CNN

The upper bound is set by RPN: proposal recall (IoU >= 0.5) grows from 66% for the top-100 to 87% for the top-2000 proposals and from 56% to 77% for small objects. At the second stage objects are filtered and classified, dropping recall to 79% for any class and 76% for correct class. With more proposals *AP_smal* was raised from 0.105 to 0.137 at the same mAP, but the speed slowed down (13 -> 10 img/s). Adding GIoU gave the best mAP of 0.250. The peak performance was achieved by epoch 3-6 for every variant.

#### Failure cases

Small and blocked off objects are the most common to be missed: recall drops from 0.90 for large to 0.52 for small and 0.13 for tiny objects; and from 0.81 that aren't obstructed to 0.22 for heavily obstructed objects.

![Missed objects](images/failure_missed.png)

*Figure 4. Failure case 1 — the validation frame with the most missed objects (Faster R-CNN, score ≥ 0.3). Green: correct detections, red: false positives, yellow dashed: missed objects, grey: ignored regions.*

Roofs, shadows or road parks sometimes look like objects from above, such mistakes occupy 22% of all FP predictions on background (Figure 5).

![Background false positives](images/failure_background.png)

*Figure 5. Failure case 2 — the frame with the most false positives on background (same colours as in Figure 4).*

Similar classes are sometimes confused: `motor` and `person` (610 cases as rider and motorbike share one box), `van` and `car` (715), `bicycle` and `motor` (383). With only 53 validation instances, the `bus` has recall 0, being predicted as a `car`, `van` or `truck` (Figure 6).

![Class confusion](images/failure_class.png)

*Figure 6. Failure case 3 — the frame with the most class confusions (same colours as in Figure 4).*

#### RetinaNet

Without a proposal stage, RetinaNet scores ~100k anchors directly, almost all of them background; focal loss handles this imbalance, but there is no second box refinement, so localisation is worse (AP75 0.180 vs 0.222 for Faster R-CNN). Smaller anchors improved coverage but lowered mAP (0.193 → 0.171) and AP_large (0.355 → 0.300), because large objects lost the biggest anchors; gamma 1.5 recovered part of it (0.179). RetinaNet is ~1.8× faster than Faster R-CNN but clearly weaker (0.193 vs 0.250).

#### FCOS

The search covered the P2 level (off/on) and the box loss (GIoU/CIoU). Both changes stayed within noise (0.190 → 0.196 → 0.194), and AP_small did not improve (0.094 → 0.089); only AP_large grew (0.344 → 0.365). The new, randomly initialised P2 layers probably needed more than 8 epochs. FCOS is on par with RetinaNet and below Faster R-CNN.

#### RT-DETR

Neither run converged: the 90-minute budget stopped them after 4–5 of 8 epochs while the loss was still falling. Even so, RT-DETR matches the best CNN (0.253 vs 0.250) and is ahead of it at every epoch (Figure 7). Without NMS it keeps neighbouring objects, so AR is higher (0.48 vs 0.40) and mAP on the densest frames is slightly better (0.356 vs 0.350); 500 queries added +0.005 mAP.

![RT-DETR vs Faster R-CNN](images/rtdetr_vs_cnn.png)

*Figure 7. Validation mAP during training: RT-DETR (300 and 500 queries) vs the best CNN (frcnn_props_giou), by epoch (left) and by training time (right). RT-DETR was trained on a slower GPU in fp32.*

### Pipeline

**Table 2.** Pipeline steps on the validation set. Steps A change only inference (same weights); steps B change the model.

| Step | Change | val mAP | AP_small | AR |
|---|---|---|---|---|
| 0 | frcnn_props_giou as trained | 0.250 | 0.121 | 0.399 |
| A1 | score ≥ 0.001, ≤ 500 dets, NMS 0.5 | 0.256 | 0.125 | 0.457 |
| A2 | NMS 0.6 / 0.7 | 0.256 / 0.249 | 0.126 / 0.124 | 0.465 / 0.469 |
| A3 | + horizontal-flip TTA | 0.255 | 0.122 | 0.466 |
| B1 | smaller RPN anchors (16–256 px) | 0.227 | 0.120 | 0.343 |
| B2 | A1 + rtdetr_q500 ensemble, NMS 0.6 | **0.266** | **0.143** | **0.497** |

A1 gave +0.006 mAP and +0.058 AR without retraining. NMS 0.6 equalled 0.5, while 0.7 added duplicates. Flip TTA did not help, because NMS keeps one of two shifted boxes instead of averaging them. B1 was worse: small objects already received proposals, while large ones lost the 512-px anchors. B2 was best (+0.010 over A1), since the CNN and the transformer make different errors.

**Table 3.** Kaggle submissions (public leaderboard).

| Submission | Public score |
|---|---|
| Baseline | 0.2216 |
| frcnn_props_giou (as trained) | 0.2242 |
| Final: Faster R-CNN (A1) + RT-DETR q500 ensemble, ≤ 300 dets/image | **0.2820** |
| Faster R-CNN (A1), ≤ 500 dets/image | scoring error (file too large) |

### Unsuccessful attempts

- Smaller anchors in RetinaNet and Faster R-CNN (B1).
- P2 level and CIoU loss in FCOS (no significant gain).
- Horizontal-flip TTA.
- fp16 training on the local GTX 16xx GPU: NaN losses in RT-DETR and mAP 0.000 for B1 (the same code reached 0.227 on a T4).
- A submission with 500 detections per image (Kaggle internal scoring error).

## Conclusion

<!-- 3-4 main findings and next steps -->

### Final model and how to run

<!-- which models, inference settings and merging form the final submission (notebook section 9.4) -->

<!-- ```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
``` -->
