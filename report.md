---
title: Homework 2 -- Object Detection on Drone Videos (VisDrone2019-VID)
author: Pavlunyshyn Bohdan
course: Artificial Intelligence · UCU
date: 2026-10-12
---

## Introduction

Write your report here using regular **Markdown**. This text is an example — replace it.

## Data and preprocessing

![Figure 1. Object size per class and after resizing to 800 px; dashed lines mark 16 / 32 / 96 px.](images/object_sizes.png)

![Figure 2. Object density between scenes and clips.](images/density_per_clip.png)

![Figure 3. Class distribution in both validation and training data.](images/class_distribution.png)

## Method

Inline math works with single dollars: $f(x) = \sigma(Wx + b)$. Display math uses double dollars:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} y_i \log \hat{y}_i
$$

Code blocks are highlighted:

```python
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
```

### Training protocol

### Detectors

**Table 1.** Validation results of all required detectors (same frames and metric; best epoch by val mAP).

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


![Figure 4.](images/failure_missed.png)

![Figure 5.](images/failure_background.png)

![Figure 6.](images/failure_class.png)

![Figure 7.](images/rtdetr_vs_cnn.png)

### Your pipeline

**Table 2.** 

| Step | Change | val mAP | AP_small | AR |
|---|---|---|---|---|
| 0 | frcnn_props_giou as trained | 0.250 | 0.121 | 0.399 |
| A1 | score ≥ 0.001, ≤ 500 dets, NMS 0.5 | 0.256 | 0.125 | 0.457 |
| A2 | NMS 0.6 / 0.7 | 0.256 / 0.249 | 0.126 / 0.124 | 0.465 / 0.469 |
| A3 | + horizontal-flip TTA | 0.255 | 0.122 | 0.466 |
| B1 | smaller RPN anchors (16–256) | 0.227 | 0.120 | 0.343 |
| B2 | A1 + rtdetr_q500 ensemble, NMS 0.6 | **0.266** | **0.143** | **0.497** |


**Table 3.** Kaggle submission results

| Kaggle submission | Public score |
|---|---|
| Baseline | 0.2216 |
| frcnn_props_giou (as trained) | 0.2242 |
| Final: Faster R-CNN (A1) + RT-DETR q500 ensemble | **0.2820** |
| Faster R-CNN A1, 500 dets/image | scoring error (file too large) |

## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.
