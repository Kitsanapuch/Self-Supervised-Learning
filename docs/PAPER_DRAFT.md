# Label Budget and Sensor-Outage Sensitivity of Masked Reconstruction in Human Activity Recognition

**Research manuscript for advisor review.** Authors and affiliations must be supplied by the researcher. This report does not claim a new masking algorithm or state-of-the-art performance. All numerical tables are generated from completed runs.

## Abstract

We compare supervised training with three masked-reconstruction pretraining objectives on subject-disjoint UCI HAR data. Forty-eight training configurations cover four label budgets and three random seeds; each checkpoint is evaluated under three signal conditions. At a 1% training-label budget, supervised training obtains 68.52 +/- 0.84% Macro-F1, temporal masking obtains 72.00 +/- 1.47% (+3.48 percentage points), and combined temporal-channel masking obtains 68.95 +/- 2.06%. These findings apply to one fixed subject split and do not establish a new algorithm or state-of-the-art performance. Full validation labels are used for selection and are excluded from the stated training-label budgets.

## 1. Introduction

Wearable human activity recognition maps multichannel inertial measurements to activity labels. This study asks whether a compact encoder pretrained to reconstruct masked measurements improves classification when training labels are limited, and whether any improvement persists after a complete sensor group is suppressed. The contribution is a transparent experimental comparison with paired label subsets and explicit annotation-budget accounting.

## 2. Related work

TS-TCC learns time-series representations through temporal and contextual contrastive tasks [2]. TS2Vec uses hierarchical contrasting [3]. Channel and combined time-channel masking have already been investigated for HAR [4], and RobustHAR studies multiscale masked pretraining for missing-data robustness [5]. Consequently, neither masking across time and channels nor studying sensor outages is claimed as a new idea here. Our independently written reconstruction baselines are not reproductions of these systems, and their published scores are not directly compared with ours because protocols differ.

## 3. Data and evaluation protocol

We use UCI HAR [1]. The dataset provides pre-windowed 128-sample inertial sequences at 50 Hz, with six activity classes. We retain total accelerometer and gyroscope XYZ channels; derived body-acceleration channels are excluded. Official training subjects are subdivided into 17 training and four validation subjects using split seed 2026. Validation subjects are 26, 19, 25 and 1. The official nine test subjects are untouched during optimization. The resulting sample counts are 5,844 training, 1,508 validation and 2,947 test windows. All normalization statistics are computed exclusively from training windows.

A teaching mirror was used after slow access to the canonical UCI endpoint. Archive provenance and SHA-256 are supplied; dimensions and subject identities were verified, but byte identity to a completed origin download was not independently established. No raw data are redistributed with the code package.

Training-label fractions are 1%, 5%, 10% and 100%. Each class contributes ceil(fraction × class count), with a minimum of one window. A fixed permutation per class and seed produces nested subsets across fractions and identical subsets across methods. Existing ground-truth labels are used to construct these simulated stratified budgets. All validation labels remain available for model selection: fractions refer only to training labels, not total annotation costs. Overlapping windows within a subject are not treated as independent human participants.

## 4. Models and training

The encoder comprises Conv1D(6,32,k=7), GroupNorm(4,32), GELU, Conv1D(32,64,k=5), GroupNorm(8,64), and GELU. Same padding preserves temporal length. Global average pooling and a six-output linear layer perform classification. The classifier has 12,262 trainable parameters. A 1×1 convolution maps encoder outputs to six reconstruction channels during pretraining.

Given normalized signal x, a Boolean mask m marks missing positions, and the corrupted input is x multiplied by (1−m). The reconstruction objective is the mean squared error at masked positions only. Temporal masking removes one contiguous 32-sample span across all channels. Channel masking removes one randomly selected channel over the whole window. Combined masking takes their union. Their masked proportions are respectively 25%, 16.67% and 37.5%; this experiment does not isolate mask geometry from total coverage.

The supervised baseline starts from a randomly initialized encoder. Three SSL variants pretrain on all training windows with label targets excluded. Each uses 30 pretraining epochs; compatible pretrained encoders are reused across label budgets. All methods then train for 30 classification epochs with Adam, learning rate 0.001, batch size 128, without a learning-rate schedule. The classifier head initialization and shuffled labeled-example order are paired by seed. The encoder is fully fine-tuned. Checkpoints are selected by clean validation Macro-F1, with the first checkpoint retained in a tie. No test score changes hyperparameters or checkpoint selection. SSL has additional pretraining computation; the comparison is not compute matched.

Three seeds (0,1,2), four label budgets and four methods yield 48 training configurations. Each selected checkpoint is evaluated on clean test windows, all accelerometer channels replaced by their training mean, and all gyroscope channels replaced by their training mean. These are simulated group outages, not observed hardware failures. Normalized zero replacement uses training statistics, with no test-time refitting.

## 5. Results

### Complete signals

Macro-F1 in percent, mean +/- sample SD over three seeds.

| Labels | Supervised | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|---|
| 1% | 68.52 +/- 0.84 | 72.00 +/- 1.47 | 71.30 +/- 3.58 | 68.95 +/- 2.06 |
| 5% | 82.85 +/- 4.79 | 84.86 +/- 0.72 | 84.61 +/- 1.86 | 82.46 +/- 0.64 |
| 10% | 87.68 +/- 2.68 | 89.01 +/- 1.47 | 87.46 +/- 0.29 | 87.01 +/- 1.87 |
| 100% | 90.92 +/- 0.46 | 91.26 +/- 2.02 | 90.84 +/- 0.87 | 90.31 +/- 2.05 |

### Accelerometer unavailable

Macro-F1 in percent, mean +/- sample SD over three seeds.

| Labels | Supervised | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|---|
| 1% | 31.09 +/- 2.81 | 34.76 +/- 5.56 | 29.38 +/- 4.64 | 28.87 +/- 7.20 |
| 5% | 44.84 +/- 6.75 | 40.88 +/- 6.69 | 28.75 +/- 1.77 | 27.18 +/- 8.71 |
| 10% | 47.91 +/- 0.74 | 39.73 +/- 0.81 | 28.32 +/- 2.57 | 25.85 +/- 5.51 |
| 100% | 27.54 +/- 5.48 | 30.49 +/- 3.94 | 22.80 +/- 4.01 | 24.74 +/- 2.86 |

### Gyroscope unavailable

Macro-F1 in percent, mean +/- sample SD over three seeds.

| Labels | Supervised | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|---|
| 1% | 35.26 +/- 2.23 | 38.02 +/- 1.76 | 41.88 +/- 3.27 | 38.04 +/- 1.02 |
| 5% | 37.40 +/- 2.80 | 45.05 +/- 5.08 | 50.07 +/- 3.63 | 43.86 +/- 5.47 |
| 10% | 47.00 +/- 3.92 | 51.43 +/- 5.79 | 51.03 +/- 2.67 | 49.08 +/- 8.25 |
| 100% | 62.42 +/- 1.32 | 60.05 +/- 7.32 | 60.51 +/- 4.46 | 60.44 +/- 5.76 |

### Paired clean-condition differences

| Labels | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|
| 1% | 3.48 +/- 1.56 | 2.78 +/- 2.86 | 0.43 +/- 2.44 |
| 5% | 2.01 +/- 5.02 | 1.76 +/- 3.24 | -0.38 +/- 4.88 |
| 10% | 1.33 +/- 1.25 | -0.23 +/- 2.42 | -0.67 +/- 3.21 |
| 100% | 0.35 +/- 2.34 | -0.07 +/- 1.31 | -0.61 +/- 1.61 |

Differences are percentage points and use paired seeds. Three seeds are not sufficient for strong significance claims. Test-score leaders are descriptive only; model choice must use validation results.

### Exploratory paired subject bootstrap

95% percentile intervals, 5,000 resamples of nine test subjects; conditional on the fitted models, without multiplicity correction.

| Labels | Method | Difference (pp) | 95% interval (pp) |
|---|---|---|---|
| 1% | SSL-time | +3.48 | [+0.77, +5.99] |
| 1% | SSL-channel | +2.78 | [+0.85, +4.63] |
| 1% | SSL-combined | +0.43 | [-2.27, +2.83] |
| 5% | SSL-time | +2.01 | [+0.02, +4.26] |
| 5% | SSL-channel | +1.76 | [-0.39, +3.72] |
| 5% | SSL-combined | -0.38 | [-2.90, +1.75] |
| 10% | SSL-time | +1.33 | [+0.43, +2.27] |
| 10% | SSL-channel | -0.23 | [-1.92, +1.44] |
| 10% | SSL-combined | -0.67 | [-2.13, +0.75] |
| 100% | SSL-time | +0.35 | [-0.42, +1.23] |
| 100% | SSL-channel | -0.07 | [-2.06, +1.78] |
| 100% | SSL-combined | -0.61 | [-1.88, +0.61] |

## 6. Limitations

The study uses one dataset and one subject partition with three seeds. It does not reproduce contrastive or modern masked-model baselines, tune every method under an equal search budget, or match total computation. Full validation labels add annotation costs outside the advertised training-label budgets. Combined masking has a larger coverage fraction, and single-channel masking during pretraining differs from whole-group suppression during evaluation. Training windows are correlated and class-balanced subsampling assumes labels are already known. The protocol cannot establish generalization to new devices, placements, demographic groups, or clinical use. An empirical advantage would not establish architectural novelty or universal robustness; a negative result would not invalidate SSL in general.

## 7. Reproducibility and research integrity

Source, exact installed package versions, data checksum, split identities, labeled indices, learning curves, checkpoint weights, confusion matrices and per-class F1 are provided. A synthetic smoke test and unit tests check the pipeline separately from the research results. Generative AI assisted with code, analysis and manuscript preparation. The researcher and advisor must verify the work and comply with their institution and venue's disclosure rules before submission. No authorship, affiliation, ethics approval or acceptance is asserted.

## References

[1] Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., and Parra, X. (2013). Human Activity Recognition Using Smartphones. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K

[2] Eldele, E., et al. (2021). Time-Series Representation Learning via Temporal and Contextual Contrasting. IJCAI. https://arxiv.org/abs/2106.14112

[3] Yue, Z., et al. (2022). TS2Vec: Towards Universal Representation of Time Series. AAAI. https://arxiv.org/abs/2106.10466

[4] Wang, J., Zhu, T., and Ning, H. (2023). An Improved Masking Strategy for Self-supervised Masked Reconstruction in Human Activity Recognition. arXiv:2312.04147. https://arxiv.org/abs/2312.04147

[5] Liu, X., Yuan, G., Zhang, Y., Liu, S., and Yan, Q. (2025). RobustHAR: Multi-scale Spatial-temporal Masked Self-supervised Pre-training for Robust Human Activity Recognition. IJCAI, 8563-8571. https://www.ijcai.org/proceedings/2025/0952.pdf
