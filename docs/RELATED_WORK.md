# Related work and novelty assessment

Checked 2026-10-14. This is a targeted literature check, not a systematic review. The delivered implementation does not reproduce the cited models.

| Source | What is established | Consequence for this study |
|---|---|---|
| Eldele et al., TS-TCC (IJCAI 2021), https://arxiv.org/abs/2106.14112 | Temporal/contextual contrastive learning for unlabeled time series | Contrastive SSL is an important future external baseline |
| Yue et al., TS2Vec (AAAI 2022), https://arxiv.org/abs/2106.10466 | Hierarchical contrastive time-series representations | Do not describe time-series SSL as new |
| Wang, Zhu and Ning (2023), https://arxiv.org/abs/2312.04147 | Channel masking and combined time-channel masking for HAR | The proposed initial idea already has direct precedent; do not claim masking novelty |
| Liu et al., RobustHAR (IJCAI 2025), https://www.ijcai.org/proceedings/2025/0952.pdf | Multiscale masked pretraining for missing-data robustness in wearable HAR | Missing-sensor robustness is also an established topic |
| Self-supervised learning with randomized cross-sensor masked reconstruction for human activity recognition (2024), https://doi.org/10.1016/j.engappai.2023.107478 | Cross-sensor reconstruction with large-scale pretraining | Current small, single-dataset experiment does not establish superiority over large pretrained models |

Defensible description: a reproducible empirical comparison of simple temporal, channel and combined reconstruction objectives under a fixed architecture, subject-disjoint split and label budgets. This is a research report for advisor review, not a demonstrated novel algorithm paper. Any publication claim requires agreement on venue, stronger baselines, further datasets and a clearly justified contribution.
