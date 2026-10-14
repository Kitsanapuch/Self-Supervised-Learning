# Experimental results

We compare supervised training with three masked-reconstruction pretraining objectives on subject-disjoint UCI HAR data. Forty-eight training configurations cover four label budgets and three random seeds; each checkpoint is evaluated under three signal conditions. At a 1% training-label budget, supervised training obtains 68.52 +/- 0.84% Macro-F1, temporal masking obtains 72.00 +/- 1.47% (+3.48 percentage points), and combined temporal-channel masking obtains 68.95 +/- 2.06%. These findings apply to one fixed subject split and do not establish a new algorithm or state-of-the-art performance. Full validation labels are used for selection and are excluded from the stated training-label budgets.

## Complete signals

Macro-F1 in percent, mean +/- sample SD over three seeds.

| Labels | Supervised | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|---|
| 1% | 68.52 +/- 0.84 | 72.00 +/- 1.47 | 71.30 +/- 3.58 | 68.95 +/- 2.06 |
| 5% | 82.85 +/- 4.79 | 84.86 +/- 0.72 | 84.61 +/- 1.86 | 82.46 +/- 0.64 |
| 10% | 87.68 +/- 2.68 | 89.01 +/- 1.47 | 87.46 +/- 0.29 | 87.01 +/- 1.87 |
| 100% | 90.92 +/- 0.46 | 91.26 +/- 2.02 | 90.84 +/- 0.87 | 90.31 +/- 2.05 |

## Accelerometer unavailable

Macro-F1 in percent, mean +/- sample SD over three seeds.

| Labels | Supervised | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|---|
| 1% | 31.09 +/- 2.81 | 34.76 +/- 5.56 | 29.38 +/- 4.64 | 28.87 +/- 7.20 |
| 5% | 44.84 +/- 6.75 | 40.88 +/- 6.69 | 28.75 +/- 1.77 | 27.18 +/- 8.71 |
| 10% | 47.91 +/- 0.74 | 39.73 +/- 0.81 | 28.32 +/- 2.57 | 25.85 +/- 5.51 |
| 100% | 27.54 +/- 5.48 | 30.49 +/- 3.94 | 22.80 +/- 4.01 | 24.74 +/- 2.86 |

## Gyroscope unavailable

Macro-F1 in percent, mean +/- sample SD over three seeds.

| Labels | Supervised | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|---|
| 1% | 35.26 +/- 2.23 | 38.02 +/- 1.76 | 41.88 +/- 3.27 | 38.04 +/- 1.02 |
| 5% | 37.40 +/- 2.80 | 45.05 +/- 5.08 | 50.07 +/- 3.63 | 43.86 +/- 5.47 |
| 10% | 47.00 +/- 3.92 | 51.43 +/- 5.79 | 51.03 +/- 2.67 | 49.08 +/- 8.25 |
| 100% | 62.42 +/- 1.32 | 60.05 +/- 7.32 | 60.51 +/- 4.46 | 60.44 +/- 5.76 |

## Paired clean-condition differences

| Labels | SSL-time | SSL-channel | SSL-combined |
|---|---|---|---|
| 1% | 3.48 +/- 1.56 | 2.78 +/- 2.86 | 0.43 +/- 2.44 |
| 5% | 2.01 +/- 5.02 | 1.76 +/- 3.24 | -0.38 +/- 4.88 |
| 10% | 1.33 +/- 1.25 | -0.23 +/- 2.42 | -0.67 +/- 3.21 |
| 100% | 0.35 +/- 2.34 | -0.07 +/- 1.31 | -0.61 +/- 1.61 |

Differences are percentage points and use paired seeds. Three seeds are not sufficient for strong significance claims. Test-score leaders are descriptive only; model choice must use validation results.

## Exploratory paired subject bootstrap

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
