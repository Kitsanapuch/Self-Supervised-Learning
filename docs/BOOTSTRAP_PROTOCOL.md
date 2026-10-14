# Supplemental uncertainty analysis

Specified while the experiment matrix was still running, before aggregate test results were inspected. This supplements the original fixed-seed protocol; it does not change training or model selection.

For each of four label budgets and three SSL methods versus supervised, compute paired clean-test Macro-F1 differences. Resample the nine test subjects with replacement 5,000 times using seed 20261014. Retain all windows of each selected subject and pair the resampled subject identities across methods and seeds. Calculate pooled-window Macro-F1 separately for each fitted seed and average the difference across three seeds. Report percentile 95% intervals.

These are exploratory intervals conditional on the fixed training partition and fitted models. They address test-subject sampling variability, not training-split or hyperparameter uncertainty. Nine subjects are few. No multiple-comparison correction is applied, and no claim of universal significance is made. The primary reported score remains the original pooled-window test Macro-F1 averaged across seeds.
