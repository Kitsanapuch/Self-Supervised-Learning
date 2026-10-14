# Reproduce the delivered study

Run commands from the repository root. The delivered experiment used Python 3.14.6, CPU and two PyTorch CPU threads per process. Install the recorded training environment if supported on your platform; the general requirements allow compatible newer environments, which must be recorded separately.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m src.download_data
python -m unittest discover -v
python run_matrix.py --runs runs/full --epochs 30 --pretrain-epochs 30
```

Do not mix protocols in the same output directory. Complete output folders are skipped on resume, while incomplete folders require inspection and relocation before retrying. Downloaded data are checked against the exact mirror archive checksum used in the results.

## Rebuild the report and audit

```bash
python -m pip install -r requirements-report.txt
python -m scripts.subject_bootstrap
python -m scripts.build_report
python -m scripts.audit_runs
```

The report builder requires the complete 48-run matrix under `runs/full`, and deliberately refuses incomplete data. It generates the numerical tables and PDF directly from saved JSON; there are no manually typed performance values. `src.summarize` supports other output roots for exploratory runs.

The scientific experiment scripts and training source are recorded in the Git commit cited by run metadata. Later commits add reporting and convenience tools; they do not retroactively change the training source. Exact weight reproducibility across different operating systems, PyTorch releases or hardware is not guaranteed.

## Verification scope

- unit checks: split disjointness and scaling, paired/nested subsets, masking and gradients, Macro-F1 behavior
- end-to-end smoke: synthetic data only, excluded from research scores
- run audit: all 48 completed, all losses/weights finite, all 144 metric records recomputed from confusion matrices
- replay: reload a real trained checkpoint and independently reproduce all three condition scores
- paper QA: extract PDF text, render pages and visually inspect layout
- source ZIP: extraction and unit tests checked
- Colab notebook and GitHub publication: prepared, not executed in those external services
