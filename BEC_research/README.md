# BEC detection experiment

This package contains the executed primary-seed code and aggregate summaries. It is a code/data-access deliverable, not evidence that a public Git repository has been published. Raw Enron correspondence, review text, fitted models, private databases and per-message predictions are excluded.

## Measured experiment

Retained pool: 3,664 records (1,751 synthetic BEC; 1,913 presumed legitimate Enron). Shared train/validation/test sizes: 2,617/523/524. Only training seed 42 was run. Both models have clean F1 0.9939879759519038. Interpret robustness through the summaries, including token-ID changes.

## Obtain sources separately

- Synthetic BEC: https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset . Obtain the unmodified original CSV and record its version and licence. Dataset documentation describes CC BY 4.0; check the downloaded version. Do not substitute the poisoned file for originals.
- Enron: https://www.cs.cmu.edu/~enron/ . The experiment used the May 7, 2015 distribution. Respect usage terms and privacy; do not redistribute raw email text.
- DistilBERT: distilbert/distilbert-base-uncased revision 12040accade4e8a0f71eabdb258fecc2e7e948be. Model weights download when the script first runs.

Original synthetic CSV SHA256: c6151b4fca463a48f5141303a911f7fb177af2c7b6314344fd52498603dd932e

## Environment

The measured runs used Windows, Python 3.14.7, scikit-learn 1.9.1, NumPy 2.5.3, PyTorch 2.14.1+cpu and Transformers 5.19.0. The core dependency file pins verified versions. A complete installed dependency freeze was not supplied; obtain it on the execution computer using `py -m pip freeze > full_environment.txt`. The scripts seed their RNGs but exact cross-platform determinism is not guaranteed.

## Staged preparation

Each script exposes `--help`. Execute in order: inventory_enron.py; prepare_email_candidates.py; screen_email_eligibility.py; export_bec_review.py; curate_bec_candidates.py; build_provisional_pool.py; run_primary_lr.py; run_distilbert_cpu.py. The original curation stage consumes a reviewed workbook whose decisions are not included in this code-only bundle. Private manifests and recorded decisions are required to reconstruct the exact upstream selection. Do not invent completed human review.

The final two scripts reproduce the core model runs from an independently reconstructed provisional_pool.sqlite. Keep both scripts in the same folder: the transformer imports perturbation and metric functions from run_primary_lr.py.

Windows commands (use a new output directory):

    py -m pip install -r requirements_core.txt
    py scripts/run_primary_lr.py --pool PATH_TO/provisional_pool.sqlite --output NEW_RESULTS/primary_experiment
    py scripts/run_distilbert_cpu.py --experiment NEW_RESULTS/primary_experiment --output NEW_RESULTS/distilbert_experiment

Linux/macOS entry point:

    bash run.sh PATH_TO/provisional_pool.sqlite NEW_RESULTS

## Protocol and limitations

The complete independent human audit was not finished. Final exclusions were automated and exploratory, influenced by an AI-assisted audit sample before partitioning. High-similarity pairs were conservatively grouped without manual relationship confirmation. Template families remain unidentified. These facts limit claims about leakage and label validity.

LR searched eight configurations and chose C=100, no class weighting, threshold 0.74. DistilBERT used fixed learning rate 2e-5, at most two epochs, restored epoch 1 and threshold 0.99. The unequal tuning budget is a disclosed deviation. All primary perturbations use rate 0.10 and seeds 101/202/303/404/505; LR additionally evaluates 0.05 and 0.20. Group bootstrap uses 10,000 replicates and seed 606.

Clean scores do not establish production performance. Source confounding persists; no observed false positives among 273 Enron records is not a zero deployment false-positive rate. Integrated gradients, held-out styles, other training seeds, and manual error review were not completed. Test findings must not be used to retrospectively modify the frozen partitions or tune thresholds.

## Publishing

Create a separate BEC research repository and upload this code-only package after checking its contents and permissions. Keep data/model directories ignored. This package must not be mixed with unrelated private client repositories. Replace a future repository URL in the manuscript only after publication actually succeeds.
