Release History
===============

Version 0.2.0 (02/04/2026)
--------------------------

- Overhauled ``expDeconv`` API: new AnnData-based interface (``adata_ref``,
  ``adata_tgt``, ``label_key``), automatic raw-counts detection, marker gene
  selection via ``topk`` DE, Leiden spatial clustering, cluster-mean
  augmentation, score-init warm-start, cluster-to-celltype mapping,
  JS-divergence loss (``wt_js``), cosine LR scheduling, and post-hoc
  calibration.
- Added progress reporting: informational print statements at each major phase
  boundary in ``expDeconv``, plus periodic epoch-level logging in
  ``fit_deconv`` for non-TTY (batch) mode.
- Updated ``expVeloImp`` with AnnData interface support and proportions
  recalibration fix.

Version 0.1.1 (21/07/2023)
--------------------------

- Alpha version of TransImp released