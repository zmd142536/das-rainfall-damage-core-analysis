# Code availability statement

The analysis code used for DAS preprocessing, CAT-RMS/RMS calculation, voxel reconstruction, connected-component damage-core analysis, threshold sensitivity tests, and model fitting is provided in the accompanying repository/Zenodo archive.

The released code is organized into two main folders:

- `code/cat_rms_and_das_processing/`: 8 renamed publication-facing scripts for preprocessing, PCA denoising, downsampling, RMS/CAT-RMS calculation, thresholding, normalization, log transformation, and strain/strain-rate conversion. The duplicate non-PCA denoising script was removed, and `DASH5_PCA.py` was retained as the main preprocessing workflow.
- `code/voxel_damage_analysis/`: 4 renamed publication-facing scripts for voxel damage reconstruction, main-damage-core identification, connected-component analysis, logistic/exponential/linear/power-law model fitting, leave-one-out tests, grid/downsampling sensitivity tests, and figure-reproduction workflows.

