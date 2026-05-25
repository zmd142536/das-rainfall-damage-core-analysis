# DAS preprocessing and CAT-RMS/RMS scripts

This folder contains the reduced, publication-facing scripts for DAS preprocessing, CAT-RMS/RMS calculation, and derived plotting-data preparation.

## Retained scripts

- `DASH5_PCA.py`  
  Main preprocessing/denoising workflow retained for release. It includes mean removal, detrending, despiking, common-mode removal, spectral denoising, notch filtering, and PCA denoising.

- `preprocess_lowpass_downsample_h5.py`  
  Low-pass filtering and downsampling of DAS HDF5 files.

- `estimate_background_rms_thresholds.py`  
  Background RMS statistics and threshold estimation for channel groups.

- `calculate_cumulative_cat_rms.py`  
  Threshold-exceedance cumulative RMS/CAT-RMS calculation using manually defined channel groups and thresholds.

- `extract_rms_snapshot_table.py`  
  Extracts a target time row from CSV/Excel RMS outputs and converts it to a plotting-friendly table.

- `normalize_cumulative_rms.py`  
  Normalizes cumulative RMS/CAT-RMS tables.

- `log_transform_rms_for_voxler.py`  
  Applies log transformation to RMS/CAT-RMS values for visualization or Voxler input.

- `strain_rate_to_strain_export.py`  
  Reads DAS strain-rate/strain files, exports aligned CSV tables, and produces diagnostic heatmaps/curves.

## Removed duplicate script

- `DASdataDenoising(ALL channe Remove random bad lanes filter ).py`: duplicate non-PCA denoising workflow. It was removed because `DASH5_PCA.py` contains the retained preprocessing workflow and adds PCA denoising.

## Note

Concrete local paths have been removed from the released scripts. Before rerunning them on another workstation, fill in the empty input HDF5/CSV paths and output folders near the configuration sections of each script, and then check the channel groups and thresholds.
