# Voxel damage analysis scripts

This folder contains the reduced, publication-facing scripts for the voxel-based damage analysis. Older duplicate or partial scripts were removed from the final upload package, and the retained scripts were renamed by function.

## Retained scripts

- `main_damage_core_growth_analysis.py`  
  Main voxel damage-core workflow: connected-component analysis, main-core size evolution, exponential/model-comparison support, and Figure 5-style growth analysis.

- `logistic_model_comparison_and_fit.py`  
  Logistic, exponential, linear, and power-law model fitting; goodness-of-fit heatmaps; AICc/BIC comparison; delta-AICc plotting.

- `p90_core_organization_metrics.py`  
  P90 connected-core organization metrics, including `N90`, `Lmax90`, `D90`, and log-scale diagnostic plots.

- `supplementary_logistic_sensitivity_analysis.py`  
  Logistic bootstrap confidence intervals, P85/P90/P95 threshold sensitivity, leave-one-out tests, and grid/downsampling sensitivity analysis.

## Removed duplicate/legacy scripts

- `sunshangfenxiN90(duotu1).py`: earlier multi-panel N90 connected-core workflow; superseded by `main_damage_core_growth_analysis.py`.
- `sunshangfenxiN90(dantu2).py`: later N90 workflow; superseded by `main_damage_core_growth_analysis.py`.
- `sunshangfenxip90p80R2AICCdeng.py`: model-fitting extension block; integrated into `main_damage_core_growth_analysis.py`.
- `sunshangfenxiD90(sunsahngheyoushidu1).py`: simpler P90/D90 diagnostic script; superseded by `p90_core_organization_metrics.py`.

## Note

Concrete local paths have been removed from the released scripts. Before rerunning them on another workstation, fill in the empty input CSV paths and output folders near the configuration sections of each script.
