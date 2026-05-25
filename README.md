# DAS rainfall-induced damage analysis: source data and code

This repository contains the source data used to plot the main and supplementary figures, together with the analysis scripts used for DAS preprocessing, CAT-RMS calculation, voxel reconstruction, damage-core identification, connectivity analysis, and model fitting.

## Contents

- `source_data/main_figures/`: source data for main-text figures.
- `source_data/supplementary_figures/`: plotting metadata/source tables for supplementary figures.
- `code/cat_rms_and_das_processing/`: 8 publication-facing DAS preprocessing and CAT-RMS/RMS calculation scripts.
- `code/voxel_damage_analysis/`: 4 publication-facing voxel damage-analysis scripts for connected-component analysis, model fitting, core-organization metrics, and sensitivity analysis.
- `metadata/file_manifest.csv`: file-level manifest for the released package.
- `metadata/data_inventory.csv`: inventory of data categories and access status.
- `restricted_data_request/`: request templates for large raw DAS data that are not included in this repository.

## Data access

The released package includes all figure source data and processed metadata needed to reproduce the published plots. The complete raw DAS recordings are not bundled because of their large volume and site-sensitive experimental metadata. They are available from the corresponding author upon reasonable request, subject to data-transfer feasibility and institutional approval where applicable.

Please replace `[corresponding_author_email]` with the final corresponding author's email before release.

## Software environment

The scripts were developed in Python and use common scientific packages. A minimal environment can be installed with:

```bash
pip install -r requirements.txt
```

Concrete local input/output paths have been removed from the released scripts. Before rerunning the full workflow, fill in the empty path fields to match your local data layout. See `metadata/code_inventory.csv` for the released code list.

## Recommended citation

If you use this material, please cite the associated paper and the archived Zenodo DOI once available.
