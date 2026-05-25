# GitHub and Zenodo upload guide

## 1. Before upload

1. Open `README.md`, `DATA_AVAILABILITY_statement.md`, `CODE_AVAILABILITY_statement.md`, `CITATION.cff`, and `restricted_data_request/DAS_raw_data_request_template.md`.
2. Replace all placeholders:
   - `[paper title]`
   - `[author names]`
   - `[corresponding_author_email]`
   - `[GitHub URL]`
   - `[Zenodo DOI]`
3. Check that no raw private DAS files, unpublished personal information, or site-sensitive unrestricted files are included.
4. Choose licenses:
   - Recommended for code: MIT or BSD-3-Clause.
   - Recommended for figure source data: CC-BY-4.0.
   - For restricted raw DAS data: do not upload publicly; describe request procedure.

## 2. Upload to GitHub

Recommended repository name:

```text
das-rainfall-damage-core-analysis
```

Command-line workflow from inside the `最终` folder:

```bash
git init
git add .
git commit -m "Release source data and analysis code"
git branch -M main
git remote add origin https://github.com/<your-user-or-org>/das-rainfall-damage-core-analysis.git
git push -u origin main
```

Then create a GitHub release:

1. Go to the repository on GitHub.
2. Click `Releases`.
3. Click `Draft a new release`.
4. Tag: `v1.0.0`.
5. Release title: `Source data and analysis code for [paper title]`.
6. Description: briefly state that this release contains figure source data, plotting metadata, and analysis code.
7. Publish the release.

GitHub browser upload has per-file and file-count limits, so the command-line route is preferred.

## 3. Archive with Zenodo

Two good options are available.

### Option A: Zenodo GitHub integration

Use this if you want Zenodo to archive the GitHub release automatically.

1. Log in to Zenodo.
2. Connect GitHub to Zenodo.
3. Enable the target repository in Zenodo.
4. Publish a GitHub release, for example `v1.0.0`.
5. Wait for Zenodo to archive the release.
6. Copy the resulting DOI into the paper and `CITATION.cff`.

### Option B: Manual Zenodo upload

Use this if you want to upload a ZIP package directly.

1. Compress the contents of the `最终` folder as a ZIP file.
2. Create a new Zenodo upload.
3. Upload the ZIP.
4. Resource type: `Dataset` if emphasizing figure source data, or `Software` if submitting code only. For this mixed package, `Dataset` plus a linked GitHub software repository is usually clearer.
5. Fill in title, creators, description, keywords, funding, license, and related identifiers.
6. Reserve a DOI before final publication if the manuscript needs the DOI in advance.
7. Publish when the metadata has been checked.

## 4. Suggested manuscript wording

Data availability:

```text
Source data and plotting metadata for all figures are available at GitHub: [GitHub URL] and archived at Zenodo: [Zenodo DOI]. The complete raw DAS recordings are available from the corresponding author upon reasonable request at [corresponding_author_email], subject to data-volume and institutional data-sharing constraints.
```

Code availability:

```text
Code used for DAS preprocessing, CAT-RMS calculation, voxel damage-core reconstruction, connectivity analysis, sensitivity analysis, and model fitting is available at GitHub: [GitHub URL] and archived at Zenodo: [Zenodo DOI].
```

## 5. Official references

- GitHub Docs: Adding locally hosted code to GitHub.
- GitHub Docs: Managing releases in a repository.
- Zenodo Help: Quick start for uploading and publishing records.
- Zenodo Support: GitHub integration FAQ.
