Source data for Figure 4 b_delta_aicc.png

Generated from the workflow in:
code/voxel_damage_analysis/logistic_model_comparison_and_fit.py

The figure shows Delta AICc relative to the Logistic model:
Delta_AICc = AICc(model) - AICc(Logistic)

Files:
- figure4b_lmax_input.csv: Lmax(t) input used for model fitting.
- figure4b_model_aicc_delta_source.csv: full model-fit source data for the bar plot.
- figure4b_plot_bars.csv: compact wide-format table containing the plotted bar heights.

Key parameters:
- MAIN_PCT = 90
- MIN_COMP_VOXELS = 5
- Fit windows:
  S1: 3.56-8.63 h
  S2: 1.91-11.61 h
  S4: 3.00-11.50 h

Model definitions:
- Logistic: L(t) = K / (1 + exp(-r * (t - t0)))
- Exponential: L(t) = a * exp(lambda * t)
- Linear: L(t) = a * t + b
- Power-law: L(t_positive) = a * t_positive^b, where t_positive = t - t_min + 0.1

AICc was calculated as:
AICc = n * ln(SSE / n) + 2k + 2k(k+1)/(n-k-1), using max(n-k-1, 1) in the denominator as in the script.
