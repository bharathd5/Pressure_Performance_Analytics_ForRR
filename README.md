# Pressure Performance Analytics
Ranks batters and bowlers by how reliably they perform when the match is on the line (IPL + T20I, 2022-2026).

**Method in one paragraph.** Pressure = leverage of a delivery (expected |win-probability swing|, cross-fitted GBM win-probability model).
Performance = run-expectancy value (runs above an average player in the same state), adjusted for phase x hand x pace/spin and opponent quality.
Rating = empirical-Bayes partial pooling of overall skill (g), pressure lift (D) and consistency (C) with cluster-robust SEs.
Identical for batters and bowlers. Full details: `reports/methodology_report.pdf`.

## Run
```
pip install -r requirements.txt
# put the six CSVs in data/raw/
python run_pipeline.py            # ~2 min: clean -> pressure -> ratings -> validation -> results/
python reports/make_figures.py && python reports/build_report.py
streamlit run dashboard/app.py
```
All knobs (thresholds, weights, eligibility, stakes) are in `src/config.py`. Without `pyarrow`, processed tables are saved as `.pkl` instead of `.parquet`.
Added beyond the requested tree: `src/models/{win_probability,run_expectancy,opponent_adjustment,shrinkage}.py`, `run_pipeline.py`.
