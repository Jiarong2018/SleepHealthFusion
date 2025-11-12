Reproducibility Package for:
"Screening Insomnia Using an Actigraphy-Based Sleep Rhythm Index with Lifestyle and Physical Function"

OVERVIEW
This archive contains all data and code required to reproduce the main analyses and figures reported in the manuscript.

DIRECTORY STRUCTURE
.
├── data/
│   └── deidentified_data.csv
├── result/                # Generated artifacts (created automatically)
└── script/
    ├── model_training_xgboost.py
    ├── feature_contribution.py
    ├── PCA_LogisticRegression.py
    ├── statistic_analysis1.py
    └── statistic_analysis2.py

Relative paths: All scripts inside /script/ read from ../data/ and write to ../result/.
Fonts & style: Figures use Arial, seaborn style=white, context=talk, and consistent viridis endpoints where applicable.

SOFTWARE REQUIREMENTS
- Python 3.9
- See requirements.txt for exact, pinned versions.

Install:
pip install -r requirements.txt

DATA
- Input file for all analyses: data/deidentified_data.csv
- Ensure the file exists before running. The Makefile checks this automatically.

SCRIPTS

1) model_training_xgboost.py
Train and evaluate an XGBoost classifier.

Expected outputs
- result/xgb_model.json (required by feature_contribution.py)
- (Optional) result/feature_names.txt — one feature per line, if the model object does not retain names.

Run
make train
# or
(cd script && python model_training_xgboost.py)


2) feature_contribution.py
Analyze and visualize feature contributions from the trained XGBoost model.

Behavior
- Output is identical to the contribution part of the training script.
- Loads model from result/xgb_model.json.
- If feature names are not embedded, provide result/feature_names.txt (one per line).

Run
make contributions
# or
(cd script && python feature_contribution.py)


3) PCA_LogisticRegression.py
PCA (PC1, PC2) + Logistic Regression decision surface.

Visualization matches SVM-style
- Smooth probability background (imshow) with a custom blue colormap
- No explicit decision boundary line
- Scatter points use viridis endpoints (Non-Insomnia first, Insomnia second)
- Colorbar labeled as P(Insomnia)

Output
- result/PCA2_LogReg_boundary.pdf

Run
make pca_logreg
# or
(cd script && python PCA_LogisticRegression.py)


4) statistic_analysis1.py
Generate KDE and box plots of SRI stratified by insomnia (PSQI), depression, and anxiety.

Reads
- data/deidentified_data.csv

Writes
- result/kde_plot_psqi.pdf
- result/box_plot_psqi.pdf
- result/kde_plot_depression.pdf
- result/box_plot_depression.pdf
- result/kde_plot_anxiety.pdf
- result/box_plot_anxiety.pdf

Run
make stat1
# or
(cd script && python statistic_analysis1.py)


5) statistic_analysis2.py
Scatter plots of SRI vs. PHQ-9 / GAD-7 / PSQI / FSS-9 with binary highlighting
(orange = above cutoff; black = otherwise).

Reads
- data/deidentified_data.csv

Writes
- result/SRI_vs_PHQ9_GAD7_PSQI_FSS9.pdf

Run
make stat2
# or
(cd script && python statistic_analysis2.py)

CUT-OFFS & PLOTTING CONVENTIONS
- Binary flags: PSQI > 5, PHQ-9 > 5, GAD-7 > 5, FSS-9 > 4
- KDE/box plots: viridis (2 endpoints), explicit legend order; anxiety panel uses fixed axis/legend ordering to match manuscript figures.
- Export: PDF, dpi=450, bbox_inches='tight'

REPRODUCTION
One command to build everything:
make all

This will:
1. Check data availability and ensure result/ exists
2. Run both statistical figure scripts
3. Train the XGBoost model
4. Generate the feature contribution outputs

Clean generated figures:
make clean-figs

Remove figures and model artifacts:
make clean

NOTES
- All scripts assume relative paths as shown above.
- If you modify directory names, also update the paths inside the scripts and the Makefile.
