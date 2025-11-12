#PCA_LogisticRegression.py
"""
PCA(PC1, PC2) + Logistic Regression
Visualization matches the SVM script style:
- Smooth probability background (imshow) with a custom blue colormap
- No explicit decision boundary line
- Scatter points use viridis endpoints: Non-Insomnia first, Insomnia second
- Colorbar labeled as P(Insomnia)

Output:
- ../result/PCA2_LogReg_boundary.pdf
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# ========== Data loading & feature marking ==========
original_data = pd.read_csv('../data/deidentified_data.csv')

marked_data = pd.DataFrame({
    'Age': original_data['age'],
    'Sleep Need': original_data['self_reported_sleep_need'],
    'Education': original_data['education'],
    'Gender': original_data['gender'] - 1  # male=0, female=1
})

# Clinical screens (same thresholds as before)
marked_data['PHQ-9'] = original_data['depression'].apply(lambda x: 1 if x > 5 else 0)
marked_data['GAD-7'] = original_data['anxiety'].apply(lambda x: 1 if x > 5 else 0)
marked_data['PSQI_marked'] = original_data['PSQI'].apply(lambda x: 1 if x > 5 else 0)

# Lifestyle flags (kept semantics)
marked_data['Snoring'] = original_data['snorning'].apply(lambda x: 1 if x > 3 else 0)
marked_data['Daytime Napping'] = original_data['midday_rest_daylight_nap'].apply(lambda x: 1 if x > 1 else 0)
marked_data['Sports'] = original_data.apply(
    lambda row: 1 if row['sport_3'] == 2 and (row['sport1h'] == 2 or row['sport_30min'] == 2) else 0, axis=1
)
marked_data['SRI'] = original_data['SRI']
marked_data['Mobil Phone Use'] = original_data.apply(
    lambda row: 1 if (row['mobile_phone_usage1h'] == 2 or row['mobile_phone_usage30min'] == 2) else 0, axis=1
)
marked_data['Overweight'] = original_data['overweighted_2yes'].apply(lambda x: 1 if x > 1 else 0)

# ========== Label construction & PCA(2D) ==========
df = marked_data.copy()

# Map label to {0,1}; preserve original meaning of PSQI_marked
psqi_raw = df['PSQI_marked']
if psqi_raw.dtype == object:
    y = psqi_raw.astype(str).str.strip().str.lower().map({'non-insomnia': 0, 'insomnia': 1})
else:
    y = psqi_raw.map({0: 0, 1: 1, False: 0, True: 1})

mask = ~y.isna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(int).values

# Remove label from features; keep numeric columns only
df = df.drop(columns=['PSQI_marked'])
X_raw = df.select_dtypes(include=[np.number]).copy()
if X_raw.shape[1] == 0:
    raise ValueError("No numeric features available for PCA.")

# Impute -> scale -> PCA(2)
pipe_pca = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, random_state=42))
])
X_pca = pipe_pca.fit_transform(X_raw.values)

# Pack for plotting
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=X_raw.index)
df_pca['PSQI'] = y
df_pca['PSQI_lbl'] = np.where(y == 1, 'Insomnia', 'Non-Insomnia')

# Viridis endpoints: Non-Insomnia first, Insomnia second
base_colors = sns.color_palette('viridis', 2)
palette = {'Non-Insomnia': base_colors[0], 'Insomnia': base_colors[1]}

# Custom smooth blue colormap for probability background
colors_bg = ['#3E4F94', '#3E90BF', '#58B6E9', '#FAF9CB']
custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors_bg)

def evaluate_model(clf, X, y, name):
    """
    Print stratified k-fold CV metrics and a 20% hold-out evaluation.
    Returns the classifier fitted on the train split (for transparency).
    """
    # Allow at least 2 folds; upper-bounded by minority class size
    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
    cv_f1  = cross_val_score(clf, X, y, cv=cv, scoring='f1').mean()
    cv_auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    clf.fit(X_tr, y_tr)

    # Use predicted probability for AUC / thresholding
    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc_hold = accuracy_score(y_te, y_pred)
    f1_hold  = f1_score(y_te, y_pred)
    auc_hold = roc_auc_score(y_te, y_prob)

    print(f"[{name}] CV: ACC={cv_acc:.3f} | F1={cv_f1:.3f} | AUC={cv_auc:.3f}")
    print(f"[{name}] Hold-out: ACC={acc_hold:.3f} | F1={f1_hold:.3f} | AUC={auc_hold:.3f}")
    return clf

def plot_prob_background(clf, X2d, y01, df_plot, out_pdf):
    """
    SVM-style visualization with a smooth probability background (imshow)
    and no explicit decision boundary line.
    """
    # Fit on all data for the smooth background field
    clf.fit(X2d, y01)

    # Grid over PC space
    pad = 0.8
    x_min, x_max = X2d[:, 0].min() - pad, X2d[:, 0].max() + pad
    y_min, y_max = X2d[:, 1].min() - pad, X2d[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Probability field
    prob = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Background: imshow to match the SVM script's look-and-feel
    plt.figure(figsize=(9, 6.5))
    im = plt.imshow(
        prob, origin='lower',
        extent=(x_min, x_max, y_min, y_max),
        cmap=custom_cmap, vmin=0, vmax=1, aspect='auto', alpha=0.9
    )

    # Scatter points (viridis endpoints)
    sns.scatterplot(
        data=df_plot, x='PC1', y='PC2',
        hue='PSQI_lbl', hue_order=['Non-Insomnia', 'Insomnia'],
        palette=palette, s=90, edgecolor='k', linewidth=0.2, alpha=0.9
    )

    # Axes, legend, and colorbar styling
    plt.xlabel('PC1', fontsize=18)
    plt.ylabel('PC2', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)

    leg = plt.legend(loc='upper right', frameon=True, title=None)
    for text in leg.get_texts():
        text.set_fontsize(14)

    cbar = plt.colorbar(im)
    cbar.set_label('P(Insomnia)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Clean spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=450, bbox_inches='tight')
    plt.show()

# ========== Train, evaluate, and plot ==========
X = df_pca[['PC1', 'PC2']].values
y01 = df_pca['PSQI'].astype(int).values

# Logistic Regression (linear model with probabilities)
logreg = LogisticRegression(
    C=1.0, class_weight='balanced',
    solver='lbfgs', max_iter=1000, random_state=42
)
_ = evaluate_model(logreg, X, y01, "Logistic Regression (PC1 & PC2)")
plot_prob_background(logreg, X, y01, df_pca, "../result/PCA2_LogReg_boundary.pdf")
