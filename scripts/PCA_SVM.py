#PCA_SVM.py
"""
Linear SVM on 2D PCA of engineered features.
- Data: '../data/deidentified_data.csv'
- Prints: 5-fold CV and 20% hold-out metrics (ACC, F1, AUC)
- Figure: '../result/PCA2_SVM_boundary.pdf' (probability background + viridis endpoints)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

# ================ Load & engineer features (kept logic) ================
# Read raw data
original_data = pd.read_csv('../data/deidentified_data.csv')

# Build the working DataFrame with the exact original feature definitions
marked_data = pd.DataFrame()
marked_data['Age'] = original_data['age']
marked_data['Sleep Need'] = original_data['self_reported_sleep_need']
marked_data['Education'] = original_data['education']

# Gender recoding: male=0, female=1 (source coding minus 1)
marked_data['Gender'] = original_data['gender'] - 1

# Clinical screens (same thresholds as the original script)
marked_data['PHQ-9'] = original_data['depression'].apply(lambda x: 1 if x > 5 else 0)
marked_data['GAD-7'] = original_data['anxiety'].apply(lambda x: 1 if x > 5 else 0)
marked_data['PSQI_marked'] = original_data['PSQI'].apply(lambda x: 1 if x > 5 else 0)

# Snoring: >3 times/week → 1
marked_data['Snoring'] = original_data['snorning'].apply(lambda x: 1 if x > 3 else 0)

# Daytime napping: >1 → 1
marked_data['Daytime Napping'] = original_data['midday_rest_daylight_nap'].apply(lambda x: 1 if x > 1 else 0)

# Sports: sport_3==2 AND (sport1h==2 OR sport_30min==2) → 1
marked_data['Sports'] = original_data.apply(
    lambda row: 1 if row['sport_3'] == 2 and (row['sport1h'] == 2 or row['sport_30min'] == 2) else 0, axis=1
)

# Sleep Regularity Index (continuous)
marked_data['SRI'] = original_data['SRI']

# Mobile phone use: either usage flag == 2 → 1
marked_data['Mobil Phone Use'] = original_data.apply(
    lambda row: 1 if row['mobile_phone_usage1h'] == 2 or row['mobile_phone_usage30min'] == 2 else 0, axis=1
)

# Overweight flag from 'overweighted_2yes': >1 → 1
marked_data['Overweight'] = original_data['overweighted_2yes'].apply(lambda x: 1 if x > 1 else 0)

# ================ Prepare label and features ================
df = marked_data.copy()

# Clean and map PSQI labels to {0,1}; keep original mapping semantics
psqi_raw = df['PSQI_marked']
if psqi_raw.dtype == object:
    y = psqi_raw.astype(str).str.strip().str.lower().map({'non-insomnia': 0, 'insomnia': 1})
else:
    y = psqi_raw.map({0: 0, 1: 1, False: 0, True: 1})

mask = ~y.isna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(int).values

# Drop label column from features; keep numeric features only
if 'PSQI_marked' in df.columns:
    df = df.drop(columns=['PSQI_marked'])
X_raw = df.select_dtypes(include=[np.number]).copy()
if X_raw.shape[1] == 0:
    raise ValueError("No numeric features available for PCA.")

# ================ PCA to 2D (impute+scale inside a pipeline) ================
pipe_pca = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, random_state=42))
])
X_pca = pipe_pca.fit_transform(X_raw.values)

# For plotting
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['PSQI'] = y
df_pca['PSQI_lbl'] = np.where(y == 1, 'Insomnia', 'Non-Insomnia')

# ================ Train & evaluate Linear SVM (probability enabled) ================
# Note: Linear SVM is deterministic here; CV splits are fixed via random_state.
svm = SVC(kernel='linear', class_weight='balanced', probability=True)

# 5-fold stratified CV (folds limited by the minority class size)
cv = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=42)
cv_acc = cross_val_score(svm, X_pca, y, cv=cv, scoring='accuracy').mean()
cv_f1  = cross_val_score(svm, X_pca, y, cv=cv, scoring='f1').mean()
cv_auc = cross_val_score(svm, X_pca, y, cv=cv, scoring='roc_auc').mean()

# 20% hold-out
X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
svm.fit(X_tr, y_tr)
y_prob = svm.predict_proba(X_te)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc_hold = accuracy_score(y_te, y_pred)
f1_hold  = f1_score(y_te, y_pred)
auc_hold = roc_auc_score(y_te, y_prob)

# Keep original (Chinese) print messages to preserve console output
print(f"[Linear SVM] CV: ACC={cv_acc:.3f} | F1={cv_f1:.3f} | AUC={cv_auc:.3f}")
print(f"[Linear SVM] Hold-out: ACC={acc_hold:.3f} | F1={f1_hold:.3f} | AUC={auc_hold:.3f}")

# ================ Visualization: probability background + endpoints ================
# Mesh grid for probability field
pad = 0.8
x_min, x_max = X_pca[:, 0].min() - pad, X_pca[:, 0].max() + pad
y_min, y_max = X_pca[:, 1].min() - pad, X_pca[:, 1].max() + pad
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
grid = np.c_[xx.ravel(), yy.ravel()]
prob = svm.predict_proba(grid)[:, 1].reshape(xx.shape)

# Custom smooth colormap for background (deep blue → blue → light blue → pale yellow)
colors = ['#3E4F94', '#3E90BF', '#58B6E9', '#FAF9CB']
custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors)

plt.figure(figsize=(9, 6.5))
# Probability background (no explicit decision boundary)
im = plt.imshow(
    prob, origin='lower',
    extent=(x_min, x_max, y_min, y_max),
    cmap=custom_cmap, vmin=0, vmax=1, aspect='auto', alpha=0.9
)

# Scatter with viridis endpoints: Non-Insomnia first, Insomnia second
base_colors = sns.color_palette('viridis', 2)
palette = {'Non-Insomnia': base_colors[0], 'Insomnia': base_colors[1]}
sns.scatterplot(
    x='PC1', y='PC2', data=df_pca,
    hue='PSQI_lbl', palette=palette, hue_order=['Non-Insomnia', 'Insomnia'],
    s=90, edgecolor='k', linewidth=0.2, alpha=0.9
)

# Axis and legend styling
plt.xlabel('PC1', fontsize=18)
plt.ylabel('PC2', fontsize=18)
plt.tick_params(axis='both', labelsize=16)

leg = plt.legend(loc='upper right', frameon=True, title=None)
for text in leg.get_texts():
    text.set_fontsize(14)

# Colorbar with probability label
cbar = plt.colorbar(im)
cbar.set_label('P(Insomnia)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig('../result/PCA2_SVM_boundary.pdf', dpi=450, bbox_inches='tight')
plt.show()
