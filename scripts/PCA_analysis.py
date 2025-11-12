import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load and mark raw data
# Input data
original_data = pd.read_csv('../data/deidentified_data.csv')

# Build the working DataFrame with engineered/marked features
# Conventions used in the source data (kept exactly as in the original script):
# - gender: make male=0, female=1 by subtracting 1 from source coding
# - binary flags: 0=no, 1=yes (thresholds exactly follow the original logic)
marked_data = pd.DataFrame()
marked_data['Age'] = original_data['age']
marked_data['Sleep Need'] = original_data['self_reported_sleep_need']
marked_data['Education'] = original_data['education']

# Gender: male=0, female=1  (keep original transformation)
marked_data['Gender'] = original_data['gender'] - 1

# Clinical screens (same thresholds as original)
marked_data['PHQ-9'] = original_data['depression'].apply(lambda x: 1 if x > 5 else 0)
marked_data['GAD-7'] = original_data['anxiety'].apply(lambda x: 1 if x > 5 else 0)
marked_data['PSQI_marked'] = original_data['PSQI'].apply(lambda x: 1 if x > 5 else 0)

# Snoring: >3 times/week → 1
marked_data['Snoring'] = original_data['snorning'].apply(lambda x: 1 if x > 3 else 0)

# Daytime napping: threshold >1 (keep original definition)
marked_data['Daytime Napping'] = original_data['midday_rest_daylight_nap'].apply(lambda x: 1 if x > 1 else 0)

# Sports: requires sport_3==2 AND (sport1h==2 OR sport_30min==2)
marked_data['Sports'] = original_data.apply(
    lambda row: 1 if row['sport_3'] == 2 and (row['sport1h'] == 2 or row['sport_30min'] == 2) else 0, axis=1
)

# Sleep Regularity Index (continuous)
marked_data['SRI'] = original_data['SRI']

# Mobile phone use: 1 if either >= threshold (keep original semantics)
marked_data['Mobil Phone Use'] = original_data.apply(
    lambda row: 1 if row['mobile_phone_usage1h'] == 2 or row['mobile_phone_usage30min'] == 2 else 0, axis=1
)

# Overweight flag (source variable "overweighted_2yes"): >1 → 1
marked_data['Overweight'] = original_data['overweighted_2yes'].apply(lambda x: 1 if x > 1 else 0)

# ========= 1) PCA to 2D + scatter by PSQI status =========
# Features = all columns except the label
feature_cols = [c for c in marked_data.columns if c != 'PSQI_marked']
X_raw = marked_data[feature_cols].copy()

# Label y (keep original mapping rules verbatim)
y_raw = marked_data['PSQI_marked']
_map = {'Non-Insomnia': 0, 'Insomnia': 1, 'non-insomnia': 0, 'insomnia': 1, True: 1, False: 0}
y = y_raw.map(_map).fillna(y_raw).astype(int).values

# Standardize features, then 2D PCA (random_state kept for reproducibility, same as original)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Build a plotting DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['PSQI'] = y
df_pca['PSQI_lbl'] = np.where(y == 1, 'Insomnia', 'Non-Insomnia')

# Colors: viridis two-color palette, Non-Insomnia first, Insomnia second (same as original)
base_colors = sns.color_palette('viridis', 2)
palette = {'Non-Insomnia': base_colors[0], 'Insomnia': base_colors[1]}

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca,
    x='PCA1', y='PCA2',
    hue='PSQI_lbl',
    hue_order=['Non-Insomnia', 'Insomnia'],
    palette=palette,
    s=100,
    linewidth=0,   # avoid edgecolor warnings
    alpha=0.7
)

# Axis labels and ticks (kept sizes)
plt.xlabel('Principal Component 1', fontsize=20)
plt.ylabel('Principal Component 2', fontsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

# Remove top/right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend styling and location (kept)
plt.legend(
    title=None,
    loc='lower left',
    frameon=True,
    edgecolor='black',
    fontsize=16,
    markerscale=2
)

# Output (same filename, dpi, bbox)
plt.savefig('../result/PCA_PSQI.pdf', dpi=450, bbox_inches='tight')
plt.show()

# ========= 2) PCA on all numeric features + Scree plot =========
df = marked_data.copy()

# Drop label column (do not include in PCA)
if 'PSQI_marked' in df.columns:
    df = df.drop(columns='PSQI_marked')

# Keep numeric cols only for PCA
X_raw = df.select_dtypes(include=[np.number]).copy()
if X_raw.shape[1] == 0:
    raise ValueError("No numeric features available for PCA. Please check data or encode categorical variables first.")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA keeping all components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_    # per-PC explained variance ratio
cum_explained = np.cumsum(explained)         # cumulative explained variance ratio

# Helper: number of PCs required to reach a given threshold
def need_k_for(threshold: float) -> int:
    return int(np.searchsorted(cum_explained, threshold) + 1)

# Print the same diagnostics
print(f"总特征数（进入PCA）：{X_raw.shape[1]}")
if len(explained) >= 2:
    print(f"前两主成分累计解释方差：{cum_explained[1]:.3f}  (PC1={explained[0]:.3f}, PC2={explained[1]:.3f})")
else:
    print(f"PC1 解释方差：{explained[0]:.3f}")

for thr in [0.80, 0.90, 0.95]:
    k = need_k_for(thr)
    print(f"达到 {int(thr*100)}% 累计解释方差需要主成分个数：K={k}")

# Scree plot (bar: per-PC; line: cumulative), styling identical to original
plt.figure(figsize=(9, 5.2))
xs = np.arange(1, len(explained) + 1)

# Bar = explained variance of each PC
plt.bar(xs, explained, alpha=0.8, color='#3E4F94', edgecolor='none', label='Explained variance ratio')

# Line = cumulative explained variance
plt.plot(xs, cum_explained, linewidth=2.5, marker='o', color='#3E4F94', label='Cumulative explained variance')

plt.xticks(xs)
plt.xlabel('Principal Component (PC)', fontsize=18)
plt.ylabel('Proportion of Variance', fontsize=18)
plt.tick_params(axis='both', labelsize=16)

# Keep left/bottom spines, hide top/right
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

plt.grid(False)
plt.ylim(0, 1.05)
plt.legend(loc='best', frameon=False)

plt.tight_layout()
plt.savefig('../result/PCA_explained_variance_450dpi.pdf', dpi=450, bbox_inches='tight')
plt.show()

# ========= 3) Feature loadings heatmap (contributions to first PCs) =========
# Use the same features as Section 1 (exclude the label column only)
features = [col for col in marked_data.columns if col != 'PSQI_marked']
X = marked_data[features].copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA with number of components equal to number of features
pca = PCA(n_components=len(features))
pca.fit(X_scaled)

# Loadings matrix: each feature's coefficient on each PC
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(features))],
    index=features
)

# Heatmap of contributions to the first 7 PCs (kept exactly as original)
plt.figure(figsize=(10, 6))
sns.heatmap(loadings.iloc[:, :7], annot=True, cmap="coolwarm", center=0)

# Clean axes, keep tick sizes
plt.xlabel("")
plt.ylabel("")
plt.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig("../result/PCA_feature_contributions.pdf", dpi=450, bbox_inches="tight")
plt.show()
