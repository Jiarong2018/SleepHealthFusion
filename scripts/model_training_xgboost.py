# --- Core Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Visualization & Colors ---
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Machine Learning ---
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score,
    recall_score, f1_score
)

# =========================================
# 1. Load and preprocess data
# =========================================
original_data = pd.read_csv('../data/deidentified_data.csv')

# Create a processed DataFrame for modeling
marked_data = pd.DataFrame()

# Basic demographic and lifestyle features
marked_data['Age'] = original_data['age']
marked_data['Sleep Need'] = original_data['self_reported_sleep_need']
marked_data['Education'] = original_data['education']
marked_data['Gender'] = original_data['gender'] - 1  # male=0, female=1

# Health outcomes (binary labels)
marked_data['PHQ-9'] = original_data['depression'].apply(lambda x: 1 if x > 5 else 0)
marked_data['GAD-7'] = original_data['anxiety'].apply(lambda x: 1 if x > 5 else 0)
marked_data['FSS-9'] = original_data['tiredness'].apply(lambda x: 1 if x > 4 else 0)
marked_data['PSQI'] = original_data['PSQI'].apply(lambda x: 1 if x > 5 else 0)

# Lifestyle-related binary features
marked_data['Snoring'] = original_data['snorning'].apply(lambda x: 1 if x > 3 else 0)
marked_data['Daytime Napping'] = original_data['midday_rest_daylight_nap'].apply(lambda x: 1 if x > 1 else 0)
marked_data['Sports'] = original_data.apply(
    lambda row: 1 if row['sport_3'] == 2 and (row['sport1h'] == 2 or row['sport_30min'] == 2) else 0, axis=1
)
marked_data['SRI'] = original_data['SRI']
marked_data['Mobile Phone Use'] = original_data.apply(
    lambda row: 1 if row['mobile_phone_usage1h'] == 2 or row['mobile_phone_usage30min'] == 2 else 0, axis=1
)
marked_data['Overweight'] = original_data['overweighted_2yes'].apply(lambda x: 1 if x > 1 else 0)

# =========================================
# 2. Prepare features and target
# =========================================
X = marked_data[['Age', 'Gender', 'Snoring',
                 'Daytime Napping', 'Sports', 'SRI',
                 'Mobile Phone Use', 'Overweight']]
y = marked_data['PSQI']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================================
# 3. Train XGBoost model
# =========================================
model = xgb.XGBClassifier(scale_pos_weight=99)  # handle class imbalance
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# =========================================
# 4. Confusion matrix visualization
# =========================================
plt.rcParams['font.family'] = 'Arial'  # use Arial font

# Custom colormap
colors = ['#3E4F94', '#3E90BF', '#58B6E9', '#FAF9CB']
custom_cmap = LinearSegmentedColormap.from_list('custom_grad', colors)

fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=custom_cmap, colorbar=False, ax=ax)
ax.set_aspect('equal')

# Enlarge numbers inside cells
for row in disp.text_:
    for txt in row:
        txt.set_fontsize(15)

# Axis labels
ax.set_xticks([0, 1])
ax.set_xticklabels(['Predict\nNon-Insomnia', 'Predict\nInsomnia'], fontsize=20, ha='center')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Non-Insomnia', 'Insomnia'], fontsize=20)
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='both', which='both', length=0)

# Add colorbar with fixed width
fixed_inch = 0.4
fig_width_inch = fig.get_figwidth()
size_pct = f"{fixed_inch / fig_width_inch * 100:.1f}%"
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size=size_pct, pad=0.5)
im = ax.images[0]
cbar = fig.colorbar(im, cax=cax)

cbar.ax.xaxis.set_label_position('top')
cbar.ax.set_xlabel('Count', fontsize=16, labelpad=8)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('../result/xgboost_confusion_matrix.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()

# =========================================
# 5. ROC curve visualization
# =========================================
roc_auc = roc_auc_score(y, y_pred_proba)
fpr, tpr, _ = roc_curve(y, y_pred_proba)

fig, ax = plt.subplots(figsize=(7, 6))
main_color = custom_cmap(0.2)
diag_color = custom_cmap(0.8)

ax.plot(fpr, tpr, color=main_color, linewidth=2.5, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color=diag_color)
ax.set_xlabel('False Positive Rate', fontsize=20)
ax.set_ylabel('True Positive Rate', fontsize=20)
ax.legend(fontsize=16, loc='lower right', frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig('../result/xgboost_ROC.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()


model.save_model("../result/xgb_model.json")
