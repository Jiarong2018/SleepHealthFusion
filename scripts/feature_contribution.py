# feature_contribution.py
"""
This script analyzes and visualizes feature contributions
from a trained XGBoost classification model.
Output is identical to the contribution part of the training script.
"""

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Load trained model
model = xgb.XGBClassifier()
model.load_model("../result/xgb_model.json")

#Define feature names (must match training)
feature_names = ['Age', 'Gender', 'Snoring',
                 'Daytime Napping', 'Sports', 'SRI',
                 'Mobile Phone Use', 'Overweight']


#  Feature importance analysis
feature_importances = model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("每个特征的贡献度排名：")
print(feature_df)

# Visualization

plt.figure(figsize=(10, 6))
plt.barh(feature_df['Feature'], feature_df['Importance'] * 100, color='#3E4F94')  # convert to %
plt.xlabel('Importance (%)', fontsize=20)
plt.gca().invert_yaxis()  # most important feature at top

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

plt.savefig('../result/xgboost_feature_contribution.pdf', dpi=450, bbox_inches='tight')
plt.show()
