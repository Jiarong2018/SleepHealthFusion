#statistic_analysis1.py
"""
Generate KDE and box plots of SRI stratified by insomnia (PSQI), depression, and anxiety.
- Reads:   ../data/deidentified_data.csv
- Writes:  ../result/kde_plot_psqi.pdf
           ../result/box_plot_psqi.pdf
           ../result/kde_plot_depression.pdf
           ../result/box_plot_depression.pdf
           ../result/kde_plot_anxiety.pdf
           ../result/box_plot_anxiety.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


# Load and prepare data

original_data = pd.read_csv('../data/deidentified_data.csv')

# Create a working table with derived indicators
marked_data = pd.DataFrame({
    'kcode': original_data['kcode'],
    'age': original_data['age'],
    'self_reported_sleep_need': original_data['self_reported_sleep_need'],
    'education': original_data['education'],
    # Map gender to 0/1: male=0, female=1 (source column assumed 1/2)
    'gender': original_data['gender'] - 1,
    # Clinical cutoffs / binary flags
    'depression_marked': (original_data['depression'] > 5).astype(int),
    'anxiety_marked': (original_data['anxiety'] > 5).astype(int),
    'tiredness_marked': (original_data['tiredness'] > 4).astype(int),
    'PSQI_marked': (original_data['PSQI'] > 5).astype(int),
    # Snoring: >3 times/week treated as positive
    'snoring': (original_data['snorning'] > 3).astype(int),
    # Daytime nap: threshold >1
    'midday_rest_daylight_nap': (original_data['midday_rest_daylight_nap'] > 1).astype(int),
    # Physical activity: sport_3==2 AND (sport1h==2 OR sport_30min==2)
    'sport_marked': original_data.apply(
        lambda r: 1 if (r['sport_3'] == 2) and (r['sport1h'] == 2 or r['sport_30min'] == 2) else 0, axis=1
    ),
    # Sleep Regularity Index
    'SRI': original_data['SRI'],
    # Mobile phone usage: either threshold met
    'mobile_phone_usage_marked': original_data.apply(
        lambda r: 1 if (r['mobile_phone_usage1h'] == 2 or r['mobile_phone_usage30min'] == 2) else 0, axis=1
    ),
    # Overweight flag
    'over_weighted': (original_data['overweighted_2yes'] > 1).astype(int),
})

# Global figure style (consistent across all panels)
plt.rcParams['font.family'] = 'Arial'
sns.set(style="white", context="talk")

# A convenient base palette (viridis, 2 levels)
base_two = sns.color_palette('viridis', 2)


# SRI vs Insomnia status (PSQI_marked → Non-Insomnia / Insomnia)

psqi_df = marked_data.copy()
psqi_df['PSQI_marked'] = psqi_df['PSQI_marked'].replace({0: 'Non-Insomnia', 1: 'Insomnia'})

psqi_order = ['Non-Insomnia', 'Insomnia']
psqi_palette = dict(zip(psqi_order, base_two))

# KDE
fig, ax1 = plt.subplots(figsize=(8, 6))
sns.kdeplot(
    data=psqi_df,
    x='SRI',
    hue='PSQI_marked',
    hue_order=psqi_order,
    fill=True,
    common_norm=False,
    alpha=0.5,
    linewidth=2,
    palette=psqi_palette,
    ax=ax1
)
ax1.set_title('')
ax1.set_xlabel('SRI', fontsize=18)
ax1.set_ylabel('Probability Density', fontsize=18)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
for spine in ax1.spines.values():
    spine.set_visible(True)

# Legend without frame
handles = [Patch(facecolor=psqi_palette[k], alpha=0.5, label=k) for k in psqi_order]
ax1.legend(handles=handles, title=' ', loc='upper left', bbox_to_anchor=(0, 1.06), frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig('../result/kde_plot_psqi.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()

# Boxplot
fig, ax2 = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=psqi_df,
    x='PSQI_marked',
    y='SRI',
    hue='PSQI_marked',
    hue_order=psqi_order,
    dodge=False,
    showfliers=False,
    palette=psqi_palette,
    ax=ax2
)
ax2.set_title('')
ax2.set_xlabel(' ', fontsize=16)
ax2.set_ylabel('SRI', fontsize=16)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
for spine in ax2.spines.values():
    spine.set_visible(True)

leg = ax2.get_legend()
if leg:
    leg.remove()

plt.tight_layout()
plt.savefig('../result/box_plot_psqi.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()


# SRI vs Depression (depression_marked → Non/Depression)

dep_df = marked_data.copy()
dep_df['depression_marked'] = dep_df['depression_marked'].replace({0: 'Non-Depression', 1: 'Depression'})

dep_order = ['Non-Depression', 'Depression']
dep_palette = dict(zip(dep_order, base_two))

# KDE
fig, ax1 = plt.subplots(figsize=(8, 6))
sns.kdeplot(
    data=dep_df,
    x='SRI',
    hue='depression_marked',
    hue_order=dep_order,
    fill=True,
    common_norm=False,
    alpha=0.5,
    linewidth=2,
    palette=dep_palette,
    ax=ax1
)
ax1.set_title('')
ax1.set_xlabel('SRI', fontsize=18)
ax1.set_ylabel('Probability Density', fontsize=18)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
for spine in ax1.spines.values():
    spine.set_visible(True)

handles = [Patch(facecolor=dep_palette[k], alpha=0.5, label=k) for k in dep_order]
ax1.legend(handles=handles, title=' ', loc='upper left', bbox_to_anchor=(0, 1.06), frameon=False, fontsize=16)

plt.tight_layout()
plt.savefig('../result/kde_plot_depression.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()

# Boxplot
fig, ax2 = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=dep_df,
    x='depression_marked',
    y='SRI',
    hue='depression_marked',
    hue_order=dep_order,
    dodge=False,
    showfliers=False,
    palette=dep_palette,
    ax=ax2
)
ax2.set_title('')
ax2.set_xlabel(' ', fontsize=16)
ax2.set_ylabel('SRI', fontsize=16)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
for spine in ax2.spines.values():
    spine.set_visible(True)

leg = ax2.get_legend()
if leg:
    leg.remove()

plt.tight_layout()
plt.savefig('../result/box_plot_depression.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()


# SRI vs Anxiety (custom order & legend behavior)

anx_df = marked_data.copy()
anx_df['anxiety_marked'] = anx_df['anxiety_marked'].replace({0: 'Non-Anxiety', 1: 'Anxiety'})

# NOTE:
# - Plot order (x-axis and hue) is ['Anxiety', 'Non-Anxiety'] to match the original script.
# - Legend is manually reordered to show ['Non-Anxiety', 'Anxiety'].
anx_order = ['Anxiety', 'Non-Anxiety']
anx_palette = {
    'Anxiety': base_two[1],       # second color (green-ish)
    'Non-Anxiety': base_two[0],   # first color (blue-ish)
}

# KDE
fig, ax1 = plt.subplots(figsize=(8, 6))
sns.kdeplot(
    data=anx_df,
    x='SRI',
    hue='anxiety_marked',
    hue_order=anx_order,  # left: Anxiety; right: Non-Anxiety (for consistency with original)
    fill=True,
    common_norm=False,
    alpha=0.5,
    linewidth=2,
    palette=anx_palette,
    ax=ax1
)
ax1.set_title('')
ax1.set_xlabel('SRI', fontsize=18)
ax1.set_ylabel('Probability Density', fontsize=18)
ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
for spine in ax1.spines.values():
    spine.set_visible(True)

# Legend shown as ['Non-Anxiety', 'Anxiety'] without frame
handles = [Patch(facecolor=anx_palette[k], alpha=0.5, label=k) for k in ['Non-Anxiety', 'Anxiety']]
ax1.legend(handles=handles, title=' ', loc='upper left', bbox_to_anchor=(0, 1.06), frameon=False, fontsize=16)

plt.savefig('../result/kde_plot_anxiety.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()

# Boxplot
fig, ax2 = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=anx_df,
    x='anxiety_marked',
    y='SRI',
    hue='anxiety_marked',
    order=anx_order,       # x-order: Anxiety (left), Non-Anxiety (right)
    hue_order=anx_order,
    dodge=False,
    showfliers=False,
    palette=anx_palette,
    ax=ax2
)
ax2.set_title('')
ax2.set_xlabel(' ', fontsize=16)
ax2.set_ylabel('SRI', fontsize=16)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)
for spine in ax2.spines.values():
    spine.set_visible(True)

leg = ax2.get_legend()
if leg:
    leg.remove()

plt.tight_layout()
plt.savefig('../result/box_plot_anxiety.pdf', dpi=450, bbox_inches='tight', pad_inches=0.2)
plt.show()
