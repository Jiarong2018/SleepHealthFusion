#statistic_analysis2.py
"""
Scatter plots of SRI vs. PHQ-9 / GAD-7 / PSQI / FSS-9 with binary highlight.
- Reads:  ../data/deidentified_data.csv
- Writes: ../result/SRI_vs_PHQ9_GAD7_PSQI_FSS9.pdf
"""
import pandas as pd
import matplotlib.pyplot as plt


# Load data and derive flags

df = pd.read_csv('../data/deidentified_data.csv')

# Binary flags based on conventional cutoffs (1 = above threshold, 0 = otherwise)
df['depression_marked'] = (df['depression'] > 5).astype(int)   # PHQ-9 > 5
df['anxiety_marked']    = (df['anxiety'] > 5).astype(int)      # GAD-7 > 5
df['tiredness_marked']  = (df['tiredness'] > 4).astype(int)    # FSS-9 > 4
df['PSQI_marked']       = (df['PSQI'] > 5).astype(int)         # PSQI > 5


# Configure panel definitions and figure

# Each tuple: (y column, flag column, y-axis label)
plots = [
    ('PSQI',       'PSQI_marked',       'PSQI'),
    ('depression', 'depression_marked', 'PHQ-9'),
    ('anxiety',    'anxiety_marked',    'GAD-7'),
    ('tiredness',  'tiredness_marked',  'FSS-9'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)


# Draw scatter panels

for ax, (y_col, mark_col, ylabel) in zip(axes.flat, plots):
    # Color mapping: flagged (1) → orange; non-flagged (0) → black
    colors = df[mark_col].map({1: 'orange', 0: 'black'})

    ax.scatter(
        df['SRI'],
        df[y_col],
        c=colors,
        alpha=0.7,
        s=100
    )

    # Axis labels and styling
    ax.set_xlabel('SRI', fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)

    # Hide top/right spines to match the original visual style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tick label sizes
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# Tight layout and export
plt.tight_layout()
plt.savefig('../result/SRI_vs_PHQ9_GAD7_PSQI_FSS9.pdf', dpi=450, bbox_inches='tight')
plt.show()
