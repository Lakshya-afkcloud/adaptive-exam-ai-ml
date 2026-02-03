import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ==========================================
# SETUP
# ==========================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 20

print("="*60)
print("NAVIKA MODEL EVALUATION")
print("="*60)

# ==========================================
# 1. LOAD DATA
# ==========================================
try:
    test_df = pd.read_csv('navika_real_test.csv')
    data_source = "Real World Data"
    
    if 'student_id' not in test_df.columns and 'user_id' in test_df.columns:
        test_df = test_df.rename(columns={'user_id': 'student_id'})
    if 'is_correct' not in test_df.columns and 'correct' in test_df.columns:
        test_df = test_df.rename(columns={'correct': 'is_correct'})
except FileNotFoundError:
    try:
        test_df = pd.read_csv('navika_synthetic_test.csv')
        data_source = "Synthetic Data"
    except FileNotFoundError:
        print("❌ No test data found!")
        exit()

print(f"✅ Loaded: {data_source}")
print(f"   {len(test_df):,} interactions from {test_df['student_id'].nunique()} students")

try:
    with open('navika_brain.pkl', 'rb') as f:
        kmeans_model, scaler_model, cluster_map = pickle.load(f)
    print("✅ Loaded trained model")
except FileNotFoundError:
    print("❌ navika_brain.pkl not found!")
    exit()

# ==========================================
# 2. BUILD PROFILES
# ==========================================
print("\nBuilding student profiles...")

test_df['speed_factor'] = test_df['response_time'] / (
    test_df['expected_time_sec'] if 'expected_time_sec' in test_df.columns 
    else test_df['response_time'].mean()
)

test_profiles = test_df.groupby('student_id').agg({
    'is_correct': ['mean', 'std'],
    'speed_factor': ['mean', 'std'],
    'response_time': 'count'
}).reset_index()

test_profiles.columns = ['student_id', 'accuracy', 'accuracy_std', 
                         'avg_speed', 'speed_consistency', 'n_interactions']

test_profiles = test_profiles.fillna(0)
test_profiles = test_profiles[test_profiles['n_interactions'] >= 5]

print(f"✅ {len(test_profiles):,} student profiles created")

# ==========================================
# 3. PREDICT COHORTS
# ==========================================
print("Predicting cohorts...")

# Match training features exactly
features = test_profiles[['accuracy', 'avg_speed', 'speed_consistency', 'accuracy_std']].copy()
features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())

scaled_features = scaler_model.transform(features)
test_profiles['predicted_cluster_id'] = kmeans_model.predict(scaled_features)
test_profiles['predicted_cohort'] = test_profiles['predicted_cluster_id'].map(cluster_map)

has_ground_truth = 'behavior_type_hidden' in test_df.columns

if has_ground_truth:
    ground_truth = test_df.groupby('student_id')['behavior_type_hidden'].first()
    test_profiles = test_profiles.merge(ground_truth.rename('true_cohort'),
                                       left_on='student_id', right_index=True, how='left')
    print("✅ Ground truth available")
else:
    print("⚠️ No ground truth (expected for real data)")

print(f"✅ Cohorts predicted")

# ==========================================
# 4. COHORT STATISTICS
# ==========================================
print("\n" + "="*60)
print("COHORT STATISTICS")
print("="*60)

cohort_summary = test_profiles.groupby('predicted_cohort').agg({
    'accuracy': ['mean', 'std', 'min', 'max'],
    'avg_speed': ['mean', 'std'],
    'speed_consistency': ['mean'],
    'student_id': 'count'
}).round(3)

cohort_summary.columns = ['acc_mean', 'acc_std', 'acc_min', 'acc_max',
                          'speed_mean', 'speed_std', 'consistency', 'count']

print(cohort_summary)

# ==========================================
# 5. EVALUATION METRICS
# ==========================================
if has_ground_truth:
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    test_profiles['predicted_clean'] = test_profiles['predicted_cohort'].str.replace('Detected: ', '')
    
    label_mapping = {
        'Fast_Guesser': 'Fast Guesser',
        'Slow_Thinker': 'Slow Thinker',
        'Normal': 'Normal'
    }
    test_profiles['true_mapped'] = test_profiles['true_cohort'].map(label_mapping)
    
    y_true = test_profiles['true_mapped']
    y_pred = test_profiles['predicted_clean']
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.2%}")
    print("\n" + classification_report(y_true, y_pred, zero_division=0))

# ==========================================
# 6. CLEAN VISUALIZATIONS
# ==========================================
print("\nGenerating separate high-resolution visualizations...")

# Configuration
cohort_order = test_profiles['predicted_cohort'].value_counts().index.tolist()
colors = ['#9B59B6', '#3498DB', '#E67E22', '#E74C3C', '#2ECC71'][:len(cohort_order)]
color_map = dict(zip(cohort_order, colors))

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12

def save_fig(fig, name):
    # Standardize saving with extra padding for external legends
    plt.subplots_adjust(right=0.8)
    fig.savefig(f'NAVIKA_Eval_{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: NAVIKA_Eval_{name}.png")
    plt.show()

# ==========================================
# A. Speed vs Accuracy Scatter
# ==========================================
fig1, ax1 = plt.subplots(figsize=(14, 8), facecolor='white')
ax1.set_facecolor('#F8F9FA')

for cohort in cohort_order:
    subset = test_profiles[test_profiles['predicted_cohort'] == cohort]
    ax1.scatter(subset['avg_speed'], subset['accuracy'], 
                s=120, alpha=0.75, label=cohort, color=color_map[cohort], 
                edgecolor='white', linewidth=1.5)

# Reference lines
ax1.axhline(0.5, color='gray', linestyle='--', linewidth=2, alpha=0.6)
ax1.axvline(1.0, color='gray', linestyle='--', linewidth=2, alpha=0.6)
ax1.text(0.1, 0.52, '50% Accuracy Threshold', fontsize=10, color='gray', fontweight='bold')
ax1.text(1.05, 0.05, 'Normal Speed (1.0x)', fontsize=10, color='gray', fontweight='bold', rotation=90)

ax1.set_xlabel('Average Speed Factor (Lower = Faster)', fontsize=LABEL_SIZE, fontweight='bold')
ax1.set_ylabel('Accuracy (0.0 - 1.0)', fontsize=LABEL_SIZE, fontweight='bold')
ax1.set_title('Student Behavior Map: Speed vs. Accuracy', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=TICK_SIZE)

# Outside Legend
ax1.legend(title="Detected Cohort", title_fontsize=LEGEND_SIZE, fontsize=LEGEND_SIZE, 
           loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, shadow=True)

save_fig(fig1, 'BehaviorMap')

# ==========================================
# B. Cohort Distribution
# ==========================================
fig2, ax2 = plt.subplots(figsize=(14, 6), facecolor='white')
ax2.set_facecolor('#F8F9FA')

counts = test_profiles['predicted_cohort'].value_counts()
y_pos = np.arange(len(counts))

bars = ax2.barh(y_pos, counts.values, color=[color_map[c] for c in counts.index], 
                edgecolor='black', linewidth=1.5, height=0.6)

# Labels
total = len(test_profiles)
for i, (bar, count) in enumerate(zip(bars, counts.values)):
    width = bar.get_width()
    pct = count / total * 100
    ax2.text(width + (total*0.01), bar.get_y() + bar.get_height()/2, 
             f'{int(count)} Students ({pct:.1f}%)', 
             ha='left', va='center', fontsize=12, fontweight='bold', color='#2C3E50')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(counts.index, fontsize=TICK_SIZE, fontweight='bold')
ax2.set_xlabel('Count', fontsize=LABEL_SIZE, fontweight='bold')
ax2.set_title('Final Cohort Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
ax2.grid(axis='x', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

save_fig(fig2, 'CohortCount')

# ==========================================
# C. Accuracy Boxplot
# ==========================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')
fig3.suptitle('Metric Distributions by Cohort', fontsize=TITLE_SIZE+2, fontweight='bold', y=0.98)

# Accuracy Boxplot
acc_data = [test_profiles[test_profiles['predicted_cohort'] == c]['accuracy'].values for c in cohort_order]
bp1 = ax3a.boxplot(acc_data, patch_artist=True, widths=0.6, showmeans=True, 
                   meanprops=dict(marker='D', markerfacecolor='white', markersize=8))

for patch, c_name in zip(bp1['boxes'], cohort_order):
    patch.set_facecolor(color_map[c_name])
    patch.set_alpha(0.7)
    patch.set_linewidth(1.5)

ax3a.set_xticklabels([c.replace('Detected: ','') for c in cohort_order], rotation=15, fontsize=TICK_SIZE)
ax3a.set_title('Accuracy Distribution', fontsize=TITLE_SIZE, fontweight='bold')
ax3a.set_ylabel('Accuracy', fontsize=LABEL_SIZE)
ax3a.grid(axis='y', alpha=0.3)

# Speed Boxplot
spd_data = [test_profiles[test_profiles['predicted_cohort'] == c]['avg_speed'].values for c in cohort_order]
bp2 = ax3b.boxplot(spd_data, patch_artist=True, widths=0.6, showmeans=True,
                   meanprops=dict(marker='D', markerfacecolor='white', markersize=8))

for patch, c_name in zip(bp2['boxes'], cohort_order):
    patch.set_facecolor(color_map[c_name])
    patch.set_alpha(0.7)
    patch.set_linewidth(1.5)

ax3b.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Expected Speed')
ax3b.set_xticklabels([c.replace('Detected: ','') for c in cohort_order], rotation=15, fontsize=TICK_SIZE)
ax3b.set_title('Speed Factor Distribution', fontsize=TITLE_SIZE, fontweight='bold')
ax3b.set_ylabel('Speed Factor', fontsize=LABEL_SIZE)
ax3b.grid(axis='y', alpha=0.3)
ax3b.legend(loc='upper right')

save_fig(fig3, 'Boxplots')

# ==========================================
# FIG 4: CONSISTENCY VS PERFORMANCE
# ==========================================
fig4, ax4 = plt.subplots(figsize=(14, 8), facecolor='white')
ax4.set_facecolor('#F8F9FA')

for cohort in cohort_order:
    subset = test_profiles[test_profiles['predicted_cohort'] == cohort]
    ax4.scatter(subset['speed_consistency'], subset['accuracy'], 
                s=100, alpha=0.75, label=cohort, color=color_map[cohort], 
                edgecolor='white', linewidth=1)

ax4.set_xlabel('Speed Variability (Standard Deviation)', fontsize=LABEL_SIZE, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=LABEL_SIZE, fontweight='bold')
ax4.set_title('Consistency vs. Performance', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=TICK_SIZE)
ax4.text(0.05, 0.05, '← More Consistent', transform=ax4.transAxes, fontsize=12, color='#2C3E50', fontweight='bold')

# Outside Legend
ax4.legend(title="Cohort", title_fontsize=LEGEND_SIZE, fontsize=LEGEND_SIZE, 
           loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, shadow=True)

save_fig(fig4, 'Consistency')

# ---------------------------------------------------------
# FIG 5: SUMMARY TABLE
# ---------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(14, 5), facecolor='white')
ax5.axis('off')

table_data = []
for cohort in cohort_order:
    subset = test_profiles[test_profiles['predicted_cohort'] == cohort]
    table_data.append([
        cohort.replace('Detected: ', ''),
        f"{len(subset)}",
        f"{subset['accuracy'].mean():.2%}",
        f"{subset['avg_speed'].mean():.2f}x",
        f"{subset['speed_consistency'].mean():.3f}"
    ])

# Add header
col_labels = ['Cohort', 'Count', 'Avg Accuracy', 'Avg Speed', 'Avg Consistency']
row_colors = []

# Create table
table = ax5.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center',
                  colColours=['#2C3E50']*5)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)  # Scale up rows

# Style loop
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('white')
    else:
        # Match border color to cohort color for left strip, else grey
        if col == 0:
            cell.set_text_props(weight='bold', color=color_map.get(table_data[row-1][0], 'black'))
        cell.set_edgecolor('#BDC3C7')
        cell.set_facecolor('#F8F9FA' if row % 2 == 0 else 'white')

ax5.set_title(f'NAVIKA Cohort Statistical Summary ({data_source})', fontsize=TITLE_SIZE, fontweight='bold', y=0.85)

save_fig(fig5, 'SummaryTable')

# ==========================================
# 7. DETAILED INSIGHTS
# ==========================================
print("\n" + "="*60)
print("BEHAVIORAL INSIGHTS")
print("="*60)

for cohort in cohort_order:
    subset = test_profiles[test_profiles['predicted_cohort'] == cohort]
    print(f"\n{cohort}:")
    print(f"  Students: {len(subset)} ({len(subset)/len(test_profiles)*100:.1f}%)")
    print(f"  Accuracy: {subset['accuracy'].mean():.2%} (σ={subset['accuracy'].std():.3f})")
    print(f"  Speed: {subset['avg_speed'].mean():.2f}x (σ={subset['avg_speed'].std():.3f})")
    print(f"  Consistency: {subset['speed_consistency'].mean():.3f}")
    
    # Insights
    if subset['avg_speed'].mean() < 0.7:
        print("  → Very fast responses - monitor for guessing")
    elif subset['avg_speed'].mean() > 1.3:
        print("  → Slower learners - may need additional support")
    
    if subset['speed_consistency'].mean() > 1.0:
        print("  → High variability - inconsistent performance")

# ==========================================
# 8. SAVE RESULTS
# ==========================================
results = {
    'data_source': data_source,
    'n_students': len(test_profiles),
    'n_interactions': len(test_df),
    'n_cohorts': len(cohort_order)
}

if has_ground_truth:
    results['accuracy'] = acc

pd.DataFrame([results]).to_csv('navika_evaluation_results.csv', index=False)
test_profiles.to_csv('navika_test_predictions.csv', index=False)

print("\n✅ Saved: navika_evaluation_results.csv")
print("✅ Saved: navika_test_predictions.csv")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)
print(f"Students: {len(test_profiles):,}")
print(f"Cohorts: {len(cohort_order)}")
if has_ground_truth:
    print(f"Accuracy: {acc:.2%}")
print("="*60)