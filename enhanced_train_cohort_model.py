import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle

# ==========================================
# 1. LOAD REAL TRAINING DATA
# ==========================================
print("="*60)
print("NAVIKA Model Training - Using Real-World Data")
print("="*60)

# Try loading real data first, fallback to synthetic
try:
    df = pd.read_csv('navika_real_train.csv')
    print(f"✅ Loaded {len(df):,} real-world training interactions")
    data_source = "Real World Data"
    
    # Standardize student ID column name
    if 'student_id' not in df.columns and 'user_id' in df.columns:
        df['student_id'] = df['user_id']
    
    # Real data uses 'user_id' and may not have 'expected_time_sec'
    if 'expected_time_sec' not in df.columns:
        # Calculate expected time as mean time per item
        item_avg_times = df.groupby('item_id')['response_time'].mean()
        df['expected_time_sec'] = df['item_id'].map(item_avg_times)
        
except FileNotFoundError:
    print("⚠️ Real data not found. Attempting to load synthetic data...")
    try:
        df = pd.read_csv('navika_synthetic_train.csv')
        print(f"✅ Loaded {len(df):,} synthetic training interactions (fallback)")
        data_source = "Synthetic Data"
        
        # Rename columns to match expected format
        if 'student_id' not in df.columns and 'user_id' in df.columns:
            df['student_id'] = df['user_id']
        if 'expected_time' in df.columns and 'expected_time_sec' not in df.columns:
            df['expected_time_sec'] = df['expected_time']
            
    except FileNotFoundError:
        print("❌ Error: No training data found!")
        print("Please run either:")
        print("  1. 'enhanced_process_datasets.py' for real data, OR")
        print("  2. 'enhanced_generate_data.py' for synthetic data")
        exit()

# Standardize column names
if 'correct' in df.columns and 'is_correct' not in df.columns:
    df['is_correct'] = df['correct']
elif 'is_correct' not in df.columns:
    print("❌ Error: No correctness column found in data!")
    exit()

# Calculate Speed Factor
df['speed_factor'] = df['response_time'] / df['expected_time_sec']

# Remove extreme outliers (speed factor > 10 or < 0.1)
original_len = len(df)
df = df[(df['speed_factor'] > 0.1) & (df['speed_factor'] < 10)]
if len(df) < original_len:
    print(f"⚠️ Filtered {original_len - len(df):,} outlier interactions")

print(f"📊 Training on {len(df):,} interactions from {df['student_id'].nunique():,} students")

# ==========================================
# 2. BUILD ENHANCED STUDENT PROFILES
# ==========================================
print("\nBuilding student behavioral profiles...")

student_profiles = df.groupby('student_id').agg({
    'is_correct': ['mean', 'std', 'count'],
    'speed_factor': ['mean', 'std'],
    'response_time': ['mean', 'std']
}).reset_index()

# Flatten column names
student_profiles.columns = [
    'student_id', 
    'accuracy', 'accuracy_std', 'n_attempts',
    'avg_speed', 'speed_consistency',
    'avg_time', 'time_std'
]

# Handle NaN values
student_profiles = student_profiles.fillna(0)

# Filter students with at least 10 interactions for reliable profiling
min_interactions = 10
student_profiles = student_profiles[student_profiles['n_attempts'] >= min_interactions]

print(f"✅ Created profiles for {len(student_profiles):,} students (min {min_interactions} interactions each)")
print(f"\nProfile Statistics:")
print(student_profiles[['accuracy', 'avg_speed', 'speed_consistency']].describe().round(3))

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("\nEngineering behavioral features...")

# Core features for clustering
features = student_profiles[[
    'accuracy', 
    'avg_speed', 
    'speed_consistency',
    'accuracy_std'
]].copy()

# Add derived features for better separation
student_profiles['speed_score'] = (1 / (student_profiles['avg_speed'] + 0.1)) * student_profiles['accuracy']
student_profiles['efficiency'] = student_profiles['accuracy'] / (student_profiles['avg_speed'] + 0.1)

# Extended feature set (optional - use if clusters aren't separating well)
# features_extended = student_profiles[[
#     'accuracy', 
#     'avg_speed', 
#     'speed_consistency',
#     'speed_score',
#     'confidence'
# ]].copy()

print(f"Feature matrix shape: {features.shape}")

# ==========================================
# 4. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ==========================================
print("\nDetermining optimal cluster count...")

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Test different cluster counts
inertias = []
silhouette_scores = []
gap_stats = []

# Test 3-5 clusters (force more granularity)
K_range = range(3, 6)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    kmeans_temp.fit(scaled_features)
    
    inertias.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(scaled_features, kmeans_temp.labels_)
    silhouette_scores.append(sil_score)
    
    # Calculate gap between inertias (elbow method)
    if len(inertias) > 1:
        gap = inertias[-2] - inertias[-1]
        gap_stats.append(gap)

# IMPROVED: Select based on BOTH silhouette AND practical cohort needs
# Prefer 3-4 clusters for educational cohorts
if silhouette_scores[0] > 0.30:  # If K=3 is decent
    optimal_k = 3
elif len(silhouette_scores) > 1 and silhouette_scores[1] > 0.28:  # If K=4 is decent
    optimal_k = 4
else:
    optimal_k = K_range[np.argmax(silhouette_scores)]

print(f"✅ Optimal cluster count: {optimal_k}")
print(f"   Silhouette Score: {silhouette_scores[optimal_k - 3]:.3f}")

print("\nCluster Quality Scores:")
for k, score in zip(K_range, silhouette_scores):
    marker = " ← SELECTED" if k == optimal_k else ""
    print(f"  K={k}: Silhouette={score:.3f}{marker}")

# ==========================================
# 5. TRAIN FINAL CLUSTERING MODEL
# ==========================================
print(f"\nTraining final K-Means model with {optimal_k} clusters...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=30, max_iter=500)
student_profiles['cluster_id'] = kmeans.fit_predict(scaled_features)

print(f"✅ Model trained successfully")

# ==========================================
# 6. ANALYZE AND LABEL CLUSTERS
# ==========================================
print("\n" + "="*60)
print("CLUSTER ANALYSIS")
print("="*60)

cluster_stats = student_profiles.groupby('cluster_id').agg({
    'accuracy': ['mean', 'std'],
    'avg_speed': ['mean', 'std'],
    'speed_consistency': ['mean', 'std'],
    'n_attempts': ['mean', 'count']
}).round(3)

cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
print(cluster_stats)

# Sort clusters by speed for logical labeling
cluster_summary = student_profiles.groupby('cluster_id').agg({
    'accuracy': 'mean',
    'avg_speed': 'mean',
    'speed_consistency': 'mean',
    'accuracy_std': 'mean',
    'n_attempts': 'count'
}).reset_index()

cluster_summary = cluster_summary.sort_values('avg_speed')

# ==========================================
# 7. DYNAMIC CLUSTER LABELING
# ==========================================
print("\n" + "="*60)
print("INTELLIGENT CLUSTER LABELING")
print("="*60)

cluster_map = {}

for idx, row in cluster_summary.iterrows():
    cluster_id = row['cluster_id']
    acc = row['accuracy']
    speed = row['avg_speed']
    consistency = row['speed_consistency']
    acc_std = row['accuracy_std']
    count = row['n_attempts']
    
    # Detailed labeling logic based on behavioral patterns
    if speed < 0.7:  # Very fast
        if acc < 0.50:
            label = "Fast Guesser (Low Accuracy)"
        elif acc < 0.65:
            label = "Fast Solver (Medium Accuracy)"
        else:
            label = "Fast Solver (High Accuracy)"
            
    elif speed > 1.3:  # Slow
        if consistency > 0.8:
            label = "Slow Thinker (Highly Variable)"
        elif acc < 0.55:
            label = "Struggling Learner (Slow & Low Accuracy)"
        else:
            label = "Methodical Learner (Slow & Careful)"
            
    else:  # Normal speed
        if acc > 0.75:
            label = "High Achiever (Strong Performance)"
        elif acc < 0.50:
            label = "Struggling (Average Speed, Low Accuracy)"
        else:
            if consistency > 0.6:
                label = "Normal (Variable Performance)"
            else:
                label = "Normal (Consistent Performance)"
    
    cluster_map[cluster_id] = label
    
    print(f"\nCluster {cluster_id}: {label}")
    print(f"  Students: {count}")
    print(f"  Avg Accuracy: {acc:.2%}")
    print(f"  Avg Speed: {speed:.2f}x")
    print(f"  Consistency (σ): {consistency:.3f}")
    print(f"  Accuracy Std: {acc_std:.3f}")

student_profiles['cluster_name'] = student_profiles['cluster_id'].map(cluster_map)

# ==========================================
# 8. VALIDATION (If ground truth available)
# ==========================================
if 'behavior_type_hidden' in df.columns:
    print("\n" + "="*60)
    print("VALIDATION AGAINST GROUND TRUTH")
    print("="*60)
    
    # Get ground truth for students
    ground_truth = df.groupby('student_id')['behavior_type_hidden'].first()
    student_profiles = student_profiles.merge(
        ground_truth.rename('true_label'),
        left_on='student_id',
        right_index=True,
        how='left'
    )
    
    # Confusion matrix
    if not student_profiles['true_label'].isna().all():
        conf_matrix = pd.crosstab(
            student_profiles['true_label'], 
            student_profiles['cluster_name'],
            margins=True
        )
        print(conf_matrix)
        
        # Calculate accuracy by cohort
        for behavior in student_profiles['true_label'].unique():
            if pd.notna(behavior):
                subset = student_profiles[student_profiles['true_label'] == behavior]
                # Find most common predicted label for this behavior
                most_common = subset['cluster_name'].mode()
                if len(most_common) > 0:
                    correct = len(subset[subset['cluster_name'] == most_common[0]])
                    print(f"\n{behavior}: {correct}/{len(subset)} ({correct/len(subset)*100:.1f}%) correctly identified")

# ==========================================
# 9. VISUALIZATIONS
# ==========================================
print("\nGenerating training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Speed vs Accuracy with cluster coloring
ax1 = axes[0, 0]
for cluster_id, label in cluster_map.items():
    cluster_data = student_profiles[student_profiles['cluster_id'] == cluster_id]
    ax1.scatter(
        cluster_data['avg_speed'],
        cluster_data['accuracy'],
        s=80,
        alpha=0.6,
        label=label.replace('Detected: ', ''),
        edgecolors='white',
        linewidth=0.5
    )

ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1.5, label='50% Accuracy')
ax1.axvline(x=1.0, color='blue', linestyle='--', alpha=0.3, linewidth=1.5, label='Expected Speed')
ax1.set_xlabel('Average Speed Factor (Response Time / Expected Time)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (Proportion Correct)', fontsize=11, fontweight='bold')
ax1.set_title('Student Cohorts: Speed vs Accuracy', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Plot 2: Consistency analysis
ax2 = axes[0, 1]
for cluster_id, label in cluster_map.items():
    cluster_data = student_profiles[student_profiles['cluster_id'] == cluster_id]
    ax2.scatter(
        cluster_data['speed_consistency'],
        cluster_data['accuracy'],
        s=80,
        alpha=0.6,
        label=label.replace('Detected: ', ''),
        edgecolors='white',
        linewidth=0.5
    )

ax2.set_xlabel('Speed Consistency (Std Dev)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Behavioral Consistency Analysis', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Cluster size distribution
ax3 = axes[1, 0]
cluster_counts = student_profiles['cluster_name'].value_counts()
colors = plt.cm.Set3(range(len(cluster_counts)))
bars = ax3.barh(range(len(cluster_counts)), cluster_counts.values, color=colors, edgecolor='black')
ax3.set_yticks(range(len(cluster_counts)))
ax3.set_yticklabels([label.replace('Detected: ', '') for label in cluster_counts.index], fontsize=9)
ax3.set_xlabel('Number of Students', fontsize=11, fontweight='bold')
ax3.set_title('Cluster Distribution', fontsize=13, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Add count labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, 
            f' {int(width)}', ha='left', va='center', fontsize=9, fontweight='bold')

# Plot 4: Optimization curves
ax4 = axes[1, 1]
ax4_twin = ax4.twinx()

line1 = ax4.plot(K_range, inertias, 'b-o', label='Inertia', linewidth=2, markersize=6)
line2 = ax4_twin.plot(K_range, silhouette_scores, 'r-s', label='Silhouette Score', linewidth=2, markersize=6)

ax4.axvline(x=optimal_k, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal K={optimal_k}')

ax4.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color='b', fontsize=11, fontweight='bold')
ax4_twin.set_ylabel('Silhouette Score (Cluster Quality)', color='r', fontsize=11, fontweight='bold')
ax4.set_title('Cluster Optimization Analysis', fontsize=13, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='b')
ax4_twin.tick_params(axis='y', labelcolor='r')
ax4.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.suptitle(f'NAVIKA Cohort Analysis - {data_source}', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('Figure_train.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization: Figure_train.png")
plt.show()

# ==========================================
# 10. SAVE THE TRAINED MODEL
# ==========================================
print("\n" + "="*60)
print("Saving trained model and artifacts...")

# Save the trained model
with open('navika_brain.pkl', 'wb') as f:
    pickle.dump((kmeans, scaler, cluster_map), f)
print("✅ Model saved: navika_brain.pkl")

# Save student profiles for reference
student_profiles.to_csv('navika_student_profiles.csv', index=False)
print("✅ Profiles saved: navika_student_profiles.csv")

# Save cluster statistics for documentation
cluster_summary.to_csv('navika_cluster_statistics.csv', index=False)
print("✅ Statistics saved: navika_cluster_statistics.csv")

# ==========================================
# 11. TRAINING SUMMARY
# ==========================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"Data Source: {data_source}")
print(f"Students Trained: {len(student_profiles):,}")
print(f"Total Interactions: {df['n_attempts'].sum() if 'n_attempts' in df.columns else len(df):,}")
print(f"Clusters Identified: {optimal_k}")
print(f"Cluster Quality (Silhouette): {max(silhouette_scores):.3f}")
print("\nIdentified Cohorts:")
for cluster_id, label in sorted(cluster_map.items()):
    count = len(student_profiles[student_profiles['cluster_id'] == cluster_id])
    pct = count / len(student_profiles) * 100
    print(f"  • {label.replace('Detected: ', '')}: {count} students ({pct:.1f}%)")
print("\n" + "="*60)
print("Next step: Run 'enhanced_test_suite.py' to evaluate the model")
print("="*60)