import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
print("Loading dataset...")
try:
    df = pd.read_csv('navika_synthetic_data.csv')
except FileNotFoundError:
    print("Error: 'navika_synthetic_data.csv' not found. Please run generate_data.py first.")
    exit()

# Calculate "Speed Factor" for every interaction
# Speed Factor = Actual Time / Expected Time
# 0.5 means they took half the expected time (FAST)
# 2.0 means they took double the expected time (SLOW)
df['speed_factor'] = df['response_time'] / df['expected_time']

# ==========================================
# 2. FEATURE ENGINEERING (Build Student Profiles)
# ==========================================
print("Building Student Profiles...")

student_profiles = df.groupby('student_id').agg({
    'is_correct': 'mean',           # Average Accuracy
    'speed_factor': ['mean', 'std'], # Average Speed & Consistency (Std Dev)
    'behavior_type_hidden': 'first'  # Keep the label for validation later (AI won't see this)
}).reset_index()

# Flatten the multi-level columns
student_profiles.columns = ['student_id', 'accuracy', 'avg_speed', 'consistency', 'true_label']

# Handle NaN values (if a student answered only 1 question, std is NaN)
student_profiles = student_profiles.fillna(0)

print(f"Profiles built for {len(student_profiles)} students.")
print(student_profiles.head())

# ==========================================
# 3. TRAIN THE AI MODEL (K-Means Clustering)
# ==========================================
print("\nTraining Cohort Detection Model...")

# Select the features the AI is allowed to see
features = student_profiles[['accuracy', 'avg_speed', 'consistency']]

# Scale the data (Normalize) so "Speed" and "Accuracy" have equal weight
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means
# We ask for 3 clusters because we suspect 3 types of students
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
student_profiles['cluster_id'] = kmeans.fit_predict(scaled_features)

# ==========================================
# 4. ANALYZE THE CLUSTERS (The "Research" Part)
# ==========================================
print("\n--- CLUSTER ANALYSIS ---")
# Check the average stats for each cluster to identify what they represent
cluster_stats = student_profiles.groupby('cluster_id')[['accuracy', 'avg_speed', 'consistency']].mean().reset_index()
print(cluster_stats)

# Sort by Speed to logically identify the groups
# (Fastest = Guesser, Slowest = Thinker)
cluster_stats_sorted = cluster_stats.sort_values('avg_speed')

print("Cluster Centroids (Sorted by Speed):")
print(cluster_stats_sorted)

# Map Cluster IDs to Human Names based on logic
# (Note: The cluster IDs 0, 1, 2 change every run, so we define logic to name them)
# cluster_map = {}
# for cluster_id, row in cluster_stats.iterrows():
#     if row['avg_speed'] < 0.6: 
#         cluster_map[cluster_id] = "Detected: Fast Guesser"
#     elif row['avg_speed'] > 1.3:
#         cluster_map[cluster_id] = "Detected: Slow Thinker"
#     else:
#         cluster_map[cluster_id] = "Detected: Normal"

cluster_map = {}
cluster_map[cluster_stats_sorted.iloc[0]['cluster_id']] = "Detected: Fast Guesser"
cluster_map[cluster_stats_sorted.iloc[1]['cluster_id']] = "Detected: Normal"
cluster_map[cluster_stats_sorted.iloc[2]['cluster_id']] = "Detected: Slow Thinker"

print("\nDynamic Cluster Mapping:")
print(cluster_map)

student_profiles['cluster_name'] = student_profiles['cluster_id'].map(cluster_map)

# ==========================================
# 5. VALIDATION (Did it work?)
# ==========================================
print("\n--- CONFUSION MATRIX (True vs Detected) ---")
# Compare the AI's "Detected" name with the "True Hidden" label
conf_matrix = pd.crosstab(student_profiles['true_label'], student_profiles['cluster_name'])
print(conf_matrix)

# ==========================================
# 6. VISUALIZATION (For your Paper)
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=student_profiles, 
    x='avg_speed', 
    y='accuracy', 
    hue='cluster_name', 
    size='consistency',  # Use Consistency as bubble size!
    sizes=(20, 200),
    style='true_label', 
    palette='viridis',
    alpha=0.8,
    edgecolor='w'
)
plt.title('NAVIKA Cohort Analysis: Speed vs. Accuracy (Size = Consistency)', fontsize=14)
plt.xlabel('Average Speed Factor (Lower = Faster)', fontsize=12)
plt.ylabel('Average Accuracy', fontsize=12)
plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Expected Speed Baseline')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nâœ… Model Trained & Visualized!")

# Save the trained model and the scaler
with open('navika_brain.pkl', 'wb') as f:
    pickle.dump((kmeans, scaler, cluster_map), f)
    
print("âœ… Brain saved to 'navika_brain.pkl'")