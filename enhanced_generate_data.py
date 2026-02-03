import numpy as np
import pandas as pd
import os

# ==========================================
# LOAD REAL-WORLD STATISTICS (if available)
# ==========================================
print("="*60)
print("NAVIKA Synthetic Data Generator (Enhanced)")
print("="*60)

# Try to load real statistics
try:
    with open('navika_real_stats.txt', 'r') as f:
        stats_lines = f.readlines()
        REAL_MEAN_TIME = float(stats_lines[0].split('=')[1].strip())
        REAL_STD_TIME = float(stats_lines[1].split('=')[1].strip())
        REAL_ACCURACY = float(stats_lines[2].split('=')[1].strip())
    print("✅ Using REAL WORLD statistics from processed datasets")
    print(f"   Mean Time: {REAL_MEAN_TIME:.2f}s, Std: {REAL_STD_TIME:.2f}s, Accuracy: {REAL_ACCURACY:.2%}")
except:
    print("⚠️ Real statistics not found. Using default educational testing standards.")
    REAL_MEAN_TIME = 39.77
    REAL_STD_TIME = 56.98
    REAL_ACCURACY = 0.68

# ==========================================
# CONFIGURATION
# ==========================================
NUM_STUDENTS = 2000  # Increased for better distribution
NUM_QUESTIONS = 150  # More questions for comprehensive testing
np.random.seed(42)

print(f"\nGenerating dataset:")
print(f"  Students: {NUM_STUDENTS}")
print(f"  Questions: {NUM_QUESTIONS}")
print("="*60)

# ==========================================
# 1. GENERATE REALISTIC QUESTIONS
# ==========================================
print("\nGenerating question bank...")

# Difficulty follows normal distribution
difficulties = np.random.normal(0, 1, NUM_QUESTIONS)

# Expected times with realistic distribution
# Using lognormal to better match real-world response times (right-skewed)
expected_times = np.random.lognormal(
    mean=np.log(REAL_MEAN_TIME),
    sigma=0.8,
    size=NUM_QUESTIONS
)
expected_times = np.clip(expected_times, 10, 300)  # Reasonable bounds

questions_data = {
    'question_id': range(NUM_QUESTIONS),
    'difficulty': difficulties,
    'discrimination': np.random.uniform(0.8, 2.2, NUM_QUESTIONS),  # Realistic range
    'guessing': np.random.uniform(0.05, 0.25, NUM_QUESTIONS),
    'expected_time_sec': expected_times.round(2),
}

questions = pd.DataFrame(questions_data)

print(f"✅ Created {NUM_QUESTIONS} questions")
print(f"   Difficulty range: [{questions['difficulty'].min():.2f}, {questions['difficulty'].max():.2f}]")

# ==========================================
# 2. GENERATE REALISTIC STUDENT COHORTS
# ==========================================
print("\nGenerating student population...")

student_ids = range(NUM_STUDENTS)
abilities = []
behavior_types = []
consistency_levels = []

for i in student_ids:
    # Ability: Normal distribution with realistic spread
    theta = np.random.normal(0, 1.2)
    abilities.append(theta)
    
    # Behavior distribution (more realistic proportions)
    rand_val = np.random.rand()
    
    if rand_val < 0.12:  # 12% Fast Guessers
        behavior_types.append('Fast_Guesser')
        consistency_levels.append(np.random.uniform(0.4, 0.8))  # Inconsistent
    elif rand_val < 0.25:  # 13% Slow Thinkers
        behavior_types.append('Slow_Thinker')
        consistency_levels.append(np.random.uniform(0.3, 0.7))  # Variable consistency
    else:  # 75% Normal
        behavior_types.append('Normal')
        consistency_levels.append(np.random.uniform(0.1, 0.4))  # More consistent

students = pd.DataFrame({
    'student_id': student_ids,
    'true_ability': abilities,
    'behavior_type': behavior_types,
    'consistency_level': consistency_levels
})

print(f"✅ Created {NUM_STUDENTS} students")
print(f"   Behavior distribution:")
for btype in students['behavior_type'].unique():
    count = len(students[students['behavior_type'] == btype])
    print(f"     {btype}: {count} ({count/NUM_STUDENTS*100:.1f}%)")

# ==========================================
# 3. SIMULATE REALISTIC EXAM INTERACTIONS
# ==========================================
print("\nSimulating realistic exam interactions...")

interactions = []

for student_idx, student in students.iterrows():
    if student_idx % 200 == 0:
        print(f"  Processing student {student_idx}/{NUM_STUDENTS}...")
    
    # Extract student characteristics
    theta = student['true_ability']
    behavior = student['behavior_type']
    consistency = student['consistency_level']
    
    # Vectorized IRT calculation
    a = questions['discrimination'].values
    b = questions['difficulty'].values
    c = questions['guessing'].values
    
    z = a * (theta - b)
    prob_correct = c + (1 - c) / (1 + np.exp(-z))
    
    # Generate correctness with realistic noise
    is_correct_arr = np.random.binomial(1, prob_correct)
    
    # Calculate base response times
    difficulty_gap = b - theta
    
    # More realistic time modeling
    # Harder questions (relative to ability) take longer
    time_difficulty_factor = 1 + 0.3 * np.tanh(difficulty_gap)  # Smooth scaling
    
    base_times = questions['expected_time_sec'].values * time_difficulty_factor
    
    # Add human variability
    human_noise = np.random.lognormal(0, 0.25, NUM_QUESTIONS)
    base_times *= human_noise
    
    # Apply behavioral modifiers with realistic variability
    if behavior == 'Fast_Guesser':
        # Fast but variable speed (20% to 50% of expected)
        speed_modifier = np.random.uniform(0.2, 0.5, NUM_QUESTIONS)
        # Add consistency variation
        speed_modifier *= np.random.normal(1, consistency, NUM_QUESTIONS)
        simulated_times = base_times * speed_modifier
        
        # Penalty for rushing hard questions
        hard_mask = b > theta + 1.0
        rush_penalty = np.random.rand(NUM_QUESTIONS) < 0.75
        is_correct_arr[hard_mask & rush_penalty] = 0
        
    elif behavior == 'Slow_Thinker':
        # Slow and variable (150% to 250% of expected)
        speed_modifier = np.random.uniform(1.5, 2.5, NUM_QUESTIONS)
        # Add anxiety-induced variability
        speed_modifier *= np.random.normal(1, consistency, NUM_QUESTIONS)
        simulated_times = base_times * speed_modifier
        
        # Occasional "freeze" on hard questions
        very_hard_mask = b > theta + 1.5
        freeze_mask = np.random.rand(NUM_QUESTIONS) < 0.3
        simulated_times[very_hard_mask & freeze_mask] *= np.random.uniform(1.5, 2.0, NUM_QUESTIONS)[very_hard_mask & freeze_mask]
        
    else:  # Normal
        # Normal variation (75% to 125% with small consistency variation)
        speed_modifier = np.random.uniform(0.75, 1.25, NUM_QUESTIONS)
        speed_modifier *= np.random.normal(1, consistency, NUM_QUESTIONS)
        simulated_times = base_times * speed_modifier
    
    # Ensure realistic bounds
    simulated_times = np.clip(simulated_times, 5, 600)
    
    # Create interaction records
    for q_idx in range(NUM_QUESTIONS):
        interactions.append({
            'student_id': student['student_id'],
            'question_id': questions.iloc[q_idx]['question_id'],
            'is_correct': is_correct_arr[q_idx],
            'response_time': round(simulated_times[q_idx], 2),
            'expected_time': questions.iloc[q_idx]['expected_time_sec'],
            'difficulty': round(b[q_idx], 3),
            'true_ability_hidden': round(theta, 3),
            'behavior_type_hidden': behavior
        })

# ==========================================
# 4. CREATE DATAFRAME AND VALIDATE
# ==========================================
df_interactions = pd.DataFrame(interactions)

# Validate generated data matches real-world statistics
print("\n" + "="*60)
print("VALIDATION: Generated vs Target Statistics")
print("="*60)
print(f"Response Time:")
print(f"  Target Mean: {REAL_MEAN_TIME:.2f}s | Generated: {df_interactions['response_time'].mean():.2f}s")
print(f"  Target Std:  {REAL_STD_TIME:.2f}s | Generated: {df_interactions['response_time'].std():.2f}s")
print(f"\nAccuracy:")
print(f"  Target: {REAL_ACCURACY:.2%} | Generated: {df_interactions['is_correct'].mean():.2%}")

# Check behavior patterns
print(f"\nBehavior Cohort Statistics:")
for behavior in df_interactions['behavior_type_hidden'].unique():
    subset = df_interactions[df_interactions['behavior_type_hidden'] == behavior]
    avg_time = subset['response_time'].mean()
    avg_acc = subset['is_correct'].mean()
    speed_factor = (subset['response_time'] / subset['expected_time']).mean()
    print(f"  {behavior}:")
    print(f"    Accuracy: {avg_acc:.2%}, Speed Factor: {speed_factor:.2f}x")

# ==========================================
# 5. EXPORT DATASETS
# ==========================================
print("\n" + "="*60)
print("Exporting datasets...")

df_interactions.to_csv('navika_synthetic_data.csv', index=False)
questions.to_csv('navika_questions_meta.csv', index=False)

print("✅ navika_synthetic_data.csv")
print("✅ navika_questions_meta.csv")

# ==========================================
# 6. CREATE TRAIN/TEST SPLIT
# ==========================================
from sklearn.model_selection import train_test_split

unique_students = df_interactions['student_id'].unique()
train_students, test_students = train_test_split(unique_students, test_size=0.2, random_state=42)

train_df = df_interactions[df_interactions['student_id'].isin(train_students)]
test_df = df_interactions[df_interactions['student_id'].isin(test_students)]

train_df.to_csv('navika_synthetic_train.csv', index=False)
test_df.to_csv('navika_synthetic_test.csv', index=False)

print("✅ navika_synthetic_train.csv")
print("✅ navika_synthetic_test.csv")

print("\n" + "="*60)
print("✅ DATASET GENERATION COMPLETE!")
print("="*60)
print(f"Total Interactions: {len(df_interactions):,}")
print(f"  Training: {len(train_df):,} ({len(train_students)} students)")
print(f"  Testing: {len(test_df):,} ({len(test_students)} students)")
print("\nDataset Statistics:")
print(df_interactions[['is_correct', 'response_time', 'difficulty']].describe())
print("="*60)