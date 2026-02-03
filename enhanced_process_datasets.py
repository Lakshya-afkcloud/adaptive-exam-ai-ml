import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. DEFINE THE STANDARD SCHEMA
# ==========================================
common_columns = ['user_id', 'item_id', 'correct', 'response_time', 'source']

# ==========================================
# 2. PROCESS ASSISTMENTS (Skill Builder)
# ==========================================
print("Processing ASSISTments...")
try:
    df_assist = pd.read_csv('skill_builder_data.csv', encoding='latin-1', low_memory=False)
    
    # Filter: Valid times only (0 to 600 seconds)
    df_assist = df_assist[(df_assist['ms_first_response'] > 0) & 
                          (df_assist['ms_first_response'] < 600000)]
    
    # Map columns
    clean_assist = pd.DataFrame()
    clean_assist['user_id'] = "AST_" + df_assist['user_id'].astype(str)
    clean_assist['item_id'] = "AST_" + df_assist['problem_id'].astype(str)
    clean_assist['correct'] = df_assist['correct']
    clean_assist['response_time'] = df_assist['ms_first_response'] / 1000.0
    clean_assist['source'] = 'ASSISTments'
    
    # Add skill/topic if available
    # if 'skill_name' in df_assist.columns:
    #     clean_assist['topic'] = df_assist['skill_name']
    
    print(f"✅ ASSISTments: {len(clean_assist)} records loaded")
    
except Exception as e:
    print(f"⚠️ Skipping ASSISTments (Error: {e})")
    clean_assist = pd.DataFrame()

# ==========================================
# 3. PROCESS JUNYI ACADEMY
# ==========================================
print("Processing Junyi Academy...")
try:
    # Sample first 500k for manageable processing
    df_junyi = pd.read_csv('Log_Problem.csv', nrows=1000000)
    
    if 'time_taken' in df_junyi.columns:
        df_junyi = df_junyi[(df_junyi['time_taken'] > 0) & (df_junyi['time_taken'] < 600)]
        
        clean_junyi = pd.DataFrame()
        clean_junyi['user_id'] = "JUN_" + df_junyi['user_id'].astype(str)
        clean_junyi['item_id'] = "JUN_" + df_junyi['exercise'].astype(str) if 'exercise' in df_junyi.columns else "JUN_" + df_junyi['problem_id'].astype(str)
        clean_junyi['correct'] = df_junyi['is_correct']
        clean_junyi['response_time'] = df_junyi['time_taken']
        clean_junyi['source'] = 'Junyi'
        
        # if 'topic_mode' in df_junyi.columns:
        #     clean_junyi['topic'] = df_junyi['topic_mode']
        
        print(f"✅ Junyi Academy: {len(clean_junyi)} records loaded")
    else:
        print("⚠️ Warning: 'time_taken' column not found in Junyi file.")
        clean_junyi = pd.DataFrame()

except Exception as e:
    print(f"⚠️ Skipping Junyi (Error: {e})")
    clean_junyi = pd.DataFrame()

# ==========================================
# 4. PROCESS KDD CUP (Algebra)
# ==========================================
print("Processing KDD Cup Algebra...")
try:
    df_kdd = pd.read_csv('algebra_2008_2009_train.txt', sep='\t', low_memory=False) 
    
    df_kdd = df_kdd.dropna(subset=['Step Duration (sec)', 'Correct First Attempt'])
    
    clean_kdd = pd.DataFrame()
    clean_kdd['user_id'] = "KDD_" + df_kdd['Anon Student Id'].astype(str)
    clean_kdd['item_id'] = "KDD_" + df_kdd['Problem Name'].astype(str) + "_" + df_kdd['Step Name'].astype(str)
    clean_kdd['correct'] = df_kdd['Correct First Attempt']
    clean_kdd['response_time'] = pd.to_numeric(df_kdd['Step Duration (sec)'], errors='coerce')
    clean_kdd['source'] = 'KDD_Algebra'
    
    # Extract topic from Problem Hierarchy if available
    # if 'Problem Hierarchy' in df_kdd.columns:
    #     clean_kdd['topic'] = df_kdd['Problem Hierarchy'].str.split(',').str[0]
    
    # Filter bad data
    clean_kdd = clean_kdd.dropna()
    clean_kdd = clean_kdd[(clean_kdd['response_time'] > 0) & (clean_kdd['response_time'] < 600)]
    
    print(f"✅ KDD Algebra: {len(clean_kdd)} records loaded")

except Exception as e:
    print(f"⚠️ Skipping KDD (Error: {e})")
    clean_kdd = pd.DataFrame()

# ==========================================
# 5. MERGE ALL DATASETS
# ==========================================
print("\n" + "="*60)
print("Merging all datasets...")
final_df = pd.concat([clean_assist, clean_junyi, clean_kdd], ignore_index=True)

# Remove any remaining invalid data
final_df = final_df.dropna(subset=['user_id', 'item_id', 'correct', 'response_time'])
final_df = final_df[(final_df['response_time'] > 0) & (final_df['response_time'] < 600)]

print(f"\n📊 Total Combined Records: {len(final_df):,}")
print("\nRecords per source:")
print(final_df.groupby('source').size())

# ==========================================
# 6. CALCULATE REALISTIC PRIORS
# ==========================================
real_mean_time = final_df['response_time'].mean()
real_std_time = final_df['response_time'].std()
real_accuracy = final_df['correct'].mean()

print("\n" + "="*60)
print("📈 REALISTIC DATASET STATISTICS")
print("="*60)
print(f"Mean Response Time: {real_mean_time:.2f} seconds")
print(f"Std Response Time:  {real_std_time:.2f} seconds")
print(f"Average Accuracy:   {real_accuracy:.2%}")
print("="*60)

# ==========================================
# 7. MAP TO NAVIKA TOPICS (Intelligent Mapping)
# ==========================================
# print("\nMapping to NAVIKA topic categories...")

# # Define NAVIKA topics
# navika_topics = ['Algebra', 'Geometry', 'Calculus', 'Statistics']

# # Create topic mapping based on keywords
# def map_to_navika_topic(original_topic):
#     if pd.isna(original_topic):
#         return np.random.choice(navika_topics)
    
#     topic_lower = str(original_topic).lower()
    
#     # Keyword-based mapping
#     if any(k in topic_lower for k in ['algebra', 'equation', 'polynomial', 'factor', 'linear']):
#         return 'Algebra'
#     elif any(k in topic_lower for k in ['geometry', 'triangle', 'circle', 'angle', 'shape']):
#         return 'Geometry'
#     elif any(k in topic_lower for k in ['calculus', 'derivative', 'integral', 'limit', 'function']):
#         return 'Calculus'
#     elif any(k in topic_lower for k in ['statistics', 'probability', 'data', 'mean', 'median']):
#         return 'Statistics'
#     else:
#         # Random assignment for unclear topics
#         return np.random.choice(navika_topics)

# if 'topic' in final_df.columns:
#     final_df['navika_topic'] = final_df['topic'].apply(map_to_navika_topic)
# else:
#     final_df['navika_topic'] = np.random.choice(navika_topics, size=len(final_df))

# print("Topic distribution:")
# print(final_df['navika_topic'].value_counts())

# ==========================================
# 8. ESTIMATE DIFFICULTY (IRT-based) - FIXED
# ==========================================
print("\nEstimating question difficulty using IRT principles...")

# Group by item and calculate difficulty proxy
item_stats = final_df.groupby('item_id').agg({
    'correct': ['mean', 'count'],
    'response_time': 'mean'
}).reset_index()

item_stats.columns = ['item_id', 'p_correct', 'n_responses', 'avg_time']

# Filter items with sufficient responses (at least 5) BEFORE calculating difficulty
item_stats = item_stats[item_stats['n_responses'] >= 5]
print(f"Items with 5+ responses: {len(item_stats):,}")

# CRITICAL FIX: Clip p_correct to avoid infinity in logit transformation
# Ensure values are strictly between 0.01 and 0.99
item_stats['p_correct_clipped'] = item_stats['p_correct'].clip(0.01, 0.99)

# Difficulty estimation: logit of proportion correct (inverse)
# Higher p_correct = easier = lower difficulty
# This will now never produce infinity or NaN
item_stats['difficulty'] = -np.log(
    item_stats['p_correct_clipped'] / (1 - item_stats['p_correct_clipped'])
)

# Check for any infinite or NaN values (should be none now)
n_invalid = np.sum(~np.isfinite(item_stats['difficulty']))
if n_invalid > 0:
    print(f"⚠️ Warning: {n_invalid} items with invalid difficulty, removing...")
    item_stats = item_stats[np.isfinite(item_stats['difficulty'])]

print(f"Difficulty range before normalization: [{item_stats['difficulty'].min():.2f}, {item_stats['difficulty'].max():.2f}]")

# Normalize to -3 to +3 range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-3, 3))
item_stats['difficulty'] = scaler.fit_transform(item_stats[['difficulty']])

print(f"Difficulty range after normalization: [{item_stats['difficulty'].min():.2f}, {item_stats['difficulty'].max():.2f}]")

# Discrimination: items with more variance are more discriminating
# Use a more sophisticated approach based on response variance
item_variance = final_df.groupby('item_id')['correct'].std().fillna(0)
item_stats = item_stats.merge(
    item_variance.rename('response_variance'),
    left_on='item_id',
    right_index=True,
    how='left'
)

# Scale variance to discrimination parameter range (0.5 to 2.5)
# Higher variance = better discrimination
item_stats['discrimination'] = 0.5 + (item_stats['response_variance'] * 2.0)
item_stats['discrimination'] = item_stats['discrimination'].clip(0.5, 2.5)

# Guessing parameter (for 3PL IRT)
# Lower for harder items (less chance of guessing correctly)
# Use p_correct as a guide: easier items might have higher guessing
item_stats['guessing'] = (item_stats['p_correct_clipped'] * 0.25).clip(0, 0.25)

# Clean up temporary columns
item_stats = item_stats.drop(columns=['p_correct_clipped', 'response_variance'])

print(f"\n✅ Question statistics calculated for {len(item_stats):,} items")
print("\nIRT Parameter Summary:")
print(item_stats[['difficulty', 'discrimination', 'guessing']].describe())

# ==========================================
# 9. MERGE DIFFICULTY BACK TO MAIN DATAFRAME
# ==========================================
final_df = final_df.merge(
    item_stats[['item_id', 'difficulty']], 
    on='item_id', 
    how='left'
)

# Remove rows with missing difficulty (items with <5 responses)
before_filter = len(final_df)
final_df = final_df.dropna(subset=['difficulty'])
after_filter = len(final_df)

if before_filter > after_filter:
    print(f"\n⚠️ Filtered {before_filter - after_filter:,} interactions from items with <5 responses")

# ==========================================
# 10. CREATE TRAIN/TEST SPLIT (Student-level)
# ==========================================
print("\n" + "="*60)
print("Creating Train/Test Split (80/20)...")

# Get unique students
unique_students = final_df['user_id'].unique()

# Split students into train/test (80/20)
train_students, test_students = train_test_split(
    unique_students, 
    test_size=0.2, 
    random_state=42
)

# Create train and test dataframes
train_df = final_df[final_df['user_id'].isin(train_students)].copy()
test_df = final_df[final_df['user_id'].isin(test_students)].copy()

train_df = train_df.rename(columns={'user_id': 'student_id', 'item_id': 'question_id', 'correct': 'is_correct'})
test_df = test_df.rename(columns={'user_id': 'student_id', 'item_id': 'question_id', 'correct': 'is_correct'})

# Add expected_time_sec column
train_df['expected_time_sec'] = train_df['question_id'].map(
    item_stats.set_index('item_id')['avg_time']
)
test_df['expected_time_sec'] = test_df['question_id'].map(
    item_stats.set_index('item_id')['avg_time']
)

print(f"Training set: {len(train_students):,} students, {len(train_df):,} interactions")
print(f"Test set: {len(test_students):,} students, {len(test_df):,} interactions")

# ==========================================
# 11. SAVE ALL FILES
# ==========================================
print("\n" + "="*60)
print("Saving files...")

# Save combined dataset
final_df_renamed = final_df.rename(columns={'user_id': 'student_id', 'item_id': 'question_id', 'correct': 'is_correct'})
final_df_renamed['expected_time_sec'] = final_df_renamed['question_id'].map(
    item_stats.set_index('item_id')['avg_time']
)
final_df_renamed.to_csv('navika_real_world_combined.csv', index=False)
print("✅ Saved: navika_real_world_combined.csv")

# Save train/test splits
train_df.to_csv('navika_real_train.csv', index=False)
test_df.to_csv('navika_real_test.csv', index=False)
print("✅ Saved: navika_real_train.csv")
print("✅ Saved: navika_real_test.csv")

# Save question metadata (for adaptive engine)
question_meta = item_stats[['item_id', 'difficulty', 'discrimination', 'guessing', 'avg_time']].copy()
question_meta = question_meta.rename(columns={
    'item_id': 'question_id',
    'avg_time': 'expected_time_sec'
})

# Add auto-incrementing question_id as integer for easier handling
question_meta = question_meta.reset_index(drop=True)
question_meta.insert(0, 'question_id_int', range(len(question_meta)))

question_meta.to_csv('navika_questions_meta_real.csv', index=False)
print("✅ Saved: navika_questions_meta_real.csv")

# Save statistics for generate_data.py
with open('navika_real_stats.txt', 'w') as f:
    f.write(f"REAL_MEAN_TIME = {real_mean_time:.2f}\n")
    f.write(f"REAL_STD_TIME = {real_std_time:.2f}\n")
    f.write(f"REAL_ACCURACY = {real_accuracy:.4f}\n")
print("✅ Saved: navika_real_stats.txt")

print("\n" + "="*60)
print("✅ PROCESSING COMPLETE!")
print("="*60)
print(f"\nDataset Summary:")
print(f"  Total students: {len(unique_students):,}")
print(f"  Total interactions: {len(final_df):,}")
print(f"  Total unique items: {len(item_stats):,}")
print(f"  Average interactions per student: {len(final_df) / len(unique_students):.1f}")
print(f"  Average responses per item: {final_df.groupby('question_id' if 'question_id' in final_df.columns else 'item_id').size().mean():.1f}")
print("\nDifficulty Distribution:")
print(f"  Easy (< -1.0): {(item_stats['difficulty'] < -1.0).sum()} items")
print(f"  Medium (-1.0 to 1.0): {((item_stats['difficulty'] >= -1.0) & (item_stats['difficulty'] <= 1.0)).sum()} items")
print(f"  Hard (> 1.0): {(item_stats['difficulty'] > 1.0).sum()} items")
print("\nNext steps:")
print("1. Run 'enhanced_train_cohort_model.py' with real training data")
print("2. Run 'enhanced_test_suite.py' with real test data")
print("="*60)