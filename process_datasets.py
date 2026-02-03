import pandas as pd
import numpy as np

# ==========================================
# 1. DEFINE THE STANDARD SCHEMA
# ==========================================
# All datasets must end up looking like this:
# [user_id, item_id, correctness, response_time_sec, source]

common_columns = ['user_id', 'item_id', 'correct', 'response_time', 'source']

# ==========================================
# 2. PROCESS ASSISTMENTS (Skill Builder)
# ==========================================
print("Processing ASSISTments...")
try:
    # Adjust filename if needed
    df_assist = pd.read_csv('skill_builder_data.csv', encoding='latin-1', low_memory=False)
    
    # Filter: Valid times only (0 to 600 seconds)
    df_assist = df_assist[(df_assist['ms_first_response'] > 0) & 
                          (df_assist['ms_first_response'] < 600000)]
    
    # Map columns
    clean_assist = pd.DataFrame()
    clean_assist['user_id'] = "AST_" + df_assist['user_id'].astype(str) # Add prefix to avoid ID collisions
    clean_assist['item_id'] = "AST_" + df_assist['problem_id'].astype(str)
    clean_assist['correct'] = df_assist['correct']
    clean_assist['response_time'] = df_assist['ms_first_response'] / 1000.0 # Convert ms to seconds
    clean_assist['source'] = 'ASSISTments'
    
except Exception as e:
    print(f"Skipping ASSISTments (Error: {e})")
    clean_assist = pd.DataFrame()

# ==========================================
# 3. PROCESS JUNYI ACADEMY (Log_Problem)
# ==========================================
print("Processing Junyi Academy...")
try:
    # This file is huge, so we might read just the first 1M rows for testing
    df_junyi = pd.read_csv('Log_Problem.csv', nrows=1000000)
    
    # Junyi doesn't always have exact "response time" in all versions. 
    # If 'time_taken' exists, use it. Otherwise, we might skip time or infer it.
    # Checking common column names for Junyi: 'time_taken' or 'duration'
    if 'time_taken' in df_junyi.columns:
        df_junyi = df_junyi[(df_junyi['time_taken'] > 0) & (df_junyi['time_taken'] < 600)]
        
        clean_junyi = pd.DataFrame()
        clean_junyi['user_id'] = "JUN_" + df_junyi['user_id'].astype(str)
        clean_junyi['item_id'] = "JUN_" + df_junyi['problem_id'].astype(str)
        clean_junyi['correct'] = df_junyi['is_correct'] # Usually 1 or 0
        clean_junyi['response_time'] = df_junyi['time_taken']
        clean_junyi['source'] = 'Junyi'
    else:
        print("Warning: 'time_taken' column not found in Junyi file.")
        clean_junyi = pd.DataFrame()

except Exception as e:
    print(f"Skipping Junyi (Error: {e})")
    clean_junyi = pd.DataFrame()

# ==========================================
# 4. PROCESS KDD CUP (Algebra 2008-2009)
# ==========================================
print("Processing KDD Cup Algebra...")
try:
    # Separator is often tab-delimited for KDD files
    df_kdd = pd.read_csv('algebra_2008_2009_train.txt', sep='\t', low_memory=False) 
    
    # Column mapping for KDD
    # 'Correct First Attempt': 1 = correct, 0 = incorrect
    # 'Step Duration (sec)': Response time
    
    df_kdd = df_kdd.dropna(subset=['Step Duration (sec)', 'Correct First Attempt'])
    
    clean_kdd = pd.DataFrame()
    clean_kdd['user_id'] = "KDD_" + df_kdd['Anon Student Id'].astype(str)
    clean_kdd['item_id'] = "KDD_" + df_kdd['Problem Name'] + "_" + df_kdd['Step Name']
    clean_kdd['correct'] = df_kdd['Correct First Attempt']
    clean_kdd['response_time'] = pd.to_numeric(df_kdd['Step Duration (sec)'], errors='coerce')
    clean_kdd['source'] = 'KDD_Algebra'
    
    # Filter bad data
    clean_kdd = clean_kdd.dropna()
    clean_kdd = clean_kdd[clean_kdd['response_time'] > 0]

except Exception as e:
    print(f"Skipping KDD (Error: {e})")
    clean_kdd = pd.DataFrame()

# ==========================================
# 5. MERGE AND EXPORT
# ==========================================
print("Merging datasets...")
final_df = pd.concat([clean_assist, clean_junyi, clean_kdd], ignore_index=True)

print(f"Total Combined Records: {len(final_df)}")
print(final_df.groupby('source').size())

# Save extraction of stats for the Generator
final_df.to_csv('navika_real_world_combined.csv', index=False)
print("âœ… Saved to navika_real_world_combined.csv")

# ==========================================
# 6. EXTRACT "PRIORS" FOR YOUR GENERATOR
# ==========================================
# We calculate the Mean and Std Dev to update your generate_data.py
real_mean_time = final_df['response_time'].mean()
real_std_time = final_df['response_time'].std()
real_accuracy = final_df['correct'].mean()

print("\n--- COPIABLE PARAMETERS FOR GENERATOR ---")
print(f"REAL_MEAN_TIME = {real_mean_time:.2f}")
print(f"REAL_STD_TIME = {real_std_time:.2f}")
print(f"REAL_AVG_ACCURACY = {real_accuracy:.2f}")