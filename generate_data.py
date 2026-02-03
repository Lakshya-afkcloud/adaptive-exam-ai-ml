import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION & CALIBRATION
# ==========================================
# These are the "Real World" priors. 
# If you ran 'process_datasets.py', replace these numbers with your specific results.
# Otherwise, these defaults are statistically standard for educational tests.
REAL_MEAN_TIME = 39.77   # Average time to answer a question (seconds)
REAL_STD_TIME = 56.98    # Standard deviation of time
REAL_ACCURACY = 0.68    # Global average accuracy (65%)

NUM_STUDENTS = 1000
NUM_QUESTIONS = 100
np.random.seed(42)  # Ensures reproducible results for your paper

# ==========================================
# 1. GENERATE QUESTIONS (The Exam Paper)
# ==========================================
print("Generating Enhanced Question Bank...")

# Define Topics and their inherent difficulty offset
topics = ['Algebra', 'Geometry', 'Calculus', 'Statistics']
topic_offsets = {'Algebra': 0.0, 'Geometry': 0.2, 'Calculus': 0.5, 'Statistics': -0.2}

# Difficulty (b): Real tests follow a Normal Distribution centered at 0
difficulties = np.random.normal(0, 1, NUM_QUESTIONS)

# Expected Time: Uses the "Real World" statistics we defined above
# We use absolute value to prevent negative times, and clip it to be reasonable (min 10s)
expected_times = np.abs(np.random.normal(REAL_MEAN_TIME, REAL_STD_TIME, NUM_QUESTIONS))
expected_times = np.maximum(expected_times, 10) # Minimum 10 seconds

questions_data = {
    'question_id': range(NUM_QUESTIONS),
    'difficulty': [],               # b parameter
    'discrimination': np.random.uniform(0.5, 2.5, NUM_QUESTIONS), # a parameter (High disc = good question)
    'guessing': np.random.uniform(0, 0.25, NUM_QUESTIONS),        # c parameter (Chance of lucky guess)
    'expected_time_sec': expected_times.round(2),
    'topic': []
}

# Assign Topics and Adjust Difficulty
for i in range(NUM_QUESTIONS):
    # Cyclic assignment: Algebra -> Geometry -> Calculus -> Stats
    t = topics[i % 4]
    
    # Base difficulty + Topic offset (Calculus is harder than Stats)
    final_diff = difficulties[i] + topic_offsets[t]
    
    questions_data['topic'].append(t)
    questions_data['difficulty'].append(final_diff)

questions = pd.DataFrame(questions_data)

# ==========================================
# 2. GENERATE STUDENTS (The Dynamic Cohorts)
# ==========================================
print("Generating Student Cohorts...")

student_ids = range(NUM_STUDENTS)
abilities = []
behavior_types = []

for i in student_ids:
    # True Ability (Theta): Normal distribution (-3 to +3)
    theta = np.random.normal(0, 1)
    abilities.append(theta)
    
    # Assign Behavior Profiles (Crucial for NAVIKA's "Consistency" check)
    rand_val = np.random.rand()
    
    # 15% are Fast Guessers (The "Cheater/Rusher" Cohort)
    if rand_val < 0.15:
        behavior_types.append('Fast_Guesser')
    # 15% are Slow Thinkers (The "Anxious/Perfectionist" Cohort)
    elif rand_val < 0.30:
        behavior_types.append('Slow_Thinker')
    # 70% are Normal
    else:
        behavior_types.append('Normal')

students = pd.DataFrame({
    'student_id': student_ids,
    'true_ability': abilities,
    'behavior_type': behavior_types
})

# ==========================================
# 3. SIMULATE THE EXAM (Interaction Loop)
# ==========================================
print("Simulating Exam Interactions...")

interactions = []

for _, student in students.iterrows():
    # Vectorized calculation for speed (instead of double loop)
    # 1. Calculate 3PL Probability for all questions at once
    # P(theta) = c + (1-c) / (1 + e^(-a(theta - b)))
    
    a = questions['discrimination'].values
    b = questions['difficulty'].values
    c = questions['guessing'].values
    theta = student['true_ability']
    
    z = a * (theta - b)
    prob_correct = c + (1 - c) / (1 + np.exp(-z))
    
    # Determine Correctness (Bernoulli Trial)
    # Returns 1 or 0 based on probability
    is_correct_arr = np.random.binomial(1, prob_correct)
    
    # 2. Calculate Response Times
    # Base: The question's expected time
    # Factor: Harder questions take longer (Exponential penalty)
    difficulty_gap = b - theta
    variable_factor = np.random.uniform(0.25, 0.35, NUM_QUESTIONS)
    time_factor = np.exp(difficulty_gap * variable_factor)
    
    simulated_times = questions['expected_time_sec'].values * time_factor
    
    # Add "Human Noise" (Students aren't robots, +/- 20% randomness)
    human_noise = np.random.uniform(0.8, 1.2, NUM_QUESTIONS)
    simulated_times *= human_noise
    
    # 3. Apply Behavioral Modifiers (The "NAVIKA" Logic)
    if student['behavior_type'] == 'Fast_Guesser':
        # Speed: Varies between 25% and 45% of normal time (Randomized)
        speed_modifier = np.random.uniform(0.25, 0.45, NUM_QUESTIONS)
        simulated_times = simulated_times * speed_modifier
        
        # If they rushed a hard question, force their accuracy down (Cheating/Guessing check)
        # Identify indices where difficulty is high (> 1)
        hard_indices = np.where(b > 1.0)[0]
        # 80% chance to fail hard questions when rushing
        fail_mask = np.random.rand(len(hard_indices)) > 0.2 
        is_correct_arr[hard_indices[fail_mask]] = 0
                
    elif student['behavior_type'] == 'Slow_Thinker':
        # Speed: Varies between 140% and 220% (Anxious/Careful)
        speed_modifier = np.random.uniform(1.4, 2.2, NUM_QUESTIONS)
        simulated_times = simulated_times * speed_modifier
    
    # Add human noise (Jitter) +/- 10%
    # jitter = np.random.uniform(0.9, 1.1, NUM_QUESTIONS)
    # simulated_times = simulated_times * jitter
    
    # Create rows
    for q_idx in range(NUM_QUESTIONS):
        interactions.append({
            'student_id': student['student_id'],
            'question_id': questions.iloc[q_idx]['question_id'],
            'is_correct': is_correct_arr[q_idx],
            'response_time': round(simulated_times[q_idx], 2),
            'expected_time': questions.iloc[q_idx]['expected_time_sec'],
            'difficulty': round(b[q_idx], 3),
            'topic': questions.iloc[q_idx]['topic'],
            'true_ability_hidden': round(theta, 3),        # Ground Truth (Hidden from Model)
            'behavior_type_hidden': student['behavior_type'] # Ground Truth (Hidden from Model)
        })

# ==========================================
# 4. EXPORT DATA
# ==========================================
df_interactions = pd.DataFrame(interactions)

# Save files
df_interactions.to_csv('navika_synthetic_data.csv', index=False)
questions.to_csv('navika_questions_meta.csv', index=False)

print("\nâœ… Dataset Generation Complete!")
print("-" * 30)
print(f"Total Interactions: {len(df_interactions)}")
print(f"Students: {NUM_STUDENTS}")
print(f"Questions per Student: {NUM_QUESTIONS}")
print("-" * 30)
print("Preview of generated data:")
print(df_interactions.head())
print(df_interactions.tail())