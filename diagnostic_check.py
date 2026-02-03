"""
Diagnostic script to check if your data and model are set up correctly
Run this BEFORE running the test suite
"""
import pandas as pd
import numpy as np
import pickle

print("="*60)
print("NAVIKA DIAGNOSTIC CHECK")
print("="*60)

# 1. Check question database
print("\n1. CHECKING QUESTION DATABASE...")
try:
    questions = pd.read_csv('navika_questions_meta_real.csv')
    print(f"✅ Loaded {len(questions)} questions")
    
    # Check for required columns
    required_cols = ['difficulty', 'discrimination', 'guessing', 'expected_time_sec']
    missing = [col for col in required_cols if col not in questions.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
    else:
        print("✅ All required columns present")
    
    # Check difficulty distribution
    print(f"\nDifficulty Statistics:")
    print(f"  Range: [{questions['difficulty'].min():.2f}, {questions['difficulty'].max():.2f}]")
    print(f"  Mean: {questions['difficulty'].mean():.2f}")
    print(f"  Std: {questions['difficulty'].std():.2f}")
    
    # Check for variety
    easy = len(questions[questions['difficulty'] < -1.0])
    medium = len(questions[(questions['difficulty'] >= -1.0) & (questions['difficulty'] <= 1.0)])
    hard = len(questions[questions['difficulty'] > 1.0])
    
    print(f"\nDifficulty Distribution:")
    print(f"  Easy (< -1.0): {easy} ({easy/len(questions)*100:.1f}%)")
    print(f"  Medium (-1.0 to 1.0): {medium} ({medium/len(questions)*100:.1f}%)")
    print(f"  Hard (> 1.0): {hard} ({hard/len(questions)*100:.1f}%)")
    
    if easy == 0 or hard == 0:
        print("⚠️ WARNING: Insufficient variety in question difficulties!")
        print("   This will cause flat learning curves.")
    
except FileNotFoundError:
    print("❌ Question database not found!")
    print("   Run 'enhanced_process_datasets.py' or 'enhanced_generate_data.py' first")

# 2. Check test data
print("\n2. CHECKING TEST DATA...")
try:
    test_df = pd.read_csv('navika_real_test.csv')
    print(f"✅ Loaded {len(test_df)} test interactions")
    
    # Standardize column names
    if 'student_id' not in test_df.columns and 'user_id' in test_df.columns:
        test_df['student_id'] = test_df['user_id']
    if 'is_correct' not in test_df.columns and 'correct' in test_df.columns:
        test_df['is_correct'] = test_df['correct']
    
    print(f"  Students: {test_df['student_id'].nunique()}")
    print(f"  Avg interactions per student: {len(test_df) / test_df['student_id'].nunique():.1f}")
    
    # Check data quality
    print(f"\nData Quality:")
    print(f"  Accuracy: {test_df['is_correct'].mean():.2%}")
    print(f"  Avg response time: {test_df['response_time'].mean():.1f}s")
    
    # Check for diversity in student performance
    student_acc = test_df.groupby('student_id')['is_correct'].mean()
    print(f"\nStudent Performance Diversity:")
    print(f"  Min accuracy: {student_acc.min():.2%}")
    print(f"  Max accuracy: {student_acc.max():.2%}")
    print(f"  Std dev: {student_acc.std():.3f}")
    
    if student_acc.std() < 0.15:
        print("⚠️ WARNING: Students are too similar!")
        print("   This will cause cohort classification to fail.")
    
    # Check students with sufficient data
    student_counts = test_df['student_id'].value_counts()
    sufficient = len(student_counts[student_counts >= 30])
    print(f"\nStudents with 30+ interactions: {sufficient}")
    
    if sufficient < 6:
        print("⚠️ WARNING: Not enough students with sufficient data!")
        print("   Recommended: At least 10 students with 30+ interactions")
    
except FileNotFoundError:
    print("❌ Test data not found!")

# 3. Check trained model
print("\n3. CHECKING TRAINED MODEL...")
try:
    with open('navika_brain.pkl', 'rb') as f:
        kmeans, scaler, cluster_map = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"  Number of clusters: {len(cluster_map)}")
    print(f"\nCluster Labels:")
    for cluster_id, label in sorted(cluster_map.items()):
        print(f"    {cluster_id}: {label}")
    
    # Check cluster separation
    print(f"\nCluster Centers (scaled):")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"  Cluster {i}: {center}")
    
    # Check if clusters are too similar
    from scipy.spatial.distance import pdist
    distances = pdist(kmeans.cluster_centers_)
    min_dist = distances.min()
    
    print(f"\nCluster Separation:")
    print(f"  Min distance between clusters: {min_dist:.3f}")
    
    if min_dist < 0.5:
        print("⚠️ WARNING: Clusters are too close together!")
        print("   Model may have difficulty distinguishing between cohorts.")
        print("   Consider retraining with different parameters.")
    
except FileNotFoundError:
    print("❌ Model not found!")
    print("   Run 'enhanced_train_cohort_model.py' first")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. Test theta estimation
print("\n4. TESTING THETA ESTIMATION...")
try:
    from adaptive_engine import estimate_theta
    questions = pd.read_csv('navika_questions_meta_real.csv')
    
    # Test with perfect score
    perfect_responses = [(questions.iloc[i]['question_id'], 1) for i in range(10)]
    theta_perfect = estimate_theta(perfect_responses, questions)
    print(f"  Perfect responses (10/10): θ = {theta_perfect:+.3f}")
    
    # Test with all wrong
    zero_responses = [(questions.iloc[i]['question_id'], 0) for i in range(10)]
    theta_zero = estimate_theta(zero_responses, questions)
    print(f"  All wrong (0/10): θ = {theta_zero:+.3f}")
    
    # Test with mixed
    mixed_responses = [(questions.iloc[i]['question_id'], i % 2) for i in range(10)]
    theta_mixed = estimate_theta(mixed_responses, questions)
    print(f"  Mixed (5/10): θ = {theta_mixed:+.3f}")
    
    # Check if estimates are reasonable
    if abs(theta_perfect - theta_zero) < 1.0:
        print("⚠️ WARNING: Theta estimation not working properly!")
        print("   Perfect and zero scores should be far apart.")
    else:
        print("✅ Theta estimation working correctly")
        
except Exception as e:
    print(f"❌ Error testing theta: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nRECOMMENDATIONS:")
print("1. If question variety is low, regenerate with better difficulty spread")
print("2. If student diversity is low, use more diverse training data")
print("3. If clusters are too similar, retrain with adjusted parameters")
print("4. If theta estimation fails, check IRT parameters in questions")
print("="*60)