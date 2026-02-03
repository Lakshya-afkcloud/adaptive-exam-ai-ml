# Adaptive engine module
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import minimize
import os

# ==========================================
# 1. LOAD THE BRAIN & DATA
# ==========================================
print("Loading NAVIKA System...")

# Try to load real question metadata first, fallback to synthetic
try:
    questions_db = pd.read_csv('navika_questions_meta_real.csv')
    print("✅ Loaded real-world question bank")
except FileNotFoundError:
    try:
        questions_db = pd.read_csv('navika_questions_meta.csv')
        print("✅ Loaded synthetic question bank")
    except FileNotFoundError:
        print("❌ No question bank found! Please run data generation first.")
        exit()

# Load trained model
try:
    with open('navika_brain.pkl', 'rb') as f:
        kmeans_model, scaler_model, cluster_map = pickle.load(f)
    print("✅ AI Brain Loaded Successfully.")
except FileNotFoundError:
    print("⚠️ Warning: navika_brain.pkl not found.")
    print("   Run 'enhanced_train_cohort_model.py' first for full functionality.")
    print("   Continuing with limited capability...")
    kmeans_model, scaler_model, cluster_map = None, None, None
except ValueError:
    print("❌ Error: Old brain file detected. Please run 'enhanced_train_cohort_model.py' first.")
    exit()

# ==========================================
# 2. HELPER MATH FUNCTIONS (IRT)
# ==========================================
def get_prob_correct(theta, a, b, c):
    """3PL IRT Probability Formula"""
    z = a * (theta - b)
    z = np.clip(z, -10, 10)  # Prevent overflow
    return c + (1 - c) / (1 + np.exp(-z))

def estimate_theta(responses, questions):
    """
    Given a list of (question_id, is_correct), find the best Theta (Ability).
    We maximize the Likelihood of the answers given the difficulty.
    """
    if not responses:
        return 0.0  # Start at average ability

    def negative_log_likelihood(theta):
        log_like = 0
        for q_id, correct in responses:
            # Handle both integer and string question IDs
            if 'item_id' in questions.columns:
                q_match = questions[questions['item_id'] == q_id]
            else:
                q_match = questions[questions['question_id'] == q_id]
            
            if q_match.empty:
                continue
                
            q_row = q_match.iloc[0]
            a, b, c = q_row['discrimination'], q_row['difficulty'], q_row['guessing']
            p = get_prob_correct(theta[0], a, b, c)
            
            # Avoid log(0) error
            p = np.clip(p, 0.01, 0.99)
            
            if correct:
                log_like += np.log(p)
            else:
                log_like += np.log(1 - p)
        return -log_like if np.isfinite(log_like) else 1e10

    # Minimize the negative likelihood to find best Theta
    prop_correct = sum(r[1] for r in responses) / len(responses)
    # Map proportion to theta: 0.5 correct -> theta=0, higher/lower scales appropriately
    initial_theta = np.log(prop_correct / (1 - prop_correct + 0.01))
    initial_theta = np.clip(initial_theta, -2, 2)
    
    # Minimize with better bounds
    result = minimize(
        negative_log_likelihood, 
        x0=[initial_theta], 
        bounds=[(-3, 3)],
        method='L-BFGS-B'
    )
    
    return float(result.x[0]) if result.success else initial_theta

# ==========================================
# 3. THE NAVIKA ENGINE CLASS
# ==========================================
class NavikaEngine:
    def __init__(self, student_id=0):
        self.student_id = student_id
        self.history = []  # Stores (q_id, correct, time, expected_time)
        
        self.questions_asked = set()
        self.cohort = "Normal"
        
        # Global ability estimate
        self.theta = 0.0
        self.theta_history = [0.0]
        
        # Difficulty zones for adaptive selection
        self.current_difficulty_zone = 'medium'  # 'easy', 'medium', 'hard'
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        
        self.cohort_history = ['Normal']
        
    def update_cohort(self):
        """Runs the AI model to classify the student based on history"""
        if len(self.history) < 3:
            return  # Need data to decide
        
        # Check if model is loaded
        if kmeans_model is None or scaler_model is None:
            # Simple rule-based fallback
            df_hist = pd.DataFrame(self.history, columns=['q_id', 'correct', 'time', 'exp_time'])
            df_hist['speed_factor'] = df_hist['time'] / df_hist['exp_time']
            
            avg_acc = df_hist['correct'].mean()
            avg_speed = df_hist['speed_factor'].mean()
            
            if avg_speed < 0.7 and avg_acc < 0.5:
                self.cohort = "Fast Guesser"
            elif avg_speed < 0.7 and avg_acc >= 0.5:
                self.cohort = "Fast Solver"
            elif avg_speed > 1.6:
                self.cohort = "Slow Thinker"
            else:
                self.cohort = "Normal"
            
            self.cohort_history.append(self.cohort)
            return

        # 1. Extract Features from history
        df_hist = pd.DataFrame(self.history, columns=['q_id', 'correct', 'time', 'exp_time'])
        df_hist['speed_factor'] = df_hist['time'] / df_hist['exp_time']
        
        avg_acc = df_hist['correct'].mean()
        avg_speed = df_hist['speed_factor'].mean()
        consistency = df_hist['speed_factor'].std()
        
        if np.isnan(consistency):
            consistency = 0
        
        # Add derived feature to match scaler's expected input (4 features)
        # speed_score: higher accuracy and lower time = higher score
        speed_score = (1.0 / max(avg_speed, 0.1)) * avg_acc  # Avoid division by zero

        # 2. Predict using the Saved Brain (4 features: accuracy, avg_speed, consistency, speed_score)
        features = np.array([[avg_acc, avg_speed, consistency, speed_score]])
        scaled_features = scaler_model.transform(features)
        
        # Get Cluster ID
        cluster_id = kmeans_model.predict(scaled_features)[0]
        
        # Map ID to Human Name
        raw_label = cluster_map[cluster_id]
        
        # Clean the label string for internal logic
        if "Fast Guesser" in raw_label or "Fast Solver" in raw_label:
            # Distinguish between fast guesser and fast solver
            if avg_acc < 0.5:
                self.cohort = "Fast Guesser"
            else:
                self.cohort = "Normal"  # Fast but accurate is normal
        elif "Slow" in raw_label or "Anxious" in raw_label or "Methodical" in raw_label:
            self.cohort = "Slow Thinker"
        else:
            self.cohort = "Normal"
        
        self.cohort_history.append(self.cohort)
        
        # Update global ability estimate
        if len(self.history) >= 3:
            responses = [(x[0], x[1]) for x in self.history]
            new_theta = estimate_theta(responses, questions_db)
            
            # Only update if change is reasonable (not jumping wildly)
            if abs(new_theta - self.theta) < 2.0 or len(self.history) <= 5:
                self.theta = new_theta
                self.theta_history.append(self.theta)

    def adapt_difficulty_zone(self):
        """Dynamically adjust difficulty zone based on recent performance"""
        if len(self.history) < 2:
            return 'medium'
        
        # Get last 5 answers
        window = min(5, len(self.history))
        recent_answers = [x[1] for x in self.history[-window:]]
        recent_accuracy = sum(recent_answers) / len(recent_answers)
        
        # RULE 1: Fast Guessers - give harder questions to slow them down
        if self.cohort == "Fast Guesser":
            if recent_accuracy < 0.3:
                return 'easy'  # They're struggling, ease up
            else:
                return 'hard'  # Challenge them
        
        # RULE 2: Slow Thinkers - adaptive based on recent performance
        elif self.cohort == "Slow Thinker":
            if self.consecutive_incorrect >= 2:
                return 'easy'
            elif recent_accuracy > 0.7:
                return 'medium'
            else:
                return 'easy'
        
        # RULE 3: Normal students - standard adaptive logic with more sensitivity
        else:
            if self.consecutive_correct >= 2:
                return 'hard'
            elif self.consecutive_incorrect >= 2:
                return 'easy'
            elif recent_accuracy > 0.75:
                return 'hard'
            elif recent_accuracy < 0.35:
                return 'easy'
            else:
                return 'medium'

    def get_next_question(self):
        """Select the next optimal question for this student"""
        # 1. Determine target difficulty zone
        zone = self.adapt_difficulty_zone()
        
        # 2. Map zones to difficulty ranges
        if zone == 'easy':
            target_diff = self.theta - 1.0
            search_range = 0.8
        elif zone == 'hard':
            target_diff = self.theta + 1.0
            search_range = 0.8
        else:  # medium
            target_diff = self.theta
            search_range = 0.6
        
        # 3. Handle both 'question_id' and 'item_id' columns
        id_col = 'question_id' if 'question_id' in questions_db.columns else 'item_id'
        
        # 4. Find candidate questions (Unasked + Within difficulty range)
        candidates = questions_db[
            (~questions_db[id_col].isin(self.questions_asked)) & 
            (questions_db['difficulty'] >= target_diff - search_range) &
            (questions_db['difficulty'] <= target_diff + search_range)
        ].copy()
        
        # Fallback 1: If no candidates in range, expand search
        if candidates.empty:
            candidates = questions_db[
                ~questions_db[id_col].isin(self.questions_asked)
            ].copy()
        
        # Fallback 2: If all questions exhausted, allow repeats
        if candidates.empty:
            candidates = questions_db.copy()
            # Reset the questions_asked set to start fresh
            self.questions_asked.clear()
        
        # 5. Pick closest difficulty to target
        candidates['dist'] = abs(candidates['difficulty'] - target_diff)
        candidates = candidates.sort_values('dist')
        
        # Pick from top 5 closest instead of always the closest
        top_n = min(5, len(candidates))
        best_q = candidates.head(top_n).sample(1).iloc[0]
        
        self.current_difficulty_zone = zone
        return best_q
    
    def submit_answer(self, q_id, is_correct, time_taken, exp_time):
        """Record student's answer and update model"""
        self.history.append((q_id, is_correct, time_taken, exp_time))
        self.questions_asked.add(q_id)
        
        # Track consecutive patterns
        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0
        
        # Update cohort classification
        self.update_cohort()

# ==========================================
# 4. SIMULATION: RUN A LIVE TEST
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NAVIKA ENGINE - Live Test Simulation")
    print("="*60)
    print("Scenario: 'Calculus Struggle' - Student struggles with Calculus but excels in Algebra\n")
    
    engine = NavikaEngine(student_id=999)
    
    # --- HELPER: SIMULATE REALISTIC USER ---
    def simulate_real_response(difficulty, expected_time, question_num, student_skill_curve):
        """Simulate realistic student response based on progress and difficulty"""
        # Simulate learning curve: starts weak, improves over time
        # base_skill = -1.0 + (question_num / 30) * 2.0  # -1.0 to +1.0 over 30 questions
        
        # # Add some noise for realism
        # skill = base_skill + np.random.normal(0, 0.3)
        
        # Get current skill level
        skill = student_skill_curve(question_num)
        
        # Add realistic noise
        skill_noisy = skill + np.random.normal(0, 0.25)
        
        # 1. Determine Accuracy using IRT
        prob = 1 / (1 + np.exp(-(skill_noisy - difficulty)))
        is_correct = 1 if np.random.rand() < prob else 0
        
        # 2. Speed depends on difficulty relative to skill
        difficulty_gap = difficulty - skill
        
        if difficulty_gap > 1.5:  # Way too hard
            speed_factor = np.random.uniform(1.8, 3.0)
        elif difficulty_gap > 0.5:  # Somewhat hard
            speed_factor = np.random.uniform(1.2, 1.8)
        elif difficulty_gap < -1.0:  # Way too easy
            speed_factor = np.random.uniform(0.4, 0.7)
        else:  # Appropriate difficulty
            speed_factor = np.random.uniform(0.8, 1.3)
            
        return is_correct, expected_time * speed_factor

    # Define a learning curve: student improves over time
    def learning_curve(step):
        """Student starts weak, improves steadily"""
        base = -1.2  # Start below average
        growth = 2.5 / 30  # Grow to +1.3 over 30 questions
        return base + (step * growth) + np.random.normal(0, 0.15)

    # Run simulation
    for i in range(1, 31):
        q = engine.get_next_question()
        
        q_id = q['question_id'] if 'question_id' in q.index else q['item_id']
        diff = q['difficulty']
        exp_time = q['expected_time_sec']
        
        # Simulate response
        correct, time = simulate_real_response(diff, exp_time, i, learning_curve)
        
        # Submit to engine
        engine.submit_answer(q_id, correct, time, exp_time)
        
        # Log output
        status_icon = "✅" if correct else "❌"
        speed_factor = time / exp_time
        zone_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[engine.current_difficulty_zone]
        
        print(f"Q{i:02}: {zone_icon} Diff={diff:+5.2f} | θ={engine.theta:+5.2f} | Time={time:5.1f}s ({speed_factor:.2f}x) | {status_icon} | Cohort: {engine.cohort}")
        
    # Final summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Final Cohort Assessment: {engine.cohort}")
    print(f"Final Ability Estimate (θ): {engine.theta:+.3f}")
    print(f"\nLearning Progression:")
    if len(engine.theta_history) > 1:
        print(f"  Initial θ: {engine.theta_history[0]:+.3f}")
        print(f"  Final θ:   {engine.theta_history[-1]:+.3f}")
        print(f"  Improvement: {engine.theta_history[-1] - engine.theta_history[0]:+.3f}")
        print(f"  Theta updates: {len(engine.theta_history)}")
    print(f"\nCohort Transitions: {' → '.join(set(engine.cohort_history))}")
    print("="*60)