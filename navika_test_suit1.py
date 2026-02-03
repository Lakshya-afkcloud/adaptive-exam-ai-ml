import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

# ==========================================
# 1. SETUP & LOAD SYSTEM
# ==========================================
print("Loading NAVIKA System Resources...")
questions_db = pd.read_csv('navika_questions_meta.csv')
with open('navika_brain.pkl', 'rb') as f:
    kmeans_model, scaler_model = pickle.load(f)

# Helper: IRT Probability
def get_prob_correct(theta, a, b, c):
    z = a * (theta - b)
    return c + (1 - c) / (1 + np.exp(-z))

# Helper: Theta Estimator
def estimate_theta(responses, questions):
    if not responses: return 0.0
    def neg_log_like(theta):
        log_like = 0
        for q_id, correct in responses:
            q_row = questions[questions['question_id'] == q_id].iloc[0]
            p = get_prob_correct(theta, q_row['discrimination'], q_row['difficulty'], q_row['guessing'])
            p = np.clip(p, 0.001, 0.999)
            log_like += np.log(p) if correct else np.log(1 - p)
        return -log_like
    res = minimize(neg_log_like, x0=[0.0], bounds=[(-3, 3)])
    return res.x[0]

# ==========================================
# 2. THE ENGINE (With Data Logging)
# ==========================================
class NavikaEngine:
    def __init__(self):
        self.history = [] 
        self.current_theta = 0.0 
        self.cohort = "Normal" # Start Assumption
        self.questions_asked = set()
        # Logger for plotting later
        self.logs = {'q_idx': [], 'theta': [], 'difficulty': [], 'cohort': [], 'correct': []}

    def update_cohort(self):
        if len(self.history) < 3: return
        df_hist = pd.DataFrame(self.history, columns=['q_id', 'correct', 'time', 'exp_time'])
        df_hist['speed'] = df_hist['time'] / df_hist['exp_time']
        
        # Extract features
        feats = np.array([[df_hist['correct'].mean(), df_hist['speed'].mean(), df_hist['speed'].std()]])
        scaled = scaler_model.transform(feats)
        pred = kmeans_model.predict(scaled)[0]
        
        # Map Logic (must match your training logic)
        avg_speed = feats[0][1]
        if avg_speed < 0.6: self.cohort = "Fast Guesser"
        elif avg_speed > 1.3: self.cohort = "Slow Thinker"
        else: self.cohort = "Normal"

    def get_next_question(self):
        target = self.current_theta
        # Dynamic Strategy
        if self.cohort == "Fast Guesser": target += 0.7  # Penalty
        elif self.cohort == "Slow Thinker": target -= 0.4 # Confidence boost
        
        candidates = questions_db[~questions_db['question_id'].isin(self.questions_asked)].copy()
        candidates['dist'] = abs(candidates['difficulty'] - target)
        return candidates.sort_values('dist').iloc[0]

    def submit_answer(self, q, is_correct, time, exp_time, step):
        self.history.append((q['question_id'], is_correct, time, exp_time))
        self.questions_asked.add(q['question_id'])
        self.current_theta = estimate_theta([(x[0], x[1]) for x in self.history], questions_db)
        
        if len(self.history) % 3 == 0: self.update_cohort()
        
        # Log for Plotting
        self.logs['q_idx'].append(step)
        self.logs['theta'].append(self.current_theta)
        self.logs['difficulty'].append(q['difficulty'])
        self.logs['cohort'].append(self.cohort)
        self.logs['correct'].append(is_correct)

# ==========================================
# 3. SCENARIO RUNNER
# ==========================================
def run_simulation(scenario_name, behavior_func):
    engine = NavikaEngine()
    print(f"\nRunning Scenario: {scenario_name}")
    
    for i in range(1, 21): # 20 Questions
        q = engine.get_next_question()
        
        # Get behavior for this step (is_correct, time_factor)
        is_correct, time_factor = behavior_func(i, q['difficulty'])
        
        actual_time = q['expected_time_sec'] * time_factor
        engine.submit_answer(q, is_correct, actual_time, q['expected_time_sec'], i)
        
        print(f"  Q{i}: Diff={q['difficulty']:.2f} | Ans={is_correct} | Time={time_factor:.2f}x | Cohort={engine.cohort}")
        
    return engine.logs

# ==========================================
# 4. DEFINE TEST SCENARIOS
# ==========================================

# Scenario A: The Normal Student (Control)
# Consistently takes 1.0x time, answers correctly if Difficulty < Skill (Skill=0.5)
def scenario_normal(step, diff):
    true_skill = 0.5
    time_factor = np.random.uniform(0.9, 1.1) # Normal time
    # Probability of correct based on skill gap
    prob = 1 / (1 + np.exp(-(true_skill - diff))) 
    correct = 1 if np.random.rand() < prob else 0
    return correct, time_factor

# Scenario B: The "Redemption Arc" (Guesser -> Normal)
# Q1-8: Guesses fast (0.3x time). Q9-20: Works normally.
def scenario_redemption(step, diff):
    if step <= 8:
        return (1 if np.random.rand() > 0.5 else 0), 0.3 # Fast & Random
    else:
        # Becomes a distinct "Smart" student (Skill=1.0)
        prob = 1 / (1 + np.exp(-(1.0 - diff)))
        return (1 if np.random.rand() < prob else 0), 1.1 # Normal time

# Scenario C: The "Panic" (Normal -> Slow/Struggling)
# Q1-8: Normal. Q9-20: Slow (2.0x time) and error-prone.
def scenario_panic(step, diff):
    if step <= 8:
        prob = 1 / (1 + np.exp(-(0.0 - diff)))
        return (1 if np.random.rand() < prob else 0), 1.0
    else:
        return 0, 2.5 # Always wrong, very slow (Struggling/Anxious)

# ==========================================
# 5. EXECUTE & PLOT
# ==========================================
results_A = run_simulation("Normal Baseline", scenario_normal)
results_B = run_simulation("Redemption Arc", scenario_redemption)
results_C = run_simulation("Anxiety Collapse", scenario_panic)

# PLOTTING FUNCTION
def plot_results(logs, title, ax):
    x = logs['q_idx']
    
    # 1. Plot Cohort Background Zones
    # Convert cohort names to numeric codes for creating colored zones
    cohort_map = {"Normal": 0, "Fast Guesser": 1, "Slow Thinker": 2}
    codes = [cohort_map[c] for c in logs['cohort']]
    
    # Fill background based on cohort
    # We iterate to find segments where cohort is constant
    for i in range(len(x)-1):
        c = logs['cohort'][i]
        color = 'white'
        if c == "Fast Guesser": color = '#ffebee' # Light Red
        if c == "Slow Thinker": color = '#e8f5e9' # Light Green
        ax.axvspan(x[i], x[i]+1, color=color, alpha=0.5)

    # 2. Plot Lines
    ax.plot(x, logs['theta'], label='Est. Ability ($\Theta$)', color='blue', linewidth=2)
    ax.plot(x, logs['difficulty'], label='Q. Difficulty', color='red', linestyle='--', marker='o')
    
    # 3. Add Labels
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-3.5, 3.5)
    ax.set_ylabel("Skill / Difficulty")
    ax.set_xlabel("Question Number")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

# Create the figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

plot_results(results_A, "Scenario A: Consistent Normal Student", ax1)
plot_results(results_B, "Scenario B: Redemption (Guesser $\to$ Normal)", ax2)
plot_results(results_C, "Scenario C: Panic (Normal $\to$ Slow/Struggling)", ax3)

plt.tight_layout()
plt.show()

print("\n✅ Test Suite Complete. Check the popup graph for visual proof of dynamic adaptation.")