import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adaptive_engine import NavikaEngine 

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot') # Fallback if seaborn is not available

# ==========================================
# 1. SETUP LOGGING
# ==========================================
class LoggingEngine(NavikaEngine):
    def __init__(self):
        super().__init__()
        self.logs = {'q_idx': [], 'topic': [], 'cohort': [], 'diff': [], 'correct': []}
        
    def log_step(self, step, diff, correct):
        self.logs['q_idx'].append(step)
        self.logs['topic'].append(self.current_topic)
        self.logs['cohort'].append(self.cohort)
        self.logs['diff'].append(diff)
        self.logs['correct'].append(correct)

# ==========================================
# 2. HELPER: REALISTIC HUMAN SIMULATOR
# ==========================================
def simulate_human_response(difficulty, true_skill, speed_profile):
    """
    Generates a realistic response based on IRT probability + Noise.
    speed_profile: 'fast', 'normal', 'slow'
    """
    # 1. Calculate Probability of Correctness (IRT)
    # P = 1 / (1 + e^-(theta - b))
    prob_correct = 1 / (1 + np.exp(-(true_skill - difficulty)))
    
    # Add slight randomness to the "roll" (Luck factor)
    is_correct = 1 if np.random.rand() < prob_correct else 0
    
    # 2. Calculate Realistic Response Time with Jitter
    if speed_profile == 'fast':
        # Fast Guesser: 10% to 30% of expected time
        time_factor = np.random.uniform(0.1, 0.3)
    elif speed_profile == 'slow':
        # Panic/Anxious: 160% to 280% of expected time
        time_factor = np.random.uniform(1.6, 2.8)
    else:
        # Normal: 80% to 120% of expected time
        time_factor = np.random.uniform(0.8, 1.2)
        
    return is_correct, time_factor

# ==========================================
# 3. SCENARIO RUNNER
# ==========================================
def run_scenario(scenario_name, logic_func):
    engine = LoggingEngine()
    print(f"\nRunning {scenario_name}...")
    
    # Run for 30 Questions to allow patterns to emerge naturally
    for i in range(1, 31): 
        q = engine.get_next_question()
        
        # Get Student 'State' (Skill & Speed) for this specific topic
        skill, speed_type = logic_func(q['topic'])
        
        # Generate Realistic Response
        is_correct, time_factor = simulate_human_response(q['difficulty'], skill, speed_type)
        
        actual_time = q['expected_time_sec'] * time_factor
        engine.submit_answer(q['question_id'], is_correct, actual_time, q['expected_time_sec'], q['topic'])
        engine.log_step(i, q['difficulty'], is_correct)
        
    return engine.logs

# ==========================================
# 4. DEFINE REALISTIC PROFILES
# ==========================================

# A: The "High Achiever"
# Consistently good across all topics. Normal speed.
def profile_genius(topic):
    # Skill is high (2.5) for everything. Speed is Normal.
    return 2.5, 'normal'

# B: The "Calculus Anxiety"
# Skill is high for Algebra (2.0), but terrible for Calculus (-2.0).
# Speed becomes 'slow' (panic) when facing Calculus.
def profile_panic(topic):
    if topic == "Calculus":
        return -2.0, 'slow' # Low skill, Panic speed
    elif topic == "Geometry":
        return 1.0, 'normal' # Decent
    else:
        return 2.0, 'normal' # Strong in Algebra

# C: The "Strategic Guesser"
# Skill is low (-1.0) but tries to game the system by answering fast.
def profile_guesser(topic):
    # Skill is effectively 0 because they are guessing (random). Speed is Fast.
    return -1.0, 'fast'

# ==========================================
# 5. ADVANCED PLOTTING (Research Grade)
# ==========================================
def plot_results(logs, title, ax):
    df = pd.DataFrame(logs)
    
    topics = ['Algebra', 'Geometry', 'Calculus', 'Statistics']
    topic_map = {t: i for i, t in enumerate(topics)}
    df['topic_num'] = df['topic'].map(topic_map)
    
    # 1. Background Zones (Cohorts)
    x = df['q_idx']
    for i in range(len(x)-1):
        c = df['cohort'][i]
        color = 'white'
        if c == "Slow Thinker": color = '#ffebee' # Red Tint
        if c == "Fast Guesser": color = '#fff3e0' # Orange Tint
        ax.axvspan(x[i], x[i]+1, color=color, alpha=0.6)

    # 2. Topic Path (The Main Line)
    ax.plot(df['q_idx'], df['topic_num'], color='#3f51b5', linewidth=2.5, alpha=0.8, label='Topic Path')
    
    # 3. Scatter Points (Green=Correct, Red=Wrong)
    # This adds "Texture" to the graph making it look realistic
    ax.scatter(df[df['correct']==1]['q_idx'], df[df['correct']==1]['topic_num'], 
               color='green', s=80, edgecolors='white', zorder=5, label='Correct Answer')
    ax.scatter(df[df['correct']==0]['q_idx'], df[df['correct']==0]['topic_num'], 
               color='#d32f2f', marker='X', s=80, edgecolors='white', zorder=5, label='Incorrect Answer')

    # Formatting
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontweight='bold', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Question Sequence")
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend only to the first plot to keep it clean
    if "Ideal" in title: 
        ax.legend(loc='lower right', frameon=True, framealpha=0.9)
        
# ==========================================
# 6. EXECUTE
# ==========================================
if __name__ == "__main__":
    # Run Scenarios
    logs_A = run_scenario("Scenario A: High Achiever", profile_genius)
    logs_B = run_scenario("Scenario B: Panic Response", profile_panic)
    logs_C = run_scenario("Scenario C: Serial Guesser", profile_guesser)

    # Setup the Canvas
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot Data
    plot_results(logs_A, "A: Ideal Progression\n(Steady Climb)", ax1)
    plot_results(logs_B, "B: Dynamic Adaptation\n(Calculus Panic -> Algebra Reset)", ax2)
    plot_results(logs_C, "C: Anomaly Detection\n(Guesser Locked in Algebra)", ax3)
    
    # Final Touches
    plt.suptitle("NAVIKA Dynamic Navigation System: Live Simulation Results", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Show
    print("\nâœ… Simulation Complete. Displaying Realistic Research Graphs...")
    plt.show()
    # Run Scenarios with Randomness
    logs_A = run_scenario("Scenario A: High Achiever", profile_genius)
    logs_B = run_scenario("Scenario B: Panic Response", profile_panic)
    logs_C = run_scenario("Scenario C: Serial Guesser", profile_guesser)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    plot_results(logs_A, "A: Ideal Progression\n(Consistently High Performance)", ax1)
    plot_results(logs_B, "B: Dynamic Adaptation\n(Calculus Panic -> Algebra Reset)", ax2)
    plot_results(logs_C, "C: Anomaly Detection\n(Guesser Locked in Algebra)", ax3)
    
    plt.suptitle("NAVIKA Dynamic Navigation System: Stochastic Simulation Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("\nâœ… Simulation Complete. Graphs generated with realistic noise.")