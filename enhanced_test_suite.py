import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adaptive_engine import NavikaEngine
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

# ==========================================
# 1. LOAD TEST DATA AND MODEL
# ==========================================
print("="*60)
print("NAVIKA MODEL EVALUATION - Real World Test Data")
print("="*60)

# Load test data
try:
    test_df = pd.read_csv('navika_real_test.csv')
    print(f"✅ Loaded {len(test_df):,} test interactions")
    use_real_data = True
    
    # Standardize column names
    if 'user_id' in test_df.columns and 'student_id' not in test_df.columns:
        test_df['student_id'] = test_df['user_id']
    if 'correct' in test_df.columns and 'is_correct' not in test_df.columns:
        test_df['is_correct'] = test_df['correct']
    if 'item_id' in test_df.columns and 'question_id' not in test_df.columns:
        test_df['question_id'] = test_df['item_id']
        
except FileNotFoundError:
    print("⚠️ Real test data not found. Using synthetic fallback.")
    test_df = pd.read_csv('navika_synthetic_test.csv')
    use_real_data = False
    
    if 'student_id' not in test_df.columns:
        test_df['student_id'] = test_df.get('user_id', range(len(test_df)))
    if 'is_correct' not in test_df.columns:
        test_df['is_correct'] = test_df.get('correct', 0)

# Load trained model
try:
    with open('navika_brain.pkl', 'rb') as f:
        kmeans_model, scaler_model, cluster_map = pickle.load(f)
    print("✅ Loaded trained NAVIKA model")
except FileNotFoundError:
    print("❌ Error: navika_brain.pkl not found!")
    print("Please run 'train_cohort_model_enhanced.py' first.")
    exit()

# Load question metadata
try:
    questions_db = pd.read_csv('navika_questions_meta_real.csv')
    print(f"✅ Loaded {len(questions_db)} question items")
except FileNotFoundError:
    questions_db = pd.read_csv('navika_questions_meta.csv')
    print(f"✅ Loaded synthetic questions (fallback)")
    
# Standardize question DB columns
if 'item_id' in questions_db.columns and 'question_id' not in questions_db.columns:
    questions_db['question_id'] = questions_db['item_id']

# ==========================================
# 2. SETUP ENHANCED LOGGING ENGINE
# ==========================================
class EnhancedLoggingEngine(NavikaEngine):
    def __init__(self):
        super().__init__()
        self.logs = {
            'q_idx': [],
            'cohort': [],
            'diff': [],
            'correct': [],
            'pred_correct': [],
            'response_time': [],
            'theta': [],
            'zone': [],
            'expected_time': []
        }
        
    def log_step(self, step, diff, correct, pred_prob, time, exp_time, theta, zone):
        self.logs['q_idx'].append(step)
        self.logs['cohort'].append(self.cohort)
        self.logs['diff'].append(diff)
        self.logs['correct'].append(correct)
        self.logs['pred_correct'].append(pred_prob)
        self.logs['response_time'].append(time)
        self.logs['expected_time'].append(exp_time)
        self.logs['theta'].append(theta)
        self.logs['zone'].append(zone)

# ==========================================
# 3. TEST ON REAL STUDENT DATA
# ==========================================
def test_real_students(test_df, n_students=10, questions_per_student=30):
    """
    Test NAVIKA on actual student interaction sequences from the test set
    """
    print(f"\n{'='*60}")
    print(f"Testing on {n_students} real students from test set")
    print(f"{'='*60}\n")
    
    # Select random students with sufficient interactions
    student_counts = test_df['student_id'].value_counts()
    eligible_students = student_counts[student_counts >= questions_per_student].index
    
    if len(eligible_students) < n_students:
        print(f"⚠️ Only {len(eligible_students)} students with {questions_per_student}+ interactions")
        n_students = min(n_students, len(eligible_students))
    
    # Select diverse students (some high performers, some low performers)
    student_accuracy = test_df.groupby('student_id')['is_correct'].mean()
    eligible_with_acc = student_accuracy[eligible_students]
    
    # Sort by accuracy and pick from different ranges
    sorted_students = eligible_with_acc.sort_values()
    
    test_students = []
    # Bottom 20% (struggling)
    n_low = max(2, n_students // 5)
    test_students.extend(sorted_students.head(n_low).index.tolist())
    
    # Top 20% (high achievers)
    n_high = max(2, n_students // 5)
    test_students.extend(sorted_students.tail(n_high).index.tolist())
    
    # Middle 60% (average)
    remaining_needed = n_students - len(test_students)
    middle_students = sorted_students.iloc[n_low:-n_high]
    if remaining_needed > 0 and len(middle_students) > 0:
        test_students.extend(middle_students.sample(min(remaining_needed, len(middle_students))).index.tolist())
    
    # Ensure we have exactly n_students
    test_students = test_students[:n_students]
    
    results = {
        'student_id': [],
        'initial_cohort': [],
        'final_cohort': [],
        'accuracy': [],
        'avg_speed': [],
        'initial_theta': [],
        'final_theta': [],
        'theta_change': [],
        'transitions': []
    }
    
    all_logs = []
    
    for idx, student_id in enumerate(test_students):
        print(f"\nStudent {idx+1}/{n_students}: {student_id}")
        
        # Get student's real interaction sequence
        student_data = test_df[test_df['student_id'] == student_id].head(questions_per_student)
        
        # Initialize engine
        engine = EnhancedLoggingEngine()
        
        # Track cohort transitions
        cohort_history = []
        
        # Replay student's real interactions
        for i, (_, row) in enumerate(student_data.iterrows(), 1):
            # IMPROVED: Engine picks question adaptively
            q = engine.get_next_question()
            
            # Use actual student response data
            is_correct = int(row['is_correct'])
            actual_time = row['response_time']
            exp_time = q['expected_time_sec']
            
            # Calculate IRT probability
            theta = engine.theta
            a, b, c = q['discrimination'], q['difficulty'], q['guessing']
            z = a * (theta - b)
            z = np.clip(z, -10, 10)
            pred_prob = c + (1 - c) / (1 + np.exp(-z))
            
            # Submit answer
            engine.submit_answer(q['question_id'], is_correct, actual_time, exp_time)
            engine.log_step(i, q['difficulty'], is_correct, pred_prob, actual_time, 
                          exp_time, engine.theta, engine.current_difficulty_zone)
            
            # Track cohort
            if i == 1 or engine.cohort != cohort_history[-1]:
                cohort_history.append(engine.cohort)
        
        # Store results
        results['student_id'].append(student_id)
        results['initial_cohort'].append(cohort_history[0] if cohort_history else 'Unknown')
        results['final_cohort'].append(engine.cohort)
        results['accuracy'].append(np.mean([x[1] for x in engine.history]))
        
        times = [x[2] / x[3] for x in engine.history if x[3] > 0]
        results['avg_speed'].append(np.mean(times) if times else 1.0)
        
        initial_theta = engine.theta_history[0] if len(engine.theta_history) > 0 else 0.0
        final_theta = engine.theta_history[-1] if len(engine.theta_history) > 0 else 0.0
        
        results['initial_theta'].append(initial_theta)
        results['final_theta'].append(final_theta)
        results['theta_change'].append(final_theta - initial_theta)
        results['transitions'].append(len(set(cohort_history)))
        
        all_logs.append(engine.logs)
        
        print(f"  Initial: {cohort_history[0]} → Final: {engine.cohort}")
        print(f"  Accuracy: {results['accuracy'][-1]:.2%}, Speed: {results['avg_speed'][-1]:.2f}x")
        print(f"  θ: {results['initial_theta'][-1]:+.2f} → {results['final_theta'][-1]:+.2f} (Δ{results['theta_change'][-1]:+.2f})")
    
    return pd.DataFrame(results), all_logs

# ==========================================
# 4. SYNTHETIC SCENARIO TESTING
# ==========================================
def simulate_scenario(scenario_name, skill_profile_func):
    """
    Simulate specific behavioral scenarios for comparison
    """
    print(f"\nRunning Scenario: {scenario_name}")
    
    engine = EnhancedLoggingEngine()
    
    for i in range(1, 31):
        q = engine.get_next_question()
        
        # Get skill and speed based on progress
        skill, speed_type = skill_profile_func(i, q['difficulty'], engine.theta)
        
        # Calculate probability
        prob = 1 / (1 + np.exp(-(skill - q['difficulty'])))
        prob = np.clip(prob, 0.05, 0.95)  # Add realism
        is_correct = 1 if np.random.rand() < prob else 0
        
        # Generate time
        if speed_type == 'fast':
            time_factor = np.random.uniform(0.3, 0.6)
        elif speed_type == 'slow':
            time_factor = np.random.uniform(1.6, 2.5)
        else:
            time_factor = np.random.uniform(0.85, 1.25)
        
        actual_time = q['expected_time_sec'] * time_factor
        
        theta = engine.theta
        a, b, c = q['discrimination'], q['difficulty'], q['guessing']
        z = a * (theta - b)
        z = np.clip(z, -10, 10)
        pred_prob = c + (1 - c) / (1 + np.exp(-z))
        
        engine.submit_answer(q['question_id'], is_correct, actual_time, q['expected_time_sec'])
        engine.log_step(i, q['difficulty'], is_correct, pred_prob, actual_time, q['expected_time_sec'], engine.theta, engine.current_difficulty_zone)
    
    return engine.logs

# Scenario profiles with learning curves
def high_achiever_learning(step, diff, theta):
    """High achiever: consistent high performance"""
    base = 1.8 + (step / 30) * 0.5  # Starts high, improves slightly
    return base + np.random.normal(0, 0.2), 'normal'

def struggling_improver(step, diff, theta):
    """Starts weak, shows clear improvement"""
    progress = (step - 1) / 30
    skill = -1.2 + progress * 2.2  # -1.2 to +1.0
    speed = 'slow' if step < 12 else ('normal' if step < 24 else 'fast')
    return skill + np.random.normal(0, 0.3), speed

def fast_guesser_profile(step, diff, theta):
    """Fast but inaccurate, slight improvement"""
    base = -0.8 + (step / 30) * 0.4
    return base + np.random.normal(0, 0.35), 'fast'

def anxious_learner(step, diff, theta):
    """Capable but slow, gains confidence"""
    skill = 1.0 + np.random.normal(0, 0.3)
    progress = (step - 1) / 30
    # Gets faster as confidence builds
    if progress < 0.4:
        speed = 'slow'
    else:
        speed = 'normal'
    return skill, speed

def variable_performer(step, diff, theta):
    """Inconsistent performance - the interesting case"""
    # Performance varies with question difficulty
    if abs(diff - theta) < 0.5:  # Questions near their level
        skill = theta + np.random.normal(0, 0.4)
        speed = 'normal'
    elif diff > theta:  # Hard questions
        skill = theta - 0.5 + np.random.normal(0, 0.5)
        speed = 'slow'
    else:  # Easy questions
        skill = theta + 0.5 + np.random.normal(0, 0.3)
        speed = 'fast'
    return skill, speed

# ==========================================
# 5. RUN TESTS
# ==========================================
print("\n" + "="*60)
print("PHASE 1: Real Student Testing")
print("="*60)

real_results, real_logs = test_real_students(test_df, n_students=6, questions_per_student=30)

print("\n" + "="*60)
print("REAL STUDENT RESULTS SUMMARY")
print("="*60)
print(real_results.to_string(index=False))

print("\n" + "="*60)
print("PHASE 2: Synthetic Scenario Testing")
print("="*60)

scenario_logs = {
    'High Achiever': simulate_scenario("High Achiever", high_achiever_learning),
    'Struggling Improver': simulate_scenario("Struggling Improver", struggling_improver),
    'Fast Guesser': simulate_scenario("Fast Guesser", fast_guesser_profile),
    'Anxious Learner': simulate_scenario("Anxious Learner", anxious_learner)
}

# ==========================================
# 6. ENHANCED VISUALIZATIONS (SPLIT ROW-WISE)
# ==========================================
print("\nGenerating comprehensive visualizations across 4 separate images...")

# Set beautiful style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- FONT & STYLE CONFIGURATION ---
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 10
BG_COLOR = '#f8f9fa'

def setup_row_figure(title):
    """Helper to create a wide figure for a single row"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    fig.suptitle(title, fontsize=TITLE_SIZE+2, fontweight='bold', color='#2C3E50', y=0.98)
    # Adjust layout to make room for legends on the right of each subplot
    plt.subplots_adjust(left=0.05, right=0.90, top=0.85, bottom=0.15, wspace=0.4)
    return fig, axes

def add_outside_legend(ax, lines, labels=None):
    """Helper to place legend outside to the right of the plot"""
    if labels is None:
        labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=LEGEND_SIZE, framealpha=0.95, edgecolor='#BDC3C7', fancybox=True)

# ==============================================================================
# ROW 1: Real Student Learning Paths
# ==============================================================================
print("Generating Row 1: Real Student Paths...")
fig1, axes1 = setup_row_figure('Row 1: Real Student Learning Paths & Cohort Evolution')

for idx, ax in enumerate(axes1):
    if idx >= len(real_logs): break
    logs = real_logs[idx]
    df_log = pd.DataFrame(logs)
    ax.set_facecolor(BG_COLOR)
    ax2 = ax.twinx()
    
    # 1. Plot Lines (Unpacking the list to get the object)
    line1, = ax.plot(df_log['q_idx'], df_log['theta'], linestyle='-', linewidth=3.5, alpha=0.9, 
            label='Student Ability (θ)', color='#2E86AB', zorder=3)
    ax.fill_between(df_log['q_idx'], df_log['theta'], alpha=0.2, color='#2E86AB', zorder=1)
    
    line2, = ax2.plot(df_log['q_idx'], df_log['diff'], linestyle='--', linewidth=3, 
            alpha=0.7, label='Question Difficulty', color='#E63946', zorder=3)
    
    # 2. Plot Markers
    correct_mask = df_log['correct'] == 1
    scatter1 = ax.scatter(df_log[correct_mask]['q_idx'], df_log[correct_mask]['theta'], 
              color='#06D6A0', s=160, marker='o', edgecolors='white', linewidths=2.5, zorder=6, label='Correct Answer')
    scatter2 = ax.scatter(df_log[~correct_mask]['q_idx'], df_log[~correct_mask]['theta'],
              color='#EF476F', s=160, marker='X', edgecolors='white', linewidths=2.5, zorder=6, label='Incorrect Answer')
    
    # 3. Background Cohort Shading
    for i in range(len(df_log)-1):
        cohort = df_log['cohort'].iloc[i]
        c_color = '#FFE5E5' if ('Slow' in cohort or 'Anxious' in cohort) else ('#FFF4E5' if ('Fast' in cohort or 'Guesser' in cohort) else None)
        if c_color: ax.axvspan(df_log['q_idx'].iloc[i], df_log['q_idx'].iloc[i]+1, color=c_color, alpha=0.4, zorder=0)
    
    # 4. Styling
    ax.set_xlabel('Question Number', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Estimated Ability (θ)', fontsize=LABEL_SIZE, fontweight='600', color='#2E86AB')
    ax2.set_ylabel('Question Difficulty', fontsize=LABEL_SIZE, fontweight='600', color='#E63946')
    ax.tick_params(labelsize=TICK_SIZE, labelcolor='#2E86AB', axis='y')
    ax2.tick_params(labelsize=TICK_SIZE, labelcolor='#E63946', axis='y')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=2)
    
    res = real_results.iloc[idx]
    ax.set_title(f'Student {idx+1}\n{res["initial_cohort"]} → {res["final_cohort"]}\nΔθ = {res["theta_change"]:+.2f}',
                 fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.7, color='#BDC3C7')

    # 5. Legend
    add_outside_legend(ax, [line1, line2, scatter1, scatter2])

fig1.savefig('NAVIKA_Row1_RealStudents.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved Row 1")


# ==============================================================================
# ROW 2: Synthetic Scenarios
# ==============================================================================
print("Generating Row 2: Synthetic Scenarios...")
fig2, axes2 = setup_row_figure('Row 2: Synthetic Behavioral Scenarios')

scenario_names = list(scenario_logs.keys())
scenario_colors = ['#9B59B6', '#3498DB', '#E74C3C']

for idx, (name, logs) in enumerate(list(scenario_logs.items())[:3]):
    ax = axes2[idx]
    df_log = pd.DataFrame(logs)
    ax.set_facecolor(BG_COLOR)
    ax2 = ax.twinx()
    color = scenario_colors[idx]
    
    line1, = ax.plot(df_log['q_idx'], df_log['theta'], linestyle='-', linewidth=3.5, alpha=0.9, color=color, zorder=3, label=f'{name} Ability')
    ax.fill_between(df_log['q_idx'], df_log['theta'], alpha=0.25, color=color, zorder=1)
    
    line2, = ax2.plot(df_log['q_idx'], df_log['diff'], linestyle='--', linewidth=3, alpha=0.7, color='#E67E22', zorder=3, label='Difficulty')
    
    c_mask = df_log['correct'] == 1
    s1 = ax.scatter(df_log[c_mask]['q_idx'], df_log[c_mask]['theta'], color='#27AE60', s=140, marker='o', edgecolors='white', linewidths=2, zorder=6, label='Correct')
    s2 = ax.scatter(df_log[~c_mask]['q_idx'], df_log[~c_mask]['theta'], color='#C0392B', s=140, marker='X', edgecolors='white', linewidths=2, zorder=6, label='Incorrect')

    for i in range(len(df_log)-1):
        cohort = df_log['cohort'].iloc[i]
        bc = '#FFE5E5' if 'Slow' in cohort else ('#FFF4E5' if 'Fast' in cohort else None)
        if bc: ax.axvspan(df_log['q_idx'].iloc[i], df_log['q_idx'].iloc[i]+1, color=bc, alpha=0.35, zorder=0)
    
    ax.set_xlabel('Question Number', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
    ax.set_ylabel('Ability (θ)', fontsize=LABEL_SIZE, fontweight='600', color=color)
    ax2.set_ylabel('Difficulty', fontsize=LABEL_SIZE, fontweight='600', color='#E67E22')
    ax.tick_params(labelcolor=color, axis='y', labelsize=TICK_SIZE)
    ax2.tick_params(labelcolor='#E67E22', axis='y', labelsize=TICK_SIZE)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    theta_change = df_log['theta'].iloc[-1] - df_log['theta'].iloc[0]
    ax.set_title(f'{name}\nFinal: {df_log["cohort"].iloc[-1]}\nΔθ = {theta_change:+.2f}', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
    ax.grid(True, alpha=0.3, linewidth=0.7)
    
    add_outside_legend(ax, [line1, line2, s1, s2])

fig2.savefig('NAVIKA_Row2_Synthetic.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved Row 2")


# ==============================================================================
# ROW 3: Analysis Plots
# ==============================================================================
print("Generating Row 3: Analysis Plots...")
fig3, axes3 = setup_row_figure('Row 3: Cohort Analysis & Performance Metrics')

# 3.1: Cohort distribution
ax_cohort = axes3[0]
ax_cohort.set_facecolor(BG_COLOR)
cohort_counts = real_results['final_cohort'].value_counts()
colors_cohort = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6'][:len(cohort_counts)]
bars = ax_cohort.barh(range(len(cohort_counts)), cohort_counts.values, color=colors_cohort, edgecolor='white', linewidth=2, height=0.7)
for bar in bars:
    width = bar.get_width()
    ax_cohort.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=12, fontweight='bold', color='#2C3E50')
ax_cohort.set_yticks(range(len(cohort_counts)))
ax_cohort.set_yticklabels(cohort_counts.index, fontsize=TICK_SIZE, color='#2C3E50')
ax_cohort.set_xlabel('Count of Students', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_cohort.set_title('Final Cohort Distribution', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
ax_cohort.grid(axis='x', alpha=0.3)
ax_cohort.spines['top'].set_visible(False); ax_cohort.spines['right'].set_visible(False)
# No legend needed for bar chart

# 3.2: Learning gains
ax_gains = axes3[1]
ax_gains.set_facecolor(BG_COLOR)
colors_gain = ['#2ECC71' if x >= 0 else '#E74C3C' for x in real_results['theta_change']]
ax_gains.barh(range(len(real_results)), real_results['theta_change'], color=colors_gain, edgecolor='white', linewidth=2, height=0.7)
ax_gains.set_yticks(range(len(real_results)))
ax_gains.set_yticklabels([f'Student {i+1}' for i in range(len(real_results))], fontsize=TICK_SIZE, color='#2C3E50')
ax_gains.set_xlabel('Ability Change (Δθ)', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_gains.set_title('Learning Gains by Student', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
ax_gains.axvline(x=0, color='#34495E', linestyle='-', linewidth=2, alpha=0.8)
ax_gains.grid(axis='x', alpha=0.3)
# Legend for gains
from matplotlib.patches import Patch
g_patches = [Patch(facecolor='#2ECC71', label='Improved (Δθ > 0)'), Patch(facecolor='#E74C3C', label='Regressed (Δθ < 0)')]
add_outside_legend(ax_gains, g_patches)

# 3.3: Speed vs Accuracy
ax_perf = axes3[2]
ax_perf.set_facecolor(BG_COLOR)
scatter = ax_perf.scatter(real_results['avg_speed'], real_results['accuracy'], s=250, c=real_results['theta_change'], cmap='RdYlGn', alpha=0.9, edgecolors='black', linewidths=1.5, zorder=5)
l_acc = ax_perf.axhline(y=0.5, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=2.5, label='50% Accuracy Line')
l_spd = ax_perf.axvline(x=1.0, color='#3498DB', linestyle='--', alpha=0.7, linewidth=2.5, label='Expected Speed (1.0)')
ax_perf.set_xlabel('Avg Speed Factor (Lower=Faster)', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_perf.set_ylabel('Accuracy', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_perf.set_title('Performance Map (Color=Gain)', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
ax_perf.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax_perf, pad=0.02)
cbar.set_label('Learning Gain (Δθ)', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
add_outside_legend(ax_perf, [l_acc, l_spd])

fig3.savefig('NAVIKA_Row3_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved Row 3")


# ==============================================================================
# ROW 4: Detailed Metrics
# ==============================================================================
print("Generating Row 4: Detailed Metrics...")
fig4, axes4 = setup_row_figure('Row 4: Detailed Interaction Metrics Over Time')

line_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

# 4.1: Response time
ax_time = axes4[0]
ax_time.set_facecolor(BG_COLOR)
time_lines = []
for idx, logs in enumerate(real_logs[:4]):
    df_log = pd.DataFrame(logs)
    smoothed = (df_log['response_time'] / df_log['expected_time']).rolling(window=5, min_periods=1).mean()
    l, = ax_time.plot(df_log['q_idx'], smoothed, alpha=0.85, linewidth=3, label=f'Student {idx+1}', color=line_colors[idx])
    time_lines.append(l)

ref_time = ax_time.axhline(y=1.0, color='#34495E', linestyle='--', linewidth=2.5, alpha=0.8, label='Expected Baseline')
time_lines.append(ref_time)
ax_time.set_ylabel('Speed Factor (Smoothed)', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_time.set_title('Response Speed Evolution', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
add_outside_legend(ax_time, time_lines)

# 4.2: Difficulty adaptation
ax_adapt = axes4[1]
ax_adapt.set_facecolor(BG_COLOR)
adapt_lines = []
for idx, logs in enumerate(real_logs[:4]):
    df_log = pd.DataFrame(logs)
    l, = ax_adapt.plot(df_log['q_idx'], df_log['diff'] - df_log['theta'], alpha=0.85, linewidth=3, color=line_colors[idx], label=f'Student {idx+1}')
    adapt_lines.append(l)

ref_adapt = ax_adapt.axhline(y=0, color='#34495E', linestyle='--', linewidth=2.5, alpha=0.8, label='Perfect Match (0)')
fill_adapt = ax_adapt.fill_between([0, 30], -0.5, 0.5, alpha=0.15, color='#2ECC71', label='Optimal Zone (±0.5)')
# Dummy handle for fill
from matplotlib.patches import Patch
p_fill = Patch(facecolor='#2ECC71', alpha=0.3, label='Optimal Zone')

adapt_lines.extend([ref_adapt, p_fill])
ax_adapt.set_ylabel('Gap (Difficulty - Ability)', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_adapt.set_title('Adaptive Difficulty Targeting', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
add_outside_legend(ax_adapt, adapt_lines)

# 4.3: Cumulative accuracy
ax_cum = axes4[2]
ax_cum.set_facecolor(BG_COLOR)
acc_lines = []
for idx, logs in enumerate(real_logs[:4]):
    df_log = pd.DataFrame(logs)
    l, = ax_cum.plot(df_log['q_idx'], df_log['correct'].expanding().mean(), alpha=0.85, linewidth=3, color=line_colors[idx], label=f'Student {idx+1}')
    acc_lines.append(l)

ref_acc = ax_cum.axhline(y=0.5, color='#E74C3C', linestyle='--', linewidth=2.5, alpha=0.8, label='50% Threshold')
acc_lines.append(ref_acc)
ax_cum.set_ylabel('Cumulative Accuracy', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
ax_cum.set_title('Learning Progress', fontsize=TITLE_SIZE, fontweight='bold', color='#2C3E50')
ax_cum.set_ylim([0, 1])
add_outside_legend(ax_cum, acc_lines)

# Common styling for Row 4
for ax in axes4:
    ax.set_xlabel('Question Number', fontsize=LABEL_SIZE, fontweight='600', color='#2C3E50')
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.tick_params(labelsize=TICK_SIZE)

fig4.savefig('NAVIKA_Row4_Metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved Row 4")

print("\n" + "="*60)
print("✅ EVALUATION & VISUALIZATION COMPLETE!")
print("="*60)
print(f"Tested on {len(real_results)} real students")
print(f"+ {len(scenario_logs)} synthetic scenarios")
print("\nKey Findings:")
print(f"  Average Learning Gain: {real_results['theta_change'].mean():+.3f}")
print(f"  Students with Positive Gain: {(real_results['theta_change'] > 0).sum()}/{len(real_results)}")
print(f"  Average Final Accuracy: {real_results['accuracy'].mean():.2%}")
print("="*60)