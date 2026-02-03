import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Define the Data
data = {
    'Model': ['NAVIKA', 'Strielkowski', 'Rizvi et al.', 'Lin et al.', 
              'Contrino', 'Meylani', 'du Plooy', 'FTC 2025'],
    'Adaptability': [9.5, 2.0, 5.0, 7.0, 5.0, 4.0, 3.0, 4.0],
    'Efficiency': [8.0, 9.0, 7.0, 5.0, 8.0, 4.0, 0.0, 0.0],
    'Personalization': [9.5, 3.0, 4.0, 8.0, 5.0, 5.0, 4.0, 5.0],
    'Interoperability': [8.5, 8.0, 9.0, 7.0, 5.0, 5.0, 8.0, 0.0],
    'Scalability': [9.0, 8.0, 9.0, 7.0, 5.0, 5.0, 0.0, 0.0],
    'Complexity': [9.0, 2.0, 5.0, 6.0, 4.0, 7.0, 2.0, 3.0]
}

df = pd.DataFrame(data)

# 2. Setup Radar Chart Parameters
categories = list(df.columns[1:])
N = len(categories)

# Calculate angles for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Initialize Plot with a compact figure size (9x8.5 removes extra white space)
fig, ax = plt.subplots(figsize=(9, 8.5), subplot_kw=dict(polar=True))

# Set background color inside the circle
ax.set_facecolor('#fafafa') 

# 3. Plot Each Model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

for i, row in df.iterrows():
    values = row[1:].tolist()
    values += values[:1]  # Close the loop
    model_name = row['Model']
    
    # Conditional Styling: Highlight NAVIKA
    if model_name == 'NAVIKA':
        line_width = 2.5
        line_alpha = 1.0
        fill_alpha = 0.25
        z_order = 10 
        color = "#306dae" # Strong Blue
        marker = 'o'
    else:
        line_width = 1.5
        line_alpha = 0.6
        fill_alpha = 0.0
        z_order = 1
        color = colors[i]
        marker = None

    ax.plot(angles, values, linewidth=line_width, linestyle='solid', 
            label=model_name, color=color, alpha=line_alpha, zorder=z_order, marker=marker)
    
    if fill_alpha > 0:
        ax.fill(angles, values, color=color, alpha=fill_alpha, zorder=z_order)

# 4. Styling & Labels

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold', color='#333333')
# 'pad=25' keeps labels outside but brings them closer than before to reduce spacing
ax.tick_params(axis='x', which='major', pad=25)

# Grid
ax.grid(color='#AAAAAA', alpha=0.3, linestyle='--') 
ax.spines['polar'].set_visible(False) 

# Y-Axis Labels
ax.set_rlabel_position(0)
plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=9)
plt.ylim(0, 10.5)

# Title: Moved closer to graph (y=1.05)
plt.title("NAVIKA vs. Contemporary Frameworks", 
          size=16, weight='bold', y=1.05, color='#222222')

# Legend: Horizontal at Bottom
# bbox_to_anchor places it below the axis. ncol=4 makes it horizontal (2 rows).
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
           fancybox=True, shadow=True, ncol=4, fontsize=10)

# Show Plot
plt.tight_layout()
plt.savefig('Comparative_analysis_figure')
print("Saved figure")
plt.show()