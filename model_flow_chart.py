import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

# ==========================================
# 1. SETUP & STYLING (COMPACT & BALANCED)
# ==========================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), facecolor='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis('off')

BOX_STYLE = "round,pad=0.1,rounding_size=0.2"
TITLE_FONT = {'size': 9, 'weight': 'bold'}
TEXT_FONT = {'size': 9.5} # Slightly smaller text for tighter boxes
ARROW_PROPS = dict(arrowstyle='-|>', lw=1.1, color='black', mutation_scale=10)

# ==========================================
# 2. DRAWING FUNCTIONS
# ==========================================
def draw_box(x, y, w, h, text, title=None, fc='white', ec='black', ls='solid'):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=BOX_STYLE,
        facecolor=fc, edgecolor=ec, linewidth=1.1, linestyle=ls))
    if title:
        ax.text(x + w/2, y + h - 0.7, title.upper(), 
                ha='center', va='center', **TITLE_FONT)
        ax.text(x + w/2, y + h/2 - 0.45, text, 
                ha='center', va='center', **TEXT_FONT, wrap=True)
    else:
        ax.text(x + w/2, y + h/2, text, 
                ha='center', va='center', **TEXT_FONT, wrap=True)

def draw_db(x, y, w, h, text):
    ax.add_patch(Ellipse((x+w/2, y), w, h*0.22, fc='white', ec='black', lw=1.1))
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec='none'))
    ax.add_patch(Ellipse((x+w/2, y+h), w, h*0.22, fc='white', ec='black', lw=1.1))
    ax.plot([x, x], [y, y+h], lw=1.1, c='black')
    ax.plot([x+w, x+w], [y, y+h], lw=1.1, c='black')
    ax.text(x+w/2, y+h/2, text, 
            ha='center', va='center', fontsize=8.5, weight='bold')

def arrow(x1, y1, x2, y2, rad=0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        **ARROW_PROPS,
        connectionstyle=f"arc3,rad={rad}"))

def diamond(x, y, r, text):
    ax.add_patch(mpatches.RegularPolygon(
        (x, y), 4, radius=r, 
        orientation=0, facecolor='white', 
        edgecolor='black', linewidth=1.1))
    ax.text(x, y, text, 
            ha='center', va='center', 
            fontsize=8.5, weight='bold')

# ==========================================
# 3. TITLE
# ==========================================
ax.text(50, 54.5, "NAVIKA SYSTEM ARCHITECTURE", 
        ha='center', fontsize=13, weight='bold')
ax.plot([32, 68], [53.2, 53.2], lw=1.4, c='black')

# ==========================================
# 4. PHASE 1 – OFFLINE (NARROWER)
# ==========================================
# Background
ax.add_patch(FancyBboxPatch((4, 6), 28, 43, 
                            boxstyle="round,pad=0.2",
                            fc='#f9f9f9', ec='black', ls='--', lw=1))
ax.text(18, 47.5, "PHASE 1: OFFLINE INTELLIGENCE", 
        ha='center', fontsize=8.5, weight='bold')

# Boxes (Width 20 instead of 22)
draw_box(8, 39.5, 20, 5.2, "Historical Logs", "Data Ingestion")
arrow(18, 39.5, 18, 36)

draw_box(8, 29, 20, 6.2, 
         "Calc Speed Factor\n(Normalize Time)", "Preprocessing")
arrow(18, 29, 18, 25.5)

draw_box(8, 18.5, 20, 6.2, 
         "K-Means Algorithm\n(Identify Groups)", "Unsupervised Learning")
arrow(18, 18.5, 18, 15)

draw_db(13, 8, 10, 6, "COHORT\nMODEL")

# ==========================================
# 5. PHASE 2 – ONLINE (NARROWER)
# ==========================================
# Background
ax.add_patch(FancyBboxPatch((35, 4), 62, 45, 
                            boxstyle="round,pad=0.2",
                            fc='#eef6fb', ec='black', lw=1.4))
ax.text(66, 47.5, "PHASE 2: ONLINE ADAPTIVE ENGINE", 
        ha='center', fontsize=8.5, weight='bold')

# Top Box (Student Response)
# Centered at x=66. Width=22 (start 55, end 77)
draw_box(55, 41, 22, 5.2, 
         "Student Response\n(Correctness + Time)", fc='#e8f6f3')

# Split Arrows (adjusted for new box width)
# Start from bottom corners of top box
arrow(60, 41, 49, 36.5, 0.15)
arrow(72, 41, 83, 36.5, -0.15)

# --- Micro loop (Left Column) ---
ax.text(49, 38, "Micro-Loop (Cognitive)", 
        fontsize=7.5, style='italic', weight='bold', ha='center')

# Left Box: Centered at ~49. Width 18 (Start 40)
draw_box(40, 29, 18, 6.2, 
         "IRT Engine\n(3PL Model)", "Psychometrics")
arrow(49, 29, 49, 25.5)

draw_box(40, 21, 18, 4.8, "Update Ability (θ)")

# --- Macro loop (Right Column) ---
ax.text(83, 38, "Macro-Loop (Behavioral)", 
        fontsize=7.5, style='italic', weight='bold', ha='center')

# Right Box: Centered at ~83. Width 18 (Start 74)
draw_box(74, 29, 18, 6.2, 
         "Cohort Classifier\n(Dist to Centroid)", "Behavior Analysis")
arrow(83, 29, 83, 25.5)

draw_box(74, 21, 18, 4.8, 
         "Update Cohort\n(e.g., Fast Guesser)")

# Connector: Offline DB -> Online Macro Loop
# DB Center x=18, y=8. Target x=74 (left edge of classifier), y=32
arrow(23, 10, 74, 31, 0.15)

# --- Convergence ---
# From boxes (center 49, 83) to Diamond (center 66)
arrow(49, 21, 61, 15.5, 0.08)
arrow(83, 21, 71, 15.5, -0.08)

# Diamond
diamond(66, 13.5, 5.0, "Strategy\nSelector")

arrow(66, 9.2, 66, 5)

# Bottom Box
# Centered at 66. Width 22 (Start 55)
draw_box(55, 1, 22, 3.6, 
         "Select Next Question\n& Deliver", fc='#2f2f2f')
ax.text(66, 2.7, 
        "Select Next Question\n& Deliver", 
        ha='center', va='center', 
        fontsize=9, color='white', weight='bold')

# ==========================================
# FEEDBACK LOOP (ALIGNED TO RIGHT EDGE)
# ==========================================
# Bottom Box Right Edge: x = 55 + 22 = 77
# Top Box Right Edge:    x = 55 + 22 = 77
# Loop path: x=77 -> out to 90 -> up -> back to 77

ax.plot([77, 96, 96, 78],       # x coords: Box Edge -> Margin -> Margin -> Top Box
        [2.8, 2.8, 43.6, 43.6], # y coords
        ls='--', lw=1.1, color='black')

ax.add_patch(FancyArrowPatch(
    (78, 43.6), (77, 43.6),
    **ARROW_PROPS,
    connectionstyle="arc3,rad=0"
))
# ==========================================
plt.tight_layout()
plt.savefig("NAVIKA_Architecture_Compact.png", 
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ Compact, fixed-width architecture diagram generated.")