import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Euro 2024 Fullback Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1c15;
    }
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #00140D;
    }
    /* Text Styling */
    h1, h2, h3, p, div, span, label, li {
        color: #e0e0e0 !important;
    }
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1a2e25;
        color: white;
        border: 1px solid #2d5e4d;
    }
    /* Images centered */
    div[data-testid="stImage"] {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    /* Player Card Styling */
    .player-card {
        background: #072015;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #243d33;
        margin-bottom: 10px;
    }
    .player-card b {
        color: #00ff85; /* Highlight label color */
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA LOADING & MAPPING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Load files
    stats = pd.read_csv("players_stats.csv")
    events = pd.read_csv("player_events.csv")
    
    # 1. Normalize Columns (Lowercase + Underscore) to match your provided index safely
    # 'Player Name' -> 'player_name', 'Image URL' -> 'image_url'
    stats.columns = [c.strip().lower().replace(' ', '_') for c in stats.columns]
    events.columns = [c.strip().lower().replace(' ', '_') for c in events.columns]
    
    return stats, events

try:
    stats_df, events_df = load_data()
except FileNotFoundError:
    st.error("Data files not found. Ensure 'players_stats.csv' and 'player_events.csv' are in the app directory.")
    st.stop()

# ---------------------------------------------------------
# 3. DEFINE DATA MAPPING
# ---------------------------------------------------------
# Based on your index: ['Player Name', 'Image URL', 'Position', 'Footed', 'Physical', 'Born', 'National Team', 'Club', 'age_years'...]
# Normalized keys become:
KEY_NAME = 'player_name'
KEY_IMG = 'image_url'
KEY_POS = 'position'
KEY_FOOT = 'footed'
KEY_PHYS = 'physical'
KEY_BORN = 'born'
KEY_NATION = 'national_team'
KEY_CLUB = 'club'
KEY_AGE = 'age_years'

# Verify ID column exists
if KEY_NAME not in stats_df.columns:
    st.error(f"Column '{KEY_NAME}' not found in stats file. Available: {list(stats_df.columns)}")
    st.stop()

# Get Player List
player_list = sorted(stats_df[KEY_NAME].dropna().unique())

# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Scouting Tool")
st.sidebar.markdown("---")

# Smart defaults
default_a = "Marc Cucurella Saseta"
default_b = "Joshua Kimmich"

def get_index(name):
    return player_list.index(name) if name in player_list else 0

p1 = st.sidebar.selectbox("Select Player 1 (Left)", player_list, index=get_index(default_a))
p2 = st.sidebar.selectbox("Select Player 2 (Right)", ["None"] + player_list, index=player_list.index(default_b)+1 if default_b in player_list else 0)

st.sidebar.markdown("---")
st.sidebar.info("Comparison of Euro 2024 Fullbacks based on Z-Scores per 90.")

# ---------------------------------------------------------
# 5. VISUALIZATION FUNCTIONS
# ---------------------------------------------------------

# Theme colors
THEME = {
    'background': '#0e1c15',
    'pitch_lines': '#243d33',
    'pass': '#ffffff',
    'carry': '#00d4ff',
    'defense': '#00ff85',
    'text': '#ffffff',
    'context': '#ffffff'
}

def get_player_info(player_name):
    """Extracts player details safely"""
    row = stats_df[stats_df[KEY_NAME] == player_name]
    if row.empty:
        return None
    return row.iloc[0]

def draw_horizontal_pitch(ax, player_name, events_df):
    """Draws the mplsoccer pitch with stats below it"""
    pitch = Pitch(pitch_type='statsbomb', pitch_color=THEME['background'], 
                  line_color=THEME['pitch_lines'], linewidth=1.5, goal_type='box')
    pitch.draw(ax=ax)

    # Data Filter
    cdf = events_df[events_df["player_name"] == player_name].copy()

    # Logic Filters
    mask_carries = (cdf.type_name == 'Carry') & (cdf["prog_carry"] == 1)
    mask_passes = (cdf.type_name == 'Pass') & (cdf.outcome_name.isna()) & ((cdf.prog_pass == 1) | (cdf.pass_into_box == 1))
    mask_defense = ((cdf.type_name == "Duel") & (cdf.outcome_name == "Won")) | (cdf.type_name.isin(["Interception", "Ball Recovery"]))
    
    prog_carries = cdf[mask_carries]
    prog_passes = cdf[mask_passes]
    defense_actions = cdf[mask_defense]

    # Drawing
    # 1. Carries (Cyan Dashed)
    pitch.lines(prog_carries.x, prog_carries.y, prog_carries.end_x, prog_carries.end_y,
                ax=ax, color=THEME['carry'], lw=2.5, linestyle='--', alpha=0.9, zorder=2)
    # 2. Passes (White Arrow)
    pitch.arrows(prog_passes.x, prog_passes.y, prog_passes.end_x, prog_passes.end_y,
                 ax=ax, width=2, headwidth=5, color=THEME['pass'], alpha=0.95, zorder=3)
    pitch.scatter(prog_passes.x, prog_passes.y, ax=ax, s=50, color=THEME['pass'], zorder=3)
    # 3. Defense (Green Hex)
    pitch.scatter(defense_actions.x, defense_actions.y, ax=ax, s=250, marker='h',
                  facecolors=THEME['defense'], edgecolors='black', linewidth=1.5, zorder=4)

    # Title
    ax.set_title(player_name, color='white', fontsize=16, fontweight='bold')

    # Stats Text (Relative axes coordinates for positioning)
    # Position: x=0.5 (center), y=-0.05 (below axis)
    stats_str = f"PASSES: {len(prog_passes)}  |  CARRIES: {len(prog_carries)}  |  DEFENSE: {len(defense_actions)}"
    ax.text(0.5, -0.1, stats_str, transform=ax.transAxes, 
            color=THEME['text'], fontsize=12, fontweight='bold', ha='center', va='top')
    
    # Direction Arrow
    ax.arrow(45, 83, 30, 0, color='white', width=0.6, head_width=1, alpha=0.6, zorder=10)
    ax.text(60, 88, "DIRECTION OF PLAY", color='white', ha='center', fontsize=8, fontweight='bold', alpha=0.8)

    return ax

def plot_radar_strip(df, p1_name, p2_name=None):
    """Plotly Strip Chart for Z-Scores"""
    # Metrics to display
    metrics = ['defense_score_z', 'retention_score_z', 'width_score_z', 
               'progression_score_z', 'chance_score_z', 'penetration_score_z', 
               'discipline_score_z']
    
    pretty_labels = {
        'progression_score_z': 'Progression', 'penetration_score_z': 'Penetration',
        'chance_score_z': 'Chance Creation', 'width_score_z': 'Width',
        'retention_score_z': 'Retention', 'defense_score_z': 'Defence',
        'discipline_score_z': 'Discipline'
    }

    # Melt for Plotly
    df_melt = df[[KEY_NAME] + metrics].melt(id_vars=KEY_NAME, var_name='Metric', value_name='Score')
    
    # Subsets
    df_bg = df_melt[~df_melt[KEY_NAME].isin([p1_name, p2_name] if p2_name else [p1_name])]
    df_p1 = df_melt[df_melt[KEY_NAME] == p1_name]
    
    fig = go.Figure()
    
    # Avg Line
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color='#88c9a1', opacity=0.5)

    # Background Dots
    fig.add_trace(go.Scatter(
        x=df_bg['Score'], y=df_bg['Metric'], mode='markers', name='Pool',
        text=df_bg[KEY_NAME], hovertemplate='<b>%{text}</b><br>Z-Score: %{x:.2f}<extra></extra>',
        marker=dict(color='#109648', size=8, opacity=0.4)
    ))

    # Player 2 (Yellow Hex)
    if p2_name and p2_name != "None":
        df_p2 = df_melt[df_melt[KEY_NAME] == p2_name]
        fig.add_trace(go.Scatter(
            x=df_p2['Score'], y=df_p2['Metric'], mode='markers', name=p2_name,
            text=df_p2[KEY_NAME], hovertemplate='<b>%{text}</b>: %{x:.2f}<extra></extra>',
            marker=dict(color='#f1c40f', size=22, symbol='hexagon', line=dict(color='#032f20', width=1))
        ))

    # Player 1 (White Hex)
    fig.add_trace(go.Scatter(
        x=df_p1['Score'], y=df_p1['Metric'], mode='markers', name=p1_name,
        text=df_p1[KEY_NAME], hovertemplate='<b>%{text}</b>: %{x:.2f}<extra></extra>',
        marker=dict(color='#ffffff', size=22, symbol='hexagon', line=dict(color='#032f20', width=1))
    ))

    # Custom Layout
    annotations = []
    for m in metrics:
        annotations.append(dict(
            x=0, y=m, text=pretty_labels.get(m, m), xref="x", yref="y", 
            yshift=22, showarrow=False, font=dict(color='#e0e0e0', size=14)
        ))

    fig.update_layout(
        plot_bgcolor='#032f20', paper_bgcolor='#032f20', height=750,
        showlegend=True, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(color='#88c9a1')),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4.5, 5.5]),
        yaxis=dict(showgrid=True, gridcolor='#2d5e4d', showticklabels=False, 
                   categoryorder='array', categoryarray=metrics[::-1]),
        annotations=annotations,
        margin=dict(l=10, r=10, t=30, b=80)
    )
    return fig

# ---------------------------------------------------------
# 6. RENDER APP LAYOUT
# ---------------------------------------------------------
st.title("Euro 2024 Fullback Analysis")

# --- A. PLAYER PROFILES ---
left_info = get_player_info(p1)
right_info = get_player_info(p2) if (p2 and p2 != "None") else None

c1, c2 = st.columns(2)

def render_profile(col, info):
    with col:
        if info is not None:
            st.markdown(f"### {info[KEY_NAME]}")
            
            # Image handling
            img_url = info.get(KEY_IMG)
            if pd.isna(img_url):
                st.image("https://via.placeholder.com/150?text=No+Image", width=120)
            else:
                st.image(img_url, width=120)
            
            # Stats Card
            st.markdown(f"""
            <div class="player-card">
                <b>Club:</b> {info.get(KEY_CLUB, '-')}<br>
                <b>Nation:</b> {info.get(KEY_NATION, '-')}<br>
                <b>Age:</b> {info.get(KEY_AGE, '-')} | 
                <b>Foot:</b> {info.get(KEY_FOOT, '-')}<br>
                <b>Position:</b> {info.get(KEY_POS, '-')}
            </div>
            """, unsafe_allow_html=True)
        else:
            if col == c2: # Right column placeholder
                st.markdown("### Comparison")
                st.info("Select a second player from the sidebar to compare.")

render_profile(c1, left_info)
render_profile(c2, right_info)

st.markdown("---")

# --- B. PITCH MAPS ---
st.subheader("Match Events")

# Create figure based on selection
if p2 and p2 != "None":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(THEME['background'])
    draw_horizontal_pitch(ax1, p1, events_df)
    draw_horizontal_pitch(ax2, p2, events_df)
else:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(THEME['background'])
    draw_horizontal_pitch(ax, p1, events_df)

# Common Legend for Matplotlib
legend_elements = [
    Line2D([0], [0], color=THEME['pass'], lw=2, label='Progressive Pass'),
    Line2D([0], [0], color=THEME['carry'], lw=2, linestyle='--', label='Progressive Carry'),
    Line2D([0], [0], marker='h', color='none', markerfacecolor=THEME['defense'], 
           markeredgecolor='black', markersize=10, label='Defensive Action'),
]
fig.legend(handles=legend_elements, loc='upper center',
           ncol=3, fontsize=10, facecolor='#1a2e25', edgecolor='white', labelcolor='white')

st.pyplot(fig)

st.markdown("---")

# --- C. METRICS CHART & GLOSSARY ---
st.subheader("Performance Metrics (Z-Scores)")

fig_radar = plot_radar_strip(stats_df, p1, (p2 if p2 != "None" else None))
st.plotly_chart(fig_radar, use_container_width=True)

# Glossary inside Expander BELOW the chart
with st.expander("📖 Metric Glossary (Click to Expand)"):
    st.markdown("""
    <div style="background-color: #072015; padding: 15px; border-radius: 5px; color: #e0e0e0;">
    <div class="glossary-title">1. Progression</div>
            Moves ball significantly closer to goal.
            <ul>
                <li><b>Prog Passes:</b> >15m gain toward goal.</li>
                <li><b>Prog Carries:</b> >8m gain toward goal.</li>
                <li><b>Final Third:</b> Actions entering attacking 3rd.</li>
            </ul>
            <div class="glossary-title" style="margin-top:10px;">2. Penetration</div>
            Breaks into dangerous zones.
            <ul>
                <li><b>Box Entry:</b> Pass or Carry into Penalty Area.</li>
                <li><b>Cross:</b> Pass from wide into box.</li>
                <li><b>Cutback:</b> Backward pass from byline.</li>
            </ul>
            <div class="glossary-title" style="margin-top:10px;">3. Chance Creation</div>
            Direct contribution to shots/goals.
            <ul>
                <li><b>Key Pass:</b> Leads directly to a shot.</li>
                <li><b>Goal Assist:</b> Leads directly to a goal.</li>
            </ul>
             <div class="glossary-title" style="margin-top:10px;">4. Width</div>
            Stretching play horizontally.
            <ul>
                <li><b>Wide Touches:</b> Touches near touchlines.</li>
                <li><b>Y-IQR:</b> Vertical spread of player position.</li>
            </ul>
            <div class="glossary-title" style="margin-top:10px;">5. Retention</div>
            Ball security under pressure.
            <ul>
                <li><b>Negative:</b> Miscontrols & Dispossessions.</li>
                <li><b>Positive:</b> Pass completion under pressure.</li>
            </ul>
            <div class="glossary-title" style="margin-top:10px;">6. Defence</div>
            Defensive activity volume.
            <ul>
                <li><b>Actions:</b> Tackles + Interceptions + Blocks + Recoveries.</li>
                <li><b>Duels:</b> Aerial & Ground duels won.</li>
            </ul>
            <div class="glossary-title" style="margin-top:10px;">7. Discipline</div>
            Defensive reliability (Higher = Better).
            <ul>
                <li><b>Negative Factors:</b> Fouls, Yellow/Red Cards, Errors leading to shots, Dribbled Past.</li>
            </ul>
    """, unsafe_allow_html=True)


st.markdown("""
    <br><br>
    <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #2d5e4d;">
        <p style="font-size: 16px; color: #88c9a1 !important;">
            Presented by - <b>Hitesh Gautam using AI tools </b>
        </p>
        <a href="https://www.linkedin.com/in/hiteshgautam026/" target="_blank" style="text-decoration: none;">
            <button style="
                background-color: #0077b5; 
                color: white; 
                border: none; 
                padding: 8px 15px; 
                border-radius: 5px; 
                cursor: pointer; 
                font-weight: bold;
                display: inline-flex; 
                align-items: center; 
                gap: 8px;">
                <span>Connect on LinkedIn</span>
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" style="filter: brightness(0) invert(1);">
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)