import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from streamlit_plotly_events import plotly_events
from fpdf import FPDF

# --- 1. APP CONFIG ---
st.set_page_config(page_title="Golf Dispersion Pro", layout="wide")

# --- 2. DATABASE CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    try:
        return conn.read(worksheet="Sheet1")
    except Exception:
        return pd.DataFrame(columns=["Tournament", "Range", "X", "Y"])

if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Radii helper (Meters)
def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

# --- 3. THE VISUAL ENGINE (WITH INVISIBLE GRID HACK) ---
def draw_dispersion(df_filtered, label):
    fig = go.Figure()
    r_b, r_p = get_radii(label)
    limit = r_p + 2

    # --- THE INVISIBLE TOUCH GRID ---
    # This creates a dense net of transparent dots so you can click anywhere
    grid_vals = np.linspace(-limit, limit, 60) # 60x60 = 3600 clickable points
    gx, gy = np.meshgrid(grid_vals, grid_vals)
    fig.add_trace(go.Scatter(
        x=gx.flatten(), y=gy.flatten(), mode='markers',
        marker=dict(size=15, color='rgba(0,0,0,0)'), # Completely transparent
        hoverinfo='none', showlegend=False
    ))

    # Birdie Circle: Blue outline, Light Blue Opaque Fill
    fig.add_shape(type="circle", x0=-r_b, y0=-r_b, x1=r_b, y1=r_b,
                  line_color="blue", line_width=2, fillcolor="rgba(173, 216, 230, 0.4)")
    
    # Par Circle: Blue outline, No Fill
    fig.add_shape(type="circle", x0=-r_p, y0=-r_p, x1=r_p, y1=r_p,
                  line_color="blue", line_width=2)
    
    # Metric Labels
    fig.add_annotation(x=0, y=r_b, text=f"{r_b}m", showarrow=False, yshift=10, font=dict(color="blue"))
    fig.add_annotation(x=0, y=r_p, text=f"{r_p}m", showarrow=False, yshift=10, font=dict(color="blue"))

    # Crosshairs
    fig.add_shape(type="line", x0=-limit, y0=0, x1=limit, y1=0, line=dict(color="rgba(0,0,0,0.1)", dash="dash"))
    fig.add_shape(type="line", x0=0, y0=-limit, x1=0, y1=limit, line=dict(color="rgba(0,0,0,0.1)", dash="dash"))

    # Plot Actual Shots
    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        # Colors: Birdie=Red, Par=Blue, Bogey=Black
        df['c'] = df['dist'].apply(lambda d: 'red' if d <= r_b else ('blue' if d <= r_p else 'black'))
        
        fig.add_trace(go.Scatter(
            x=df['X'], y=df['Y'], mode='markers', 
            marker=dict(size=14, color=df['c'], line=dict(width=1, color='white')),
            hoverinfo='skip'
        ))

    fig.update_layout(
        template="plotly_white", 
        # RECTANGLE FRAME & NO NUMBERS
        xaxis=dict(range=[-limit, limit], fixedrange=True, showticklabels=False, showgrid=False, zeroline=False, mirror=True, showline=True, linecolor='black', linewidth=2),
        yaxis=dict(range=[-limit, limit], fixedrange=True, showticklabels=False, showgrid=False, zeroline=False, mirror=True, showline=True, linecolor='black', linewidth=2),
        width=400, height=400, showlegend=False, margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# --- 4. NAVIGATION ---
if 'page' not in st.session_state: st.session_state.page = "Home"
if 'active_t' not in st.session_state: st.session_state.active_t = None

menu = st.sidebar.radio("Navigation", ["Home", "Master Sheet", "Stats"])
if menu != "Home": st.session_state.page = menu

# --- PAGE: HOME ---
if st.session_state.page == "Home":
    st.header("🏌️‍♂️ My Tournaments")
    with st.expander("➕ New Tournament"):
        t_name = st.text_input("Tournament Name:")
        if st.button("Create & Start"):
            if t_name:
                st.session_state.active_t = t_name
                st.session_state.page = "Record"
                st.rerun()

    st.divider()
    all_t = st.session_state.data['Tournament'].unique().tolist()
    if not all_t:
        st.info("Create a tournament to begin.")
    else:
        for t in all_t:
            c1, c2 = st.columns([4, 1])
            if c1.button(f"⛳ {t}", use_container_width=True):
                st.session_state.active_t = t
                st.session_state.page = "Record"
                st.rerun()
            if c2.button("🗑️", key=f"del_{t}"):
                st.session_state.data = st.session_state.data[st.session_state.data['Tournament'] != t]
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()

# --- PAGE: RECORD (TOUCH-TO-ADD) ---
elif st.session_state.page == "Record":
    st.button("⬅️ Back to Home", on_click=lambda: setattr(st.session_state, 'page', "Home"))
    st.title(f"Editing: {st.session_state.active_t}")
    
    t_tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
    r_list = ["50-100", "101-150", "151-200"]
    
    for i, r_label in enumerate(r_list):
        with t_tabs[i]:
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == st.session_state.active_t) & (st.session_state.data['Range'] == r_label)]
            
            st.write("👆 **Tap inside the box to record a shot.**")
            
            # THE MAGIC TOUCH COMPONENT
            # We add len(df_v) to the key. This forces the chart to "refresh" instantly after every tap.
            event_data = plotly_events(
                draw_dispersion(df_v, r_label), 
                click_event=True, 
                hover_event=False, 
                key=f"plot_{r_label}_{len(df_v)}"
            )
            
            if event_data:
                # Grab the X and Y from the invisible dot you tapped
                new_x = round(event_data[0]['x'], 2)
                new_y = round(event_data[0]['y'], 2)
                
                new_row = pd.DataFrame([{"Tournament": st.session_state.active_t, "Range": r_label, "X": new_x, "Y": new_y}])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()

            if not df_v.empty and st.button("Undo Last Shot", key=f"un_{r_label}"):
                st.session_state.data = st.session_state.data.drop(df_v.index[-1])
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()

# --- PAGE: MASTER SHEET ---
elif st.session_state.page == "Master Sheet":
    st.header("Master Accumulated View")
    for r in ["50-100", "101-150", "151-200"]:
        st.subheader(f"Global {r}m Dispersion")
        # Reuse the exact same drawing function
        st.plotly_chart(draw_dispersion(st.session_state.data[st.session_state.data['Range'] == r], r))

# --- PAGE: STATS ---
elif st.session_state.page == "Stats":
    st.header("Overall Performance")
    for r in ["50-100", "101-150", "151-200"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, rp = get_radii(r)
            tot = len(sub)
            b = len(sub[sub['d'] <= rb])
            p = len(sub[(sub['d'] > rb) & (sub['d'] <= rp)])
            st.write(f"**{r}m:** {tot} Shots | Birdies: {b} | Pars: {p} | Bogeys: {tot-(b+p)}")
