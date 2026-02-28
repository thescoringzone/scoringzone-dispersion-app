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
    except:
        return pd.DataFrame(columns=["Tournament", "Range", "X", "Y"])

if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Helper to find radii
def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

# --- 3. DRAWING ENGINE ---
def draw_dispersion(df_filtered, label):
    fig = go.Figure()
    r_b, r_p = get_radii(label)

    # Birdie Circle (Blue, Opaque Fill)
    fig.add_shape(type="circle", x0=-r_b, y0=-r_b, x1=r_b, y1=r_b,
                  line_color="blue", line_width=2, fillcolor="rgba(173, 216, 230, 0.4)")
    # Par Circle (Blue, No Fill)
    fig.add_shape(type="circle", x0=-r_par, y0=-r_par, x1=r_par, y1=r_par, # Error check: use r_p
                  line_color="blue", line_width=2)
    
    # Fix: Ensure r_p is used
    fig.add_shape(type="circle", x0=-r_p, y0=-r_p, x1=r_p, y1=r_p,
                  line_color="blue", line_width=2)

    # Labels
    fig.add_annotation(x=0, y=r_b, text=f"{r_b}m", showarrow=False, yshift=10)
    fig.add_annotation(x=0, y=r_p, text=f"{r_p}m", showarrow=False, yshift=10)

    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        df['color'] = df['dist'].apply(lambda d: 'red' if d <= r_b else ('blue' if d <= r_p else 'black'))
        fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='markers', 
                                 marker=dict(size=14, color=df['color'], line=dict(width=1, color='white'))))

    limit = r_p + 3
    fig.update_layout(template="plotly_white", xaxis=dict(range=[-limit, limit], fixedrange=True),
                      yaxis=dict(range=[-limit, limit], fixedrange=True), width=500, height=500, showlegend=False)
    return fig

# --- 4. NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'active_tournament' not in st.session_state:
    st.session_state.active_tournament = None

# Sidebar navigation (Override for manual jumps)
menu = st.sidebar.radio("Navigation", ["Home", "Master Sheet", "Stats"])
if menu != "Home":
    st.session_state.page = menu

# --- 5. PAGE: HOME (List & Select) ---
if st.session_state.page == "Home":
    st.header("🏌️‍♂️ My Tournaments")
    
    # Create New Tournament
    with st.expander("➕ Create New Tournament"):
        new_t = st.text_input("Tournament Name:")
        if st.button("Save Tournament"):
            if new_t:
                # To initialize a tournament, we add a dummy row or just track it in session
                st.success(f"'{new_t}' created!")
                st.session_state.active_tournament = new_t
                st.session_state.page = "Record Shots"
                st.rerun()

    st.divider()
    
    # List Existing Tournaments
    tourneys = st.session_state.data['Tournament'].unique().tolist()
    if not tourneys:
        st.info("No tournaments found. Create one above!")
    else:
        st.subheader("Select a Tournament to Edit/View")
        for t in tourneys:
            col1, col2 = st.columns([4, 1])
            if col1.button(f"⛳ {t}", use_container_width=True):
                st.session_state.active_tournament = t
                st.session_state.page = "Record Shots"
                st.rerun()
            
            # Delete Option
            if col2.button("🗑️", key=f"del_{t}"):
                st.session_state.data = st.session_state.data[st.session_state.data['Tournament'] != t]
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()

# --- 6. PAGE: RECORD SHOTS ---
elif st.session_state.page == "Record Shots":
    curr_t = st.session_state.active_tournament
    st.button("⬅️ Back to Tournament List", on_click=lambda: setattr(st.session_state, 'page', 'Home'))
    st.title(f"Tournament: {curr_t}")
    
    tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
    ranges = ["50-100", "101-150", "151-200"]
    
    for i, r in enumerate(ranges):
        with tabs[i]:
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == curr_t) & (st.session_state.data['Range'] == r)]
            
            # Capture Touch
            selected = plotly_events(draw_dispersion(df_v, r), click_event=True, override_height=500)
            
            if selected:
                new_row = pd.DataFrame([{"Tournament": curr_t, "Range": r, "X": round(selected[0]['x'], 2), "Y": round(selected[0]['y'], 2)}])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()
            
            if not df_v.empty and st.button(f"Undo Last Shot", key=f"u_{r}"):
                st.session_state.data = st.session_state.data.drop(df_v.index[-1])
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()

# --- 7. MASTER & STATS (Simplified) ---
elif st.session_state.page == "Master Sheet":
    st.header("Master Metric Accumulation")
    for r in ["50-100", "101-150", "151-200"]:
        st.subheader(f"Master {r}m View")
        st.plotly_chart(draw_dispersion(st.session_state.data[st.session_state.data['Range'] == r], r))

elif st.session_state.page == "Stats":
    st.header("Performance Analytics")
    for r in ["50-100", "101-150", "151-200"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            r_b, r_p = get_radii(r)
            tot, b = len(sub), len(sub[sub['d'] <= r_b])
            p = len(sub[(sub['d'] > r_b) & (sub['d'] <= r_p)])
            st.write(f"**{r}m:** {tot} Shots | Birdies: {b} ({b/tot:.1%}) | Pars: {p} ({p/tot:.1%})")
