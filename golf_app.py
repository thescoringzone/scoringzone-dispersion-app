import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from streamlit_plotly_events import plotly_events
from fpdf import FPDF
import io

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

# Helper to find radii (in meters)
def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

# --- 3. DYNAMIC DRAWING ENGINE (FIXED) ---
def draw_dispersion(df_filtered, label):
    fig = go.Figure()
    r_b, r_p = get_radii(label) # r_b = Birdie, r_p = Par

    # Birdie Circle (Blue, Opaque Fill)
    fig.add_shape(type="circle", x0=-r_b, y0=-r_b, x1=r_b, y1=r_b,
                  line_color="blue", line_width=2, fillcolor="rgba(173, 216, 230, 0.4)")
    
    # Par Circle (Blue, No Fill)
    fig.add_shape(type="circle", x0=-r_p, y0=-r_p, x1=r_p, y1=r_p,
                  line_color="blue", line_width=2)
    
    # Labels at the top of the circles
    fig.add_annotation(x=0, y=r_b, text=f"{r_b}m", showarrow=False, yshift=10, font=dict(color="blue"))
    fig.add_annotation(x=0, y=r_p, text=f"{r_p}m", showarrow=False, yshift=10, font=dict(color="blue"))

    # Crosshairs (Subtle)
    limit = r_p + 3
    fig.add_shape(type="line", x0=-limit, y0=0, x1=limit, y1=0, line=dict(color="rgba(0,0,0,0.1)", dash="dash"))
    fig.add_shape(type="line", x0=0, y0=-limit, x1=0, y1=limit, line=dict(color="rgba(0,0,0,0.1)", dash="dash"))

    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        
        # Color Logic: Birdie=Red, Par=Blue, Bogey=Black
        df['dot_color'] = df['dist'].apply(lambda d: 'red' if d <= r_b else ('blue' if d <= r_p else 'black'))
        
        fig.add_trace(go.Scatter(
            x=df['X'], y=df['Y'], mode='markers', 
            marker=dict(size=14, color=df['dot_color'], line=dict(width=1, color='white')),
            hoverinfo='skip'
        ))

    fig.update_layout(
        template="plotly_white", 
        xaxis=dict(range=[-limit, limit], fixedrange=True, zeroline=False, showgrid=False),
        yaxis=dict(range=[-limit, limit], fixedrange=True, zeroline=False, showgrid=False),
        width=500, height=500,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# --- 4. PDF GENERATOR ---
def create_pdf(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Golf Dispersion Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for r in ["50-100", "101-150", "151-200"]:
        sub = df[df['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, rp = get_radii(r)
            tot = len(sub)
            b = len(sub[sub['d'] <= rb])
            p = len(sub[(sub['d'] > rb) & (sub['d'] <= rp)])
            pdf.cell(200, 8, txt=f"{r}m: {tot} shots | Birdies: {b} | Pars: {p}", ln=True)
    return pdf.output()

# --- 5. NAVIGATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'active_tournament' not in st.session_state:
    st.session_state.active_tournament = None

# Sidebar always visible for Master/Stats
menu = st.sidebar.radio("Main Menu", ["Home", "Master Sheet", "Stats"])
if menu != "Home":
    st.session_state.page = menu

# --- PAGE: HOME (The List) ---
if st.session_state.page == "Home":
    st.header("🏌️‍♂️ My Tournaments")
    
    # Create Section
    with st.expander("➕ New Tournament"):
        new_name = st.text_input("Name:")
        if st.button("Save & Open"):
            if new_name:
                st.session_state.active_tournament = new_name
                st.session_state.page = "Record Shots"
                st.rerun()

    st.divider()
    
    # Display List
    all_t = st.session_state.data['Tournament'].unique().tolist()
    if not all_t:
        st.info("No tournaments yet. Click the + above!")
    else:
        for t in all_t:
            c1, c2 = st.columns([4, 1])
            if c1.button(f"⛳ {t}", use_container_width=True):
                st.session_state.active_tournament = t
                st.session_state.page = "Record Shots"
                st.rerun()
            if c2.button("🗑️", key=f"del_{t}"):
                st.session_state.data = st.session_state.data[st.session_state.data['Tournament'] != t]
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()
    
    # Global PDF Export
    if not st.session_state.data.empty:
        pdf_out = create_pdf(st.session_state.data)
        st.download_button("📄 Download PDF Stats", data=pdf_out, file_name="golf_stats.pdf")

# --- PAGE: RECORD SHOTS ---
elif st.session_state.page == "Record Shots":
    curr_t = st.session_state.active_tournament
    if st.button("⬅️ Home List"):
        st.session_state.page = "Home"
        st.rerun()
        
    st.title(f"Tournament: {curr_t}")
    tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
    ranges = ["50-100", "101-150", "151-200"]
    
    for i, r_label in enumerate(ranges):
        with tabs[i]:
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == curr_t) & (st.session_state.data['Range'] == r_label)]
            
            # Interactive Chart
            sel = plotly_events(draw_dispersion(df_v, r_label), click_event=True, override_height=500)
            
            if sel:
                new_x, new_y = round(sel[0]['x'], 2), round(sel[0]['y'], 2)
                row = pd.DataFrame([{"Tournament": curr_t, "Range": r_label, "X": new
