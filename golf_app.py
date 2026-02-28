import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection

# --- APP CONFIG ---
st.set_page_config(page_title="Pro-Link Golf", layout="wide", page_icon="⛳")

# --- DATABASE CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    try:
        return conn.read(worksheet="Sheet1")
    except:
        return pd.DataFrame(columns=["Tournament", "Range", "X", "Y"])

if 'data' not in st.session_state:
    st.session_state.data = load_data()

# --- DRAWING ENGINE ---
def draw_circles(df_filtered, label):
    fig = go.Figure()
    # Circles: Birdie (1yd), Par (3yds), Bogey (5yds)
    zones = [{"r": 1, "c": "Gold"}, {"r": 3, "c": "#00FF00"}, {"r": 5, "c": "Red"}]
    for z in zones:
        fig.add_shape(type="circle", x0=-z['r'], y0=-z['r'], x1=z['r'], y1=z['r'],
                      line_color=z['c'], line_width=2)
    
    # Crosshairs
    fig.add_shape(type="line", x0=-6, y0=0, x1=6, y1=0, line=dict(color="gray", dash="dash"))
    fig.add_shape(type="line", x0=0, y0=-6, x1=0, y1=6, line=dict(color="gray", dash="dash"))

    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        df['color'] = df['dist'].apply(lambda d: 'Gold' if d<=1 else ('#00FF00' if d<=3 else 'Red'))
        fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='markers', 
                                 marker=dict(size=14, color=df['color'], line=dict(width=1, color='white'))))

    fig.update_layout(template="plotly_dark", title=f"{label} Dispersion", 
                      xaxis=dict(range=[-6, 6]), yaxis=dict(range=[-6, 6]), width=450, height=450)
    return fig

# --- NAVIGATION ---
menu = st.sidebar.radio("Menu", ["Home", "Record Shots", "Master Sheet", "Stats"])

if menu == "Home":
    st.header("⛳ Golf Dispersion Tracker")
    new_t = st.text_input("New Tournament Name:")
    if st.button("Add Tournament"):
        st.success(f"Tournament '{new_t}' active. Go to 'Record Shots'.")

elif menu == "Record Shots":
    tourneys = st.session_state.data['Tournament'].unique().tolist()
    curr_t = st.selectbox("Tournament", tourneys if tourneys else ["Practice"])
    tabs = st.tabs(["50-100", "101-150", "151-200"])
    ranges = ["50-100", "101-150", "151-200"]
    
    for i, r in enumerate(ranges):
        with tabs[i]:
            with st.form(f"f{r}"):
                c1, c2 = st.columns(2)
                x = c1.slider("Left/Right", -5.0, 5.0, 0.0)
                y = c2.slider("Short/Long", -5.0, 5.0, 0.0)
                if st.form_submit_button("Confirm Shot"):
                    new_row = pd.DataFrame([{"Tournament": curr_t, "Range": r, "X": x, "Y": y}])
                    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                    conn.update(worksheet="Sheet1", data=st.session_state.data)
                    st.rerun()
            
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == curr_t) & (st.session_state.data['Range'] == r)]
            st.plotly_chart(draw_circles(df_v, r))
            if not df_v.empty and st.button(f"Undo Last {r}"):
                st.session_state.data = st.session_state.data.drop(df_v.index[-1])
                conn.update(worksheet="Sheet1", data=st.session_state.data)
                st.rerun()

elif menu == "Master Sheet":
    for r in ["50-100", "101-150", "151-200"]:
        st.plotly_chart(draw_circles(st.session_state.data[st.session_state.data['Range'] == r], f"MASTER {r}"))

elif menu == "Stats":
    st.header("Performance Analytics")
    scope = st.radio("View", ["Tournament", "Master"])
    df_s = st.session_state.data if scope == "Master" else st.session_state.data[st.session_state.data['Tournament'] == st.selectbox("Select", st.session_state.data['Tournament'].unique())]
    
    for r in ["50-100", "101-150", "151-200"]:
        sub = df_s[df_s['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            tot = len(sub)
            b = len(sub[sub['d'] <= 1])
            p = len(sub[(sub['d'] > 1) & (sub['d'] <= 3)])
            bog = tot - (b + p)
            st.write(f"**{r} Yards:** {tot} Shots | Birdies: {b} ({b/tot:.0%}) | Pars: {p} ({p/tot:.0%}) | Bogeys+: {bog} ({bog/tot:.0%})")
