import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
from streamlit_plotly_events import plotly_events
from fpdf import FPDF
import io

# --- 1. APP CONFIG ---
st.set_page_config(page_title="Pro-Link PDF Elite", layout="wide")

# --- 2. DATABASE CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)

def load_data():
    try:
        return conn.read(worksheet="Sheet1")
    except:
        return pd.DataFrame(columns=["Tournament", "Range", "X", "Y"])

if 'data' not in st.session_state:
    st.session_state.data = load_data()

# --- 3. DYNAMIC DRAWING ENGINE ---
def get_radii(label):
    if label == "50-100": return 3, 6
    if label == "101-150": return 4, 8
    return 5, 10

def draw_dispersion(df_filtered, label, is_pdf=False):
    fig = go.Figure()
    r_b, r_p = get_radii(label)

    # Birdie Circle (Blue, Opaque Fill)
    fig.add_shape(type="circle", x0=-r_b, y0=-r_b, x1=r_b, y1=r_b,
                  line_color="blue", line_width=2, fillcolor="rgba(173, 216, 230, 0.4)")
    # Par Circle (Blue, No Fill)
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
                                 marker=dict(size=12, color=df['color'], line=dict(width=1, color='white'))))

    limit = r_p + 3
    fig.update_layout(template="plotly_white", xaxis=dict(range=[-limit, limit], fixedrange=True),
                      yaxis=dict(range=[-limit, limit], fixedrange=True), width=500, height=500, showlegend=False)
    return fig

# --- 4. PDF GENERATOR ---
def create_pdf(df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Summary Stats
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Golf Dispersion Master Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for r in ["50-100", "101-150", "151-200"]:
        sub = df[df['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            r_b, r_p = get_radii(r)
            tot = len(sub)
            b = len(sub[sub['d'] <= r_b])
            p = len(sub[(sub['d'] > r_b) & (sub['d'] <= r_p)])
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=f"Range: {r} Meters", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 8, txt=f"Total Shots: {tot} | Birdies: {b} ({b/tot:.1%}) | Pars: {p} ({p/tot:.1%})", ln=True)
            pdf.ln(5)
            
    return pdf.output()

# --- 5. NAVIGATION ---
menu = st.sidebar.radio("Navigation", ["Home", "Record Shots", "Master Sheet", "Stats"])

if menu == "Home":
    st.header("🏌️‍♂️ Golf Dispersion Control")
    
    # NEW PDF EXPORT
    if not st.session_state.data.empty:
        st.subheader("Export Report")
        pdf_data = create_pdf(st.session_state.data)
        st.download_button(label="📄 Download Full PDF Report", data=pdf_data, file_name="golf_report.pdf", mime="application/pdf")
    
    new_t = st.text_input("New Tournament Name:")
    if st.button("Create Tournament"):
        st.success(f"Tournament '{new_t}' is ready.")

elif menu == "Record Shots":
    tourneys = st.session_state.data['Tournament'].unique().tolist()
    curr_t = st.selectbox("Tournament", tourneys if tourneys else ["Practice"])
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

elif menu == "Master Sheet":
    st.header("Master Metric Accumulation")
    for r in ["50-100", "101-150", "151-200"]:
        st.plotly_chart(draw_dispersion(st.session_state.data[st.session_state.data['Range'] == r], r))

elif menu == "Stats":
    st.header("Performance Analytics")
    for r in ["50-100", "101-150", "151-200"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            r_b, r_p = get_radii(r)
            tot, b = len(sub), len(sub[sub['d'] <= r_b])
            p = len(sub[(sub['d'] > r_b) & (sub['d'] <= r_p)])
            st.write(f"**{r}m:** {tot} Shots | Birdies: {b} ({b/tot:.1%}) | Pars: {p} ({p/tot:.1%})")
