import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from supabase import create_client, Client

# --- 1. APP CONFIG & SECRETS ---
st.set_page_config(page_title="Golf Dispersion Elite", layout="wide")

# Connect to the Supabase Vault
@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# Fetch data directly from the vault
def load_data():
    response = supabase.table("shots").select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame(columns=["id", "Tournament", "Range", "X", "Y"])

if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Helper for circles
def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

# --- 2. THE VISUAL ENGINE ---
def create_target_image(df_filtered, label):
    r_b, r_p = get_radii(label)
    limit = r_p + 2 

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off') 

    rect = patches.Rectangle((-limit, -limit), limit*2, limit*2, linewidth=4, edgecolor='black', facecolor='white')
    ax.add_patch(rect)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)

    circle_b = patches.Circle((0, 0), r_b, linewidth=2, edgecolor='blue', facecolor='#ADD8E6', alpha=0.4)
    circle_p = patches.Circle((0, 0), r_p, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(circle_b)
    ax.add_patch(circle_p)

    ax.text(0, r_b + 0.3, f"{r_b}m", color='blue', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(0, r_p + 0.3, f"{r_p}m", color='blue', ha='center', va='bottom', fontsize=12, fontweight='bold')

    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        colors = df['dist'].apply(lambda d: 'red' if d <= r_b else ('blue' if d <= r_p else 'black'))
        ax.scatter(df['X'], df['Y'], c=colors, s=130, edgecolors='white', linewidths=1.5, zorder=5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf) 
    plt.close(fig)
    return img, limit

# --- 3. PDF ENGINE ---
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, txt="Golf Performance Report", ln=True, align='C')
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
            pdf.cell(190, 8, txt=f"{r}m: {tot} Shots | Birdies: {b} | Pars: {p}", ln=True)
            
    pdf_out = pdf.output()
    if isinstance(pdf_out, str): return pdf_out.encode('latin-1')
    return bytes(pdf_out)

# --- 4. NAVIGATION LOGIC ---
if 'page' not in st.session_state: st.session_state.page = "Home"
if 'active_t' not in st.session_state: st.session_state.active_t = None

st.sidebar.title("🧭 Menu")
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.page = "Home"
    st.session_state.data = load_data() # Refresh from database

if st.session_state.active_t:
    if st.sidebar.button(f"🎯 Edit: {st.session_state.active_t}", use_container_width=True):
        st.session_state.page = "Record"

if st.sidebar.button("🌐 Master Sheet", use_container_width=True):
    st.session_state.page = "Master Sheet"
    st.session_state.data = load_data()

if st.sidebar.button("📊 Stats & Analytics", use_container_width=True):
    st.session_state.page = "Stats"
    st.session_state.data = load_data()


# --- PAGE: HOME ---
if st.session_state.page == "Home":
    st.header("🏌️‍♂️ My Tournaments")
    with st.expander("➕ New Tournament"):
        t_name = st.text_input("Name:")
        if st.button("Create & Open"):
            if t_name:
                st.session_state.active_t = t_name
                st.session_state.page = "Record"
                st.rerun()

    st.divider()
    all_t = st.session_state.data['Tournament'].unique().tolist() if not st.session_state.data.empty else []
    
    if not all_t:
        st.info("No tournaments yet. Create one above.")
    else:
        for t in all_t:
            c1, c2 = st.columns([4, 1])
            if c1.button(f"⛳ {t}", use_container_width=True):
                st.session_state.active_t = t
                st.session_state.page = "Record"
                st.rerun()
            if c2.button("🗑️", key=f"del_{t}"):
                # Delete from Supabase Database
                supabase.table("shots").delete().eq("Tournament", t).execute()
                if st.session_state.active_t == t:
                    st.session_state.active_t = None
                st.session_state.data = load_data() # Refresh
                st.rerun()
                
    if not st.session_state.data.empty:
        st.divider()
        st.subheader("💾 Export Data")
        pdf_file = create_pdf(st.session_state.data)
        st.download_button("📄 Download PDF Report", data=pdf_file, file_name="golf_stats.pdf")

# --- PAGE: RECORD (TOUCH + SUPABASE) ---
elif st.session_state.page == "Record":
    st.button("⬅️ Back to Home", on_click=lambda: setattr(st.session_state, 'page', "Home"))
    st.title(f"Target: {st.session_state.active_t}")
    
    t_tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
    r_list = ["50-100", "101-150", "151-200"]
    
    for i, r_label in enumerate(r_list):
        with t_tabs[i]:
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == st.session_state.active_t) & (st.session_state.data['Range'] == r_label)]
            
            st.write("👆 **Tap inside the frame to record a shot.**")
            
            img_obj, limit = create_target_image(df_v, r_label)
            value = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
            
            if value is not None:
                px, py = value['x'], value['y']
                x_meters = round((px / 500.0) * (2 * limit) - limit, 2)
                y_meters = round(limit - (py / 500.0) * (2 * limit), 2)
                
                # Push instantly to Supabase Vault
                new_shot = {"Tournament": st.session_state.active_t, "Range": r_label, "X": x_meters, "Y": y_meters}
                try:
                    supabase.table("shots").insert(new_shot).execute()
                    # Refresh data
                    st.session_state.data = load_data()
                    st.rerun()
                except Exception as e:
                    st.error(f"🚨 THE EXACT DATABASE ERROR IS: {e}")
                    st.stop()
            
            if not df_v.empty and st.button(f"Undo Last Shot", key=f"un_{r_label}"):
                # Find the database ID of the very last shot and delete it
                last_id = int(df_v.iloc[-1]['id'])
                supabase.table("shots").delete().eq("id", last_id).execute()
                st.session_state.data = load_data()
                st.rerun()

# --- PAGE: MASTER SHEET ---
elif st.session_state.page == "Master Sheet":
    st.header("Master Accumulated Data")
    for r in ["50-100", "101-150", "151-200"]:
        st.subheader(f"Global {r}m Dispersion")
        img_obj, _ = create_target_image(st.session_state.data[st.session_state.data['Range'] == r], r)
        st.image(img_obj)

# --- PAGE: STATS ---
elif st.session_state.page == "Stats":
    st.header("Performance Stats")
    for r in ["50-100", "101-150", "151-200"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, rp = get_radii(r)
            tot = len(sub)
            b = len(sub[sub['d'] <= rb])
            p = len(sub[(sub['d'] > rb) & (sub['d'] <= rp)])
            st.write(f"**{r}m:** {tot} Shots | Birdies: {b} | Pars: {p} | Bogeys: {tot-(b+p)}")
