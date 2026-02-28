import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# --- 1. APP CONFIG ---
st.set_page_config(page_title="Golf Dispersion Elite", layout="wide")

# --- 2. LOCAL MEMORY STORAGE ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Tournament", "Range", "X", "Y"])

def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

# --- 3. THE VISUAL ENGINE (IMAGE BASED TOUCH) ---
def create_target_image(df_filtered, label):
    r_b, r_p = get_radii(label)
    limit = r_p + 2 

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]) 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axis('off') 

    # 1. The Rectangle Frame
    rect = patches.Rectangle((-limit, -limit), limit*2, limit*2, linewidth=4, edgecolor='black', facecolor='white')
    ax.add_patch(rect)

    # 2. Crosshairs
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)

    # 3. Birdie & Par Circles
    circle_b = patches.Circle((0, 0), r_b, linewidth=2, edgecolor='blue', facecolor='#ADD8E6', alpha=0.4)
    circle_p = patches.Circle((0, 0), r_p, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(circle_b)
    ax.add_patch(circle_p)

    # 4. Metric Labels
    ax.text(0, r_b + 0.3, f"{r_b}m", color='blue', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(0, r_p + 0.3, f"{r_p}m", color='blue', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 5. Plot Recorded Shots
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

# --- 4. PDF ENGINE ---
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
    return pdf.output()

# --- 5. NAVIGATION LOGIC ---
if 'page' not in st.session_state: st.session_state.page = "Home"
if 'active_t' not in st.session_state: st.session_state.active_t = None

menu = st.sidebar.radio("Navigation", ["Home", "Master Sheet", "Stats"])
if menu != "Home": st.session_state.page = menu

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
    
    # LOAD DATA COMPONENT
    st.subheader("📂 Load Previous Data")
    uploaded_file = st.file_uploader("Upload your saved CSV backup:", type="csv")
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Master Sheet Restored successfully!")
        
    st.divider()
    
    all_t = st.session_state.data['Tournament'].unique().tolist()
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
                st.session_state.data = st.session_state.data[st.session_state.data['Tournament'] != t]
                st.rerun()
                
    if not st.session_state.data.empty:
        st.divider()
        st.subheader("💾 Backup Your Data")
        st.info("⚠️ Data resets if you close the app. Tap Download CSV before you leave!")
        
        col1, col2 = st.columns(2)
        pdf_file = create_pdf(st.session_state.data)
        col1.download_button("📄 Download PDF", data=pdf_file, file_name="golf_stats.pdf")
        
        csv_file = st.session_state.data.to_csv(index=False).encode('utf-8')
        col2.download_button("📊 Download CSV", data=csv_file, file_name="golf_data_backup.csv")

# --- PAGE: RECORD (BULLETPROOF TOUCH) ---
elif st.session_state.page == "Record":
    st.button("⬅️ Home List", on_click=lambda: setattr(st.session_state, 'page', "Home"))
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
                
                new_row = pd.DataFrame([{"Tournament": st.session_state.active_t, "Range": r_label, "X": x_meters, "Y": y_meters}])
                st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                st.rerun()
            
            if not df_v.empty and st.button(f"Undo Last Shot", key=f"un_{r_label}"):
                st.session_state.data = st.session_state.data.drop(df_v.index[-1])
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
