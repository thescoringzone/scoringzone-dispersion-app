import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from supabase import create_client, Client

# --- 1. APP CONFIG & SECRETS ---
st.set_page_config(page_title="Golf Dispersion Elite", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# Only loads data belonging to the current user
def load_data(current_user):
    response = supabase.table("shots").select("*").eq("User", current_user).execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame(columns=["id", "User", "Tournament", "Range", "X", "Y"])

# --- 2. USER PROFILE SIDEBAR ---
st.sidebar.title("👤 My Profile")
input_user = st.sidebar.text_input("Enter Your Name:", value="Guest").strip()
st.session_state.current_user = input_user if input_user else "Guest"

if 'data' not in st.session_state or st.session_state.get('last_user') != st.session_state.current_user:
    st.session_state.data = load_data(st.session_state.current_user)
    st.session_state.last_user = st.session_state.current_user

def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

# --- 3. THE VISUAL ENGINE (WITH RADIUS LABELS) ---
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

    # ADD RADIUS LABELS BACK IN
    ax.text(0, r_b + 0.2, f"{r_b}m", color='blue', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(0, r_p + 0.2, f"{r_p}m", color='blue', ha='center', va='bottom', fontsize=10, fontweight='bold')

    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        colors = df['dist'].apply(lambda d: 'red' if d <= r_b else ('blue' if d <= r_p else 'black'))
        ax.scatter(df['X'], df['Y'], c=colors, s=65, edgecolors='white', linewidths=1.5, zorder=5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf) 
    plt.close(fig)
    return img

# --- 4. SINGLE-PAGE PDF ENGINE ---
def create_one_page_pdf(df, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 8, txt=title, ln=True, align='C')
    pdf.ln(2)
    
    y_start = 25
    for r in ["50-100", "101-150", "151-200"]:
        sub = df[df['Range'] == r].copy()
        pdf.set_font("Arial", 'B', 10)
        pdf.set_xy(10, y_start)
        pdf.cell(100, 5, txt=f"Range: {r}m", ln=True)
        
        pdf.set_font("Arial", size=8)
        tot = len(sub)
        if tot > 0:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, rp = get_radii(r)
            b = len(sub[sub['d'] <= rb])
            p = len(sub[(sub['d'] > rb) & (sub['d'] <= rp)])
            misses = sub[sub['d'] > rb]
            ll = len(misses[(misses['X'] < 0) & (misses['Y'] > 0)])
            lr = len(misses[(misses['X'] >= 0) & (misses['Y'] > 0)])
            sl = len(misses[(misses['X'] < 0) & (misses['Y'] <= 0)])
            sr = len(misses[(misses['X'] >= 0) & (misses['Y'] <= 0)])
            stats_text = f"Shots: {tot} | Birdies: {b} | Pars: {p} | SL: {(sl/tot)*100:.0f}% LL: {(ll/tot)*100:.0f}% SR: {(sr/tot)*100:.0f}% LR: {(lr/tot)*100:.0f}%"
        else:
            stats_text = "No shots recorded."
            
        pdf.cell(190, 4, txt=stats_text, ln=True)
        img = create_target_image(sub, r)
        temp_fn = f"temp_{r}.png"
        img.save(temp_fn)
        pdf.image(temp_fn, x=70, y=pdf.get_y(), w=60)
        
        if os.path.exists(temp_fn): os.remove(temp_fn)
        y_start += 85 
        
    return bytes(pdf.output())

# --- 5. NAVIGATION ---
if 'page' not in st.session_state: st.session_state.page = "Home"
if 'active_t' not in st.session_state: st.session_state.active_t = None

st.sidebar.divider()
st.sidebar.title("🧭 Menu")
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.page = "Home"
    st.session_state.data = load_data(st.session_state.current_user)

if st.session_state.active_t:
    if st.sidebar.button(f"🎯 Edit: {st.session_state.active_t}", use_container_width=True):
        st.session_state.page = "Record"

if st.sidebar.button("📊 Stats & Master", use_container_width=True):
    st.session_state.page = "Stats"
    st.session_state.data = load_data(st.session_state.current_user)

# --- PAGE: HOME ---
if st.session_state.page == "Home":
    st.header(f"🏌️‍♂️ {st.session_state.current_user}'s Tournaments")
    with st.expander("➕ New Tournament"):
        t_name = st.text_input("Name:")
        if st.button("Create & Open"):
            if t_name:
                st.session_state.active_t = t_name
                st.session_state.page = "Record"
                st.rerun()
    st.divider()
    all_t = st.session_state.data['Tournament'].unique().tolist() if not st.session_state.data.empty else []
    for t in all_t:
        c1, c2 = st.columns([4, 1])
        if c1.button(f"⛳ {t}", use_container_width=True):
            st.session_state.active_t = t
            st.session_state.page = "Record"
            st.rerun()
        if c2.button("🗑️", key=f"del_{t}"):
            supabase.table("shots").delete().eq("Tournament", t).eq("User", st.session_state.current_user).execute()
            st.session_state.data = load_data(st.session_state.current_user)
            st.rerun()

# --- PAGE: RECORD ---
elif st.session_state.page == "Record":
    st.button("⬅️ Back", on_click=lambda: setattr(st.session_state, 'page', "Home"))
    st.title(f"Target: {st.session_state.active_t}")
    st.caption(f"Recording as: **{st.session_state.current_user}**")
    
    t_tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
    for i, r_label in enumerate(["50-100", "101-150", "151-200"]):
        with t_tabs[i]:
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == st.session_state.active_t) & (st.session_state.data['Range'] == r_label)]
            img_obj = create_target_image(df_v, r_label)
            value = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
            if value:
                px, py = value['x'], value['y']
                _, limit = get_radii(r_label); limit += 2
                x_m = round((px / 500.0) * (2 * limit) - limit, 2)
                y_m = round(limit - (py / 500.0) * (2 * limit), 2)
                supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                st.session_state.data = load_data(st.session_state.current_user); st.rerun()
            if not df_v.empty and st.button(f"Undo", key=f"un_{r_label}"):
                supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                st.session_state.data = load_data(st.session_state.current_user); st.rerun()
    
    st.divider()
    t_df = st.session_state.data[st.session_state.data['Tournament'] == st.session_state.active_t]
    if not t_df.empty:
        st.download_button("📄 Download 1-Page Tournament Report", 
                           data=create_one_page_pdf(t_df, f"Report: {st.session_state.active_t} ({st.session_state.current_user})"), 
                           file_name=f"{st.session_state.active_t}_report.pdf", 
                           on_click=lambda: st.toast("Download starting..."))

# --- PAGE: STATS ---
elif st.session_state.page == "Stats":
    st.header(f"Analytics for {st.session_state.current_user}")
    for r in ["50-100", "101-150", "151-200"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            st.subheader(f"⛳ {r}m Range")
            st.image(create_target_image(sub, r))
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, _ = get_radii(r); tot = len(sub); b = len(sub[sub['d'] <= rb])
            misses = sub[sub['d'] > rb]
            ll = len(misses[(misses['X'] < 0) & (misses['Y'] > 0)])
            lr = len(misses[(misses['X'] >= 0) & (misses['Y'] > 0)])
            sl = len(misses[(misses['X'] < 0) & (misses['Y'] <= 0)])
            sr = len(misses[(misses['X'] >= 0) & (misses['Y'] <= 0)])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Short Left", f"{(sl/tot)*100:.0f}%")
            c2.metric("Long Left", f"{(ll/tot)*100:.0f}%")
            c3.metric("Short Right", f"{(sr/tot)*100:.0f}%")
            c4.metric("Long Right", f"{(lr/tot)*100:.0f}%")
            st.divider()

    if not st.session_state.data.empty:
        st.download_button("📄 Download 1-Page Master Report", 
                           data=create_one_page_pdf(st.session_state.data, f"Master Performance: {st.session_state.current_user}"), 
                           file_name="master_report.pdf",
                           on_click=lambda: st.toast("Download starting..."))
