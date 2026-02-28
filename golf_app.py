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

def load_data(current_user):
    response = supabase.table("shots").select("*").eq("User", current_user).execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame(columns=["id", "User", "Tournament", "Range", "X", "Y"])

# --- 2. USER PROFILE SIDEBAR ---
# st.sidebar.image("logo.png") # Uncomment if you uploaded your logo!
st.sidebar.title("👤 My Profile")
input_user = st.sidebar.text_input("Enter Your Name:", value="Guest").strip()
st.session_state.current_user = input_user if input_user else "Guest"

if 'data' not in st.session_state or st.session_state.get('last_user') != st.session_state.current_user:
    st.session_state.data = load_data(st.session_state.current_user)
    st.session_state.last_user = st.session_state.current_user

# --- 3. APPROACH VISUAL ENGINE ---
def get_radii(label):
    if "50-100" in label: return 3, 6
    if "101-150" in label: return 4, 8
    return 5, 10

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

# --- 4. OFF THE TEE VISUAL ENGINE ---
def create_tee_image(df_filtered, label):
    # Updated label matching
    y_min, y_max = (270, 320) if label == "OTT: Driver" else (220, 270)
    x_limit = 30 
    
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    
    rect = patches.Rectangle((-x_limit, y_min), x_limit*2, y_max-y_min, linewidth=4, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    ax.axvspan(-10, 10, facecolor='#ADD8E6', alpha=0.4)
    
    ax.axvline(0, color='blue', linestyle='solid', linewidth=2)
    ax.axvline(-10, color='blue', linestyle='dashed', linewidth=2)
    ax.axvline(10, color='blue', linestyle='dashed', linewidth=2)
    ax.axvline(-20, color='blue', linestyle='dotted', linewidth=2)
    ax.axvline(20, color='blue', linestyle='dotted', linewidth=2)
    
    for y in range(y_min, y_max, 10):
        ax.axhline(y, color='gray', linestyle='--', alpha=0.4)
        ax.text(-29, y+0.5, f"{y}m", color='gray', fontsize=8)
        
    label_y = y_max - 2
    ax.text(0, label_y, "Centre", color='blue', ha='center', fontweight='bold')
    ax.text(-10, label_y, "10m", color='blue', ha='center', fontweight='bold')
    ax.text(10, label_y, "10m", color='blue', ha='center', fontweight='bold')
    ax.text(-20, label_y, "20m", color='blue', ha='center', fontweight='bold')
    ax.text(20, label_y, "20m", color='blue', ha='center', fontweight='bold')
    
    if not df_filtered.empty:
        df = df_filtered.copy()
        df['dist_x'] = df['X'].abs()
        colors = df['dist_x'].apply(lambda d: 'red' if d <= 10 else ('blue' if d <= 20 else 'black'))
        ax.scatter(df['X'], df['Y'], c=colors, s=65, edgecolors='white', linewidths=1.5, zorder=5)
        
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

# --- 5. PDF ENGINES ---
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
            bog = tot - (b + p)
            to_par_score = (b * -1) + (bog * 1)
            sign = "+" if to_par_score > 0 else ""
            to_par_text = f"To Par: {sign}{to_par_score} per {tot} shots"
            
            misses = sub[sub['d'] > rb]
            ll = len(misses[(misses['X'] < 0) & (misses['Y'] > 0)])
            lr = len(misses[(misses['X'] >= 0) & (misses['Y'] > 0)])
            sl = len(misses[(misses['X'] < 0) & (misses['Y'] <= 0)])
            sr = len(misses[(misses['X'] >= 0) & (misses['Y'] <= 0)])
            stats_text = f"Shots: {tot} | Birdies: {b} | Pars: {p} | {to_par_text}"
            miss_text = f"SL: {(sl/tot)*100:.0f}% LL: {(ll/tot)*100:.0f}% SR: {(sr/tot)*100:.0f}% LR: {(lr/tot)*100:.0f}%"
            pdf.cell(190, 4, txt=stats_text, ln=True)
            pdf.cell(190, 4, txt=miss_text, ln=True)
        else:
            pdf.cell(190, 4, txt="No shots recorded.", ln=True)
            
        img = create_target_image(sub, r)
        temp_fn = f"temp_{r}.png"
        img.save(temp_fn)
        pdf.image(temp_fn, x=70, y=pdf.get_y(), w=55)
        if os.path.exists(temp_fn): os.remove(temp_fn)
        y_start += 85 
    return bytes(pdf.output())

def create_tee_pdf(df, title):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 8, txt=title, ln=True, align='C')
    pdf.ln(5)
    
    y_start = 25
    for r in ["OTT: Driver", "OTT: Others"]:
        sub = df[df['Range'] == r].copy()
        pdf.set_font("Arial", 'B', 12)
        pdf.set_xy(10, y_start)
        pdf.cell(100, 6, txt=f"{r}", ln=True)
        
        pdf.set_font("Arial", size=9)
        tot = len(sub)
        if tot > 0:
            sub['dx'] = sub['X'].abs()
            in_10 = len(sub[sub['dx'] <= 10])
            in_20 = len(sub[(sub['dx'] > 10) & (sub['dx'] <= 20)])
            out_20 = len(sub[sub['dx'] > 20])
            avg_dist = sub['Y'].mean()
            
            stats_txt = f"Shots: {tot} | Avg Dist: {avg_dist:.1f}m | <10m: {(in_10/tot)*100:.0f}% | 10-20m: {(in_20/tot)*100:.0f}% | 20m+: {(out_20/tot)*100:.0f}%"
            pdf.cell(190, 5, txt=stats_txt, ln=True)
        else:
            pdf.cell(190, 5, txt="No shots recorded.", ln=True)
            
        img = create_tee_image(sub, r)
        temp_fn = f"temp_{r}.png"
        img.save(temp_fn)
        pdf.image(temp_fn, x=65, y=pdf.get_y()+2, w=80) 
        
        if os.path.exists(temp_fn): os.remove(temp_fn)
        y_start += 125 
        
    return bytes(pdf.output())

# --- 6. NAVIGATION ---
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
    
    # UPDATED TABS
    tabs_list = ["50-100m", "101-150m", "151-200m", "OTT: Driver", "OTT: Others"]
    t_tabs = st.tabs(tabs_list)
    
    for i, r_label in enumerate(tabs_list):
        with t_tabs[i]:
            df_v = st.session_state.data[(st.session_state.data['Tournament'] == st.session_state.active_t) & (st.session_state.data['Range'] == r_label)]
            
            # --- TEE LOGIC ---
            if r_label in ["OTT: Driver", "OTT: Others"]:
                img_obj = create_tee_image(df_v, r_label)
                value = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
                if value:
                    px, py = value['x'], value['y']
                    y_min, y_max = (270, 320) if r_label == "OTT: Driver" else (220, 270)
                    x_m = round((px / 500.0) * 60 - 30, 2)
                    y_m = round(y_max - (py / 500.0) * 50, 2)
                    supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                    st.session_state.data = load_data(st.session_state.current_user); st.rerun()
                
                # Tee Stats with Average Distance
                if not df_v.empty:
                    tot = len(df_v)
                    dx = df_v['X'].abs()
                    in_10 = len(df_v[dx <= 10])
                    in_20 = len(df_v[(dx > 10) & (dx <= 20)])
                    out_20 = len(df_v[dx > 20])
                    avg_dist = df_v['Y'].mean()
                    st.subheader(f"Avg Dist: {avg_dist:.1f}m | <10m: {(in_10/tot)*100:.0f}% | 10-20m: {(in_20/tot)*100:.0f}% | 20m+: {(out_20/tot)*100:.0f}%")
            
            # --- APPROACH LOGIC ---
            else:
                img_obj = create_target_image(df_v, r_label)
                value = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
                if value:
                    px, py = value['x'], value['y']
                    _, limit = get_radii(r_label); limit += 2
                    x_m = round((px / 500.0) * (2 * limit) - limit, 2)
                    y_m = round(limit - (py / 500.0) * (2 * limit), 2)
                    supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                    st.session_state.data = load_data(st.session_state.current_user); st.rerun()
                
                if not df_v.empty:
                    df_v['d'] = np.sqrt(df_v['X']**2 + df_v['Y']**2)
                    rb, rp = get_radii(r_label)
                    b = len(df_v[df_v['d'] <= rb])
                    bog = len(df_v[df_v['d'] > rp])
                    tot = len(df_v)
                    to_par = (b * -1) + (bog * 1)
                    sign = "+" if to_par > 0 else ""
                    st.subheader(f"To Par: {sign}{to_par} per {tot} shots")

            if not df_v.empty and st.button(f"Undo", key=f"un_{r_label}"):
                supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                st.session_state.data = load_data(st.session_state.current_user); st.rerun()
                
    st.divider()
    t_df = st.session_state.data[st.session_state.data['Tournament'] == st.session_state.active_t]
    if not t_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Download Approach Report", data=create_one_page_pdf(t_df, f"Approach: {st.session_state.active_t}"), file_name=f"{st.session_state.active_t}_approach.pdf")
        with col2:
            st.download_button("📄 Download Off the Tee Report", data=create_tee_pdf(t_df, f"Off the Tee: {st.session_state.active_t}"), file_name=f"{st.session_state.active_t}_tee.pdf")

# --- PAGE: STATS ---
elif st.session_state.page == "Stats":
    st.header(f"Master Analytics: {st.session_state.current_user}")
    
    st.subheader("🎯 Approach Performance")
    for r in ["50-100", "101-150", "151-200"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            st.markdown(f"**⛳ {r}m Range**")
            st.image(create_target_image(sub, r))
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, rp = get_radii(r); tot = len(sub); b = len(sub[sub['d'] <= rb]); bog = len(sub[sub['d'] > rp])
            to_par = (b * -1) + (bog * 1)
            sign = "+" if to_par > 0 else ""
            st.write(f"**To Par: {sign}{to_par} per {tot} shots**")
            
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
            
    st.subheader("🚀 Off the Tee Performance")
    for r in ["OTT: Driver", "OTT: Others"]:
        sub = st.session_state.data[st.session_state.data['Range'] == r].copy()
        if not sub.empty:
            st.markdown(f"**🏌️‍♂️ {r}**")
            st.image(create_tee_image(sub, r))
            tot = len(sub)
            dx = sub['X'].abs()
            in_10 = len(sub[dx <= 10])
            in_20 = len(sub[(dx > 10) & (dx <= 20)])
            out_20 = len(sub[dx > 20])
            avg_dist = sub['Y'].mean()
            
            # Added a 4th column for Average Distance
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Distance", f"{avg_dist:.1f}m")
            c2.metric("Within 10m", f"{(in_10/tot)*100:.0f}%")
            c3.metric("10-20m", f"{(in_20/tot)*100:.0f}%")
            c4.metric("20m+", f"{(out_20/tot)*100:.0f}%")
            st.divider()

    if not st.session_state.data.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Master Approach Report", data=create_one_page_pdf(st.session_state.data, "Master Approach Report"), file_name="master_approach.pdf")
        with col2:
            st.download_button("📄 Master Off the Tee Report", data=create_tee_pdf(st.session_state.data, "Master Off the Tee Report"), file_name="master_tee.pdf")
