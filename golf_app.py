import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from supabase import create_client, Client

# --- 1. APP CONFIG & SECRETS ---
st.set_page_config(page_title="ECGA Elite Tracker", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# --- 2. DATA LOADING ---
def load_shots(current_user):
    response = supabase.table("shots").select("*").eq("User", current_user).execute()
    if response.data:
        return pd.DataFrame(response.data)
    else:
        return pd.DataFrame(columns=["id", "User", "Tournament", "Round", "Range", "X", "Y"])

def load_round_stats(current_user, tournament, round_num):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).eq("tournament", tournament).eq("round_num", round_num).execute()
    if response.data:
        return response.data[0]
    else:
        # Create a blank record if it doesn't exist yet
        blank = {
            "user_name": current_user, "tournament": tournament, "round_num": round_num,
            "gir": 0, "gir_less_5": 0, "sg_total": 0, "sg_inside_6": 0, "sg_inside_3": 0, "sg_ud": 0, "sgz_score": 0
        }
        supabase.table("round_stats").insert(blank).execute()
        return blank

# --- 3. STATE MANAGEMENT ---
st.sidebar.title("👤 Player Profile")
input_user = st.sidebar.text_input("Username:", value="Guest").strip()
st.session_state.current_user = input_user if input_user else "Guest"

if 'shots_data' not in st.session_state or st.session_state.get('last_user') != st.session_state.current_user:
    st.session_state.shots_data = load_shots(st.session_state.current_user)
    st.session_state.last_user = st.session_state.current_user

if 'active_t' not in st.session_state: st.session_state.active_t = None
if 'active_r' not in st.session_state: st.session_state.active_r = "Round 1"
if 'workflow_step' not in st.session_state: st.session_state.workflow_step = "Driving"

# --- 4. VISUAL ENGINES ---
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

def create_tee_image(df_filtered, label):
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

# --- 5. NAVIGATION LOGIC ---
st.sidebar.divider()
st.sidebar.header("Tournament Setup")
with st.sidebar.expander("➕ New Tournament"):
    t_name = st.text_input("Name:")
    if st.button("Create & Open"):
        if t_name:
            st.session_state.active_t = t_name
            st.session_state.active_r = "Round 1"
            st.rerun()

all_t = st.session_state.shots_data['Tournament'].unique().tolist() if not st.session_state.shots_data.empty else []
t_select = st.sidebar.selectbox("Select Tournament:", ["None"] + all_t, index=0 if st.session_state.active_t is None else all_t.index(st.session_state.active_t) + 1)
if t_select != "None" and t_select != st.session_state.active_t:
    st.session_state.active_t = t_select
    st.session_state.active_r = "Round 1"
    st.rerun()

if st.session_state.active_t:
    st.sidebar.divider()
    st.sidebar.header("Select Round")
    r_select = st.sidebar.radio("Round:", ["Round 1", "Round 2", "Round 3", "Round 4"], index=["Round 1", "Round 2", "Round 3", "Round 4"].index(st.session_state.active_r))
    if r_select != st.session_state.active_r:
        st.session_state.active_r = r_select
        st.rerun()

# --- 6. MAIN WORKFLOW ---
if st.session_state.active_t:
    st.title(f"{st.session_state.active_t} - {st.session_state.active_r}")
    
    # Workflow Navigation
    steps = ["Driving", "Scoring Zone", "Short Game"] # Putting & Mental coming in Phase 2
    selected_step = st.radio("Input Phase:", steps, horizontal=True, index=steps.index(st.session_state.workflow_step))
    if selected_step != st.session_state.workflow_step:
        st.session_state.workflow_step = selected_step
        st.rerun()
    st.divider()

    # Get current round stats
    current_stats = load_round_stats(st.session_state.current_user, st.session_state.active_t, st.session_state.active_r)

    # --- PHASE: DRIVING ---
    if st.session_state.workflow_step == "Driving":
        t_tabs = st.tabs(["OTT: Driver", "OTT: Others"])
        for i, r_label in enumerate(["OTT: Driver", "OTT: Others"]):
            with t_tabs[i]:
                df_v = st.session_state.shots_data[(st.session_state.shots_data['Tournament'] == st.session_state.active_t) & (st.session_state.shots_data['Round'] == st.session_state.active_r) & (st.session_state.shots_data['Range'] == r_label)]
                img_obj = create_tee_image(df_v, r_label)
                value = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
                if value:
                    px, py = value['x'], value['y']
                    y_min, y_max = (270, 320) if r_label == "OTT: Driver" else (220, 270)
                    x_m = round((px / 500.0) * 60 - 30, 2)
                    y_m = round(y_max - (py / 500.0) * 50, 2)
                    supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Round": st.session_state.active_r, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                    st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()
                
                if not df_v.empty:
                    tot = len(df_v)
                    dx = df_v['X'].abs()
                    fwys = len(df_v[dx <= 10])
                    pens = len(df_v[dx > 20])
                    st.success(f"**Tournament Sheet Stat:** {(fwys/tot)*100:.0f}% ({pens})") # e.g., 75% (1)
                    if st.button(f"Undo Last", key=f"un_{r_label}"):
                        supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                        st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()

    # --- PHASE: SCORING ZONE ---
    elif st.session_state.workflow_step == "Scoring Zone":
        t_tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
        for i, r_label in enumerate(["50-100", "101-150", "151-200"]):
            with t_tabs[i]:
                df_v = st.session_state.shots_data[(st.session_state.shots_data['Tournament'] == st.session_state.active_t) & (st.session_state.shots_data['Round'] == st.session_state.active_r) & (st.session_state.shots_data['Range'] == r_label)]
                img_obj = create_target_image(df_v, r_label)
                value = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
                if value:
                    px, py = value['x'], value['y']
                    _, limit = get_radii(r_label); limit += 2
                    x_m = round((px / 500.0) * (2 * limit) - limit, 2)
                    y_m = round(limit - (py / 500.0) * (2 * limit), 2)
                    supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Round": st.session_state.active_r, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                    st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()
                
                if not df_v.empty:
                    df_v['d'] = np.sqrt(df_v['X']**2 + df_v['Y']**2)
                    rb, rp = get_radii(r_label)
                    b = len(df_v[df_v['d'] <= rb])
                    bog = len(df_v[df_v['d'] > rp])
                    tot = len(df_v)
                    to_par = (b * -1) + (bog * 1)
                    st.info(f"**Tournament Sheet Stat:** {to_par}({tot})") # e.g., -1(3)
                    if st.button(f"Undo Last", key=f"un_{r_label}"):
                        supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                        st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()
        
        st.divider()
        st.subheader("Manual Inputs (Scoring Zone)")
        col1, col2 = st.columns(2)
        with col1:
            gir_5_val = st.number_input("GIR < 5", min_value=0, max_value=18, value=current_stats.get('gir_less_5', 0))
        with col2:
            gir_val = st.number_input("Total GIR", min_value=0, max_value=18, value=current_stats.get('gir', 0))
        
        if st.button("Save GIR Stats"):
            supabase.table("round_stats").update({"gir": gir_val, "gir_less_5": gir_5_val}).eq("id", current_stats['id']).execute()
            st.success("Saved successfully!")

    # --- PHASE: SHORT GAME ---
    elif st.session_state.workflow_step == "Short Game":
        st.subheader("Short Game (SG)")
        st.caption("Set your total SG shots first. This denominator applies to all categories below.")
        
        # The Master Denominator
        sg_total = st.number_input("Total SG Shots (#)", min_value=0, value=current_stats.get('sg_total', 0))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Successes:**")
            sg_6 = st.number_input("< 6ft (Shots inside 6ft)", min_value=0, max_value=sg_total, value=current_stats.get('sg_inside_6', 0))
            sg_3 = st.number_input("< 3ft (Shots inside 3ft)", min_value=0, max_value=sg_total, value=current_stats.get('sg_inside_3', 0))
            sg_ud = st.number_input("U&D (Up & Downs)", min_value=0, max_value=sg_total, value=current_stats.get('sg_ud', 0))
            sgz = st.number_input("SGZ Score (Relative to Par)", value=current_stats.get('sgz_score', 0))
        
        with col2:
            st.markdown("**Tournament Sheet Output:**")
            if sg_total > 0:
                st.write(f"**< 6:** {(sg_6/sg_total)*100:.0f}%")
                st.write(f"**< 3:** {(sg_3/sg_total)*100:.0f}%")
                st.write(f"**U&D:** {(sg_ud/sg_total)*100:.0f}%")
                st.write(f"**SGZ:** {sgz}({sg_total})")
            else:
                st.write("Enter Total SG Shots to see percentages.")

        if st.button("Save Short Game Stats"):
            update_data = {
                "sg_total": sg_total, "sg_inside_6": sg_6, "sg_inside_3": sg_3, "sg_ud": sg_ud, "sgz_score": sgz
            }
            supabase.table("round_stats").update(update_data).eq("id", current_stats['id']).execute()
            st.success("Saved successfully!")

else:
    st.info("Create or select a tournament in the sidebar to begin.")
