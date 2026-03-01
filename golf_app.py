import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os
import json
from fpdf import FPDF
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

pga_putts_baseline = {
    1: 1.00, 2: 1.01, 3: 1.04, 4: 1.13, 5: 1.23, 6: 1.34, 7: 1.43, 8: 1.50, 
    9: 1.56, 10: 1.61, 15: 1.78, 20: 1.87, 25: 1.94, 30: 2.01, 40: 2.13, 50: 2.26, 60: 2.38,
    70: 2.48, 80: 2.58, 90: 2.65, 100: 2.71
}

def get_expected_putts(distance):
    closest_dist = min(pga_putts_baseline.keys(), key=lambda k: abs(k - distance))
    return pga_putts_baseline[closest_dist]

# --- 2. DATA LOADING ---
def load_shots(current_user):
    response = supabase.table("shots").select("*").eq("User", current_user).execute()
    if response.data:
        return pd.DataFrame(response.data)
    return pd.DataFrame(columns=["id", "User", "Tournament", "Round", "Range", "X", "Y"])

def load_round_stats(current_user, tournament, round_num):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).eq("tournament", tournament).eq("round_num", round_num).execute()
    if response.data:
        return response.data[0]
    blank = {
        "user_name": current_user, "tournament": tournament, "round_num": round_num,
        "gir": 0, "gir_less_5": 0, "sg_total": 0, "sg_inside_6": 0, "sg_inside_3": 0, "sg_ud": 0, "sgz_score": 0,
        "putts_total": 0, "sg_putting": 0.0, "lag_success": 0, "lag_total": 0, "mental_score": 0, "judgement_score": 0,
        "putting_holes": None
    }
    supabase.table("round_stats").insert(blank).execute()
    return blank

def load_all_tournament_stats(current_user, tournament):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).eq("tournament", tournament).execute()
    return response.data if response.data else []

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

# --- 5. DATA AGGREGATION ENGINE ---
def build_master_dataframe(df_shots, list_stats):
    rounds = ["Round 1", "Round 2", "Round 3", "Round 4"]
    r_stats = {s['round_num']: s for s in list_stats}
    data = []

    def add_section_header(title):
        data.append({
            "Category": title,
            "Round 1": "", "Round 2": "", "Round 3": "", "Round 4": "", "AV / TOTAL": ""
        })

    def add_row(category, logic_type, param=""):
        row_dict = {"Category": category}
        raw_sums = {'num': 0, 'den': 0, 'extra': 0}
        
        for r_name in rounds:
            cell_txt = "-"
            df_r = df_shots[(df_shots['Round'] == r_name)]
            stat_r = r_stats.get(r_name, {})
            
            if logic_type == "driving":
                df_d = df_r[df_r['Range'] == param]
                tot = len(df_d)
                if tot > 0:
                    fwys = len(df_d[df_d['X'].abs() <= 10])
                    pens = len(df_d[df_d['X'].abs() > 20])
                    cell_txt = f"{(fwys/tot)*100:.0f}% ({pens})"
                    raw_sums['num'] += fwys; raw_sums['den'] += tot; raw_sums['extra'] += pens
            elif logic_type == "approach":
                df_a = df_r[df_r['Range'] == param]
                tot = len(df_a)
                if tot > 0:
                    df_a['d'] = np.sqrt(df_a['X']**2 + df_a['Y']**2)
                    rb, rp = get_radii(param)
                    b = len(df_a[df_a['d'] <= rb])
                    bog = len(df_a[df_a['d'] > rp])
                    to_par = (b * -1) + (bog * 1)
                    sign = "+" if to_par > 0 else ""
                    cell_txt = f"{sign}{to_par}({tot})"
                    raw_sums['num'] += to_par; raw_sums['den'] += tot
            elif logic_type == "abs":
                val = stat_r.get(param, 0)
                if val != 0: cell_txt = str(val); raw_sums['num'] += val; raw_sums['den'] += 1
            elif logic_type == "sg_perc":
                num = stat_r.get(param, 0); den = stat_r.get('sg_total', 0)
                if den > 0:
                    cell_txt = f"{(num/den)*100:.0f}%"
                    raw_sums['num'] += num; raw_sums['den'] += den
            elif logic_type == "sgz":
                sgz = stat_r.get('sgz_score', 0); den = stat_r.get('sg_total', 0)
                if den > 0:
                    cell_txt = f"{sgz}({den})"
                    raw_sums['num'] += sgz; raw_sums['den'] += den
            elif logic_type == "lag":
                num = stat_r.get('lag_success', 0); den = stat_r.get('lag_total', 0)
                if den > 0:
                    cell_txt = f"{(num/den)*100:.0f}%"
                    raw_sums['num'] += num; raw_sums['den'] += den
            elif logic_type == "sg_putt":
                val = stat_r.get('sg_putting', 0.0)
                if val != 0:
                    sign = "+" if val > 0 else ""
                    cell_txt = f"{sign}{val:.2f}"
                    raw_sums['num'] += val; raw_sums['den'] += 1
            
            row_dict[r_name] = cell_txt
            
        av_txt = "-"
        if raw_sums['den'] > 0:
            if logic_type == "driving": av_txt = f"{(raw_sums['num']/raw_sums['den'])*100:.0f}% ({raw_sums['extra']})"
            elif logic_type == "approach":
                sign = "+" if raw_sums['num'] > 0 else ""
                av_txt = f"{sign}{raw_sums['num']}({raw_sums['den']})"
            elif logic_type in ["abs", "sg_putt"]: 
                val = raw_sums['num'] / raw_sums['den']
                if logic_type == "sg_putt":
                    sign = "+" if val > 0 else ""
                    av_txt = f"{sign}{val:.2f}"
                else: av_txt = f"{val:.1f}"
            elif logic_type in ["sg_perc", "lag"]: av_txt = f"{(raw_sums['num']/raw_sums['den'])*100:.0f}%"
            elif logic_type == "sgz": av_txt = f"{raw_sums['num']}({raw_sums['den']})"
            
        row_dict["AV / TOTAL"] = av_txt
        data.append(row_dict)

    add_section_header("LONG GAME")
    add_row("OTT: Driver", "driving", "OTT: Driver")
    add_row("OTT: Others", "driving", "OTT: Others")
    
    add_section_header("SCORING ZONE")
    add_row("151-200m", "approach", "151-200")
    add_row("101-150m", "approach", "101-150")
    add_row("50-100m", "approach", "50-100")
    add_row("GIR / 5", "abs", "gir_less_5")
    add_row("GIR", "abs", "gir")
    
    add_section_header("SHORT GAME")
    add_row("< 6", "sg_perc", "sg_inside_6")
    add_row("< 3", "sg_perc", "sg_inside_3")
    add_row("U&D", "sg_perc", "sg_ud")
    add_row("SGZ", "sgz")
    
    add_section_header("PUTTING")
    add_row("Putts (#)", "abs", "putts_total")  # Replaced massive # markdown
    add_row("SG Putting", "sg_putt")
    add_row("Lag", "lag")
    
    add_section_header("MENTAL & JUDGEMENTS")
    add_row("M", "abs", "mental_score")
    add_row("J", "abs", "judgement_score")

    return pd.DataFrame(data)

# --- 6. ECGA 2-PAGE PDF GENERATOR ---
def create_ecga_pdf(tournament, df_master, df_shots):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)
    
    # === PAGE 1: MASTER TABLE ===
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"ECGA Tournament Overview: {tournament}", ln=True, align='C')
    pdf.ln(5)
    
    col_w = [40, 42, 42, 42, 42, 42] 
    row_h = 7 
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(200, 220, 255)
    headers = ["Category", "Round 1", "Round 2", "Round 3", "Round 4", "AV / TOTAL"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], row_h, txt=h, border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font("Arial", '', 10)
    
    for index, row in df_master.iterrows():
        cat = row['Category']
        
        # If it's a section header (Round 1 is blank), print a shaded divider row
        if row['Round 1'] == "":
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(sum(col_w), row_h, txt=cat, border=1, fill=True, ln=True, align='L')
            pdf.set_font("Arial", '', 10) # Reset font
        else:
            pdf.cell(col_w[0], row_h, txt=cat, border=1)
            pdf.cell(col_w[1], row_h, txt=str(row['Round 1']), border=1, align='C')
            pdf.cell(col_w[2], row_h, txt=str(row['Round 2']), border=1, align='C')
            pdf.cell(col_w[3], row_h, txt=str(row['Round 3']), border=1, align='C')
            pdf.cell(col_w[4], row_h, txt=str(row['Round 4']), border=1, align='C')
            pdf.cell(col_w[5], row_h, txt=str(row['AV / TOTAL']), border=1, align='C')
            pdf.ln()

    # === PAGE 2: DISPERSION CHARTS ===
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"Tournament Dispersion Charts: {tournament}", ln=True, align='C')
    pdf.ln(2)

    y_start = 22
    x_offsets = [10, 105, 200]
    ranges_sz = ["50-100", "101-150", "151-200"]
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Scoring Zone (Approach)", ln=True, align='L')
    pdf.set_font("Arial", '', 9)

    for i, r in enumerate(ranges_sz):
        sub = df_shots[df_shots['Range'] == r].copy()
        tot = len(sub)
        stats_txt1 = f"Range: {r}m | Shots: {tot}"
        stats_txt2 = "No shots recorded."
        
        if tot > 0:
            sub['d'] = np.sqrt(sub['X']**2 + sub['Y']**2)
            rb, rp = get_radii(r)
            b = len(sub[sub['d'] <= rb])
            p = len(sub[(sub['d'] > rb) & (sub['d'] <= rp)])
            bog = tot - (b + p)
            to_par = (b * -1) + (bog * 1)
            sign = "+" if to_par > 0 else ""
            
            misses = sub[sub['d'] > rb]
            sl = len(misses[(misses['X'] < 0) & (misses['Y'] <= 0)])
            ll = len(misses[(misses['X'] < 0) & (misses['Y'] > 0)])
            sr = len(misses[(misses['X'] >= 0) & (misses['Y'] <= 0)])
            lr = len(misses[(misses['X'] >= 0) & (misses['Y'] > 0)])
            
            stats_txt1 = f"Range: {r}m | Shots: {tot} | To Par: {sign}{to_par}"
            stats_txt2 = f"SL: {(sl/tot)*100:.0f}% LL: {(ll/tot)*100:.0f}% SR: {(sr/tot)*100:.0f}% LR: {(lr/tot)*100:.0f}%"

        pdf.set_xy(x_offsets[i], y_start + 8)
        pdf.cell(85, 4, txt=stats_txt1, align='C')
        pdf.set_xy(x_offsets[i], y_start + 12)
        pdf.cell(85, 4, txt=stats_txt2, align='C')
        
        img = create_target_image(sub, r)
        temp_fn = f"temp_app_{i}.png"
        img.save(temp_fn)
        pdf.image(temp_fn, x=x_offsets[i]+12, y=y_start+18, w=60)
        if os.path.exists(temp_fn): os.remove(temp_fn)

    y_start = 110
    x_offsets_tee = [25, 165]
    ranges_tee = ["OTT: Driver", "OTT: Others"]
    pdf.set_xy(10, y_start)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Long Game (Off the Tee)", ln=True, align='L')
    pdf.set_font("Arial", '', 9)

    for i, r in enumerate(ranges_tee):
        sub = df_shots[df_shots['Range'] == r].copy()
        tot = len(sub)
        stats_txt1 = f"{r} | Shots: {tot}"
        stats_txt2 = "No shots recorded."
        
        if tot > 0:
            sub['dx'] = sub['X'].abs()
            in_10 = len(sub[sub['dx'] <= 10])
            in_20 = len(sub[(sub['dx'] > 10) & (sub['dx'] <= 20)])
            out_20 = len(sub[sub['dx'] > 20])
            avg_dist = sub['Y'].mean()
            
            stats_txt1 = f"{r} | Shots: {tot} | Avg Dist: {avg_dist:.1f}m"
            stats_txt2 = f"<10m: {(in_10/tot)*100:.0f}% | 10-20m: {(in_20/tot)*100:.0f}% | 20m+: {(out_20/tot)*100:.0f}%"

        pdf.set_xy(x_offsets_tee[i], y_start + 8)
        pdf.cell(100, 4, txt=stats_txt1, align='C')
        pdf.set_xy(x_offsets_tee[i], y_start + 12)
        pdf.cell(100, 4, txt=stats_txt2, align='C')
        
        img = create_tee_image(sub, r)
        temp_fn = f"temp_tee_{i}.png"
        img.save(temp_fn)
        pdf.image(temp_fn, x=x_offsets_tee[i]+10, y=y_start+18, w=80)
        if os.path.exists(temp_fn): os.remove(temp_fn)

    return bytes(pdf.output())

# --- 7. NAVIGATION LOGIC ---
st.sidebar.divider()
st.sidebar.header("Tournament Setup")
with st.sidebar.expander("➕ New Tournament"):
    t_name = st.text_input("Name:")
    if st.button("Create & Open"):
        if t_name:
            st.session_state.active_t = t_name
            st.session_state.active_r = "Round 1"
            st.session_state.workflow_step = "Driving"
            st.rerun()

all_t = st.session_state.shots_data['Tournament'].unique().tolist() if not st.session_state.shots_data.empty else []
t_select = st.sidebar.selectbox("Select Tournament:", ["None"] + all_t, index=0 if st.session_state.active_t is None else all_t.index(st.session_state.active_t) + 1)
if t_select != "None" and t_select != st.session_state.active_t:
    st.session_state.active_t = t_select
    st.session_state.active_r = "Round 1"
    st.session_state.workflow_step = "Driving"
    st.rerun()

if st.session_state.active_t:
    st.sidebar.divider()
    st.sidebar.header("Select Round")
    r_select = st.sidebar.radio("Round:", ["Round 1", "Round 2", "Round 3", "Round 4"], index=["Round 1", "Round 2", "Round 3", "Round 4"].index(st.session_state.active_r))
    if r_select != st.session_state.active_r:
        st.session_state.active_r = r_select
        st.rerun()
        
    st.sidebar.divider()
    if st.sidebar.button("📊 ECGA Master Dashboard", use_container_width=True):
        st.session_state.workflow_step = "Dashboard"
        st.rerun()

# --- 8. MAIN WORKFLOW ---
if st.session_state.active_t:
    st.title(f"{st.session_state.active_t} - {st.session_state.active_r if st.session_state.workflow_step != 'Dashboard' else 'Master Dashboard'}")
    
    if st.session_state.workflow_step != "Dashboard":
        steps = ["Driving", "Scoring Zone", "Short Game", "Putting", "Mental & Judgement"] 
        selected_step = st.radio("Input Phase:", steps, horizontal=True, index=steps.index(st.session_state.workflow_step) if st.session_state.workflow_step in steps else 0)
        if selected_step != st.session_state.workflow_step:
            st.session_state.workflow_step = selected_step
            st.rerun()
        st.divider()

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
                    st.success(f"**Tournament Sheet Stat:** {(fwys/tot)*100:.0f}% ({pens})") 
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
                    st.info(f"**Tournament Sheet Stat:** {to_par}({tot})") 
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

        if st.button("Save Short Game Stats"):
            update_data = {"sg_total": sg_total, "sg_inside_6": sg_6, "sg_inside_3": sg_3, "sg_ud": sg_ud, "sgz_score": sgz}
            supabase.table("round_stats").update(update_data).eq("id", current_stats['id']).execute()
            st.success("Saved successfully!")

    # --- PHASE: PUTTING ---
    elif st.session_state.workflow_step == "Putting":
        st.subheader("18-Hole SG Putting Calculator")
        
        raw_grid = current_stats.get('putting_holes')
        if raw_grid and isinstance(raw_grid, list) and len(raw_grid) == 18:
            df_putts = pd.DataFrame(raw_grid)
        else:
            df_putts = pd.DataFrame({"Hole": [f"Hole {i}" for i in range(1, 19)], "Distance (ft)": [0]*18, "Putts": [0]*18})

        edited_df = st.data_editor(df_putts, hide_index=True, use_container_width=True, num_rows="fixed")
        
        total_putts = int(edited_df["Putts"].sum())
        total_sg = 0.0
        
        for index, row in edited_df.iterrows():
            dist = row["Distance (ft)"]
            putts = row["Putts"]
            if dist > 0 and putts > 0:
                total_sg += (get_expected_putts(dist) - putts)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Putts", total_putts)
        col2.metric("Total SG Putting", f"{total_sg:+.2f}")
        
        st.divider()
        st.markdown("### Lag Putting")
        lag_total = st.number_input("Total Lag Putts", min_value=0, value=current_stats.get('lag_total', 0))
        lag_success = st.number_input("Lags inside putter length", min_value=0, max_value=lag_total if lag_total > 0 else 0, value=current_stats.get('lag_success', 0))
        if lag_total > 0: st.write(f"**Lag:** {(lag_success/lag_total)*100:.0f}%")

        if st.button("Save Putting Stats"):
            update_data = {
                "putting_holes": edited_df.to_dict('records'), "putts_total": total_putts, 
                "sg_putting": round(total_sg, 2), "lag_total": lag_total, "lag_success": lag_success
            }
            supabase.table("round_stats").update(update_data).eq("id", current_stats['id']).execute()
            st.success("Putting stats saved successfully!")

    # --- PHASE: MENTAL & JUDGEMENT ---
    elif st.session_state.workflow_step == "Mental & Judgement":
        st.subheader("Mental (M) & Judgements (J)")
        m_score = st.slider("Mental Score (M)", min_value=0, max_value=100, value=current_stats.get('mental_score', 0))
        j_score = st.slider("Judgement Score (J)", min_value=0, max_value=100, value=current_stats.get('judgement_score', 0))
        
        if st.button("Save Mental Stats"):
            supabase.table("round_stats").update({"mental_score": m_score, "judgement_score": j_score}).eq("id", current_stats['id']).execute()
            st.success("Saved successfully!")

    # --- PHASE: MASTER DASHBOARD ---
    elif st.session_state.workflow_step == "Dashboard":
        st.success(f"Aggregated data for **{st.session_state.active_t}**")
        
        all_tournament_shots = st.session_state.shots_data[st.session_state.shots_data['Tournament'] == st.session_state.active_t]
        all_round_stats = load_all_tournament_stats(st.session_state.current_user, st.session_state.active_t)
        
        df_master = build_master_dataframe(all_tournament_shots, all_round_stats)
        
        # Bolds the category in the UI if it's a section header
        df_ui = df_master.copy()
        df_ui['Category'] = df_ui.apply(lambda r: f"**{r['Category']}**" if r['Round 1'] == "" else r['Category'], axis=1)
        
        st.markdown("""
            <style>
            .stTable table { width: 100%; }
            .stTable th, .stTable td { white-space: nowrap !important; text-align: center !important; }
            .stTable th:first-child, .stTable td:first-child { width: 15% !important; text-align: left !important; }
            </style>
        """, unsafe_allow_html=True)
        
        st.table(df_ui.set_index('Category'))
        
        st.divider()
        
        pdf_bytes = create_ecga_pdf(st.session_state.active_t, df_master, all_tournament_shots)
        
        st.download_button(
            label="📄 Download 2-Page Tour-Grade ECGA Overview",
            data=pdf_bytes,
            file_name=f"{st.session_state.active_t}_ECGA_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:
    st.info("Create or select a tournament in the sidebar to begin.")
