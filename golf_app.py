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
from supabase import create_client

# --- 1. APP CONFIG, SECRETS & CSS ---
st.set_page_config(page_title="The Score Code", layout="wide")

st.markdown("""
    <style>
    /* 1. Import Premium Fonts from Google */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Montserrat:wght@300;400;600&display=swap');

    /* 2. Apply crisp Montserrat to general text */
    html, body, [class*="css"], [class*="st-"], .stMarkdown, .stText {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* 3. PROTECT UI ICONS: Force Streamlit icons to keep their native font */
    .material-symbols-rounded, .material-icons, [data-testid="stIconMaterial"], [class*="stIcon"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }
    
    /* 4. Apply sophisticated Playfair Display to all Headers and Titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
    }

    /* 5. Keep our larger slider thumb styling */
    div[data-baseweb="slider"] div[role="slider"] {
        height: 24px !important;
        width: 24px !important;
        border-radius: 50% !important;
        box-shadow: 0 0 4px rgba(0,0,0,0.3) !important;
    }
    
    div[data-baseweb="slider"] div[data-testid="stThumbValue"] {
        font-size: 16px !important;
        font-weight: bold !important;
        transform: translateY(-8px) !important; 
        font-family: 'Montserrat', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# PGA Tour High-Density Expected Putts Baseline (ShotLink Calibrated)
pga_putts_baseline = {
    1: 1.00, 2: 1.01, 3: 1.04, 4: 1.11, 5: 1.23, 6: 1.34, 7: 1.43, 8: 1.50, 
    9: 1.56, 10: 1.61, 11: 1.65, 12: 1.69, 13: 1.72, 14: 1.75, 15: 1.78, 
    16: 1.81, 17: 1.83, 18: 1.85, 19: 1.87, 20: 1.88, 21: 1.89, 22: 1.90, 
    23: 1.91, 24: 1.92, 25: 1.94, 26: 1.95, 27: 1.96, 28: 1.97, 29: 1.99, 
    30: 2.00, 31: 2.01, 32: 2.03, 33: 2.04, 34: 2.05, 35: 2.06, 36: 2.08, 
    37: 2.09, 38: 2.10, 39: 2.12, 40: 2.13, 45: 2.19, 50: 2.25, 55: 2.31, 
    60: 2.37, 65: 2.42, 70: 2.47, 75: 2.51, 80: 2.55, 85: 2.58, 90: 2.61, 
    95: 2.64, 100: 2.67
}

def get_expected_putts(distance):
    xp = sorted(list(pga_putts_baseline.keys()))
    fp = [pga_putts_baseline[x] for x in xp]
    return float(np.interp(distance, xp, fp))

# --- 2. DATA LOADING & AUTO-SAVE CALLBACKS ---
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
        "user_name": current_user, 
        "tournament": tournament, 
        "round_num": round_num,
        "gross_score": 0, 
        "to_par": 0, 
        "gir": 0, 
        "gir_less_5": 0, 
        "sg_total": 0, 
        "sg_inside_6": 0, 
        "sg_inside_3": 0, 
        "sg_ud": 0, 
        "sgz_score": 0,
        "putts_total": 0, 
        "sg_putting": 0.0, 
        "lag_success": 0, 
        "lag_total": 0, 
        "mental_score": 0, 
        "judgement_score": 0, 
        "cm_score": 0, 
        "putting_holes": None,
        "d_hit": 0, 
        "d_tot": 0, 
        "d_pen": 0, 
        "o_hit": 0, 
        "o_tot": 0, 
        "o_pen": 0
    }
    res = supabase.table("round_stats").insert(blank).execute()
    return res.data[0]

def load_all_stats(current_user):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).execute()
    return response.data if response.data else []

def load_all_tournament_stats(current_user, tournament):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).eq("tournament", tournament).execute()
    return response.data if response.data else []

def auto_save_stat(db_column, widget_key, record_id):
    val = st.session_state[widget_key]
    supabase.table("round_stats").update({db_column: val}).eq("id", record_id).execute()
    st.toast("☁️ Saved securely to cloud", icon="✅")

# --- 3. VISUAL ENGINES ---
def get_radii(label):
    if "50-100" in label: 
        return 3, 6
    if "101-150" in label: 
        return 4, 8
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
    
    # --- 4. DATA AGGREGATION ENGINE ---
def format_score_cell(list_s):
    valid_rounds = [s for s in list_s if (s.get('gross_score') or 0) > 0]
    if not valid_rounds: 
        return "-"
    
    avg_gross = sum((s.get('gross_score') or 0) for s in valid_rounds) / len(valid_rounds)
    avg_to_par = sum((s.get('to_par') or 0) for s in valid_rounds) / len(valid_rounds)
    
    gross_str = f"{avg_gross:.1f}".replace(".0", "")
    to_par_str = f"{avg_to_par:.1f}".replace(".0", "")
    
    if avg_to_par > 0: 
        return f"{gross_str} (+{to_par_str})"
    elif avg_to_par == 0: 
        return f"{gross_str} (E)"
    else: 
        return f"{gross_str} ({to_par_str})"

def calc_metrics(df_s, list_s, logic_type, param):
    num, den, extra = 0, 0, 0
    
    if logic_type == "driving":
        for s in list_s:
            if param == "OTT: Driver":
                num += (s.get('d_hit') or 0)
                den += (s.get('d_tot') or 0)
                extra += (s.get('d_pen') or 0)
            elif param == "OTT: Others":
                num += (s.get('o_hit') or 0)
                den += (s.get('o_tot') or 0)
                extra += (s.get('o_pen') or 0)
                
    elif logic_type == "approach":
        df_a = df_s[df_s['Range'] == param]
        den = len(df_a)
        if den > 0:
            df_a = df_a.copy()
            df_a['d'] = np.sqrt(df_a['X']**2 + df_a['Y']**2)
            rb, rp = get_radii(param)
            b_count = len(df_a[df_a['d'] <= rb])
            bog_count = len(df_a[df_a['d'] > rp])
            num = (b_count * -1) + (bog_count * 1)
            
    elif logic_type == "abs":
        for s in list_s:
            v = s.get(param) or 0
            if v != 0: 
                num += v
                den += 1
                
    elif logic_type == "sg_perc":
        for s in list_s: 
            num += (s.get(param) or 0)
            den += (s.get('sg_total') or 0)
            
    elif logic_type == "sgz":
        for s in list_s: 
            num += (s.get('sgz_score') or 0)
            den += (s.get('sg_total') or 0)
            
    elif logic_type == "lag":
        for s in list_s: 
            num += (s.get('lag_success') or 0)
            den += (s.get('lag_total') or 0)
            
    elif logic_type == "sg_putt":
        for s in list_s:
            v = s.get('sg_putting') or 0.0
            if v != 0: 
                num += v
                den += 1
                
    return num, den, extra

def format_cell(logic_type, num, den, extra):
    if den == 0: 
        return "-"
    if logic_type == "driving": 
        return f"{(num/den)*100:.0f}% ({extra})"
    if logic_type == "approach": 
        sign = "+" if num > 0 else ""
        return f"{sign}{num}({den})"
    if logic_type in ["abs", "sg_putt"]:
        val = num / den
        if logic_type == "sg_putt": 
            sign = "+" if val > 0 else ""
            return f"{sign}{val:.2f}"
        return f"{val:.1f}"
    if logic_type in ["sg_perc", "lag"]: 
        return f"{(num/den)*100:.0f}%"
    if logic_type == "sgz": 
        return f"{num}({den})"
    return "-"

def build_master_dataframe(df_shots, list_stats, mode="tournament"):
    headers = ["Round 1", "Round 2", "Round 3", "Round 4"] if mode == "tournament" else []
    data = []
    
    row_score = {"Category": "Score (To Par)"}
    for h in headers:
        s_r = [s for s in list_stats if s['round_num'] == h]
        row_score[h] = format_score_cell(s_r)
    row_score["AV / TOTAL"] = format_score_cell(list_stats)
    data.append(row_score)

    def add_section_header(title):
        row = {"Category": title, "AV / TOTAL": ""}
        for h in headers: 
            row[h] = ""
        data.append(row)

    def add_row(cat, logic_type, param=""):
        row = {"Category": cat}
        for h in headers:
            if h.strip() == "": 
                row[h] = "-"
                continue
            df_s = df_shots[df_shots['Round'] == h]
            list_s = [s for s in list_stats if s['round_num'] == h]
            n, d, e = calc_metrics(df_s, list_s, logic_type, param)
            row[h] = format_cell(logic_type, n, d, e)
            
        n_all, d_all, e_all = calc_metrics(df_shots, list_stats, logic_type, param)
        row["AV / TOTAL"] = format_cell(logic_type, n_all, d_all, e_all)
        data.append(row)

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
    add_row("Putts (#)", "abs", "putts_total")
    add_row("SG Putting", "sg_putt")
    add_row("Lag", "lag")
    
    add_section_header("MENTAL & JUDGEMENTS")
    add_row("M", "abs", "mental_score")
    add_row("J", "abs", "judgement_score")
    add_row("CM", "abs", "cm_score")
    
    return pd.DataFrame(data)

# --- 5. ECGA 2-PAGE PDF GENERATOR ---
def create_ecga_pdf(title, df_master, df_shots):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)
    
    # PAGE 1: Master Table
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"The Score Code Overview: {title}", ln=True, align='C')
    pdf.ln(5)
    
    headers = list(df_master.columns)
    col_w = [138, 138] if len(headers) == 2 else [40, 47, 47, 47, 47, 47]
    row_h = 7 
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(200, 220, 255)
    for i, h in enumerate(headers): 
        pdf.cell(col_w[i], row_h, txt=h.strip(), border=1, align='C', fill=True)
    pdf.ln()
    
    pdf.set_font("Arial", '', 10)
    
    for index, row in df_master.iterrows():
        cat = row['Category']
        if row["AV / TOTAL"] == "":
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(sum(col_w), row_h, txt=cat, border=1, fill=True, ln=True, align='L')
            pdf.set_font("Arial", '', 10) 
        else:
            for i, h in enumerate(headers):
                align = 'L' if i == 0 else 'C'
                pdf.cell(col_w[i], row_h, txt=str(row[h]), border=1, align=align)
            pdf.ln()

    # PAGE 2: Dispersion Charts
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"Dispersion Analytics: {title}", ln=True, align='C')
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
        
        if os.path.exists(temp_fn): 
            os.remove(temp_fn)

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
        
        if os.path.exists(temp_fn): 
            os.remove(temp_fn)
            
    return bytes(pdf.output())
    
    # --- 6. GLOBAL STATE LOGIC ---
if 'page' not in st.session_state: 
    st.session_state.page = "Login"
if 'current_user' not in st.session_state: 
    st.session_state.current_user = None

# --- 7. ROUTING: LOGIN GATE ---
if st.session_state.page == "Login" or not st.session_state.current_user:
    st.markdown("<h1 style='text-align: center; font-size: 4em; margin-top: 10%;'>The Score Code</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Elite Performance Platform</h3>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username_input = st.text_input("Enter Username to Access Vault", key="login_input").strip()
        if st.button("Authenticate", use_container_width=True):
            if username_input:
                st.session_state.current_user = username_input
                st.session_state.shots_data = load_shots(username_input)
                st.session_state.active_t = None
                st.session_state.page = "Season Hub"
                st.rerun()

# --- 8. ROUTING: SECURE PLATFORM ---
else:
    st.sidebar.title("👤 Player Profile")
    st.sidebar.write(f"**{st.session_state.current_user}**")
    
    if st.sidebar.button("Log Out"):
        st.session_state.page = "Login"
        st.session_state.current_user = None
        st.rerun()
        
    st.sidebar.divider()
    st.sidebar.header("🧭 Navigation")
    
    if st.sidebar.button("🏠 Season Hub", use_container_width=True):
        st.session_state.active_t = None
        st.session_state.page = "Season Hub"
        st.rerun()
        
    if st.sidebar.button("📊 Season Master Dashboard", use_container_width=True):
        st.session_state.page = "Season Master"
        st.rerun()
        
    if st.session_state.get('active_t'):
        st.sidebar.divider()
        if st.sidebar.button(f"🔙 Back to {st.session_state.active_t} Hub", use_container_width=True):
            st.session_state.page = "Tournament Hub"
            st.rerun()

    # --- PAGE: SEASON HUB ---
    if st.session_state.page == "Season Hub":
        st.title("🏠 The Score Code - Season Hub")
        st.write("Manage your events or create a new one.")
        
        with st.expander("➕ Create New Tournament"):
            new_t = st.text_input("Tournament Name:")
            if st.button("Create & Enter Hub"):
                if new_t:
                    load_round_stats(st.session_state.current_user, new_t, "Round 1")
                    st.session_state.active_t = new_t
                    st.session_state.active_r = "Round 1"
                    st.session_state.page = "Tournament Hub"
                    st.rerun()
                    
        st.divider()
        
        raw_stats = load_all_stats(st.session_state.current_user)
        t_from_shots = st.session_state.shots_data['Tournament'].unique().tolist() if not st.session_state.shots_data.empty else []
        t_from_stats = [s['tournament'] for s in raw_stats] if raw_stats else []
        all_t = sorted(list(set(t_from_shots + t_from_stats)))
        
        if all_t:
            cols = st.columns(4)
            for i, t in enumerate(all_t):
                with cols[i % 4]:
                    st.markdown(f"#### ⛳ {t}")
                    if st.button(f"Enter Hub", key=f"t_{t}", use_container_width=True):
                        st.session_state.active_t = t
                        st.session_state.active_r = "Round 1"
                        st.session_state.page = "Tournament Hub"
                        st.rerun()
        else:
            st.info("No tournaments logged yet. Create one above to get started!")

    # --- PAGE: TOURNAMENT HUB ---
    elif st.session_state.page == "Tournament Hub":
        st.title(f"⛳ {st.session_state.active_t} Hub")
        st.write("Select a round to start entering your statistics.")
        
        c1, c2, c3, c4 = st.columns(4)
        for col, r_name in zip([c1, c2, c3, c4], ["Round 1", "Round 2", "Round 3", "Round 4"]):
            with col:
                st.markdown(f"### {r_name}")
                if st.button(f"Edit Data", use_container_width=True, key=f"btn_{r_name}"):
                    st.session_state.active_r = r_name
                    st.session_state.workflow_step = "Speed Logger"
                    st.session_state.page = "Data Entry"
                    st.rerun()
                    
        st.divider()
        st.subheader("Tournament Tools")
        if st.button("📊 View Tournament Dashboard", use_container_width=True):
            st.session_state.workflow_step = "Master Dashboard"
            st.session_state.page = "Data Entry"
            st.rerun()
            
        st.divider()
        with st.expander("🗑️ Delete Entire Tournament"):
            st.warning(f"⚠️ Are you sure you want to permanently delete all data for **{st.session_state.active_t}**? This cannot be undone.")
            if st.button("Yes, Delete Tournament", type="primary", use_container_width=True):
                supabase.table("shots").delete().eq("User", st.session_state.current_user).eq("Tournament", st.session_state.active_t).execute()
                supabase.table("round_stats").delete().eq("user_name", st.session_state.current_user).eq("tournament", st.session_state.active_t).execute()
                st.session_state.shots_data = load_shots(st.session_state.current_user)
                st.session_state.active_t = None
                st.session_state.page = "Season Hub"
                st.rerun()

    # --- PAGE: SEASON MASTER DASHBOARD ---
    elif st.session_state.page == "Season Master":
        st.title("📊 Season Master Dashboard")
        st.write("Accumulated statistics across your selected events.")
        
        raw_shots = load_shots(st.session_state.current_user)
        raw_stats = load_all_stats(st.session_state.current_user)
        t_aggregates = []
        unique_ts = sorted(list(set(raw_shots['Tournament'].unique().tolist() + [s['tournament'] for s in raw_stats])))
        
        for t in unique_ts:
            t_stats = [s for s in raw_stats if s['tournament'] == t]
            if t_stats:
                v_stats = [s for s in t_stats if s.get('gross_score', 0) > 0]
                if v_stats:
                    avg_to_par = sum(s.get('to_par', 0) for s in v_stats) / len(v_stats)
                    t_aggregates.append({'name': t, 'avg': avg_to_par})
        
        t_df = pd.DataFrame(t_aggregates).sort_values('avg') if t_aggregates else pd.DataFrame()
        top_cutoff = t_df['avg'].quantile(0.3) if not t_df.empty else 0
        bottom_cutoff = t_df['avg'].quantile(0.7) if not t_df.empty else 0
        
        col_f1, col_f2 = st.columns(2)
        selected_ts = col_f1.multiselect("Filter Tournaments:", options=unique_ts, default=unique_ts)
        tier_filter = col_f2.selectbox("Filter by Performance Tier:", ["All Data", "Best 30% Tournaments", "Bottom 30% Tournaments"])
        
        final_ts_names = selected_ts
        if tier_filter == "Best 30% Tournaments" and not t_df.empty:
            final_ts_names = [t for t in selected_ts if t in t_df[t_df['avg'] <= top_cutoff]['name'].tolist()]
        elif tier_filter == "Bottom 30% Tournaments" and not t_df.empty:
            final_ts_names = [t for t in selected_ts if t in t_df[t_df['avg'] >= bottom_cutoff]['name'].tolist()]
            
        final_stats = [s for s in raw_stats if s['tournament'] in final_ts_names]
        final_shots = raw_shots[raw_shots['Tournament'].isin(final_ts_names)]
        
        if not final_shots.empty or final_stats:
            df_m = build_master_dataframe(final_shots, final_stats, mode="season")
            df_ui = df_m.copy()
            df_ui['Category'] = df_ui.apply(lambda r: f"**{r['Category']}**" if r['AV / TOTAL'] == "" else r['Category'], axis=1)
            
            st.markdown("""
                <style>
                .stTable table { width: 100%; }
                .stTable th, .stTable td { white-space: nowrap !important; text-align: center !important; }
                .stTable th:first-child, .stTable td:first-child { width: 50% !important; text-align: left !important; }
                </style>
            """, unsafe_allow_html=True)
            
            st.table(df_ui.set_index('Category'))
            
            pdf_bytes = create_ecga_pdf("Season Master (Filtered)", df_m, final_shots)
            st.download_button(
                label="📄 Download Season-Long 2-Page Report", 
                data=pdf_bytes, 
                file_name=f"{st.session_state.current_user}_Season_Report.pdf", 
                mime="application/pdf", 
                use_container_width=True
            )
            
            st.divider()
            st.subheader("Season Dispersion Analytics")
            st.write("### Scoring Zone (Approach)")
            c1, c2, c3 = st.columns(3)
            with c1: 
                st.markdown("<h4 style='text-align: center;'>50-100m</h4>", unsafe_allow_html=True)
                st.image(create_target_image(final_shots[final_shots['Range'] == '50-100'], '50-100'))
            with c2: 
                st.markdown("<h4 style='text-align: center;'>101-150m</h4>", unsafe_allow_html=True)
                st.image(create_target_image(final_shots[final_shots['Range'] == '101-150'], '101-150'))
            with c3: 
                st.markdown("<h4 style='text-align: center;'>151-200m</h4>", unsafe_allow_html=True)
                st.image(create_target_image(final_shots[final_shots['Range'] == '151-200'], '151-200'))
            
            st.write("### Long Game (Off the Tee)")
            c4, c5 = st.columns(2)
            with c4: 
                st.markdown("<h4 style='text-align: center;'>OTT: Driver</h4>", unsafe_allow_html=True)
                st.image(create_tee_image(final_shots[final_shots['Range'] == 'OTT: Driver'], 'OTT: Driver'))
            with c5: 
                st.markdown("<h4 style='text-align: center;'>OTT: Others</h4>", unsafe_allow_html=True)
                st.image(create_tee_image(final_shots[final_shots['Range'] == 'OTT: Others'], 'OTT: Others'))
        else:
            st.info("No data available for the selected filters.")

    # --- PAGE: DATA ENTRY ---
    elif st.session_state.page == "Data Entry":
        st.title(f"{st.session_state.active_t} - {st.session_state.active_r}")
        
        # Simplified Navigation: We only need two phases now!
        steps = ["Speed Logger", "Tournament Dashboard"] 
        selected_step = st.radio("Phase:", steps, horizontal=True, index=steps.index(st.session_state.workflow_step) if st.session_state.workflow_step in steps else 0)
        
        if selected_step != st.session_state.workflow_step:
            st.session_state.workflow_step = selected_step
            st.rerun()
            
        st.divider()
        current_stats = load_round_stats(st.session_state.current_user, st.session_state.active_t, st.session_state.active_r)
        cid = current_stats['id']

        if st.session_state.workflow_step == "Speed Logger":
            categories = ["Driving", "Other Club", "150-200m", "100-150m", "50-100m", "GIR", "GIR < 5m", "< 6ft", "< 3ft", "Up & Down", "SGZ", "Lag Putting", "Putt Dist (ft)", "Putts"]

            # Load existing hole-by-hole data if it exists in the DB, otherwise initialize blank
            existing_speed_data = current_stats.get('speed_logger_data')
            if existing_speed_data and isinstance(existing_speed_data, dict) and "1" in existing_speed_data:
                if "cpc_notepad" not in st.session_state:
                    st.session_state.cpc_notepad = existing_speed_data
                    st.session_state.cpc_hole = 1
            else:
                if "cpc_notepad" not in st.session_state:
                    st.session_state.cpc_notepad = {str(i): {cat: "" for cat in categories} for i in range(1, 19)}
                    st.session_state.cpc_hole = 1
            
            st.markdown("#### 🚩 Round Setup")
            c1, c2, c3 = st.columns(3)
            sl_holes = c1.radio("Holes Played:", [9, 18], index=1, horizontal=True, key="cpc_sl_holes")
            fetched_gross = current_stats.get('gross_score', 0)
            default_gross = int(fetched_gross) if fetched_gross > 0 else (72 if sl_holes==18 else 36)
            pr_gross = c2.number_input("Gross Score", min_value=0, max_value=150, value=default_gross, step=1, key="cpc_fast_gross")
            pr_to_par = c3.number_input("Score to Par (e.g., -2 or +3)", value=current_stats.get('to_par', 0), step=1, key="cpc_fast_par")
            st.divider()

            @st.fragment
            def render_speed_logger():
                active_h = str(st.session_state.cpc_hole)
                active_data = st.session_state.cpc_notepad[active_h]
                can_proceed = (active_data["Putts"] != "") and not (active_data["< 6ft"] != "" and active_data["Up & Down"] == "")

                def slim_divider(): st.markdown("<hr style='margin: 8px 0px; border: none; border-top: 1px solid #ddd;'>", unsafe_allow_html=True)
                def section_header(title): st.markdown(f"<div style='font-weight: 800; color: #1A237E; margin-top: 5px; margin-bottom: 2px; font-size: 0.9em; letter-spacing: 0.5px;'>{title}</div>", unsafe_allow_html=True)

                with st.container(border=True):
                    col_prev, col_curr, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button("⬅️ Prev", key="cpc_top_prev", use_container_width=True, disabled=(st.session_state.cpc_hole == 1)):
                            st.session_state.cpc_hole -= 1
                            st.rerun(scope="fragment")
                    with col_curr:
                        st.markdown(f"<h3 style='text-align: center; margin-top: 0px; margin-bottom: 0px;'>⛳ Hole {st.session_state.cpc_hole}</h3>", unsafe_allow_html=True)
                    with col_next:
                        if st.button("Next ➡️", key="cpc_top_next", use_container_width=True, disabled=(st.session_state.cpc_hole == sl_holes or not can_proceed)):
                            st.session_state.cpc_hole += 1
                            st.rerun(scope="fragment")
                    slim_divider()

                    def render_btn_row(category, options, subtext=None, disabled=False):
                        cols = st.columns([2.8, 1.2, 1.2, 1.2, 1.2, 1.2]) 
                        if subtext: cols[0].markdown(f"<div style='margin-top: 2px; line-height: 1.1;'><b>{category}</b><br><span style='font-size: 0.7em; color: gray;'>{subtext}</span></div>", unsafe_allow_html=True)
                        else: cols[0].markdown(f"<div style='margin-top: 8px;'><b>{category}</b></div>", unsafe_allow_html=True)
                        for i, opt in enumerate(options):
                            is_selected = st.session_state.cpc_notepad[active_h][category] == str(opt)
                            btn_type = "primary" if is_selected else "secondary"
                            if cols[i+1].button(str(opt), key=f"cpc_{active_h}_{category}_{opt}", type=btn_type, use_container_width=True, disabled=disabled):
                                if is_selected: st.session_state.cpc_notepad[active_h][category] = ""
                                else: st.session_state.cpc_notepad[active_h][category] = str(opt)
                                if category == "GIR" and st.session_state.cpc_notepad[active_h]["GIR"] == "":
                                    st.session_state.cpc_notepad[active_h]["GIR < 5m"] = ""
                                st.rerun(scope="fragment")

                    # --- 🚀 LONG GAME ---
                    section_header("🚀 LONG GAME")
                    render_btn_row("Driving", ["✅", "❌"], "Driver off the tee")
                    render_btn_row("Other Club", ["✅", "❌"], "Other club off the tee")
                    
                    active_driver = active_data["Driving"] in ["✅", "❌"]
                    active_other = active_data["Other Club"] in ["✅", "❌"]
                    
                    if active_driver or active_other:
                        r_label = "OTT: Driver" if active_driver else "OTT: Others"
                        st.markdown(f"<div style='margin-top: 8px; font-size: 0.85em; color: gray;'><b>Plot your {r_label} Dispersion:</b></div>", unsafe_allow_html=True)
                        
                        df_v = st.session_state.shots_data[
                            (st.session_state.shots_data['Tournament'] == st.session_state.active_t) & 
                            (st.session_state.shots_data['Round'] == st.session_state.active_r) & 
                            (st.session_state.shots_data['Range'] == r_label)
                        ]
                        
                        val = streamlit_image_coordinates(create_tee_image(df_v, r_label), key=f"img_{r_label}_{active_h}_{len(df_v)}")
                        
                        if val:
                            px, py = val['x'], val['y']
                            y_min, y_max = (270, 320) if r_label == "OTT: Driver" else (220, 270)
                            x_m = round((px / 500.0) * 60 - 30, 2)
                            y_m = round(y_max - (py / 500.0) * 50, 2)
                            
                            supabase.table("shots").insert({
                                "User": st.session_state.current_user, 
                                "Tournament": st.session_state.active_t, 
                                "Round": st.session_state.active_r, 
                                "Range": r_label, 
                                "X": x_m, "Y": y_m
                            }).execute()
                            
                            st.toast(f"📍 {r_label} plotted!", icon="✅")
                            st.session_state.shots_data = load_shots(st.session_state.current_user)
                            st.rerun(scope="fragment")

                        if not df_v.empty and st.button(f"Undo Last Tee Shot", key=f"un_tee_{active_h}"):
                            supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                            st.session_state.shots_data = load_shots(st.session_state.current_user)
                            st.rerun(scope="fragment")

                    slim_divider()

                    # --- 🎯 SCORING ZONE ---
                    section_header("🎯 SCORING ZONE")
                    sz_dist = st.selectbox("Approach Distance Range:", ["No Approach", "50-100m", "101-150m", "151-200m"], key=f"sz_sel_{active_h}")
                    
                    for rng in ["50-100m", "101-150m", "151-200m"]:
                        if rng != sz_dist:
                            st.session_state.cpc_notepad[active_h][rng] = ""
                            
                    if sz_dist != "No Approach":
                        r_label = sz_dist.replace("m", "") 
                        current_sz_score = active_data[sz_dist]
                        if current_sz_score:
                            st.info(f"**Current Score:** {current_sz_score} (Click chart to update)")
                        
                        df_sz = st.session_state.shots_data[
                            (st.session_state.shots_data['Tournament'] == st.session_state.active_t) & 
                            (st.session_state.shots_data['Round'] == st.session_state.active_r) & 
                            (st.session_state.shots_data['Range'] == r_label)
                        ]
                        
                        sz_val = streamlit_image_coordinates(create_target_image(df_sz, r_label), key=f"img_sz_{active_h}_{len(df_sz)}")
                        
                        if sz_val:
                            px, py = sz_val['x'], sz_val['y']
                            rb, rp = get_radii(r_label)
                            limit = rp + 2
                            x_m = round((px / 500.0) * (2 * limit) - limit, 2)
                            y_m = round(limit - (py / 500.0) * (2 * limit), 2)
                            
                            d = np.sqrt(x_m**2 + y_m**2)
                            if d <= rb: auto_score = "-1"
                            elif d <= rp: auto_score = "E"
                            else: auto_score = "+1"
                            
                            supabase.table("shots").insert({
                                "User": st.session_state.current_user, 
                                "Tournament": st.session_state.active_t, 
                                "Round": st.session_state.active_r, 
                                "Range": r_label, 
                                "X": x_m, "Y": y_m
                            }).execute()
                            
                            st.session_state.cpc_notepad[active_h][sz_dist] = auto_score
                            st.toast(f"📍 Approach scored as {auto_score}", icon="✅")
                            st.session_state.shots_data = load_shots(st.session_state.current_user)
                            st.rerun(scope="fragment")
                            
                        if not df_sz.empty and st.button(f"Undo Last Approach", key=f"un_sz_{active_h}"):
                            supabase.table("shots").delete().eq("id", int(df_sz.iloc[-1]['id'])).execute()
                            st.session_state.shots_data = load_shots(st.session_state.current_user)
                            st.session_state.cpc_notepad[active_h][sz_dist] = "" 
                            st.rerun(scope="fragment")

                    render_btn_row("GIR", ["✅"])
                    render_btn_row("GIR < 5m", ["✅"], "GIR within 5m from hole", disabled=(active_data["GIR"] != "✅"))
                    slim_divider()

                    # --- 🪤 SHORT GAME ---
                    section_header("🪤 SHORT GAME")
                    render_btn_row("< 6ft", ["✅", "❌"], "Short game shots (under 50m) hit within 6ft of hole")
                    render_btn_row("< 3ft", ["✅", "❌"], "Inside 3ft", disabled=(active_data["< 6ft"] != "✅"))
                    
                    ud_subtext = "<span style='color: #D32F2F;'>*Required</span>" if active_data["< 6ft"] != "" and active_data["Up & Down"] == "" else ""
                    render_btn_row("Up & Down", ["✅", "❌"], subtext=ud_subtext)
                    render_btn_row("SGZ", ["+2", "+1", "E", "-1", "-2"], "Short Game Zone Score")
                    slim_divider()

                    # --- ⛳ PUTTING ---
                    section_header("⛳ PUTTING")
                    render_btn_row("Lag Putting", ["✅", "❌"], "Putts over 18ft finishing within 1 putter length")
                    st.markdown("<div style='margin-top: 8px;'><b>Strokes Gained Putting</b></div>", unsafe_allow_html=True)
                    with st.container(border=True):
                        c_dist, c_putts = st.columns([3, 2])
                        c_dist.caption("1st Putt Distance (ft)")
                        current_dist = int(active_data["Putt Dist (ft)"]) if active_data["Putt Dist (ft)"] != "" else 0
                        new_dist = c_dist.slider(f"Dist_Hole_{active_h}", 0, 100, current_dist, key=f"dist_{active_h}", label_visibility="collapsed")
                        
                        c_putts.caption("Putts")
                        current_putts = int(active_data["Putts"]) if active_data["Putts"] != "" else 0
                        new_putts = c_putts.radio(f"Putts_Hole_{active_h}", [0, 1, 2, 3, 4], index=current_putts, horizontal=True, key=f"putts_{active_h}", label_visibility="collapsed")
                        
                        if new_dist != current_dist or new_putts != current_putts:
                            st.session_state.cpc_notepad[active_h]["Putt Dist (ft)"] = str(new_dist)
                            st.session_state.cpc_notepad[active_h]["Putts"] = str(new_putts)
                            st.rerun(scope="fragment")

                        if new_putts > 0 and new_dist > 0:
                            hole_sg = get_expected_putts(new_dist) - new_putts
                            st.info(f"**Hole SG Putting:** {hole_sg:+.2f}")

                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    b_col_prev, b_col_curr, b_col_next = st.columns([1, 2, 1])
                    with b_col_prev:
                        if st.button("⬅️ Prev", key="bot_prev_cpc", use_container_width=True, disabled=(st.session_state.cpc_hole == 1)):
                            st.session_state.cpc_hole -= 1
                            st.rerun(scope="fragment")
                    with b_col_next:
                        if st.button("Next ➡️", key="bot_next_cpc", use_container_width=True, disabled=(st.session_state.cpc_hole == sl_holes or not can_proceed)):
                            st.session_state.cpc_hole += 1
                            st.rerun(scope="fragment")

                st.write("<br>", unsafe_allow_html=True)
                with st.expander("📊 View Full Scorecard", expanded=False):
                    def calculate_totals(np, active_holes):
                        t = {}
                        valid_holes = {k: v for k, v in np.items() if int(k) <= active_holes}
                        hp = sum(1 for h in valid_holes.values() if h["Putts"] != "" and h["Putts"] != "0")
                        
                        for cat in categories:
                            if cat in ["Driving", "Other Club", "< 6ft", "< 3ft", "Up & Down", "Lag Putting"]:
                                hits = sum(1 for h in valid_holes.values() if h[cat] == "✅")
                                tot = hits + sum(1 for h in valid_holes.values() if h[cat] == "❌")
                                t[cat] = f"{int((hits/tot)*100)}% ({hits}/{tot})" if tot > 0 else "-"
                                
                            elif cat in ["150-200m", "100-150m", "50-100m"]:
                                sc, sh = 0, 0
                                for h in valid_holes.values():
                                    v = h[cat]
                                    if v:
                                        sh += 1
                                        if v == "-2": sc -= 2
                                        elif v == "-1": sc -= 1
                                        elif v == "+1": sc += 1
                                        elif v == "+2": sc += 2
                                t[cat] = f"{'E' if sc == 0 else f'{sc:+}'} ({sh})" if sh > 0 else "-"
                                
                            elif cat == "GIR":
                                hits = sum(1 for h in valid_holes.values() if h[cat] == "✅")
                                t[cat] = f"{int((hits/hp)*100)}% ({hits}/{hp})" if hp > 0 else "-"
                                
                            elif cat == "GIR < 5m":
                                hits = sum(1 for h in valid_holes.values() if h[cat] == "✅")
                                t[cat] = f"{hits}" if hits > 0 else "-"
                                
                            elif cat == "SGZ":
                                sgz_score = 0
                                has_data = False
                                for h in valid_holes.values():
                                    v = h[cat]
                                    if v:
                                        has_data = True
                                        if v != "E": sgz_score += int(v)
                                t[cat] = f"{'+' if sgz_score > 0 else ''}{sgz_score}" if has_data else "-"
                                
                            elif cat == "Putts":
                                p = sum(int(h[cat]) for h in valid_holes.values() if h[cat] != "")
                                total_sg = 0.0
                                for h in valid_holes.values():
                                    d = int(h.get("Putt Dist (ft)") or 0)
                                    pt = int(h.get("Putts") or 0)
                                    if d > 0 and pt > 0: total_sg += (get_expected_putts(d) - pt)
                                t[cat] = f"{p} (SG: {total_sg:+.2f})" if p > 0 else "-"
                                
                            elif cat == "Putt Dist (ft)":
                                t[cat] = "-" 
                                
                        return t
                        
                    t_dict = calculate_totals(st.session_state.cpc_notepad, sl_holes) 
                    df = pd.DataFrame(st.session_state.cpc_notepad)
                    df = df[[str(i) for i in range(1, sl_holes + 1)]] 
                    df['Total/Avg'] = df.index.map(lambda x: t_dict.get(x, "-"))
                    df = df.reindex(categories)
                    css = "<style>.compact-table { width: 100%; border-collapse: collapse; font-size: 11px; font-family: sans-serif; text-align: center; } .compact-table th, .compact-table td { border: 1px solid #e0e0e0; padding: 6px 2px; text-align: center; } .compact-table th { background-color: #f0f2f6; color: #31333F; } .compact-table tbody th { text-align: left; padding-left: 8px; background-color: #ffffff; } .compact-table tr td:last-child { font-weight: bold; background-color: #f8f9fa; color: #1A237E; }</style>"
                    st.markdown(f"<div style='overflow-x: auto;'>{css}{df.to_html(classes='compact-table', escape=False)}</div>", unsafe_allow_html=True)

            # Trigger Fragment
            render_speed_logger()

            # --- 🧠 POST-ROUND MENTAL REVIEW ---
            st.divider()
            st.markdown("### 🧠 Post-Round Mental Review")
            c_m1, c_m2, c_m3 = st.columns(3)
            
            current_ms = current_stats.get('mental_score', 0)
            current_js = current_stats.get('judgement_score', 0)
            current_cm = float(current_stats.get('cm_score', 0.0))
            
            ms_val = c_m1.slider("Mental Score (M)", 0, 100, current_ms, key=f"ms_final_{cid}")
            js_val = c_m2.slider("Judgement Score (J)", 0, 100, current_js, key=f"js_final_{cid}")
            cm_val = c_m3.slider("Course Management (CM)", 0.0, 10.0, current_cm, step=0.5, key=f"cm_final_{cid}")

            # --- 💾 MASTER SAVE FUNCTION ---
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Full Round Data", type="primary", use_container_width=True):
                np_data = st.session_state.cpc_notepad
                
                agg = {
                    "d_hit": 0, "d_tot": 0, "o_hit": 0, "o_tot": 0,
                    "gir": 0, "gir_less_5": 0,
                    "sg_total": 0, "sg_inside_6": 0, "sg_inside_3": 0, "sg_ud": 0, "sgz_score": 0,
                    "lag_success": 0, "lag_total": 0,
                    "putts_total": 0, "sg_putting": 0.0
                }
                
                valid_holes = {k: v for k, v in np_data.items() if v["Putts"] != ""}
                
                for h, data in valid_holes.items():
                    if data.get("Driving") == "✅": agg["d_hit"] += 1; agg["d_tot"] += 1
                    elif data.get("Driving") == "❌": agg["d_tot"] += 1
                    
                    if data.get("Other Club") == "✅": agg["o_hit"] += 1; agg["o_tot"] += 1
                    elif data.get("Other Club") == "❌": agg["o_tot"] += 1
                    
                    if data.get("GIR") == "✅": agg["gir"] += 1
                    if data.get("GIR < 5m") == "✅": agg["gir_less_5"] += 1
                    
                    is_sg = False
                    if data.get("< 6ft") in ["✅", "❌"]:
                        is_sg = True
                        if data["< 6ft"] == "✅": agg["sg_inside_6"] += 1
                        
                    if data.get("< 3ft") in ["✅", "❌"]:
                        is_sg = True
                        if data["< 3ft"] == "✅": agg["sg_inside_3"] += 1
                        
                    if data.get("Up & Down") in ["✅", "❌"]:
                        is_sg = True
                        if data["Up & Down"] == "✅": agg["sg_ud"] += 1
                        
                    if is_sg: agg["sg_total"] += 1
                    
                    if data.get("SGZ") and data["SGZ"] != "E":
                        try: agg["sgz_score"] += int(data["SGZ"])
                        except ValueError: pass
                        
                    if data.get("Lag Putting") == "✅": agg["lag_success"] += 1; agg["lag_total"] += 1
                    elif data.get("Lag Putting") == "❌": agg["lag_total"] += 1
                    
                    putts = int(data.get("Putts") or 0)
                    dist = int(data.get("Putt Dist (ft)") or 0)
                    
                    if putts > 0: agg["putts_total"] += putts
                    if putts > 0 and dist > 0: agg["sg_putting"] += (get_expected_putts(dist) - putts)
                
                agg["sg_putting"] = round(agg["sg_putting"], 2)

                update_payload = {
                    "gross_score": st.session_state.get("cpc_fast_gross", current_stats.get('gross_score', 0)),
                    "to_par": st.session_state.get("cpc_fast_par", current_stats.get('to_par', 0)),
                    "mental_score": ms_val,
                    "judgement_score": js_val,
                    "cm_score": cm_val,
                    "speed_logger_data": np_data
                }
                
                update_payload.update(agg)
                
                try:
                    supabase.table("round_stats").update(update_payload).eq("id", cid).execute()
                    st.success(f"✅ All statistics for {st.session_state.active_r} have been successfully aggregated and saved to the vault.")
                    st.balloons()
                except Exception as e:
                    st.error(f"Failed to save data to cloud: {e}")

        # --- KEEP ORIGINAL TOURNAMENT DASHBOARD LOGIC HERE ---
        elif st.session_state.workflow_step == "Tournament Dashboard":
            all_ts = st.session_state.shots_data[st.session_state.shots_data['Tournament'] == st.session_state.active_t]
            all_rs = load_all_tournament_stats(st.session_state.current_user, st.session_state.active_t)
            df_m = build_master_dataframe(all_ts, all_rs, mode="tournament")
            df_ui = df_m.copy()
            df_ui['Category'] = df_ui.apply(lambda r: f"**{r['Category']}**" if r.get('Round 1', '') == "" else r['Category'], axis=1)
            
            st.markdown("""
                <style>
                .stTable table { width: 100%; } 
                .stTable th, .stTable td { white-space: nowrap !important; text-align: center !important; } 
                .stTable th:first-child, .stTable td:first-child { width: 15% !important; text-align: left !important; }
                </style>
            """, unsafe_allow_html=True)
            
            st.table(df_ui.set_index('Category'))
            
            pdf_bytes = create_ecga_pdf(st.session_state.active_t, df_m, all_ts)
            st.download_button(
                label="📄 Download Tournament Report", 
                data=pdf_bytes, 
                file_name=f"{st.session_state.active_t}_Report.pdf", 
                use_container_width=True
            )
