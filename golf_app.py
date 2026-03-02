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

# --- 1. APP CONFIG & SECRETS ---
st.set_page_config(page_title="The Score Code", layout="wide")

@st.cache_resource
def init_connection():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_connection()

# PGA Tour Expected Putts Baseline
pga_putts_baseline = {
    1: 1.00, 2: 1.01, 3: 1.04, 4: 1.13, 5: 1.23, 6: 1.34, 7: 1.43, 8: 1.50, 
    9: 1.56, 10: 1.61, 15: 1.78, 20: 1.87, 25: 1.94, 30: 2.01, 40: 2.13, 50: 2.26, 60: 2.38,
    70: 2.48, 80: 2.58, 90: 2.65, 100: 2.71
}

def get_expected_putts(distance):
    closest_dist = min(pga_putts_baseline.keys(), key=lambda k: abs(k - distance))
    return pga_putts_baseline[closest_dist]

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
        "user_name": current_user, "tournament": tournament, "round_num": round_num,
        "gross_score": 0, "to_par": 0, "gir": 0, "gir_less_5": 0, "sg_total": 0, 
        "sg_inside_6": 0, "sg_inside_3": 0, "sg_ud": 0, "sgz_score": 0,
        "putts_total": 0, "sg_putting": 0.0, "lag_success": 0, "lag_total": 0, 
        "mental_score": 0, "judgement_score": 0, "cm_score": 0, "putting_holes": None
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

# --- 4. DATA AGGREGATION ENGINE ---
def format_score_cell(list_s):
    valid_rounds = [s for s in list_s if s.get('gross_score', 0) > 0]
    if not valid_rounds: 
        return "-"
    
    avg_gross = sum(s.get('gross_score', 0) for s in valid_rounds) / len(valid_rounds)
    avg_to_par = sum(s.get('to_par', 0) for s in valid_rounds) / len(valid_rounds)
    
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
        df_d = df_s[df_s['Range'] == param]
        den = len(df_d)
        if den > 0:
            num = len(df_d[df_d['X'].abs() <= 10])
            extra = len(df_d[df_d['X'].abs() > 20])
            
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
            v = s.get(param, 0)
            if v != 0: num += v; den += 1
            
    elif logic_type == "sg_perc":
        for s in list_s:
            num += s.get(param, 0)
            den += s.get('sg_total', 0)
            
    elif logic_type == "sgz":
        for s in list_s:
            num += s.get('sgz_score', 0)
            den += s.get('sg_total', 0)
            
    elif logic_type == "lag":
        for s in list_s:
            num += s.get('lag_success', 0)
            den += s.get('lag_total', 0)
            
    elif logic_type == "sg_putt":
        for s in list_s:
            v = s.get('sg_putting', 0.0)
            if v != 0: num += v; den += 1
            
    return num, den, extra

def format_cell(logic_type, num, den, extra):
    if den == 0: return "-"
    if logic_type == "driving": return f"{(num/den)*100:.0f}% ({extra})"
    if logic_type == "approach":
        sign = "+" if num > 0 else ""
        return f"{sign}{num}({den})"
    if logic_type in ["abs", "sg_putt"]:
        val = num / den
        if logic_type == "sg_putt":
            sign = "+" if val > 0 else ""
            return f"{sign}{val:.2f}"
        return f"{val:.1f}"
    if logic_type in ["sg_perc", "lag"]: return f"{(num/den)*100:.0f}%"
    if logic_type == "sgz": return f"{num}({den})"
    return "-"

def build_master_dataframe(df_shots, list_stats, mode="tournament"):
    if mode == "tournament":
        headers = ["Round 1", "Round 2", "Round 3", "Round 4"]
    else:
        headers = [] 

    data = []
    
    row_score = {"Category": "Score (To Par)"}
    for h in headers:
        s_r = [s for s in list_stats if s['round_num'] == h]
        row_score[h] = format_score_cell(s_r)
    row_score["AV / TOTAL"] = format_score_cell(list_stats)
    data.append(row_score)

    def add_section_header(title):
        row = {"Category": title, "AV / TOTAL": ""}
        for h in headers: row[h] = ""
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
    
    # PAGE 1: MASTER TABLE
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"The Score Code Overview: {title}", ln=True, align='C')
    pdf.ln(5)
    
    headers = list(df_master.columns)
    
    if len(headers) == 2:
        col_w = [138, 138] 
    else:
        col_w = [40, 47, 47, 47, 47, 47]
        
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

    # PAGE 2: DISPERSION CHARTS
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
# --- 6. GLOBAL STATE LOGIC ---
if 'page' not in st.session_state: st.session_state.page = "Login"
if 'current_user' not in st.session_state: st.session_state.current_user = None

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
    # GLOBAL SIDEBAR NAVIGATION
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
                    st.session_state.active_t = new_t
                    st.session_state.active_r = "Round 1"
                    st.session_state.page = "Tournament Hub"
                    st.rerun()
        
        st.divider()
        all_t = st.session_state.shots_data['Tournament'].unique().tolist() if not st.session_state.shots_data.empty else []
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
                    st.session_state.workflow_step = "Score & Driving"
                    st.session_state.page = "Data Entry"
                    st.rerun()
        
        st.divider()
        st.subheader("Tournament Tools")
        if st.button("📊 View Tournament Dashboard", use_container_width=True):
            st.session_state.workflow_step = "Master Dashboard"
            st.session_state.page = "Data Entry"
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
        st.title(f"{st.session_state.active_t} - {st.session_state.active_r if st.session_state.workflow_step != 'Master Dashboard' else 'Tournament Dashboard'}")
        
        steps = ["Score & Driving", "Scoring Zone", "Short Game", "Putting", "Mental & Judgement", "Master Dashboard"] 
        selected_step = st.radio("Phase:", steps, horizontal=True, index=steps.index(st.session_state.workflow_step) if st.session_state.workflow_step in steps else 0)
        
        if selected_step != st.session_state.workflow_step:
            st.session_state.workflow_step = selected_step
            st.rerun()
            
        st.divider()

        current_stats = load_round_stats(st.session_state.current_user, st.session_state.active_t, st.session_state.active_r)
        cid = current_stats['id']

        if st.session_state.workflow_step == "Score & Driving":
            st.subheader("Round Score")
            col_s1, col_s2 = st.columns(2)
            
            col_s1.number_input("Gross Score (e.g. 70)", min_value=0, max_value=150, value=current_stats.get('gross_score',0), key=f"gs_{cid}", on_change=auto_save_stat, args=("gross_score", f"gs_{cid}", cid))
            col_s2.number_input("To Par (e.g. -2, E=0, +3)", min_value=-30, max_value=30, value=current_stats.get('to_par',0), key=f"tp_{cid}", on_change=auto_save_stat, args=("to_par", f"tp_{cid}", cid))
            
            st.divider()
            st.subheader("Off The Tee")
            t_tabs = st.tabs(["OTT: Driver", "OTT: Others"])
            for i, r_label in enumerate(["OTT: Driver", "OTT: Others"]):
                with t_tabs[i]:
                    df_v = st.session_state.shots_data[(st.session_state.shots_data['Tournament'] == st.session_state.active_t) & (st.session_state.shots_data['Round'] == st.session_state.active_r) & (st.session_state.shots_data['Range'] == r_label)]
                    img_obj = create_tee_image(df_v, r_label)
                    val = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
                    
                    if val:
                        px, py = val['x'], val['y']
                        y_min, y_max = (270, 320) if r_label == "OTT: Driver" else (220, 270)
                        x_m = round((px / 500.0) * 60 - 30, 2)
                        y_m = round(y_max - (py / 500.0) * 50, 2)
                        supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Round": st.session_state.active_r, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                        st.toast("☁️ Saved securely to cloud", icon="✅")
                        st.session_state.shots_data = load_shots(st.session_state.current_user)
                        st.rerun()
                        
                    if not df_v.empty:
                        tot = len(df_v)
                        dx = df_v['X'].abs()
                        fwys = len(df_v[dx <= 10])
                        pens = len(df_v[dx > 20])
                        st.success(f"**Tournament Sheet Stat:** {(fwys/tot)*100:.0f}% ({pens})") 
                        if st.button(f"Undo Last Drive", key=f"un_{r_label}"):
                            supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                            st.session_state.shots_data = load_shots(st.session_state.current_user)
                            st.rerun()

        elif st.session_state.workflow_step == "Scoring Zone":
            t_tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
            for i, r_label in enumerate(["50-100", "101-150", "151-200"]):
                with t_tabs[i]:
                    df_v = st.session_state.shots_data[(st.session_state.shots_data['Tournament'] == st.session_state.active_t) & (st.session_state.shots_data['Round'] == st.session_state.active_r) & (st.session_state.shots_data['Range'] == r_label)]
                    img_obj = create_target_image(df_v, r_label)
                    val = streamlit_image_coordinates(img_obj, key=f"img_{r_label}_{len(df_v)}")
                    
                    if val:
                        px, py = val['x'], val['y']
                        _, limit = get_radii(r_label)
                        limit += 2
                        x_m = round((px / 500.0) * (2 * limit) - limit, 2)
                        y_m = round(limit - (py / 500.0) * (2 * limit), 2)
                        supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Round": st.session_state.active_r, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                        st.toast("☁️ Saved securely to cloud", icon="✅")
                        st.session_state.shots_data = load_shots(st.session_state.current_user)
                        st.rerun()
                        
                    if not df_v.empty:
                        df_v = df_v.copy()
                        df_v['d'] = np.sqrt(df_v['X']**2 + df_v['Y']**2)
                        rb, rp = get_radii(r_label)
                        b = len(df_v[df_v['d'] <= rb])
                        bog = len(df_v[df_v['d'] > rp])
                        tot = len(df_v)
                        to_par = (b * -1) + (bog * 1)
                        st.info(f"**Tournament Sheet Stat:** {to_par}({tot})") 
                        if st.button(f"Undo Last Shot", key=f"un_{r_label}"):
                            supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute()
                            st.session_state.shots_data = load_shots(st.session_state.current_user)
                            st.rerun()
                            
            st.divider()
            st.subheader("Manual Inputs (Scoring Zone)")
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("GIR < 5", min_value=0, max_value=18, value=current_stats.get('gir_less_5', 0), key=f"g5_{cid}", on_change=auto_save_stat, args=("gir_less_5", f"g5_{cid}", cid))
            with col2:
                st.number_input("Total GIR", min_value=0, max_value=18, value=current_stats.get('gir', 0), key=f"g_{cid}", on_change=auto_save_stat, args=("gir", f"g_{cid}", cid))

        elif st.session_state.workflow_step == "Short Game":
            st.subheader("Short Game (SG)")
            sg_tot = st.number_input("Total SG Shots (#)", min_value=0, value=current_stats.get('sg_total', 0), key=f"sgt_{cid}", on_change=auto_save_stat, args=("sg_total", f"sgt_{cid}", cid))
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Successes:**")
                st.number_input("< 6ft", min_value=0, max_value=sg_tot, value=current_stats.get('sg_inside_6', 0), key=f"sg6_{cid}", on_change=auto_save_stat, args=("sg_inside_6", f"sg6_{cid}", cid))
                st.number_input("< 3ft", min_value=0, max_value=sg_tot, value=current_stats.get('sg_inside_3', 0), key=f"sg3_{cid}", on_change=auto_save_stat, args=("sg_inside_3", f"sg3_{cid}", cid))
                st.number_input("U&D", min_value=0, max_value=sg_tot, value=current_stats.get('sg_ud', 0), key=f"sgu_{cid}", on_change=auto_save_stat, args=("sg_ud", f"sgu_{cid}", cid))
                st.number_input("SGZ Score", value=current_stats.get('sgz_score', 0), key=f"sgz_{cid}", on_change=auto_save_stat, args=("sgz_score", f"sgz_{cid}", cid))
            with col2:
                st.markdown("**Tournament Sheet Output:**")
                if sg_tot > 0:
                    st.write(f"**< 6:** {(current_stats.get('sg_inside_6',0)/sg_tot)*100:.0f}%")
                    st.write(f"**< 3:** {(current_stats.get('sg_inside_3',0)/sg_tot)*100:.0f}%")
                    st.write(f"**U&D:** {(current_stats.get('sg_ud',0)/sg_tot)*100:.0f}%")
                    st.write(f"**SGZ:** {current_stats.get('sgz_score',0)}({sg_tot})")

        elif st.session_state.workflow_step == "Putting":
            st.subheader("18-Hole SG Putting Calculator")
            
            metric_cols = st.columns(2)
            m_putts = metric_cols[0].empty()
            m_sg = metric_cols[1].empty()
            
            raw_grid = current_stats.get('putting_holes')
            if not raw_grid or len(raw_grid) != 18:
                raw_grid = [{"Hole": f"Hole {i}", "Distance (ft)": 0, "Putts": 0} for i in range(1, 19)]
            
            new_grid = []
            total_putts = 0
            total_sg = 0.0
            
            st.caption("Slide to select distance, tap to select putts (0 = Not Played).")
            
            with st.expander("⛳ Front 9", expanded=True):
                c_header1, c_header2 = st.columns([3, 2])
                c_header1.markdown("**Distance (ft)**")
                c_header2.markdown("**Putts**")
                
                for i in range(9):
                    c1, c2 = st.columns([3, 2])
                    dist = c1.slider(f"Hole {i+1} Dist", 0, 100, int(raw_grid[i]["Distance (ft)"]), key=f"dist_{cid}_{i}", label_visibility="collapsed")
                    putts = c2.radio(f"Hole {i+1} Putts", [0, 1, 2, 3, 4], index=int(raw_grid[i]["Putts"]), horizontal=True, key=f"putts_{cid}_{i}", label_visibility="collapsed")
                    new_grid.append({"Hole": f"Hole {i+1}", "Distance (ft)": dist, "Putts": putts})
                    
            with st.expander("⛳ Back 9", expanded=False):
                c_header1, c_header2 = st.columns([3, 2])
                c_header1.markdown("**Distance (ft)**")
                c_header2.markdown("**Putts**")
                
                for i in range(9, 18):
                    c1, c2 = st.columns([3, 2])
                    dist = c1.slider(f"Hole {i+1} Dist", 0, 100, int(raw_grid[i]["Distance (ft)"]), key=f"dist_{cid}_{i}", label_visibility="collapsed")
                    putts = c2.radio(f"Hole {i+1} Putts", [0, 1, 2, 3, 4], index=int(raw_grid[i]["Putts"]), horizontal=True, key=f"putts_{cid}_{i}", label_visibility="collapsed")
                    new_grid.append({"Hole": f"Hole {i+1}", "Distance (ft)": dist, "Putts": putts})
            
            for row in new_grid:
                dist = row["Distance (ft)"]
                putts = row["Putts"]
                if putts > 0:
                    total_putts += putts
                if dist > 0 and putts > 0:
                    total_sg += (get_expected_putts(dist) - putts)
                    
            m_putts.metric("Total Putts", total_putts)
            m_sg.metric("Total SG Putting", f"{total_sg:+.2f}")
            
            if new_grid != raw_grid:
                supabase.table("round_stats").update({
                    "putting_holes": new_grid, 
                    "putts_total": total_putts, 
                    "sg_putting": round(total_sg, 2)
                }).eq("id", cid).execute()
            
            st.divider()
            st.markdown("### Lag Putting")
            lt = st.number_input("Total Lag Putts", min_value=0, value=current_stats.get('lag_total', 0), key=f"lt_{cid}", on_change=auto_save_stat, args=("lag_total", f"lt_{cid}", cid))
            ls = st.number_input("Lags inside putter length", min_value=0, max_value=lt if lt>0 else 0, value=current_stats.get('lag_success', 0), key=f"ls_{cid}", on_change=auto_save_stat, args=("lag_success", f"ls_{cid}", cid))
            if lt > 0: 
                st.write(f"**Lag:** {(ls/lt)*100:.0f}%")

        elif st.session_state.workflow_step == "Mental & Judgement":
            st.subheader("Mental (M), Judgements (J), & Course Management (CM)")
            st.slider("Mental Score (M)", min_value=0, max_value=100, value=current_stats.get('mental_score', 0), key=f"ms_{cid}", on_change=auto_save_stat, args=("mental_score", f"ms_{cid}", cid))
            st.slider("Judgement Score (J)", min_value=0, max_value=100, value=current_stats.get('judgement_score', 0), key=f"js_{cid}", on_change=auto_save_stat, args=("judgement_score", f"js_{cid}", cid))
            st.slider("Course Management Score (CM)", min_value=0, max_value=100, value=current_stats.get('cm_score', 0), key=f"cm_{cid}", on_change=auto_save_stat, args=("cm_score", f"cm_{cid}", cid))

        elif st.session_state.workflow_step == "Master Dashboard":
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
