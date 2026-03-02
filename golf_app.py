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
