import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os
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

pga_putts_baseline = {
    1: 1.00, 2: 1.01, 3: 1.04, 4: 1.13, 5: 1.23, 6: 1.34, 7: 1.43, 8: 1.50, 
    9: 1.56, 10: 1.61, 15: 1.78, 20: 1.87, 25: 1.94, 30: 2.01, 40: 2.13, 50: 2.26, 60: 2.38,
    70: 2.48, 80: 2.58, 90: 2.65, 100: 2.71
}

def get_expected_putts(distance):
    closest_dist = min(pga_putts_baseline.keys(), key=lambda k: abs(k - distance))
    return pga_putts_baseline[closest_dist]

# --- 2. DATA LOADING & AUTO-SAVE ---
def load_shots(current_user):
    response = supabase.table("shots").select("*").eq("User", current_user).execute()
    return pd.DataFrame(response.data) if response.data else pd.DataFrame(columns=["id", "User", "Tournament", "Round", "Range", "X", "Y"])

def load_round_stats(current_user, tournament, round_num):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).eq("tournament", tournament).eq("round_num", round_num).execute()
    if response.data: return response.data[0]
    blank = {
        "user_name": current_user, "tournament": tournament, "round_num": round_num,
        "gross_score": 0, "course_par": 72, "gir": 0, "gir_less_5": 0, "sg_total": 0, "sg_inside_6": 0, "sg_inside_3": 0, "sg_ud": 0, "sgz_score": 0,
        "putts_total": 0, "sg_putting": 0.0, "lag_success": 0, "lag_total": 0, "mental_score": 0, "judgement_score": 0, "cm_score": 0, "putting_holes": None
    }
    res = supabase.table("round_stats").insert(blank).execute()
    return res.data[0]

def load_all_stats(current_user):
    response = supabase.table("round_stats").select("*").eq("user_name", current_user).execute()
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
    r_b, r_p = get_radii(label); limit = r_p + 2 
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100); ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.axis('off') 
    ax.add_patch(patches.Rectangle((-limit, -limit), limit*2, limit*2, linewidth=4, edgecolor='black', facecolor='white'))
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4); ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
    ax.add_patch(patches.Circle((0, 0), r_b, linewidth=2, edgecolor='blue', facecolor='#ADD8E6', alpha=0.4))
    ax.add_patch(patches.Circle((0, 0), r_p, linewidth=2, edgecolor='blue', facecolor='none'))
    if not df_filtered.empty:
        df = df_filtered.copy(); df['dist'] = np.sqrt(df['X']**2 + df['Y']**2)
        colors = df['dist'].apply(lambda d: 'red' if d <= r_b else ('blue' if d <= r_p else 'black'))
        ax.scatter(df['X'], df['Y'], c=colors, s=65, edgecolors='white', linewidths=1.5, zorder=5)
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig); return Image.open(buf)

def create_tee_image(df_filtered, label):
    y_min, y_max = (270, 320) if label == "OTT: Driver" else (220, 270); x_limit = 30 
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100); ax.set_xlim(-x_limit, x_limit); ax.set_ylim(y_min, y_max); ax.axis('off')
    ax.add_patch(patches.Rectangle((-x_limit, y_min), x_limit*2, y_max-y_min, linewidth=4, edgecolor='black', facecolor='white'))
    ax.axvspan(-10, 10, facecolor='#ADD8E6', alpha=0.4)
    ax.axvline(0, color='blue', linestyle='solid', linewidth=2); ax.axvline(-10, color='blue', linestyle='dashed', linewidth=2)
    ax.axvline(10, color='blue', linestyle='dashed', linewidth=2); ax.axvline(-20, color='blue', linestyle='dotted', linewidth=2); ax.axvline(20, color='blue', linestyle='dotted', linewidth=2)
    if not df_filtered.empty:
        df = df_filtered.copy(); colors = df['X'].abs().apply(lambda d: 'red' if d <= 10 else ('blue' if d <= 20 else 'black'))
        ax.scatter(df['X'], df['Y'], c=colors, s=65, edgecolors='white', linewidths=1.5, zorder=5)
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig); return Image.open(buf)

# --- 4. DATA ENGINE ---
def format_score_cell(list_s):
    if not list_s: return "-"
    total_gross = sum(s.get('gross_score', 0) for s in list_s)
    total_par = sum(s.get('course_par', 72) for s in list_s)
    to_par = total_gross - total_par
    avg_gross = total_gross / len(list_s)
    sign = "+" if to_par > 0 else ""
    return f"{avg_gross:.1f} ({sign}{to_par})"

def build_master_dataframe(df_shots, list_stats, mode="tournament"):
    headers = ["Round 1", "Round 2", "Round 3", "Round 4"] if mode == "tournament" else []
    data = []
    
    # TOP ROW: SCORE
    row_score = {"Category": "Score (To Par)"}
    for h in headers:
        s_r = [s for s in list_stats if s['round_num'] == h]
        row_score[h] = format_score_cell(s_r)
    row_score["AV / TOTAL"] = format_score_cell(list_stats)
    data.append(row_score)

    def add_section(title): data.append({"Category": title, "AV / TOTAL": "", **{h: "" for h in headers}})
    def add_row(cat, logic, param=""):
        row = {"Category": cat}
        for h in headers:
            n, d, e = calc_metrics(df_shots[df_shots['Round']==h], [s for s in list_stats if s['round_num']==h], logic, param)
            row[h] = format_cell(logic, n, d, e)
        n_all, d_all, e_all = calc_metrics(df_shots, list_stats, logic, param)
        row["AV / TOTAL"] = format_cell(logic, n_all, d_all, e_all)
        data.append(row)

    add_section("LONG GAME"); add_row("OTT: Driver", "driving", "OTT: Driver"); add_row("OTT: Others", "driving", "OTT: Others")
    add_section("SCORING ZONE"); add_row("151-200m", "approach", "151-200"); add_row("101-150m", "approach", "101-150"); add_row("50-100m", "approach", "50-100")
    add_row("GIR / 5", "abs", "gir_less_5"); add_row("GIR", "abs", "gir")
    add_section("SHORT GAME"); add_row("< 6", "sg_perc", "sg_inside_6"); add_row("< 3", "sg_perc", "sg_inside_3"); add_row("U&D", "sg_perc", "sg_ud"); add_row("SGZ", "sgz")
    add_section("PUTTING"); add_row("Putts (#)", "abs", "putts_total"); add_row("SG Putting", "sg_putt"); add_row("Lag", "lag")
    add_section("MENTAL & JUDGEMENTS"); add_row("M", "abs", "mental_score"); add_row("J", "abs", "judgement_score"); add_row("CM", "abs", "cm_score")
    return pd.DataFrame(data)

def calc_metrics(df_s, list_s, logic_type, param):
    num, den, extra = 0, 0, 0
    if logic_type == "driving":
        df_d = df_s[df_s['Range'] == param]; den = len(df_d)
        if den > 0: num = len(df_d[df_d['X'].abs() <= 10]); extra = len(df_d[df_d['X'].abs() > 20])
    elif logic_type == "approach":
        df_a = df_s[df_s['Range'] == param]; den = len(df_a)
        if den > 0:
            df_a = df_a.copy(); df_a['d'] = np.sqrt(df_a['X']**2 + df_a['Y']**2); rb, rp = get_radii(param)
            num = (len(df_a[df_a['d'] > rp]) - len(df_a[df_a['d'] <= rb]))
    elif logic_type == "abs":
        for s in list_s: 
            v = s.get(param, 0)
            if v != 0: num += v; den += 1
    elif logic_type == "sg_perc":
        for s in list_s: num += s.get(param, 0); den += s.get('sg_total', 0)
    elif logic_type == "sgz":
        for s in list_s: num += s.get('sgz_score', 0); den += s.get('sg_total', 0)
    elif logic_type == "lag":
        for s in list_s: num += s.get('lag_success', 0); den += s.get('lag_total', 0)
    elif logic_type == "sg_putt":
        for s in list_s:
            v = s.get('sg_putting', 0.0)
            if v != 0: num += v; den += 1
    return num, den, extra

def format_cell(logic_type, num, den, extra):
    if den == 0: return "-"
    if logic_type == "driving": return f"{(num/den)*100:.0f}% ({extra})"
    if logic_type == "approach": sign = "+" if num > 0 else ""; return f"{sign}{num}({den})"
    if logic_type in ["abs", "sg_putt"]:
        val = num / den
        if logic_type == "sg_putt": sign = "+" if val > 0 else ""; return f"{sign}{val:.2f}"
        return f"{val:.1f}"
    if logic_type in ["sg_perc", "lag"]: return f"{(num/den)*100:.0f}%"
    if logic_type == "sgz": return f"{num}({den})"
    return "-"

# --- 5. GLOBAL ROUTING ---
if 'page' not in st.session_state: st.session_state.page = "Login"
if st.session_state.page == "Login" or not st.session_state.get('current_user'):
    st.markdown("<h1 style='text-align: center; font-size: 4em; margin-top: 10%;'>The Score Code</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username_input = st.text_input("Enter Username to Access Vault").strip()
        if st.button("Authenticate", use_container_width=True):
            if username_input: st.session_state.current_user = username_input; st.session_state.shots_data = load_shots(username_input); st.session_state.page = "Season Hub"; st.rerun()
else:
    st.sidebar.title("👤 Player Profile")
    st.sidebar.write(f"**{st.session_state.current_user}**")
    if st.sidebar.button("Log Out"): st.session_state.page = "Login"; st.session_state.current_user = None; st.rerun()
    st.sidebar.divider(); st.sidebar.header("🧭 Navigation")
    if st.sidebar.button("🏠 Season Hub", use_container_width=True): st.session_state.active_t = None; st.session_state.page = "Season Hub"; st.rerun()
    if st.sidebar.button("📊 Season Master Dashboard", use_container_width=True): st.session_state.page = "Season Master"; st.rerun()
    if st.session_state.get('active_t'):
        st.sidebar.divider()
        if st.sidebar.button(f"🔙 {st.session_state.active_t} Hub", use_container_width=True): st.session_state.page = "Tournament Hub"; st.rerun()

    if st.session_state.page == "Season Hub":
        st.title("🏠 Season Hub")
        with st.expander("➕ New Tournament"):
            new_t = st.text_input("Name:")
            if st.button("Create"): st.session_state.active_t = new_t; st.session_state.active_r = "Round 1"; st.session_state.page = "Tournament Hub"; st.rerun()
        st.divider(); all_t = st.session_state.shots_data['Tournament'].unique().tolist()
        cols = st.columns(4)
        for i, t in enumerate(all_t):
            with cols[i % 4]:
                if st.button(f"⛳ {t}", use_container_width=True): st.session_state.active_t = t; st.session_state.page = "Tournament Hub"; st.rerun()

    elif st.session_state.page == "Tournament Hub":
        st.title(f"⛳ {st.session_state.active_t}")
        c1, c2, c3, c4 = st.columns(4)
        for col, r_name in zip([c1, c2, c3, c4], ["Round 1", "Round 2", "Round 3", "Round 4"]):
            if col.button(f"Edit {r_name}", use_container_width=True): st.session_state.active_r = r_name; st.session_state.workflow_step = "Score & Driving"; st.session_state.page = "Data Entry"; st.rerun()
        st.divider()
        if st.button("📊 View Master Dashboard", use_container_width=True): st.session_state.workflow_step = "Master Dashboard"; st.session_state.page = "Data Entry"; st.rerun()

    elif st.session_state.page == "Season Master":
        st.title("📊 Season Master Dashboard")
        raw_shots = load_shots(st.session_state.current_user); raw_stats = load_all_stats(st.session_state.current_user)
        
        # Scoring Tier Logic
        t_aggregates = []
        unique_ts = sorted(list(set(raw_shots['Tournament'].unique().tolist() + [s['tournament'] for s in raw_stats])))
        for t in unique_ts:
            t_stats = [s for s in raw_stats if s['tournament'] == t]
            if t_stats:
                avg_to_par = sum(s.get('gross_score',0) - s.get('course_par',72) for s in t_stats) / len(t_stats)
                t_aggregates.append({'name': t, 'avg': avg_to_par})
        
        t_df = pd.DataFrame(t_aggregates).sort_values('avg')
        top_cutoff = t_df['avg'].quantile(0.3); bottom_cutoff = t_df['avg'].quantile(0.7)
        
        col_f1, col_f2 = st.columns(2)
        selected_ts = col_f1.multiselect("Filter Tournaments:", options=unique_ts, default=unique_ts)
        tier_filter = col_f2.selectbox("Filter by Performance:", ["All Data", "Best 30% Rounds", "Bottom 30% Rounds"])
        
        final_stats = [s for s in raw_stats if s['tournament'] in selected_ts]
        if tier_filter == "Best 30% Rounds": final_stats = [s for s in final_stats if (s.get('gross_score',0) - s.get('course_par',72)) <= top_cutoff]
        elif tier_filter == "Bottom 30% Rounds": final_stats = [s for s in final_stats if (s.get('gross_score',0) - s.get('course_par',72)) >= bottom_cutoff]
        
        final_ts_names = list(set([s['tournament'] for s in final_stats]))
        final_shots = raw_shots[raw_shots['Tournament'].isin(final_ts_names)]
        
        df_m = build_master_dataframe(final_shots, final_stats, mode="season")
        st.markdown("<style>.stTable td:first-child { width: 50% !important; font-weight: bold; }</style>", unsafe_allow_html=True)
        st.table(df_m.set_index('Category'))
        
        st.divider(); st.write("### Dispersion Analytics")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("50-100m"); st.image(create_target_image(final_shots[final_shots['Range'] == '50-100'], '50-100'))
        with c2: st.markdown("101-150m"); st.image(create_target_image(final_shots[final_shots['Range'] == '101-150'], '101-150'))
        with c3: st.markdown("151-200m"); st.image(create_target_image(final_shots[final_shots['Range'] == '151-200'], '151-200'))
        c4, c5 = st.columns(2)
        with c4: st.markdown("OTT: Driver"); st.image(create_tee_image(final_shots[final_shots['Range'] == 'OTT: Driver'], 'OTT: Driver'))
        with c5: st.markdown("OTT: Others"); st.image(create_tee_image(final_shots[final_shots['Range'] == 'OTT: Others'], 'OTT: Others'))

    elif st.session_state.page == "Data Entry":
        steps = ["Score & Driving", "Scoring Zone", "Short Game", "Putting", "Mental & Judgement", "Master Dashboard"] 
        selected_step = st.radio("Phase:", steps, horizontal=True, index=steps.index(st.session_state.workflow_step))
        if selected_step != st.session_state.workflow_step: st.session_state.workflow_step = selected_step; st.rerun()
        current_stats = load_round_stats(st.session_state.current_user, st.session_state.active_t, st.session_state.active_r); cid = current_stats['id']

        if st.session_state.workflow_step == "Score & Driving":
            col_s1, col_s2 = st.columns(2)
            col_s1.number_input("Round Score (Gross)", min_value=0, value=current_stats.get('gross_score',0), key=f"gs_{cid}", on_change=auto_save_stat, args=("gross_score", f"gs_{cid}", cid))
            col_s2.number_input("Course Par", min_value=60, max_value=75, value=current_stats.get('course_par',72), key=f"cp_{cid}", on_change=auto_save_stat, args=("course_par", f"cp_{cid}", cid))
            st.divider(); t_tabs = st.tabs(["OTT: Driver", "OTT: Others"])
            for i, r_label in enumerate(["OTT: Driver", "OTT: Others"]):
                with t_tabs[i]:
                    df_v = st.session_state.shots_data[(st.session_state.shots_data['Tournament'] == st.session_state.active_t) & (st.session_state.shots_data['Round'] == st.session_state.active_r) & (st.session_state.shots_data['Range'] == r_label)]
                    val = streamlit_image_coordinates(create_tee_image(df_v, r_label), key=f"img_{r_label}_{len(df_v)}")
                    if val:
                        px, py = val['x'], val['y']; y_min, y_max = (270, 320) if r_label == "OTT: Driver" else (220, 270); x_m = round((px / 500.0) * 60 - 30, 2); y_m = round(y_max - (py / 500.0) * 50, 2)
                        supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Round": st.session_state.active_r, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                        st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()
                    if not df_v.empty and st.button(f"Undo Last Drive", key=f"un_{r_label}"): supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute(); st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()

        elif st.session_state.workflow_step == "Scoring Zone":
            t_tabs = st.tabs(["50-100m", "101-150m", "151-200m"])
            for i, r_label in enumerate(["50-100", "101-150", "151-200"]):
                with t_tabs[i]:
                    df_v = st.session_state.shots_data[(st.session_state.shots_data['Tournament'] == st.session_state.active_t) & (st.session_state.shots_data['Round'] == st.session_state.active_r) & (st.session_state.shots_data['Range'] == r_label)]
                    val = streamlit_image_coordinates(create_target_image(df_v, r_label), key=f"img_{r_label}_{len(df_v)}")
                    if val:
                        px, py = val['x'], val['y']; _, limit = get_radii(r_label); limit += 2; x_m = round((px / 500.0) * (2 * limit) - limit, 2); y_m = round(limit - (py / 500.0) * (2 * limit), 2)
                        supabase.table("shots").insert({"User": st.session_state.current_user, "Tournament": st.session_state.active_t, "Round": st.session_state.active_r, "Range": r_label, "X": x_m, "Y": y_m}).execute()
                        st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()
                    if not df_v.empty and st.button(f"Undo Last Shot", key=f"un_{r_label}"): supabase.table("shots").delete().eq("id", int(df_v.iloc[-1]['id'])).execute(); st.session_state.shots_data = load_shots(st.session_state.current_user); st.rerun()
            st.divider(); col1, col2 = st.columns(2)
            col1.number_input("GIR < 5", min_value=0, max_value=18, value=current_stats.get('gir_less_5', 0), key=f"g5_{cid}", on_change=auto_save_stat, args=("gir_less_5", f"g5_{cid}", cid))
            col2.number_input("Total GIR", min_value=0, max_value=18, value=current_stats.get('gir', 0), key=f"g_{cid}", on_change=auto_save_stat, args=("gir", f"g_{cid}", cid))

        elif st.session_state.workflow_step == "Short Game":
            sg_tot = st.number_input("Total SG Shots (#)", min_value=0, value=current_stats.get('sg_total', 0), key=f"sgt_{cid}", on_change=auto_save_stat, args=("sg_total", f"sgt_{cid}", cid))
            st.number_input("< 6ft", min_value=0, max_value=sg_tot, value=current_stats.get('sg_inside_6', 0), key=f"sg6_{cid}", on_change=auto_save_stat, args=("sg_inside_6", f"sg6_{cid}", cid))
            st.number_input("< 3ft", min_value=0, max_value=sg_tot, value=current_stats.get('sg_inside_3', 0), key=f"sg3_{cid}", on_change=auto_save_stat, args=("sg_inside_3", f"sg3_{cid}", cid))
            st.number_input("U&D", min_value=0, max_value=sg_tot, value=current_stats.get('sg_ud', 0), key=f"sgu_{cid}", on_change=auto_save_stat, args=("sg_ud", f"sgu_{cid}", cid))
            st.number_input("SGZ Score", value=current_stats.get('sgz_score', 0), key=f"sgz_{cid}", on_change=auto_save_stat, args=("sgz_score", f"sgz_{cid}", cid))

        elif st.session_state.workflow_step == "Putting":
            df_putts = pd.DataFrame(current_stats.get('putting_holes') or {"Hole":[f"Hole {i}" for i in range(1,19)], "Distance (ft)":[0]*18, "Putts":[0]*18})
            edited = st.data_editor(df_putts, hide_index=True, use_container_width=True, num_rows="fixed", key=f"grid_{cid}")
            total_putts = int(edited["Putts"].sum()); total_sg = sum((get_expected_putts(r["Distance (ft)"]) - r["Putts"]) for _,r in edited.iterrows() if r["Distance (ft)"]>0 and r["Putts"]>0)
            st.metric("Total Putts", total_putts); st.metric("Total SG Putting", f"{total_sg:+.2f}")
            if edited.to_dict('records') != current_stats.get('putting_holes'): supabase.table("round_stats").update({"putting_holes": edited.to_dict('records'), "putts_total": total_putts, "sg_putting": round(total_sg, 2)}).eq("id", cid).execute()

        elif st.session_state.workflow_step == "Mental & Judgement":
            st.slider("Mental Score (M)", 0, 100, current_stats.get('mental_score',0), key=f"ms_{cid}", on_change=auto_save_stat, args=("mental_score", f"ms_{cid}", cid))
            st.slider("Judgement Score (J)", 0, 100, current_stats.get('judgement_score',0), key=f"js_{cid}", on_change=auto_save_stat, args=("judgement_score", f"js_{cid}", cid))
            st.slider("CM Score", 0, 100, current_stats.get('cm_score',0), key=f"cm_{cid}", on_change=auto_save_stat, args=("cm_score", f"cm_{cid}", cid))

        elif st.session_state.workflow_step == "Master Dashboard":
            df_m = build_master_dataframe(st.session_state.shots_data[st.session_state.shots_data['Tournament']==st.session_state.active_t], load_all_tournament_stats(st.session_state.current_user, st.session_state.active_t), mode="tournament")
            st.table(df_m.set_index('Category'))
