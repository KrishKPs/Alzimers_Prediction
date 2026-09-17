import warnings
from datetime import date
from pathlib import Path

import joblib, numpy as np, pandas as pd
import plotly.graph_objects as go
import shap, streamlit as st
from fpdf import FPDF

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "model.pkl"

st.set_page_config(page_title="NeuroRisk Pro", page_icon="🧠", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#07111f;--panel:rgba(13,24,41,.9);--border:rgba(91,140,255,.13);
  --border-dim:rgba(255,255,255,.07);--text:#edf2ff;--muted:#94a3b8;--dim:#475569;
  --blue:#5b8cff;--purple:#8b5cf6;--green:#22c55e;--amber:#f59e0b;--red:#ef4444;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif !important;}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse at 0% 0%,rgba(91,140,255,.11) 0%,transparent 50%),
    radial-gradient(ellipse at 100% 0%,rgba(139,92,246,.09) 0%,transparent 40%),#07111f;
  color:var(--text);
}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,
footer,[data-testid="stSidebar"]{display:none !important;}
section[data-testid="stMain"]>div{padding-top:24px !important;}
div[data-testid="stForm"]{border:none;padding:0;}
p,span,div,h1,h2,h3,li{color:var(--text);}

.card{background:var(--panel);border:1px solid var(--border-dim);
  border-radius:16px;padding:20px 22px;margin-bottom:12px;}
.card-blue{border-left:3px solid var(--blue);}
.card-purple{border-left:3px solid var(--purple);}
.sec{color:var(--muted);text-transform:uppercase;letter-spacing:.13em;
  font-size:.67rem;font-weight:700;margin-bottom:12px;}

/* stepper */
.stepper{display:flex;align-items:flex-start;margin-top:14px;gap:0;}
.snode{display:flex;flex-direction:column;align-items:center;gap:5px;flex:1;}
.scircle{width:26px;height:26px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;font-size:.72rem;font-weight:700;position:relative;z-index:1;}
.sdone{background:linear-gradient(135deg,#5b8cff,#8b5cf6);color:#fff;}
.sactive{background:linear-gradient(135deg,#5b8cff,#8b5cf6);color:#fff;
  box-shadow:0 0 0 4px rgba(91,140,255,.22);}
.sidle{background:rgba(255,255,255,.06);color:var(--dim);
  border:1px solid rgba(255,255,255,.1);}
.slabel{font-size:.63rem;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.07em;text-align:center;}
.slabel-active{color:var(--text);}
.sline{height:1px;flex:1;margin:0 -6px;margin-top:-22px;
  background:rgba(255,255,255,.1);z-index:0;}
.sline-done{background:linear-gradient(90deg,#5b8cff,#8b5cf6);}

/* chips */
.chip{display:inline-flex;margin:2px 4px 2px 0;padding:5px 11px;
  border-radius:999px;background:rgba(255,255,255,.05);color:var(--text);
  border:1px solid rgba(255,255,255,.07);font-size:.78rem;}

/* metric tiles */
.tiles{display:flex;gap:7px;margin:10px 0;flex-wrap:wrap;}
.tile{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);
  border-radius:12px;padding:11px 8px;text-align:center;flex:1;min-width:60px;}
.tlbl{color:var(--muted);font-size:.6rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.08em;}
.tval{font-size:1.2rem;font-weight:800;margin:4px 0 2px;}
.tsub{color:var(--muted);font-size:.6rem;}

/* risk hero — signature element */
.risk-hero{display:flex;align-items:flex-end;justify-content:space-between;
  padding:32px 40px;border-radius:20px;margin-bottom:18px;}
.risk-eyebrow{font-size:.67rem;letter-spacing:.22em;font-weight:800;
  text-transform:uppercase;margin-bottom:6px;}
.risk-interp{font-size:.86rem;color:var(--muted);max-width:360px;line-height:1.72;}
.risk-pct{font-family:'DM Serif Display',serif;font-size:5.5rem;
  line-height:1;text-align:right;}
.risk-pct-lbl{font-size:.67rem;text-transform:uppercase;letter-spacing:.12em;
  color:var(--muted);text-align:right;margin-top:3px;}

/* prob bars */
.prow{display:flex;align-items:center;gap:10px;margin-bottom:9px;}
.pname{min-width:98px;font-size:.76rem;color:var(--muted);}
.pname-a{color:var(--text);font-weight:700;}
.pbg{flex:1;background:rgba(255,255,255,.07);border-radius:3px;height:5px;}
.ppct{font-size:.76rem;min-width:36px;text-align:right;}

/* shap */
.sfact{display:flex;align-items:flex-start;gap:8px;padding:9px 11px;
  background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
  border-radius:10px;margin-bottom:6px;}

/* disclaimer */
.disc{border-radius:12px;padding:11px 15px;margin-top:14px;
  background:rgba(59,130,246,.08);border:1px solid rgba(96,165,250,.18);
  color:#93c5fd;font-size:.77rem;line-height:1.65;}

/* inputs */
input,textarea{background:rgba(255,255,255,.05)!important;
  border:1px solid rgba(255,255,255,.1)!important;
  border-radius:9px!important;color:var(--text)!important;}
[data-baseweb="select"]>div:first-child{background:rgba(255,255,255,.05)!important;
  border:1px solid rgba(255,255,255,.1)!important;border-radius:9px!important;}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:var(--text)!important;}
[data-baseweb="popover"]{background:#0d1625!important;
  border:1px solid rgba(255,255,255,.1)!important;}
[role="option"]{background:#0d1625!important;color:var(--text)!important;}
[role="option"]:hover{background:rgba(91,140,255,.15)!important;}
[data-testid="stNumberInput"] input{background:rgba(255,255,255,.05)!important;color:var(--text)!important;}
[data-testid="stNumberInput"] button{background:rgba(255,255,255,.06)!important;
  color:var(--text)!important;border-color:rgba(255,255,255,.1)!important;}
[data-testid="stSlider"] [role="slider"]{background:#8b5cf6!important;
  border:2px solid #fff!important;box-shadow:0 0 8px rgba(139,92,246,.5)!important;}
div[data-baseweb="slider"]>div:first-child{background:rgba(255,255,255,.1)!important;}
div[data-baseweb="slider"] div[role="progressbar"]{background:#8b5cf6!important;}
[data-testid="stSlider"] [data-testid="stTickBar"]{display:none;}
[data-testid="stCheckbox"] label{background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.07);border-radius:9px;padding:9px 12px;
  cursor:pointer;display:flex!important;align-items:center;gap:8px;}
[data-testid="stCheckbox"] label:hover{border-color:#5b8cff;}
[data-testid="stCheckbox"] span{color:var(--text)!important;font-size:.83rem;font-weight:500;}
[data-testid="stExpander"]{background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.07)!important;border-radius:12px;}
[data-testid="stExpander"] summary{color:var(--text)!important;}
.stButton>button{border-radius:999px!important;padding:.7rem 1.1rem!important;
  font-weight:700!important;border:1px solid rgba(255,255,255,.1)!important;
  background:rgba(255,255,255,.04)!important;color:var(--text)!important;transition:.15s!important;}
.stButton>button:hover{border-color:#5b8cff!important;background:rgba(91,140,255,.1)!important;}
div[data-testid="stForm"] .stButton>button[kind="primaryFormSubmit"],
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#5b8cff,#8b5cf6)!important;
  color:#fff!important;border:none!important;box-shadow:0 4px 18px rgba(91,140,255,.3)!important;}
div[data-testid="stForm"] .stButton>button[kind="primaryFormSubmit"]{
  width:100%;padding:1rem!important;font-size:.96rem!important;border-radius:12px!important;}
label{color:var(--muted)!important;font-size:.72rem!important;
  font-weight:600!important;text-transform:uppercase!important;letter-spacing:.08em!important;}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RISK = {
    0: ("Low Risk",    "#22c55e", "rgba(34,197,94,.1)",  "rgba(34,197,94,.22)"),
    1: ("Medium Risk", "#f59e0b", "rgba(245,158,11,.1)", "rgba(245,158,11,.22)"),
    2: ("High Risk",   "#ef4444", "rgba(239,68,68,.1)",  "rgba(239,68,68,.22)"),
}
INTERP = {
    0: "Patient shows low probability of progression. Routine annual cognitive monitoring is recommended.",
    1: "Patient shows early signs of cognitive conversion. Biannual monitoring and preventive intervention advised.",
    2: "Patient shows high probability of dementia progression. Immediate specialist referral is strongly recommended.",
}
EDUC_MAP = {"High School":12,"Some College":13,"Associate's":14,"Bachelor's":16,"Master's":18,"Doctorate":20}
CDR_OPT  = {0.0:"0 — None",0.5:"0.5 — Questionable",1.0:"1 — Mild",2.0:"2 — Moderate"}
CDR_LBL  = {0.0:"None",0.5:"Questionable",1.0:"Mild",2.0:"Moderate"}
NUM_COLS = ["Age","EDUC","SES","MMSE","CDR","eTIV","nWBV","ASF","MR Delay (Years)",
            "MMSE_change","nWBV_change","CDR_change","baseline_MMSE","baseline_CDR","visit_count","Visit"]
FEAT_LABELS = {
    "M/F_M":"Gender (Male)","Age":"Age","EDUC":"Education (yrs)","SES":"Socioeconomic Status",
    "MMSE":"MMSE Score","CDR":"CDR Rating","eTIV":"Intracranial Volume",
    "nWBV":"Norm. Brain Volume","ASF":"Atlas Scaling Factor",
    "MR Delay (Years)":"Yrs Since 1st Visit","MMSE_change":"MMSE Δ vs Last Visit",
    "nWBV_change":"Brain Vol. Δ","CDR_change":"CDR Δ vs Last Visit",
    "baseline_MMSE":"Baseline MMSE (1st)","baseline_CDR":"Baseline CDR (1st)",
    "visit_count":"Total Visits","Visit":"Visit Number",
}

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_explainer(_p):
    return shap.TreeExplainer(_p.named_steps["model"].estimators_[1])

pipeline  = load_model()
explainer = load_explainer(pipeline) if pipeline is not None else None

for k, v in [("step",1),("fd",{}),("result",None),("vtype","first")]:
    if k not in st.session_state: st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_pt(fd):
    return pd.DataFrame([{
        "M/F":fd["gender"],"Age":fd["age"],"EDUC":fd["educ_yrs"],
        "SES":fd.get("ses",2),"MMSE":fd["mmse"],"CDR":fd["cdr"],
        "eTIV":fd.get("etiv",1470),"nWBV":fd.get("nwbv",0.73),
        "ASF":fd.get("asf",1.19),"MR Delay (Years)":fd.get("mr_delay",0.0),
        "MMSE_change":fd.get("mmse_change",0.0),"nWBV_change":fd.get("nwbv_change",0.0),
        "CDR_change":fd.get("cdr_change",0.0),"baseline_MMSE":fd.get("baseline_mmse",fd["mmse"]),
        "baseline_CDR":fd.get("baseline_cdr",fd["cdr"]),"visit_count":fd.get("visit_count",1),
        "Visit":fd.get("visit",1),
    }])

def run_model(fd):
    pt   = build_pt(fd)
    pred = int(pipeline.predict(pt)[0])
    prob = pipeline.predict_proba(pt)[0]
    Xp   = pipeline.named_steps["scaler"].transform(
               pipeline.named_steps["preprocessor"].transform(pt))
    sv   = explainer.shap_values(Xp)
    vals = sv[0,:,pred]
    ohe  = pipeline.named_steps["preprocessor"]\
               .named_transformers_["cat"].get_feature_names_out(["M/F"]).tolist()
    names = [FEAT_LABELS.get(n,n) for n in ohe+NUM_COLS]
    return pred, prob, vals, names

def make_pdf(fd, pred, prob, shap_vals, feat_names):
    risk_lbl, color, _, _ = RISK[pred]; conf = prob[pred]*100
    rc = {0:(34,197,94),1:(245,158,11),2:(239,68,68)}[pred]
    class DarkPDF(FPDF):
        def header(self):
            self.set_fill_color(7,17,31)
            self.rect(0,0,210,297,"F")
    pdf = DarkPDF(); pdf.set_auto_page_break(auto=True,margin=15); pdf.add_page()

    def s(t): return str(t).replace("—","-").replace("–","-").replace("’","'").replace("Δ","D").replace("³","3").replace("↗","+").replace("↘","-")

    def sec(t):
        pdf.set_font("Helvetica","B",11); pdf.set_text_color(91,140,255)
        pdf.cell(0,10,s(t),ln=True)
        pdf.set_draw_color(91,140,255); pdf.set_line_width(.4)
        pdf.line(10,pdf.get_y(),200,pdf.get_y()); pdf.ln(3)

    def row(l,v):
        pdf.set_font("Helvetica","",10); pdf.set_text_color(148,163,184)
        pdf.cell(68,7,s(l)+":",ln=False); pdf.set_text_color(237,242,255)
        pdf.cell(0,7,s(v),ln=True)

    pdf.set_font("Helvetica","B",18); pdf.set_text_color(237,242,255)
    pdf.cell(0,12,"NeuroRisk Pro - Clinical Report",ln=True,align="C")
    pdf.set_font("Helvetica","",10); pdf.set_text_color(148,163,184)
    pdf.cell(0,8,f"Generated: {date.today().strftime('%B %d, %Y')}",ln=True,align="C"); pdf.ln(5)

    sec("Patient Information")
    row("Patient Name",    fd.get("patient_name","—"))
    row("Patient ID / MRN",fd.get("patient_id","—"))
    row("Date of Birth",   str(fd.get("dob","—")))
    row("Visit Date",      str(fd.get("visit_date",date.today())))
    row("Visit Number",    f"{fd.get('visit',1)}" + (f" of {fd.get('visit_count','—')}" if fd.get("vtype")=="followup" else " (First Visit)"))
    row("Attending",       fd.get("doctor_name","—")); pdf.ln(3)

    sec("Clinical Inputs")
    row("Age",            f"{fd['age']} yrs")
    row("Gender",         "Male" if fd["gender"]=="M" else "Female")
    row("Education",      fd.get("educ","—"))
    row("SES",            fd.get("ses",2))
    row("MMSE (current)", f"{fd['mmse']}/30")
    row("MoCA",           f"{fd.get('moca','—')}/30")
    row("CDR",            f"{fd['cdr']} ({CDR_LBL.get(fd['cdr'],'')})")
    row("nWBV",           f"{fd.get('nwbv',0.73):.3f}")
    row("eTIV",           f"{fd.get('etiv',1470)} mm³")
    if fd.get("vtype")=="followup":
        row("MMSE Δ vs last",  f"{fd.get('mmse_change',0):+.1f}")
        row("CDR Δ vs last",   f"{fd.get('cdr_change',0):+.1f}")
        row("nWBV Δ vs last",  f"{fd.get('nwbv_change',0):+.4f}")
        row("Yrs since 1st visit", f"{fd.get('mr_delay',0):.1f}")
    pdf.ln(2)
    for n,k in [("Family Hx Alzheimer's","fam"),("Diabetes","diab"),
                ("Hypertension","htn"),("Stroke History","strk"),("Depression","dep")]:
        row(n,"Yes" if fd.get(k) else "No")
    pdf.ln(3)

    sec("AI Prediction Result")
    pdf.set_font("Helvetica","B",14); pdf.set_text_color(*rc)
    pdf.cell(0,10,f"{risk_lbl}  -  {conf:.1f}% Confidence",ln=True)
    pdf.set_font("Helvetica","",10); pdf.set_text_color(237,242,255)
    pdf.multi_cell(0,7,INTERP[pred]); pdf.ln(3)

    sec("SHAP Feature Contributions (Top 12)")
    pdf.set_font("Helvetica","",9); pdf.set_text_color(148,163,184)
    pdf.multi_cell(0,6,"Positive SHAP values increase predicted risk; negative values decrease it."); pdf.ln(2)
    for i in np.argsort(np.abs(shap_vals))[::-1][:12]:
        val=float(shap_vals[i]); col=(239,68,68) if val>0 else (34,197,94)
        pdf.set_font("Helvetica","B",9); pdf.set_text_color(*col)
        pdf.cell(7,7,"+" if val>0 else "-",ln=False)
        pdf.set_text_color(237,242,255); pdf.set_font("Helvetica","",9)
        pdf.cell(80,7,s(feat_names[i]),ln=False)
        pdf.set_text_color(148,163,184); pdf.cell(28,7,f"{val:+.4f}",ln=False)
        pdf.set_text_color(*col)
        pdf.cell(0,7,"Increases risk" if val>0 else "Decreases risk",ln=True)

    pdf.ln(3); sec("Clinical Disclaimer")
    pdf.set_font("Helvetica","I",9); pdf.set_text_color(148,163,184)
    pdf.multi_cell(0,6,"AI-assisted decision support. Does not replace clinical judgment. "
                       "Stacking Ensemble (XGBoost + RandomForest + SVM) trained on OASIS-2 longitudinal MRI dataset. Accuracy: 92%.")
    return bytes(pdf.output())

# ── Header ────────────────────────────────────────────────────────────────────
def sheet_hdr(step):
    steps_cfg = [("Patient Intake",1),("Review",2),("Results",3)]
    nodes = ""
    for i,(lbl,n) in enumerate(steps_cfg):
        if n < step:   c,sym = "sdone","✓"
        elif n==step:  c,sym = "sactive",str(n)
        else:          c,sym = "sidle",str(n)
        lc = "slabel-active slabel" if n==step else "slabel"
        if i>0:
            lk = "sline-done sline" if n<=step else "sline"
            nodes += f'<div class="{lk}"></div>'
        nodes += f'<div class="snode"><div class="scircle {c}">{sym}</div><span class="{lc}">{lbl}</span></div>'

    st.markdown(f"""
    <div style="background:rgba(13,24,41,.9);border:1px solid rgba(91,140,255,.13);
         border-radius:20px;padding:16px 22px;margin-bottom:18px">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;border-radius:9px;
               background:linear-gradient(135deg,#7c6df6,#5b8cff);
               display:flex;align-items:center;justify-content:center;font-size:16px">🧠</div>
          <div>
            <div style="font-size:.94rem;font-weight:800;color:#edf2ff">NeuroRisk Pro</div>
            <div style="font-size:.65rem;color:#94a3b8">AI Clinical Decision Support · OASIS-2</div>
          </div>
        </div>
        <div style="font-size:.68rem;color:#64748b;background:rgba(255,255,255,.04);
             border:1px solid rgba(255,255,255,.07);border-radius:999px;padding:4px 12px">
          Stacking Ensemble · 92% Accuracy
        </div>
      </div>
      <div class="stepper" style="margin-top:14px">{nodes}</div>
    </div>""", unsafe_allow_html=True)

# ── Step 1: Intake ────────────────────────────────────────────────────────────
def step1():
    if pipeline is None:
        st.error("Model not found at `outputs/model.pkl`. Run `pipeline.py` first to train and save the model.")
        return

    _, col, _ = st.columns([1,7,1])
    with col:
        sheet_hdr(1)

        # Visit type — outside form so it drives conditional rendering
        vtype = st.session_state.vtype
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🏥  First Visit", use_container_width=True,
                         type="secondary" if vtype=="followup" else "primary"):
                st.session_state.vtype = "first"; st.rerun()
        with c2:
            if st.button("🔄  Follow-up Visit", use_container_width=True,
                         type="primary" if vtype=="followup" else "secondary"):
                st.session_state.vtype = "followup"; st.rerun()

        is_fu = (st.session_state.vtype == "followup")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        with st.form("intake"):
            # ── Patient ID
            st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Patient Identification</div>', unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1: patient_name = st.text_input("Full Name *", placeholder="John Smith")
            with c2: patient_id   = st.text_input("Patient ID / MRN *", placeholder="MRN-00482")
            c1,c2,c3 = st.columns(3)
            with c1: dob         = st.date_input("Date of Birth", value=date(1950,1,1))
            with c2: visit_date  = st.date_input("Visit Date", value=date.today())
            with c3: doctor_name = st.text_input("Attending Physician", placeholder="Dr. Name")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Visit info
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Visit Information</div>', unsafe_allow_html=True)
            if is_fu:
                c1,c2,c3 = st.columns(3)
                with c1: visit_num   = st.number_input("Current Visit #", 2, 20, 2, step=1)
                with c2: visit_count = st.number_input("Total Visits (incl. this)", 2, 20, 2, step=1)
                with c3: first_visit_date = st.date_input("Date of First Visit", value=date(2020,1,1))
            else:
                st.markdown('<p style="color:var(--muted);font-size:.82rem;margin:0">First visit — longitudinal features auto-set to baseline.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Demographics
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Demographics</div>', unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            with c1: age    = st.slider("Age (years) *", 40, 100, 72)
            with c2: gender = st.selectbox("Gender *", ["Male","Female"])
            with c3: educ   = st.selectbox("Education", list(EDUC_MAP))
            with c4: ses    = st.selectbox("Socioeconomic Status", [1,2,3,4,5], index=1,
                                           help="1 = Highest, 5 = Lowest")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Current assessment
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Current Cognitive Assessment</div>', unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1: mmse = st.slider("MMSE Score * (0–30)", 0, 30, 26, help="≥24 Normal")
            with c2: moca = st.slider("MoCA Score (0–30)", 0, 30, 24, help="≥26 Normal")
            with c3: cdr  = st.selectbox("CDR Rating *", [0.0,0.5,1.0,2.0], format_func=lambda x: CDR_OPT[x])
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Previous values (follow-up only) — auto-computes deltas
            if is_fu:
                st.markdown('<div class="card card-purple">', unsafe_allow_html=True)
                st.markdown('<div class="sec">Previous Visit Values — changes calculated automatically</div>', unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                with c1: prev_mmse = st.number_input("MMSE at last visit", 0, 30, 26, step=1)
                with c2: prev_cdr  = st.selectbox("CDR at last visit", [0.0,0.5,1.0,2.0], format_func=lambda x: CDR_OPT[x])
                with c3: prev_nwbv = st.number_input("nWBV at last visit", 0.60, 0.90, 0.73, step=0.001, format="%.3f")
                st.markdown('<div class="sec" style="margin-top:14px">Baseline (1st Visit)</div>', unsafe_allow_html=True)
                c1,c2 = st.columns(2)
                with c1: bl_mmse = st.number_input("MMSE at 1st visit", 0, 30, 28, step=1)
                with c2: bl_cdr  = st.selectbox("CDR at 1st visit", [0.0,0.5,1.0,2.0], format_func=lambda x: CDR_OPT[x])
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Brain MRI (optional)
            with st.expander("⚙️  Brain MRI Biomarkers — optional, improves accuracy"):
                c1,c2,c3 = st.columns(3)
                with c1: nwbv = st.number_input("nWBV", 0.60, 0.90, 0.73, step=0.001, format="%.3f")
                with c2: etiv = st.number_input("eTIV mm³", 1000, 2100, 1470, step=10)
                with c3: asf  = st.number_input("ASF", 0.87, 1.60, 1.19, step=0.001, format="%.3f")

            # ── Clinical history
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="sec">Clinical History</div>', unsafe_allow_html=True)
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: fam  = st.checkbox("Family Hx Alzheimer's")
            with c2: diab = st.checkbox("Diabetes")
            with c3: htn  = st.checkbox("Hypertension")
            with c4: strk = st.checkbox("Stroke History")
            with c5: dep  = st.checkbox("Depression")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            sub = st.form_submit_button("Generate Risk Assessment →",
                                        use_container_width=True, type="primary")
            st.markdown('<p style="text-align:center;font-size:.7rem;color:#2a3a4a;margin-top:5px">🔒 Processed locally — no data transmitted</p>', unsafe_allow_html=True)

        if sub:
            if is_fu:
                mr_delay    = (visit_date - first_visit_date).days / 365.25
                mmse_change = float(mmse) - float(prev_mmse)
                cdr_change  = float(cdr)  - float(prev_cdr)
                nwbv_change = float(nwbv) - float(prev_nwbv)
                b_mmse      = int(bl_mmse)
                b_cdr       = float(bl_cdr)
                v_num       = int(visit_num)
                v_count     = int(visit_count)
            else:
                mr_delay    = 0.0
                mmse_change = 0.0
                cdr_change  = 0.0
                nwbv_change = 0.0
                b_mmse      = int(mmse)
                b_cdr       = float(cdr)
                v_num       = 1
                v_count     = 1

            st.session_state.fd = dict(
                vtype=st.session_state.vtype,
                patient_name=patient_name, patient_id=patient_id,
                dob=dob, visit_date=visit_date, doctor_name=doctor_name,
                age=age, gender="M" if gender=="Male" else "F", gender_d=gender,
                educ=educ, educ_yrs=EDUC_MAP[educ], ses=ses,
                mmse=int(mmse), moca=int(moca), cdr=float(cdr),
                nwbv=float(nwbv), etiv=int(etiv), asf=float(asf),
                visit=v_num, visit_count=v_count, mr_delay=mr_delay,
                mmse_change=mmse_change, cdr_change=cdr_change, nwbv_change=nwbv_change,
                baseline_mmse=b_mmse, baseline_cdr=b_cdr,
                fam=fam, diab=diab, htn=htn, strk=strk, dep=dep,
            )
            st.session_state.step = 2; st.rerun()

# ── Step 2: Review ────────────────────────────────────────────────────────────
def step2():
    fd = st.session_state.fd
    is_fu = (fd.get("vtype") == "followup")
    _, col, _ = st.columns([1,7,1])
    with col:
        sheet_hdr(2)
        st.markdown('<div style="font-size:1.35rem;font-weight:800;margin-bottom:4px">Confirm Patient Data</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--muted);font-size:.85rem;margin-bottom:16px">Review all entries before the AI assessment runs.</p>', unsafe_allow_html=True)

        def rev_sec(ico, title, chips, key):
            ca, cb = st.columns([7,1])
            with ca:
                st.markdown(f'<div class="card"><div class="sec">{ico} {title}</div>'
                            +"".join(f'<span class="chip">{c}</span>' for c in chips)
                            +'</div>', unsafe_allow_html=True)
            with cb:
                if st.button("Edit", key=key): st.session_state.step=1; st.rerun()

        rev_sec("🪪","Patient",[f"Name: {fd.get('patient_name','—')}",
            f"ID: {fd.get('patient_id','—')}",f"DOB: {fd.get('dob','—')}",
            f"Visit: {fd.get('visit_date','—')}",f"Dr: {fd.get('doctor_name','—')}"],"ep")

        visit_chips = [f"Visit #{fd.get('visit',1)}" + (f" of {fd.get('visit_count',1)}" if is_fu else " (First)")]
        if is_fu:
            visit_chips += [f"MR Delay: {fd.get('mr_delay',0):.1f} yrs (computed)",
                            f"MMSE Δ: {fd.get('mmse_change',0):+.1f} (computed)",
                            f"CDR Δ: {fd.get('cdr_change',0):+.1f} (computed)"]
        rev_sec("📋","Visit",visit_chips,"ev")
        rev_sec("👤","Demographics",[f"Age: {fd['age']} yrs",f"Gender: {fd['gender_d']}",
            f"Edu: {fd['educ']}",f"SES: {fd.get('ses',2)}"],"ed")
        rev_sec("🧠","Cognitive Assessment",[f"MMSE: {fd['mmse']}/30",
            f"MoCA: {fd.get('moca','—')}/30",f"CDR: {fd['cdr']} ({CDR_LBL.get(fd['cdr'],'')})"],"ec")
        rev_sec("❤️","Clinical History",
            [f"{'✓' if fd.get(k) else '✗'} {n}" for n,k in [("Family Hx","fam"),
             ("Diabetes","diab"),("HTN","htn"),("Stroke","strk"),("Depression","dep")]],"eh")

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            if st.button("← Edit Information", use_container_width=True):
                st.session_state.step=1; st.rerun()
        with c2:
            if st.button("Confirm & Run Assessment →", use_container_width=True, type="primary"):
                with st.spinner("Running AI model…"):
                    st.session_state.result = run_model(fd)
                st.session_state.step=3; st.rerun()
        st.markdown('<p style="text-align:center;font-size:.7rem;color:#2a3a4a;margin-top:7px">🔒 Secure local processing</p>', unsafe_allow_html=True)

# ── Step 3: Results ───────────────────────────────────────────────────────────
def step3():
    fd = st.session_state.fd
    pred, prob, shap_vals, feat_names = st.session_state.result
    risk_lbl, color, bg, border = RISK[pred]
    conf = prob[pred] * 100
    is_fu = (fd.get("vtype") == "followup")

    _, hc, _ = st.columns([1,7,1])
    with hc: sheet_hdr(3)

    _, mc, _ = st.columns([1,7,1])
    with mc:
        # Patient strip
        flags = "".join(
            f'<span style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.22);'
            f'border-radius:999px;padding:2px 9px;font-size:.7rem;color:#fca5a5;margin-left:5px">⚠ {n}</span>'
            for n,k in [("Family Hx","fam"),("Diabetes","diab"),("HTN","htn"),("Stroke","strk"),("Depression","dep")]
            if fd.get(k))
        st.markdown(f"""
        <div style="background:var(--panel);border:1px solid var(--border-dim);border-radius:14px;
             padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;
             justify-content:space-between;flex-wrap:wrap;gap:10px">
          <div>
            <div style="font-size:1rem;font-weight:800">{fd.get('patient_name','Patient')}</div>
            <div style="font-size:.74rem;color:var(--muted);margin-top:3px">
              ID: {fd.get('patient_id','—')} &nbsp;·&nbsp;
              {'Visit #'+str(fd.get('visit',1))+' of '+str(fd.get('visit_count',1)) if is_fu else 'First Visit'} &nbsp;·&nbsp;
              {fd.get('visit_date','—')} &nbsp;·&nbsp; {fd.get('doctor_name','—')}
              {flags}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Risk Hero (signature element) ─────────────────────────────────────
        st.markdown(f"""
        <div class="risk-hero" style="background:linear-gradient(135deg,{bg},{bg.replace('.1','.04')});
             border:1px solid {border};">
          <div>
            <div class="risk-eyebrow" style="color:{color}">{risk_lbl}</div>
            <div style="font-size:1.5rem;font-weight:800;color:{color};margin-bottom:6px">
              {['Nondemented','Converted','Demented'][pred]}
            </div>
            <div class="risk-interp">{INTERP[pred]}</div>
          </div>
          <div>
            <div class="risk-pct" style="color:{color}">{conf:.1f}%</div>
            <div class="risk-pct-lbl">Confidence</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Metric tiles
        mmse_c = "#22c55e" if fd['mmse']>=24 else "#ef4444"
        cdr_c  = "#22c55e" if fd['cdr']==0 else "#f59e0b" if fd['cdr']==.5 else "#ef4444"
        moca_c = "#22c55e" if fd.get('moca',0)>=26 else "#f59e0b"
        tiles  = [
            ("Age",    fd['age'],      "yrs",         "#818cf8"),
            ("MMSE",   fd['mmse'],     "Normal" if fd['mmse']>=24 else "Impaired", mmse_c),
            ("CDR",    fd['cdr'],      CDR_LBL.get(fd['cdr'],''), cdr_c),
            ("MoCA",   fd.get('moca','—'), "Normal" if fd.get('moca',0)>=26 else "Below", moca_c),
        ]
        if is_fu:
            tiles += [
                ("MMSE Δ", f"{fd.get('mmse_change',0):+.1f}", "vs last", "#22c55e" if fd.get('mmse_change',0)>=0 else "#ef4444"),
                ("CDR Δ",  f"{fd.get('cdr_change',0):+.1f}",  "vs last", "#22c55e" if fd.get('cdr_change',0)<=0 else "#ef4444"),
                ("MR Delay", f"{fd.get('mr_delay',0):.1f}", "yrs since 1st", "#94a3b8"),
            ]
        tiles.append(("nWBV", f"{fd.get('nwbv',0.73):.3f}", "brain vol.", "#94a3b8"))

        tile_html = "".join(
            f'<div class="tile"><div class="tlbl">{l}</div>'
            f'<div class="tval" style="color:{c}">{v}</div>'
            f'<div class="tsub">{s}</div></div>'
            for l,v,s,c in tiles)
        st.markdown(f'<div class="tiles">{tile_html}</div>', unsafe_allow_html=True)

        # Class probability bars
        st.markdown('<div class="card" style="margin-top:14px">', unsafe_allow_html=True)
        st.markdown('<div class="sec">Prediction Breakdown</div>', unsafe_allow_html=True)
        for i,(lbl,c,_,_) in RISK.items():
            p = prob[i]*100; nm = "pname-a pname" if i==pred else "pname"
            fw = "800" if i==pred else "400"
            st.markdown(f"""
            <div class="prow">
              <span class="{nm}">{['Nondemented','Converted','Demented'][i]}</span>
              <div class="pbg"><div style="width:{p:.1f}%;height:100%;background:{c};
                border-radius:3px;{'box-shadow:0 0 6px '+c+'50' if i==pred else ''}"></div></div>
              <span class="ppct" style="color:{c};font-weight:{fw}">{p:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── SHAP Section ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:14px">
          <div style="font-size:1.05rem;font-weight:800">🔬 SHAP Analysis</div>
          <div style="font-size:.8rem;color:var(--muted);margin-top:3px">
            Feature contributions to this prediction ·
            <span style="color:#ef4444">▲ Red increases risk</span> &nbsp;·&nbsp;
            <span style="color:#22c55e">▼ Green decreases risk</span>
          </div>
        </div>""", unsafe_allow_html=True)

        order   = np.argsort(np.abs(shap_vals))
        s_names = [feat_names[i] for i in order]
        s_vals  = [float(shap_vals[i]) for i in order]
        s_colors= ["#ef4444" if v>0 else "#22c55e" for v in s_vals]

        sl, sr = st.columns([3,2], gap="large")
        with sl:
            fig = go.Figure(go.Bar(
                y=s_names, x=s_vals, orientation="h",
                marker=dict(color=s_colors, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color="rgba(255,255,255,.12)", line_width=1.5)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8",family="Inter",size=11),
                xaxis=dict(title="SHAP value",title_font=dict(color="#64748b",size=10),
                           gridcolor="rgba(255,255,255,.05)",zerolinecolor="rgba(255,255,255,.12)",
                           tickfont=dict(color="#64748b")),
                yaxis=dict(tickfont=dict(color="#cbd5e1",size=11),gridcolor="rgba(255,255,255,.04)"),
                margin=dict(t=10,b=20,l=10,r=20), height=440, bargap=0.3,
            )
            st.plotly_chart(fig, use_container_width=True)

        with sr:
            st.markdown('<div class="sec" style="margin-bottom:10px">Top Contributing Factors</div>', unsafe_allow_html=True)
            for i in np.argsort(np.abs(shap_vals))[::-1][:10]:
                val=float(shap_vals[i]); ac="#ef4444" if val>0 else "#22c55e"
                st.markdown(f"""
                <div class="sfact">
                  <span style="color:{ac};font-size:.88rem;margin-top:2px">{'▲' if val>0 else '▼'}</span>
                  <div style="flex:1;min-width:0">
                    <div style="font-size:.8rem;font-weight:600;color:#e2e8f0;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{feat_names[i]}</div>
                    <div style="font-size:.7rem;color:{ac};margin-top:1px">
                      {'Increases' if val>0 else 'Decreases'} risk</div>
                  </div>
                  <span style="font-size:.75rem;font-weight:700;color:{ac};white-space:nowrap">{val:+.4f}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="disc">🛡️ AI-assisted — not a substitute for clinical judgment. Stacking Ensemble (XGBoost + RF + SVM), OASIS-2 dataset, accuracy 92%.</div>', unsafe_allow_html=True)

        # Nav + PDF
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("← Back to Review", use_container_width=True):
                st.session_state.step=2; st.rerun()
        with c2:
            pdf_data = make_pdf(fd, pred, prob, shap_vals, feat_names)
            fname    = f"NeuroRisk_{fd.get('patient_id','report')}_{date.today()}.pdf"
            st.download_button("⬇  Download PDF Report", data=pdf_data,
                               file_name=fname, mime="application/pdf",
                               use_container_width=True, type="primary")
        with c3:
            if st.button("🔄  New Assessment", use_container_width=True):
                for k,v in [("step",1),("fd",{}),("result",None)]:
                    st.session_state[k]=v
                st.rerun()

# ── Route ─────────────────────────────────────────────────────────────────────
[step1, step2, step3][st.session_state.step - 1]()
