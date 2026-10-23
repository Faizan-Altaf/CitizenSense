"""
CitizenSense - public feedback triage.
Loads the three models trained in CitizenSense.ipynb (category, sentiment, priority)
and shows a triage ticket for each complaint.

Run:  streamlit run citizensense_app.py
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "category": "tfidf_logistic_regression.joblib",
    "sentiment": "tfidf_sentiment_classifier.joblib",
    "priority": "priority_classifier.joblib",
}

PRIORITIES = ["Low", "Medium", "High", "Urgent"]
PRI_COLOR = {"Low": "#4C7A9E", "Medium": "#B98A1F", "High": "#D0602A", "Urgent": "#B3261E"}
SENTIMENTS = ["Negative", "Neutral", "Positive", "Friendly"]
SENT_NOTE = {
    "Negative": "Complaint or frustration",
    "Neutral": "Question or status update",
    "Positive": "Thanks for a fix",
    "Friendly": "Polite, warm request",
}
SENT_TONE = {
    "Negative": "A complaint",
    "Neutral": "A question or status update",
    "Positive": "Appreciation",
    "Friendly": "A polite request",
}

DEPARTMENTS = {
    "Roads / Infrastructure": "Roads and highways department",
    "Water Supply": "Water and sanitation authority",
    "Electricity": "Power distribution company",
    "Sanitation": "Municipal waste management",
    "Health": "Health department and district hospital",
    "Education": "School education department",
}
ISSUE_WORD = {
    "Roads / Infrastructure": "roads",
    "Water Supply": "water supply",
    "Electricity": "electricity",
    "Sanitation": "sanitation",
    "Health": "health services",
    "Education": "education",
}
ACTIONS = {
    "Urgent": "Send a response team now and alert the duty officer.",
    "High": "Assign to a field team within 24 hours.",
    "Medium": "Schedule a fix this week and update the citizen.",
    "Low": "Log it and handle in the routine queue.",
}

EXAMPLES = [
    "Hamari gali mein 3 din se pani nahi aa raha, bohat pareshani hai.",
    "Mere walid ko dil ka daura para hai aur ambulance nahi aa rahi.",
    "Bijli ki taar sarak par gir gayi hai aur chingariyan nikal rahi hain.",
    "Assalam o Alaikum, street light kharab hai, aap ki meherbani hogi agar theek ho jaye.",
    "School ki chat tapakti hai aur kitabein bheeg jati hain.",
    "Shukriya, nala saaf ho gaya hai.",
]

# Words that signal danger to life. Used by the optional safety rule.
DANGER_TERMS = [
    "maut", "faut", "intiqal", "inteqal", "jaan chali", "jaan ka khatra", "mar gaya", "mar gayi",
    "mar gaye", "dum tod", "behosh", "dil ka daura", "heart attack", "stroke", "saanp ne", "zehar",
    "zakhmi", "khoon beh", "shadeed khoon", "aag lag", "karant lag", "karant aa", "jhatka lag",
    "doob gaya", "doob gayi", "oxygen khatam", "ambulance nahi", "chingari",
    "died", "death", "dead", "electrocut", "unconscious", "bleeding", "on fire",
]

STOP = set("""
hai hain ka ki ke ko se mein main par aur ne bhi to ho tha thi the raha rahi rahe gaya gayi gaye kar karein
kya nahi nhi is iska yeh ye wo woh hum hamare hamari hamara aap ap ek koi bohat bahut bohot kai jaldi
the of and in for sir janab salam assalam alaikum shukriya meherbani hogi dein do diya di liye ab abhi tak
sirf jab jo kuch ya please aaj kal subah raat din
""".split())

# ----------------------------------------------------------------------------
# Page + styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="CitizenSense", page_icon="🏛️", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');
:root{--paper:#F3F5F7;--surface:#FFFFFF;--ink:#15222E;--muted:#5B6B7A;--line:#D9E0E6;--navy:#16365C;}
html, body, .stApp, [class*="css"]{font-family:'Public Sans',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink);}
.stApp{background:var(--paper);}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2rem;max-width:1180px;}
.masthead h1{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:2.7rem;font-weight:700;letter-spacing:-0.02em;margin:0;color:var(--navy);}
.masthead p{color:var(--muted);font-size:1.05rem;margin:.4rem 0 0;max-width:58ch;line-height:1.5;}
.rail{display:flex;height:6px;margin:1.1rem 0 1.5rem;border-radius:3px;overflow:hidden;max-width:240px;}
.rail span{flex:1;}
.ticket{background:var(--surface);border:1px solid var(--line);border-radius:8px;overflow:hidden;}
.band{padding:1.15rem 1.4rem 1.2rem;color:#fff;}
.band .small{font-size:.85rem;opacity:.92;}
.band .big{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:3.1rem;font-weight:700;line-height:1.05;letter-spacing:-0.01em;}
.band .act{margin-top:.55rem;font-size:1.02rem;line-height:1.4;}
.band .note{margin-top:.6rem;font-size:.88rem;background:rgba(255,255,255,.16);padding:.4rem .6rem;border-radius:5px;}
.row3{display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid var(--line);}
.cell{padding:.95rem 1.2rem;border-right:1px solid var(--line);}
.cell:last-child{border-right:0;}
.cell .k{font-size:.8rem;color:var(--muted);}
.cell .v{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:1.3rem;font-weight:600;line-height:1.25;margin:.15rem 0;}
.cell .n{font-size:.84rem;color:var(--muted);}
.sec{padding:.95rem 1.4rem;border-bottom:1px solid var(--line);}
.sec:last-child{border-bottom:0;}
.sec h4{font-size:.86rem;font-weight:600;color:var(--muted);margin:0 0 .55rem;}
.sec p{margin:0;line-height:1.5;}
.chip{display:inline-block;padding:.18rem .65rem;margin:0 .35rem .35rem 0;border:1px solid var(--line);border-radius:999px;background:#F7F9FA;font-size:.92rem;}
.bar-row{display:grid;grid-template-columns:78px 1fr 44px;gap:.7rem;align-items:center;font-size:.88rem;margin:.28rem 0;}
.bar{height:8px;background:#E8EDF1;border-radius:4px;overflow:hidden;}
.bar i{display:block;height:100%;}
.pct{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;}
.review{display:inline-block;margin:0 0 .9rem;padding:.45rem .75rem;border-left:4px solid #B98A1F;background:#FBF4E3;border-radius:0 5px 5px 0;font-size:.92rem;}
.empty{border:1px dashed #BFCAD3;border-radius:8px;padding:2.2rem 1.6rem;color:var(--muted);line-height:1.55;background:rgba(255,255,255,.5);}
.empty b{color:var(--ink);}
.stTextArea textarea{border-radius:8px;border:1px solid var(--line);background:#fff;font-size:1rem;line-height:1.5;}
.stButton>button{border-radius:6px;font-weight:600;border:1px solid var(--line);}
.stButton>button, .stButton>button p{color:var(--ink);}
.stButton>button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]){background:var(--surface);border:1px solid var(--line);text-align:left;justify-content:flex-start;}
.stButton>button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover{background:#EAF0F5;border-color:var(--navy);color:var(--navy);}
.stButton>button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover p{color:var(--navy);}
.stButton>button[kind="primary"] p, .stButton>button[data-testid="stBaseButton-primary"] p{color:#fff;}
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"]{background:var(--navy);border-color:var(--navy);color:#fff;}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;}
.stTabs [data-baseweb="tab"]{font-weight:600;}
@media (max-width:640px){
  .masthead h1{font-size:2.1rem;}
  .row3{grid-template-columns:1fr;}
  .cell{border-right:0;border-bottom:1px solid var(--line);}
  .band .big{font-size:2.5rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

RAIL = "".join(f'<span style="background:{PRI_COLOR[p]}"></span>' for p in PRIORITIES)
st.markdown(
    f'<div class="masthead"><h1>CitizenSense</h1>'
    f'<p>Paste a citizen complaint. See what it is about, how the citizen feels, '
    f'how urgent it is, and which office should handle it.</p></div>'
    f'<div class="rail">{RAIL}</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def default_model_dir() -> str:
    candidates = [os.environ.get("CITIZENSENSE_MODELS"), APP_DIR / "models",
                  r"E:\CitizenSense Sentiment Project\models"]
    for c in candidates:
        if c and all((Path(c) / f).exists() for f in MODEL_FILES.values()):
            return str(c)
    return str(APP_DIR / "models")


@st.cache_resource(show_spinner=False)
def load_models(folder: str):
    d = Path(folder)
    missing = [f for f in MODEL_FILES.values() if not (d / f).exists()]
    if missing:
        raise FileNotFoundError("Missing: " + ", ".join(missing))
    return {task: joblib.load(d / name) for task, name in MODEL_FILES.items()}


with st.sidebar:
    st.markdown("### Settings")
    model_dir = st.text_input("Models folder", value=default_model_dir())
    use_safety = st.toggle(
        "Escalate danger to life",
        value=True,
        help="If a complaint mentions death, injury, fire, electric shock or a medical emergency, "
             "raise its priority to Urgent even if the model scored it lower.",
    )
    review_below = st.slider(
        "Ask for human review below", 0.30, 0.90, 0.50, 0.05,
        help="If any model is less sure than this, the ticket is flagged for a person to check.",
    )
    st.caption("Predictions are suggestions to help staff sort complaints. A person makes the final call.")

try:
    MODELS = load_models(model_dir)
except Exception as err:  # noqa: BLE001
    st.error(
        f"Could not load the models from `{model_dir}`. {err}\n\n"
        "Run the training cells in CitizenSense.ipynb so the three .joblib files exist, "
        "then point **Models folder** (left sidebar) to that folder."
    )
    st.stop()

with st.sidebar:
    st.success("Three models loaded")

# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def danger_hit(text: str) -> str | None:
    t = text.lower()
    for term in DANGER_TERMS:
        if term in t:
            return term
    return None


def top_terms(pipe, text: str, label: str, n: int = 8) -> list[str]:
    """Words in this text that pushed the model toward `label`."""
    try:
        tfidf = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["classifier"]
        x = tfidf.transform([text]).tocoo()
        names = tfidf.get_feature_names_out()
        k = list(clf.classes_).index(label)
        coef = clf.coef_[k]
        scored = [(names[j], v * coef[j]) for j, v in zip(x.col, x.data) if v * coef[j] > 0]
        scored.sort(key=lambda s: -s[1])
        return [w for w, _ in scored[:n]]
    except Exception:  # noqa: BLE001
        return []


def pick_keywords(text: str, cat_label: str, pri_label: str, limit: int = 6) -> list[str]:
    candidates = top_terms(MODELS["category"], text, cat_label) + top_terms(MODELS["priority"], text, pri_label)
    out: list[str] = []
    for term in candidates:
        tokens = term.split()
        if all(tok in STOP or len(tok) < 3 for tok in tokens):
            continue
        if any(term in o or o in term for o in out):
            continue
        out.append(term)
        if len(out) == limit:
            break
    if not out:
        out = [w for w in re.findall(r"[A-Za-z']{3,}", text.lower()) if w not in STOP][:5]
    return out


def predict_one(text: str) -> dict:
    result: dict = {}
    for task in ("category", "sentiment", "priority"):
        model = MODELS[task]
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        i = int(np.argmax(proba))
        result[task] = {"label": classes[i], "conf": float(proba[i]), "dist": dict(zip(classes, map(float, proba)))}

    model_priority = result["priority"]["label"]
    final_priority, escalated_by = model_priority, None
    if use_safety:
        term = danger_hit(text)
        if term and PRIORITIES.index(model_priority) < PRIORITIES.index("Urgent"):
            final_priority, escalated_by = "Urgent", term
    keywords = pick_keywords(text, result["category"]["label"], model_priority)
    kw_text = ", ".join(keywords[:3]) if keywords else "no clear keywords"
    summary = (
        f"{SENT_TONE[result['sentiment']['label']]} about {ISSUE_WORD.get(result['category']['label'], 'a public service')}. "
        f"Main points: {kw_text}. Suggested priority: {final_priority.lower()}."
    )
    min_conf = min(result[t]["conf"] for t in ("category", "sentiment", "priority"))
    return {
        "text": text,
        "category": result["category"],
        "sentiment": result["sentiment"],
        "priority": result["priority"],
        "final_priority": final_priority,
        "model_priority": model_priority,
        "escalated_by": escalated_by,
        "keywords": keywords,
        "department": DEPARTMENTS.get(result["category"]["label"], "Relevant department"),
        "action": ACTIONS[final_priority],
        "summary": summary,
        "needs_review": min_conf < review_below,
        "min_conf": min_conf,
    }


def predict_many(texts: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"text": texts})
    for task in ("category", "sentiment", "priority"):
        model = MODELS[task]
        proba = model.predict_proba(texts)
        classes = np.array(model.classes_)
        frame[task] = classes[proba.argmax(axis=1)]
        frame[f"{task}_confidence"] = proba.max(axis=1).round(3)
    frame["model_priority"] = frame["priority"]
    frame["escalated_by"] = ""
    if use_safety:
        for i, t in enumerate(texts):
            term = danger_hit(t)
            if term and frame.at[i, "priority"] != "Urgent":
                frame.at[i, "priority"] = "Urgent"
                frame.at[i, "escalated_by"] = term
    frame["department"] = frame["category"].map(DEPARTMENTS)
    frame["suggested_action"] = frame["priority"].map(ACTIONS)
    frame["needs_review"] = frame[["category_confidence", "sentiment_confidence", "priority_confidence"]].min(axis=1) < review_below
    return frame


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x))


def bars(dist: dict, order: list[str], color_of) -> str:
    rows = []
    for label in order:
        p = dist.get(label, 0.0)
        rows.append(
            f'<div class="bar-row"><span>{esc(label)}</span>'
            f'<div class="bar"><i style="width:{p*100:.0f}%;background:{color_of(label)}"></i></div>'
            f'<span class="pct">{p*100:.0f}%</span></div>'
        )
    return "".join(rows)


def render_ticket(r: dict) -> str:
    color = PRI_COLOR[r["final_priority"]]
    note = ""
    if r["escalated_by"]:
        note = (f'<div class="note">The model scored this {esc(r["model_priority"])}. '
                f'Raised to Urgent because the text mentions "{esc(r["escalated_by"])}".</div>')
    review = ""
    if r["needs_review"]:
        review = ('<div class="review">The models are not very sure about this one '
                  f'(lowest confidence {r["min_conf"]*100:.0f}%). Please have a person check it.</div>')
    chips = "".join(f'<span class="chip">{esc(k)}</span>' for k in r["keywords"]) or "<span>No clear keywords found.</span>"
    s_label = r["sentiment"]["label"]
    return (
        review
        + '<div class="ticket">'
        + f'<div class="band" style="background:{color}"><div class="small">Priority</div>'
        + f'<div class="big">{esc(r["final_priority"])}</div>'
        + f'<div class="act">{esc(r["action"])}</div>{note}</div>'
        + '<div class="row3">'
        + f'<div class="cell"><div class="k">Topic</div><div class="v">{esc(r["category"]["label"])}</div>'
        + f'<div class="n">{r["category"]["conf"]*100:.0f}% sure</div></div>'
        + f'<div class="cell"><div class="k">Tone</div><div class="v">{esc(s_label)}</div>'
        + f'<div class="n">{esc(SENT_NOTE.get(s_label, ""))} ({r["sentiment"]["conf"]*100:.0f}% sure)</div></div>'
        + f'<div class="cell"><div class="k">Send to</div><div class="v">{esc(r["department"])}</div>'
        + '<div class="n">Suggested office</div></div></div>'
        + f'<div class="sec"><h4>Summary</h4><p>{esc(r["summary"])}</p></div>'
        + f'<div class="sec"><h4>Words that shaped the result</h4>{chips}</div>'
        + '<div class="sec"><h4>How sure the priority model is</h4>'
        + bars(r["priority"]["dist"], PRIORITIES, lambda l: PRI_COLOR[l])
        + '</div></div>'
    )


def count_chart(series: pd.Series, order: list[str], colors: dict | None = None):
    counts = series.value_counts().reindex(order, fill_value=0).rename_axis("label").reset_index(name="count")
    palette = [colors[o] for o in order] if colors else ["#16365C"] * len(order)
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("count:Q", title=None),
            y=alt.Y("label:N", sort=order, title=None),
            color=alt.Color("label:N", scale=alt.Scale(domain=order, range=palette), legend=None),
            tooltip=["label", "count"],
        )
        .properties(height=34 * len(order) + 16, width="container")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, labelFont="Public Sans", labelFontSize=13, domain=False)
    )


def show_charts(frame: pd.DataFrame):
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**Priority**")
        st.altair_chart(count_chart(frame["priority"], PRIORITIES, PRI_COLOR))
    with c2:
        st.markdown("**Topic**")
        st.altair_chart(count_chart(frame["category"], sorted(DEPARTMENTS)))
    with c3:
        st.markdown("**Tone**")
        st.altair_chart(count_chart(frame["sentiment"], SENTIMENTS))


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
st.session_state.setdefault("complaint", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)


def use_example(text: str):
    st.session_state["complaint"] = text


tab_one, tab_file, tab_insights, tab_about = st.tabs(
    ["Analyze a complaint", "Analyze a file", "Session insights", "How it works"]
)

# ----------------------------------------------------------------------------
# Tab: single complaint
# ----------------------------------------------------------------------------
with tab_one:
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.text_area(
            "Complaint text",
            key="complaint",
            height=180,
            placeholder="Write or paste a complaint in Roman Urdu or English",
        )
        analyze = st.button("Analyze complaint", type="primary")
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.caption("Or start from an example")
        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"ex{i}", on_click=use_example, args=(ex,))

    if analyze:
        text = st.session_state["complaint"].strip()
        if not text:
            st.session_state["last"] = None
            with left:
                st.warning("Type a complaint first, then select Analyze complaint.")
        else:
            res = predict_one(text)
            st.session_state["last"] = res
            st.session_state["history"].append(
                {
                    "complaint": text,
                    "topic": res["category"]["label"],
                    "tone": res["sentiment"]["label"],
                    "priority": res["final_priority"],
                    "send_to": res["department"],
                    "needs_review": res["needs_review"],
                }
            )

    with right:
        if st.session_state["last"]:
            st.markdown(render_ticket(st.session_state["last"]), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty"><b>Nothing analyzed yet.</b><br>'
                'Write a complaint on the left, or pick an example, and select <b>Analyze complaint</b>. '
                'The ticket for it will appear here.</div>',
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# Tab: file
# ----------------------------------------------------------------------------
with tab_file:
    st.markdown("Upload a CSV with one complaint per row. CitizenSense will label every row and let you download the result.")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is None:
        st.markdown(
            '<div class="empty"><b>No file yet.</b><br>Choose a CSV that has a column of complaint text.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            raw = pd.read_csv(up)
        except Exception as err:  # noqa: BLE001
            st.error(f"That file could not be read as a CSV: {err}")
            raw = None
        if raw is not None:
            if raw.empty:
                st.warning("The file has no rows.")
            else:
                text_cols = [c for c in raw.columns if raw[c].dtype == object]
                if not text_cols:
                    st.error("No text column found. The file needs at least one column of complaint text.")
                else:
                    default_i = text_cols.index("text") if "text" in text_cols else 0
                    col = st.selectbox("Which column has the complaint text?", text_cols, index=default_i)
                    st.caption(f"{len(raw):,} rows in the file")
                    if st.button("Analyze all rows", type="primary"):
                        texts = raw[col].fillna("").astype(str).str.strip().tolist()
                        with st.spinner("Reading complaints"):
                            out = predict_many(texts)
                        extra = raw.drop(columns=[col]).reset_index(drop=True)
                        st.session_state["batch"] = pd.concat([out, extra], axis=1)
                    batch = st.session_state.get("batch")
                    if batch is not None and len(batch) == len(raw):
                        st.markdown(
                            f"**{len(batch):,} complaints labeled.** "
                            f"{int((batch['priority'] == 'Urgent').sum()):,} urgent, "
                            f"{int(batch['needs_review'].sum()):,} flagged for human review."
                        )
                        show_charts(batch)
                        st.dataframe(batch)
                        st.download_button(
                            "Download labeled CSV",
                            data=batch.to_csv(index=False).encode("utf-8-sig"),
                            file_name="citizensense_labeled.csv",
                            mime="text/csv",
                        )

# ----------------------------------------------------------------------------
# Tab: insights
# ----------------------------------------------------------------------------
with tab_insights:
    hist = pd.DataFrame(st.session_state["history"])
    if hist.empty:
        st.markdown(
            '<div class="empty"><b>No complaints analyzed in this session.</b><br>'
            'Analyze a few on the first tab and the totals will show up here.</div>',
            unsafe_allow_html=True,
        )
    else:
        n_urgent = int((hist["priority"] == "Urgent").sum())
        st.markdown(f"**{len(hist)} complaints this session.** {n_urgent} urgent, {int(hist['needs_review'].sum())} flagged for review.")
        show_charts(hist.rename(columns={"topic": "category", "tone": "sentiment"}))
        st.dataframe(hist)
        st.download_button(
            "Download session history",
            data=hist.to_csv(index=False).encode("utf-8-sig"),
            file_name="citizensense_session.csv",
            mime="text/csv",
        )
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.session_state["last"] = None
            st.rerun()

# ----------------------------------------------------------------------------
# Tab: how it works
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**What happens to a complaint**

Three small text models read the same complaint. One picks the topic, one reads the tone, and one scores the priority.
Each model turns the words into numbers (TF-IDF) and then makes a choice with logistic regression.
The words that pushed the result most are shown as chips on the ticket.

**What the labels mean**

- **Urgent:** danger to life or safety, such as a death, an injury, an electric shock, a fire, a collapse, or a life-threatening illness.
- **High:** a serious problem, such as an essential service down for days, a serious illness, or a risk to children.
- **Medium:** an ongoing inconvenience, such as a billing dispute or a non-serious illness.
- **Low:** a small problem or a question for information.
- **Friendly** means the writer is warm and polite (greeting, courtesy). **Positive** means the writer is thanking staff for a fix.

**Escalate danger to life**

When this is on (left sidebar), any complaint that mentions death, injury, fire, electric shock or a medical emergency is raised to Urgent,
and the ticket says why. This is a plain word list, so it is a safety net and not a replacement for the model.

**Limits**

- The models learned from synthetic Roman Urdu complaints, so real messages with unusual spelling or mixed topics can be wrong.
- Complaints that mention two topics (for example water in a hospital) can be filed under the wrong one.
- Every result is a suggestion. A person should confirm Urgent and flagged tickets.
"""
    )
