"""Churn Risk Studio: a Streamlit app for the KKBOX churn project (SML course, BIT Mesra).

Run locally with:  streamlit run app.py
The folder artifacts/ must hold the files saved by Section 16 of the Milestone 3 notebook.
"""
import json
import os

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
CAT_COLS = ["city", "gender", "registered_via", "last_payment_method"]
NAMES = {
    "last_auto_renew": "Auto-renew at last payment",
    "auto_renew_share": "Share of payments with auto-renew",
    "last_cancel": "Last payment was a cancel",
    "last_plan_days": "Plan length (days)",
    "last_amount_paid": "Amount paid (NT$)",
    "last_discount": "Discount (NT$)",
    "days_to_expiry": "Days until membership expires",
    "n_transactions": "Number of payments",
    "n_cancels": "Number of cancels",
    "active_days_28d": "Active days (last 28 days)",
    "listening_hours_28d": "Listening hours (last 28 days)",
    "songs_per_active_day": "Songs per active day",
    "days_since_last_listen": "Days since last listen",
    "tenure_days": "Tenure (days)",
    "age": "Age",
    "city": "City code",
    "gender": "Gender",
    "registered_via": "Registration channel",
    "last_payment_method": "Payment method",
    "has_member_info": "Has a member record",
    "has_payment_history": "Has payment history",
}
BAND_COLORS = {"Low": "#2E7D4F", "Medium": "#B8860B", "High": "#B03A2E"}

st.set_page_config(page_title="Churn Risk Studio", page_icon="📉", layout="wide")


# ------------------------------------------------------------------ loading
REQUIRED = ["model_lgbm.txt", "schema.json", "meta.json", "calibrator.json", "curves.json", "results.csv",
            "cv_table.csv", "calibration_table.csv", "threshold_table.csv", "risk_bands.csv",
            "shap_importance.csv", "eda_rates.csv", "drift.csv"]
missing = [f for f in REQUIRED if not os.path.exists(os.path.join(ART, f))]
if missing:
    st.error("Some model files are missing from the artifacts folder: " + ", ".join(missing))
    st.info("Run the Milestone 3 notebook on Kaggle, download app_artifacts.zip and unzip it into the artifacts folder next to app.py.")
    st.stop()


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=os.path.join(ART, "model_lgbm.txt"))


@st.cache_data
def load_json(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(ART, name))


booster = load_model()
schema = load_json("schema.json")
meta = load_json("meta.json")
calib = load_json("calibrator.json")
COLUMNS = schema["columns"]


def stretch(fn, *args, **kwargs):
    # Full-width display on both newer and older Streamlit versions
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, **kwargs)


def nice(feature):
    return NAMES.get(feature, feature)


def calibrate(raw):
    # Platt scaling learned in Section 11 of the notebook
    raw = min(max(raw, 1e-6), 1 - 1e-6)
    z = calib["a"] * np.log(raw / (1 - raw)) + calib["b"]
    return float(1 / (1 + np.exp(-z)))


def band_of(p):
    edges, names = meta["band_edges"], meta["band_names"]
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        if lo <= p < hi:
            return name
    return names[-1]


def base_feature(col):
    for c in CAT_COLS:
        if col.startswith(c + "_"):
            return c
    return col


def build_row(values):
    row = {c: schema["medians"].get(c, 0.0) for c in COLUMNS}
    for c in COLUMNS:
        if base_feature(c) in CAT_COLS:
            row[c] = 0
    for k, v in values.items():
        if k in CAT_COLS:
            col = f"{k}_{v}"
            if col in row:
                row[col] = 1
        elif k in row and v is not None:
            row[k] = float(v)
    return pd.DataFrame([row], columns=COLUMNS)


def predict(values):
    row = build_row(values)
    raw = float(booster.predict(row)[0])
    contrib = booster.predict(row, pred_contrib=True)[0]
    parts = pd.Series(contrib[:-1], index=COLUMNS).groupby(base_feature).sum()
    return raw, calibrate(raw), parts


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Churn Risk Studio")
st.sidebar.caption("KKBOX subscriber data, WSDM Cup 2018")
page = st.sidebar.radio("Go to", ["Overview", "Member risk checker", "Retention simulator",
                                  "Model performance", "What drives churn", "About and limits"])
st.sidebar.markdown("---")
st.sidebar.caption("SML course project, BIT Mesra. Atul, Aditi and Riya.")

results = load_csv("results.csv")
oot = results[results["setting"].str.startswith("Out")].set_index("model")

# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Who will leave, why, and what should we do about it?")
    st.write("A streaming service earns money only while members keep renewing. This app predicts which members "
             "of a subscription service are likely to **churn** (not renew within 30 days after their membership "
             "expires), explains why, and estimates whether a retention offer is worth sending.")
    cohorts = meta["cohorts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Members used", f"{sum(int(c['members sampled']) for c in cohorts):,}")
    c2.metric("Churn rate, February cohort", f"{cohorts[0]['churn rate in sample']:.1%}")
    c3.metric("Churn rate, March cohort", f"{cohorts[1]['churn rate in sample']:.1%}")
    c4.metric("LightGBM ROC-AUC, out-of-time", f"{oot.loc['LightGBM', 'ROC-AUC']:.3f}")
    st.subheader("How it works")
    st.markdown(
        "1. **Data:** real KKBOX member, payment and listening records.\n"
        "2. **No leakage:** features use only data from before a cutoff date.\n"
        "3. **Model:** LightGBM, tuned with cross-validation on the February cohort.\n"
        "4. **Honest test:** trained on February 2017, tested on March 2017 (out-of-time).\n"
        "5. **Calibration:** scores turned into probabilities.\n"
        "6. **Explanations:** SHAP values show why a member is flagged.\n"
        "7. **Decision:** risk bands and a simulator for retention offers.")
    st.info("The data comes from KKBOX, a music-streaming service in Taiwan. It is not Netflix data. "
            "Netflix does not publish subscriber data.")

elif page == "Member risk checker":
    st.title("Member risk checker")
    st.caption("Enter a member's details. Fields you leave alone keep typical values from the training data.")
    med = schema["medians"]
    defaults = {"auto": "On", "cancel": "No", "plan": 30, "paid": 149, "expiry": int(med.get("days_to_expiry", 15)),
                "active": int(med.get("active_days_28d", 12)), "hours": float(round(med.get("listening_hours_28d", 10), 1)),
                "since": int(med.get("days_since_last_listen", 2)), "ntrans": int(med.get("n_transactions", 15)),
                "ncancel": int(med.get("n_cancels", 0)), "share": float(round(med.get("auto_renew_share", 1.0), 2)),
                "songs": float(round(med.get("songs_per_active_day", 20), 1)), "discount": 0,
                "tenure": int(med.get("tenure_days", 1000))}
    presets = {
        "Loyal member": {"auto": "On", "cancel": "No", "plan": 30, "paid": 149, "expiry": 20, "active": 24,
                         "hours": 40.0, "since": 1, "ntrans": 24, "ncancel": 0, "share": 1.0, "songs": 30.0,
                         "discount": 0, "tenure": 1800},
        "At-risk member": {"auto": "Off", "cancel": "Yes", "plan": 30, "paid": 149, "expiry": 2, "active": 3,
                           "hours": 2.0, "since": 20, "ntrans": 3, "ncancel": 1, "share": 0.3, "songs": 8.0,
                           "discount": 0, "tenure": 200},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    def apply(name):
        for k, v in presets[name].items():
            st.session_state[k] = v

    b1, b2, b3 = st.columns([1, 1, 4])
    b1.button("Load: loyal member", on_click=apply, args=("Loyal member",))
    b2.button("Load: at-risk member", on_click=apply, args=("At-risk member",))

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.radio("Auto-renew at last payment", ["On", "Off"], key="auto", horizontal=True)
        st.radio("Last payment was a cancel", ["No", "Yes"], key="cancel", horizontal=True)
        st.selectbox("Plan length (days)", [7, 30, 90, 180, 410], key="plan")
        st.number_input("Amount paid at last payment (NT$)", 0, 2000, step=1, key="paid")
        st.slider("Days until membership expires", -30, 60, key="expiry")
        st.slider("Active days in the last 28 days", 0, 28, key="active")
        st.number_input("Listening hours in the last 28 days", 0.0, 700.0, step=1.0, key="hours")
        st.slider("Days since last listen", 1, 29, key="since")
        with st.expander("More details"):
            st.slider("Number of payments so far", 0, 60, key="ntrans")
            st.slider("Number of cancels so far", 0, 10, key="ncancel")
            st.slider("Share of payments with auto-renew", 0.0, 1.0, step=0.05, key="share")
            st.number_input("Songs per active day", 0.0, 1000.0, step=1.0, key="songs")
            st.number_input("Discount at last payment (NT$)", 0, 500, step=1, key="discount")
            st.number_input("Tenure (days since registration)", 0, 5000, step=10, key="tenure")
            age_known = st.checkbox("Age is known", value=False)
            age = st.number_input("Age", 10, 90, value=28, disabled=not age_known)
            cats = {}
            for c in CAT_COLS:
                options = schema["categories"].get(c, [])
                if options:
                    default = schema["category_defaults"].get(c, options[0])
                    cats[c] = st.selectbox(nice(c), options, index=options.index(default) if default in options else 0)

    s = st.session_state
    values = {"last_auto_renew": 1 if s.auto == "On" else 0, "last_cancel": 1 if s.cancel == "Yes" else 0,
              "last_plan_days": s.plan, "last_amount_paid": s.paid, "last_discount": s.discount,
              "days_to_expiry": s.expiry, "active_days_28d": s.active, "listening_hours_28d": s.hours,
              "days_since_last_listen": s.since, "n_transactions": s.ntrans, "n_cancels": s.ncancel,
              "auto_renew_share": s.share, "songs_per_active_day": s.songs, "tenure_days": s.tenure,
              "age": age if age_known else None, "has_member_info": 1, "has_payment_history": 1, **cats}
    raw, prob, parts = predict(values)
    band = band_of(prob)

    with right:
        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.markdown(f"<div style='margin-top:6px;font-size:15px'>Risk band</div>"
                    f"<div style='display:inline-block;padding:6px 18px;border-radius:16px;color:white;"
                    f"font-weight:600;font-size:20px;background:{BAND_COLORS.get(band, '#555')}'>{band} risk</div>",
                    unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Raw model score {raw:.3f}, calibrated to a probability. Bands: Low below "
                   f"{meta['band_edges'][1]:.0%}, Medium up to {meta['band_edges'][2]:.0%}, High above.")
        top = parts.reindex(parts.abs().sort_values(ascending=False).index).head(8)
        reasons = pd.DataFrame({"feature": [nice(f) for f in top.index], "effect": top.values})
        reasons["direction"] = np.where(reasons["effect"] > 0, "pushes towards churn", "pushes towards staying")
        st.markdown("**Why: the strongest reasons for this member** (SHAP values, log-odds)")
        chart = alt.Chart(reasons).mark_bar().encode(
            x=alt.X("effect:Q", title="effect on the score"),
            y=alt.Y("feature:N", sort=list(reasons["feature"]), title=None,
                    axis=alt.Axis(labelLimit=300, labelOverlap=False, labelFontSize=13)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["pushes towards churn", "pushes towards staying"],
                                                           range=["#B03A2E", "#2E7D4F"]),
                            legend=alt.Legend(orient="bottom", title=None, columns=1, labelLimit=300)),
            tooltip=["feature", alt.Tooltip("effect:Q", format=".3f"), "direction"]).properties(height=44 * len(reasons))
        stretch(st.altair_chart, chart)
        action = {"Low": "No action needed. Keep the normal experience.",
                  "Medium": "Light touch: a reminder about the renewal date or a small loyalty perk.",
                  "High": "Priority: contact before expiry with a retention offer, for example a discount on the next plan."}
        st.success(f"Suggested action: {action.get(band, '')}")

elif page == "Retention simulator":
    st.title("Retention simulator")
    st.write("Every member at or above the threshold gets an offer. Offers cost money, and only some churners "
             "accept. Change the assumptions and see whether the campaign pays off. Results are per 100,000 members, "
             "based on the model's out-of-time test on March 2017.")
    d = meta["simulator_defaults"]
    c1, c2, c3, c4 = st.columns(4)
    cost = c1.slider("Cost of one offer (NT$)", 10, 300, int(d["offer_cost"]), step=10)
    value = c2.slider("Value of keeping one member (NT$)", 100, 2000, int(d["saved_value"]), step=50)
    success = c3.slider("Offer success rate", 0.05, 0.80, float(d["success_rate"]), step=0.05)
    threshold = c4.slider("Risk threshold", 0.01, 0.99, float(meta["simulator_threshold"]), step=0.01)

    thr = load_csv("threshold_table.csv")
    n = thr[["tp", "fp", "fn", "tn"]].sum(axis=1)
    scale = 100_000 / n
    sim = pd.DataFrame({"threshold": thr["threshold"], "targeted": (thr["tp"] + thr["fp"]) * scale,
                        "churners reached": thr["tp"] * scale, "offers to stayers": thr["fp"] * scale,
                        "saved": thr["tp"] * success * scale})
    sim["net benefit"] = sim["saved"] * value - sim["targeted"] * cost
    row = sim.iloc[(sim["threshold"] - threshold).abs().argmin()]
    best = sim.loc[sim["net benefit"].idxmax()]

    m = st.columns(5)
    m[0].metric("Offers sent", f"{row['targeted']:,.0f}")
    m[1].metric("Churners reached", f"{row['churners reached']:,.0f}")
    m[2].metric("Offers to stayers", f"{row['offers to stayers']:,.0f}", help="Offers sent to members who would have renewed anyway")
    m[3].metric("Members kept", f"{row['saved']:,.0f}")
    m[4].metric("Net benefit (NT$)", f"{row['net benefit']:,.0f}")

    base = alt.Chart(sim).mark_line(color="#1F3864").encode(
        x=alt.X("threshold:Q", title="risk threshold"),
        y=alt.Y("net benefit:Q", title="net benefit (NT$ per 100,000 members)"),
        tooltip=[alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("net benefit:Q", format=",.0f"),
                 alt.Tooltip("targeted:Q", format=",.0f")])
    chosen = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(strokeDash=[6, 4], color="#B8860B").encode(x="x:Q")
    best_pt = alt.Chart(pd.DataFrame({"threshold": [best["threshold"]], "net benefit": [best["net benefit"]]})).mark_point(
        size=120, filled=True, color="#B03A2E").encode(x="threshold:Q", y="net benefit:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="grey").encode(y="y:Q")
    stretch(st.altair_chart, (base + chosen + best_pt + zero).properties(height=360))
    st.write(f"Under these assumptions the best threshold is **{best['threshold']:.2f}**, with offers to "
             f"{best['targeted']:,.0f} members and a net benefit of **NT${best['net benefit']:,.0f}** "
             "(red point). The dashed line is your chosen threshold.")
    st.warning("Cost, value and success rate are assumptions, not facts from the data. The simulator assumes an offer "
               "does not change the behaviour of members who would have stayed anyway.")

elif page == "Model performance":
    st.title("Model performance")
    st.write("All scores are on the same March 2017 test members. **Random split** trains on other March members, "
             "as in Milestone 2. **Out-of-time** trains on February 2017 members, which is closer to real use.")
    show = results.copy()
    stretch(st.dataframe, show.style.format({c: "{:.4f}" for c in show.columns if c not in ("setting", "model")}),
                 hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model selection: 3-fold cross-validation on the February cohort**")
        stretch(st.dataframe, load_csv("cv_table.csv"), hide_index=True)
    with c2:
        st.markdown("**Calibration of LightGBM (March test)**")
        stretch(st.dataframe, load_csv("calibration_table.csv"), hide_index=True)

    curves = load_json("curves.json")
    roc = pd.concat([pd.DataFrame({"false positive rate": v["fpr"], "true positive rate": v["tpr"], "model": k})
                     for k, v in curves["roc"].items()])
    pr = pd.concat([pd.DataFrame({"recall": v["recall"], "precision": v["precision"], "model": k})
                    for k, v in curves["pr"].items()])
    cal = pd.concat([pd.DataFrame({"mean predicted": v["mean_predicted"], "real churn rate": v["fraction_churned"], "score": k})
                     for k, v in curves["calibration"].items()])
    g1, g2, g3 = st.columns(3)
    stretch(g1.altair_chart, alt.Chart(roc).mark_line().encode(x="false positive rate:Q", y="true positive rate:Q", color=alt.Color("model:N", legend=alt.Legend(orient="bottom", title=None, columns=1)))
                    .properties(title="ROC curve (out-of-time)", height=300))
    stretch(g2.altair_chart, alt.Chart(pr).mark_line().encode(x="recall:Q", y="precision:Q", color=alt.Color("model:N", legend=alt.Legend(orient="bottom", title=None, columns=1)))
                    .properties(title="Precision-recall curve (out-of-time)", height=300))
    diag = alt.Chart(pd.DataFrame({"mean predicted": [0, 1], "real churn rate": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="grey").encode(x="mean predicted:Q", y="real churn rate:Q")
    stretch(g3.altair_chart, (alt.Chart(cal).mark_line(point=True).encode(x="mean predicted:Q", y="real churn rate:Q", color=alt.Color("score:N", legend=alt.Legend(orient="bottom", title=None, columns=1))) + diag)
                    .properties(title="Reliability diagram", height=300))

    st.markdown("**Confusion matrix of calibrated LightGBM at a chosen threshold**")
    thr = load_csv("threshold_table.csv")
    t = st.slider("Threshold", 0.01, 0.99, float(meta["f1_threshold"]), step=0.01)
    r = thr.iloc[(thr["threshold"] - t).abs().argmin()]
    cm = pd.DataFrame({"predicted stay": [int(r["tn"]), int(r["fn"])], "predicted churn": [int(r["fp"]), int(r["tp"])]},
                      index=["really stayed", "really churned"])
    precision = r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) else 0.0
    recall = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) else 0.0
    a, b = st.columns([1, 1])
    stretch(a.dataframe, cm)
    b.metric("Precision", f"{precision:.3f}")
    b.metric("Recall", f"{recall:.3f}")

elif page == "What drives churn":
    st.title("What drives churn")
    imp = load_csv("shap_importance.csv").head(12)
    imp["feature name"] = imp["feature"].map(nice)
    st.markdown("**Overall importance in the model** (average size of SHAP values)")
    stretch(st.altair_chart, alt.Chart(imp).mark_bar(color="#2F5597").encode(
        x=alt.X("mean_abs_shap:Q", title="average effect on log-odds"),
        y=alt.Y("feature name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False))).properties(height=34 * len(imp)))
    st.caption("SHAP explains what the model relies on. It does not prove what causes churn in the real world.")

    st.markdown("**Churn rate by segment, March 2017 cohort**")
    eda = load_csv("eda_rates.csv")
    groups = list(eda["group"].unique())
    tabs = st.tabs(groups)
    for tab, g in zip(tabs, groups):
        part = eda[eda["group"] == g]
        with tab:
            stretch(st.altair_chart, alt.Chart(part).mark_bar(color="#B8860B").encode(
                x=alt.X("segment:N", title=None, sort=None), y=alt.Y("churn_rate:Q", title="churn rate", axis=alt.Axis(format="%")),
                tooltip=["segment", alt.Tooltip("churn_rate:Q", format=".1%"), alt.Tooltip("members:Q", format=",")]
            ).properties(height=280))
            stretch(st.dataframe, part[["segment", "members", "churn_rate"]].style.format({"churn_rate": "{:.1%}", "members": "{:,}"}),
                         hide_index=True)
    st.markdown("**February vs March cohort (data drift)**")
    stretch(st.dataframe, load_csv("drift.csv"), hide_index=True)
    st.markdown("**Risk bands on the March test members**")
    stretch(st.dataframe, load_csv("risk_bands.csv"), hide_index=True)

else:
    st.title("About and limits")
    st.markdown(
        "**Data.** KKBOX churn data released for the WSDM Cup 2018 on Kaggle (Chen et al., 2018). KKBOX is a music-streaming "
        "subscription service in Taiwan. The data covers 2015 to early 2017, and amounts are in New Taiwan dollars. "
        "This app holds no raw KKBOX data, only the trained model and summary tables.\n\n"
        "**Target.** A member churns if they make no new valid subscription within 30 days after the membership expires.\n\n"
        "**Model.** LightGBM trained on members expiring in February 2017, calibrated with Platt scaling and tested "
        "on members expiring in March 2017.\n\n"
        "**Limits.**\n"
        "* One company in one country, a music service and not a video service. Results may not transfer to Netflix.\n"
        "* Behaviour changes over time, so the model should be retrained regularly.\n"
        "* The simulator's cost, value and success rate are assumptions.\n"
        "* The risk checker fills fields you do not set with typical values, so it is a demonstration, not a production system.\n"
        "* We do not claim to beat the WSDM Cup winning solutions, which were scored on a different test set.")
    v = meta.get("versions", {})
    st.caption("Model trained with " + ", ".join(f"{k} {val}" for k, val in v.items()))
