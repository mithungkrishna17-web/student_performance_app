"""
Student Performance Predictor — Streamlit web app.

Run:  streamlit run app.py
"""
import html
import json
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
}
ID_COLS = ["roll_no", "name", "class"]  # shown in the app, never used by the models
FEATURES = ["study_hours", "attendance", "previous_grade", "socioeconomic_status",
            "parental_education", "internet_access", "extracurricular",
            "sleep_hours", "tutoring_sessions"]

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


# ---------- loading ----------
def ensure_trained():
    if not (MODELS_DIR / "metrics.json").exists():
        with st.spinner("First run: generating data and training models (~30 s)…"):
            if not DATA_PATH.exists():
                subprocess.run([sys.executable, str(ROOT / "generate_data.py")], check=True)
            subprocess.run([sys.executable, str(ROOT / "train.py")], check=True)


@st.cache_resource
def load_models():
    return {n: joblib.load(MODELS_DIR / f) for n, f in MODEL_FILES.items()}


@st.cache_data
def load_report():
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


ensure_trained()
models = load_models()
report = load_report()
best_name = report["best_model"]


def risk_level(p_fail: float) -> tuple[str, str]:
    if p_fail >= 0.6:
        return "High risk", "#d64545"
    if p_fail >= 0.35:
        return "Moderate risk", "#e0a030"
    return "Low risk", "#2e9e5b"


def suggestions(row: dict) -> list[str]:
    tips = []
    if row["attendance"] < 75:
        tips.append(f"Attendance is {row['attendance']:.0f}% — set up an attendance check-in plan (target ≥ 85%).")
    if row["study_hours"] < 10:
        tips.append(f"Only {row['study_hours']:.0f} study hours/week — recommend a structured study schedule or study group.")
    if row["previous_grade"] < 55:
        tips.append(f"Previous grade {row['previous_grade']:.0f} — assign remedial classes on foundational topics.")
    if row["tutoring_sessions"] < 1:
        tips.append("No tutoring sessions — enrol in peer or faculty tutoring.")
    if row["internet_access"] == "No":
        tips.append("No internet access at home — provide library/lab access or offline materials.")
    if row["socioeconomic_status"] == "Low":
        tips.append("Low socioeconomic status — check eligibility for scholarships, meals, or counselling support.")
    if not (6.5 <= row["sleep_hours"] <= 9):
        tips.append(f"Sleeps {row['sleep_hours']:.1f} h/night — discuss wellbeing and sleep habits.")
    return tips


# ---------- sidebar ----------
st.sidebar.title("🎓 Student Performance")
st.sidebar.caption("Predict pass/fail and identify at-risk students early.")
model_name = st.sidebar.selectbox(
    "Model", list(MODEL_FILES), index=list(MODEL_FILES).index(best_name),
    help=f"Best model by F1 score on the test set: {best_name}",
)
m = report["models"][model_name]["metrics"]
st.sidebar.markdown(f"**{model_name}** test metrics")
c1, c2 = st.sidebar.columns(2)
c1.metric("Accuracy", f"{m['accuracy']:.1%}")
c2.metric("F1", f"{m['f1']:.3f}")
c1.metric("Precision", f"{m['precision']:.1%}")
c2.metric("Recall", f"{m['recall']:.1%}")
st.sidebar.caption("Positive class = **Fail** (at-risk). Recall = share of failing students the model catches.")
model = models[model_name]

tab_pred, tab_batch, tab_eval, tab_data = st.tabs(
    ["🔮 Predict a student", "📋 Batch screening", "📊 Model evaluation", "🔍 Data explorer"]
)

# ---------- single prediction ----------
with tab_pred:
    st.subheader("Enter student details")
    with st.form("student"):
        n0, n1, n2 = st.columns([1, 2, 1])
        roll_no = n0.text_input("Roll number", placeholder="e.g. 23")
        student_name = n1.text_input("Student name", placeholder="e.g. Priya Sharma")
        student_class = n2.text_input("Class", placeholder="e.g. 10-A")
        a, b, c = st.columns(3)
        study_hours = a.slider("Study hours per week", 0.0, 40.0, 12.0, 0.5)
        attendance = a.slider("Attendance (%)", 30.0, 100.0, 80.0, 1.0)
        previous_grade = a.slider("Previous grade (0–100)", 20.0, 100.0, 65.0, 1.0)
        ses = b.selectbox("Socioeconomic status", ["Low", "Medium", "High"], index=1)
        pe = b.selectbox("Parental education", ["No Formal", "High School", "Bachelor", "Master+"], index=1)
        sleep = b.slider("Sleep hours per night", 3.0, 11.0, 7.0, 0.5)
        internet = c.radio("Internet access at home", ["Yes", "No"], horizontal=True)
        extra = c.radio("Extracurricular activities", ["Yes", "No"], horizontal=True, index=1)
        tutoring = c.number_input("Tutoring sessions per week", 0, 10, 1)
        compare = st.checkbox("Compare all three models", value=True)
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    if submitted:
        row = dict(study_hours=study_hours, attendance=attendance, previous_grade=previous_grade,
                   socioeconomic_status=ses, parental_education=pe, internet_access=internet,
                   extracurricular=extra, sleep_hours=sleep, tutoring_sessions=tutoring)
        X = pd.DataFrame([row])[FEATURES]
        p_fail = float(model.predict_proba(X)[0, 1])
        pred = "Fail" if model.predict(X)[0] == 1 else "Pass"
        level, color = risk_level(p_fail)
        who = html.escape(student_name.strip() or "Unnamed student")
        cls = html.escape(student_class.strip())
        roll = html.escape(roll_no.strip())
        st.session_state.setdefault("history", []).append({
            "roll_no": roll_no.strip(),
            "name": student_name.strip() or "Unnamed student", "class": student_class.strip(),
            **row, "model": model_name, "prediction": pred, "p_fail": round(p_fail, 3), "risk": level,
        })

        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"<div style='padding:1.2rem;border-radius:12px;background:{color}22;border:2px solid {color}'>"
                f"<div style='font-size:1.15rem;font-weight:600'>{who}"
                f"{f' · Class {cls}' if cls else ''}{f' · Roll No. {roll}' if roll else ''}</div>"
                f"<div style='font-size:0.9rem;opacity:.8'>{model_name} prediction</div>"
                f"<div style='font-size:2.2rem;font-weight:700;color:{color}'>{pred}</div>"
                f"<div style='font-size:1.1rem'>{level} · {p_fail:.0%} probability of failing</div></div>",
                unsafe_allow_html=True,
            )
            if compare:
                st.markdown("##### All models")
                rows = []
                for n, mdl in models.items():
                    pf = float(mdl.predict_proba(X)[0, 1])
                    rows.append({"Model": n, "Prediction": "Fail" if mdl.predict(X)[0] == 1 else "Pass",
                                 "P(fail)": f"{pf:.0%}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p_fail * 100, number={"suffix": "%"},
                title={"text": "Risk of failing"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 35], "color": "#2e9e5b33"},
                                 {"range": [35, 60], "color": "#e0a03033"},
                                 {"range": [60, 100], "color": "#d6454533"}]},
            ))
            gauge.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(gauge, width="stretch")

        tips = suggestions(row)
        st.markdown("##### Recommended support actions")
        if tips:
            for t in tips:
                st.markdown(f"- {t}")
        else:
            st.success("No specific risk factors detected — keep up the current routine.")

    hist = st.session_state.get("history", [])
    if hist:
        st.divider()
        st.markdown(f"##### Students checked this session ({len(hist)})")
        hdf = pd.DataFrame(hist)[["roll_no", "name", "class", "prediction", "p_fail", "risk", "model",
                                  "attendance", "study_hours", "previous_grade"]]
        st.dataframe(hdf, hide_index=True, width="stretch",
                     column_config={"p_fail": st.column_config.ProgressColumn(
                         "P(fail)", min_value=0, max_value=1, format="percent")})
        h1, h2 = st.columns(2)
        h1.download_button("Download this list", pd.DataFrame(hist).to_csv(index=False),
                           "checked_students.csv", "text/csv")
        if h2.button("Clear list"):
            st.session_state["history"] = []
            st.rerun()

# ---------- batch ----------
with tab_batch:
    st.subheader("Screen a whole class")
    st.write("Upload a CSV with these columns. `roll_no`, `name` and `class` are optional but recommended; "
             "a `result` column is ignored:")
    st.code(", ".join(ID_COLS + FEATURES))
    template = load_data().drop(columns=["result"]).head(20)
    st.download_button("Download sample CSV", template.to_csv(index=False),
                       "sample_students.csv", "text/csv")
    up = st.file_uploader("Upload student CSV", type="csv")
    df_in = pd.read_csv(up) if up else None
    if df_in is None and st.button("Use 20 sample students instead"):
        df_in = template
    if df_in is not None:
        missing = [c for c in FEATURES if c not in df_in.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            out = df_in.drop(columns=["result"], errors="ignore").copy()
            if "name" not in out.columns:
                out.insert(0, "name", [f"Student {i + 1}" for i in range(len(out))])
            if "roll_no" not in out.columns:
                out.insert(0, "roll_no", range(1, len(out) + 1))
            out["roll_no"] = out["roll_no"].astype(str)
            out["p_fail"] = model.predict_proba(out[FEATURES])[:, 1].round(3)
            out["prediction"] = ["Fail" if p == 1 else "Pass" for p in model.predict(out[FEATURES])]
            out["risk"] = [risk_level(p)[0] for p in out["p_fail"]]
            front = [c for c in ID_COLS if c in out.columns] + ["prediction", "p_fail", "risk"]
            out = out[front + [c for c in out.columns if c not in front]]
            out = out.sort_values("p_fail", ascending=False)

            if "class" in out.columns:
                out["class"] = out["class"].astype(str)
                classes = sorted(out["class"].unique())
                picked = st.multiselect("Filter by class", classes, default=classes)
                out = out[out["class"].isin(picked)]

            k1, k2, k3 = st.columns(3)
            k1.metric("Students", len(out))
            k2.metric("Predicted to fail", int((out["prediction"] == "Fail").sum()))
            k3.metric("High risk", int((out["risk"] == "High risk").sum()))
            st.dataframe(
                out, hide_index=True, width="stretch",
                column_config={"p_fail": st.column_config.ProgressColumn(
                    "P(fail)", min_value=0, max_value=1, format="percent")},
            )

            if "class" in out.columns and out["class"].nunique() > 1:
                st.markdown("##### At-risk students by class")
                summary = (out.groupby("class")
                           .agg(students=("name", "size"),
                                predicted_fail=("prediction", lambda s: int((s == "Fail").sum())),
                                avg_risk=("p_fail", "mean"))
                           .reset_index())
                summary["fail_rate"] = summary["predicted_fail"] / summary["students"]
                cf = px.bar(summary, x="class", y="predicted_fail", text="predicted_fail",
                            labels={"predicted_fail": "Students predicted to fail", "class": "Class"},
                            color_discrete_sequence=["#d64545"])
                cf.update_layout(height=320, margin=dict(t=20))
                st.plotly_chart(cf, width="stretch")
            st.download_button("Download results", out.to_csv(index=False),
                               "at_risk_students.csv", "text/csv", type="primary")

# ---------- evaluation ----------
with tab_eval:
    st.subheader("Model comparison on held-out test set")
    st.caption(f"Train: {report['n_train']} students · Test: {report['n_test']} students · "
               f"Hyperparameters tuned with 5-fold cross-validation (F1).")
    rows = []
    for n, r in report["models"].items():
        rows.append({"Model": n, **{k.upper() if k == "f1" else k.replace("_", " ").title(): v
                                    for k, v in r["metrics"].items()}})
    mdf = pd.DataFrame(rows).rename(columns={"Roc Auc": "ROC AUC", "Cv F1": "CV F1"})
    st.dataframe(mdf.style.highlight_max(axis=0, subset=mdf.columns[1:], color="#2e9e5b44")
                 .format(precision=3), hide_index=True, width="stretch")

    long = mdf.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"],
                    var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 text_auto=".2f", range_y=[0, 1])
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"##### Confusion matrix — {model_name}")
        cm = report["models"][model_name]["confusion_matrix"]
        cmf = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: Pass", "Pred: Fail"], y=["Actual: Pass", "Actual: Fail"])
        cmf.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(cmf, width="stretch")
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.caption(f"Caught {tp} of {tp + fn} failing students; {fp} passing students flagged unnecessarily.")
    with e2:
        st.markdown(f"##### Feature importance — {model_name}")
        fi = pd.Series(report["models"][model_name]["feature_importance"]).sort_values()
        fif = px.bar(fi, orientation="h", labels={"value": "Importance", "index": ""})
        fif.update_layout(height=340, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fif, width="stretch")

    st.markdown("##### Best hyperparameters")
    st.json({n: r["best_params"] for n, r in report["models"].items()})

    with st.expander("Metric definitions"):
        st.markdown(
            "- **Accuracy** – share of all students classified correctly.\n"
            "- **Precision** – of students flagged as *Fail*, how many actually failed.\n"
            "- **Recall** – of students who actually failed, how many were flagged (most important for early intervention).\n"
            "- **F1 score** – harmonic mean of precision and recall.\n"
            "- **ROC AUC** – how well the model ranks at-risk students above others, across all thresholds."
        )

    st.divider()
    st.markdown("##### Retrain on your own data")
    new = st.file_uploader("Upload a labelled CSV (features + `result` column with Pass/Fail)",
                           type="csv", key="train")
    if new and st.button("Retrain models", type="primary"):
        df_new = pd.read_csv(new)
        need = FEATURES + ["result"]
        if any(c not in df_new.columns for c in need):
            st.error("CSV must contain: " + ", ".join(need))
        else:
            df_new.to_csv(DATA_PATH, index=False)
            with st.spinner("Training…"):
                res = subprocess.run([sys.executable, str(ROOT / "train.py")],
                                     capture_output=True, text=True)
            if res.returncode == 0:
                st.cache_resource.clear(); st.cache_data.clear()
                st.success("Models retrained.")
                st.code(res.stdout)
                st.rerun()
            else:
                st.error(res.stderr[-2000:])

# ---------- data explorer ----------
with tab_data:
    df = load_data()
    st.subheader("Training data")
    d1, d2, d3 = st.columns(3)
    d1.metric("Students", len(df))
    d2.metric("Pass rate", f"{(df['result'] == 'Pass').mean():.0%}")
    d3.metric("Features", len(FEATURES))
    feat = st.selectbox("Feature", FEATURES)
    if df[feat].dtype == object:
        fig = px.histogram(df, x=feat, color="result", barmode="group",
                           color_discrete_map={"Pass": "#2e9e5b", "Fail": "#d64545"})
    else:
        fig = px.histogram(df, x=feat, color="result", barmode="overlay", opacity=0.65, nbins=30,
                           color_discrete_map={"Pass": "#2e9e5b", "Fail": "#d64545"})
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.head(200), width="stretch", hide_index=True)
