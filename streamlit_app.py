import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import skew, kurtosis

st.set_page_config(page_title="Fake Currency Detection", page_icon="💵", layout="wide")

FEATURES = ["variance", "skewness", "curtosis", "entropy"]


@st.cache_resource
def load_model():
    return joblib.load("models/banknote_model.joblib")


@st.cache_data
def load_samples():
    with open("models/samples.json") as f:
        return json.load(f)


@st.cache_data
def load_reference():
    return pd.read_csv("models/reference_data.csv")


@st.cache_data
def load_summary():
    with open("outputs/summary.json") as f:
        return json.load(f)


model = load_model()
samples = load_samples()
ref = load_reference()
summary = load_summary()
ranges = samples["ranges"]

# ---------- session defaults ----------
for f in FEATURES:
    if f not in st.session_state:
        st.session_state[f] = float(round(sum(ranges[f]) / 2, 2))


def set_values(d):
    for f in FEATURES:
        st.session_state[f] = float(d[f])


# ---------- header ----------
st.title("💵 Fake Currency Detection")
st.caption(
    f"Machine-learning banknote authenticator  |  Model: {samples['best_model']}  |  "
    f"Test accuracy: {summary['test_accuracy']*100:.2f}%  |  Fake recall: {summary['test_recall']*100:.2f}%"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Check a note", "📦 Batch check", "📊 Model insights", "🖼️ Image lab (demo)"]
)

# =====================================================================
# TAB 1: single note
# =====================================================================
with tab1:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Note features")
        b1, b2 = st.columns(2)
        b1.button("Load genuine sample", on_click=set_values, args=(samples["genuine"],))
        b2.button("Load fake sample", on_click=set_values, args=(samples["fake"],))

        values = {}
        for f in FEATURES:
            lo, hi = ranges[f]
            values[f] = st.slider(
                f.capitalize(),
                min_value=float(round(lo - 1, 2)),
                max_value=float(round(hi + 1, 2)),
                step=0.01,
                key=f,
            )

    row = pd.DataFrame([values])
    proba = model.predict_proba(row)[0]
    p_fake = float(proba[1])
    is_fake = p_fake >= 0.5

    with right:
        st.subheader("Verdict")
        if is_fake:
            st.error(f"🚨 FAKE note detected  (confidence {p_fake*100:.1f}%)")
        else:
            st.success(f"✅ GENUINE note  (confidence {(1-p_fake)*100:.1f}%)")
        st.progress(p_fake, text=f"Probability of being fake: {p_fake*100:.1f}%")

        if 0.2 < p_fake < 0.8:
            st.warning("⚠️ Uncertain zone: this note is close to the decision boundary. Verify manually.")

        st.markdown("**Where does this note sit among all known notes?**")
        axes_choice = st.selectbox(
            "Map axes",
            ["variance vs skewness", "variance vs curtosis", "skewness vs curtosis", "curtosis vs entropy"],
            label_visibility="collapsed",
        )
        xa, ya = axes_choice.split(" vs ")
        fig, ax = plt.subplots(figsize=(6, 4))
        g = ref[ref["class"] == 0]
        k = ref[ref["class"] == 1]
        ax.scatter(g[xa], g[ya], s=10, alpha=0.5, c="#2E8B57", label="Genuine")
        ax.scatter(k[xa], k[ya], s=10, alpha=0.5, c="#D9534F", label="Fake")
        ax.scatter([values[xa]], [values[ya]], s=260, marker="*", c="gold",
                   edgecolors="black", linewidths=1.2, label="Your note", zorder=5)
        ax.set_xlabel(xa)
        ax.set_ylabel(ya)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🔬 What-if analysis: how does one feature change the verdict?"):
        feat = st.selectbox("Feature to vary", FEATURES, key="whatif_feat")
        lo, hi = ranges[feat]
        grid = np.linspace(lo, hi, 120)
        base = pd.concat([row] * len(grid), ignore_index=True)
        base[feat] = grid
        curve = model.predict_proba(base)[:, 1]
        fig2, ax2 = plt.subplots(figsize=(7, 3.2))
        ax2.plot(grid, curve, color="#4682B4", linewidth=2)
        ax2.axhline(0.5, linestyle="--", color="gray")
        ax2.axvline(values[feat], color="gold", linewidth=2, label="Current value")
        ax2.set_xlabel(feat)
        ax2.set_ylabel("P(fake)")
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("Other three features are held at their current slider values.")

# =====================================================================
# TAB 2: batch
# =====================================================================
with tab2:
    st.subheader("Check many notes at once")
    st.caption("Upload a CSV with columns: variance, skewness, curtosis, entropy")
    template = ref[FEATURES].head(10)
    st.download_button("⬇️ Download a sample CSV to try", template.to_csv(index=False),
                       file_name="sample_notes.csv", mime="text/csv")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        data = pd.read_csv(up)
        if all(c in data.columns for c in FEATURES):
            pf = model.predict_proba(data[FEATURES])[:, 1]
            out = data.copy()
            out["fake_probability"] = pf.round(4)
            out["verdict"] = np.where(pf >= 0.5, "Fake", "Genuine")
            c1, c2, c3 = st.columns(3)
            c1.metric("Notes checked", len(out))
            c2.metric("Genuine", int((out["verdict"] == "Genuine").sum()))
            c3.metric("Fake", int((out["verdict"] == "Fake").sum()))
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ Download results", out.to_csv(index=False),
                               file_name="results.csv", mime="text/csv")
        else:
            st.warning("CSV must contain these columns: " + ", ".join(FEATURES))

# =====================================================================
# TAB 3: insights
# =====================================================================
with tab3:
    st.subheader("How good is the model?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{summary['test_accuracy']*100:.2f}%")
    m2.metric("Precision", f"{summary['test_precision']*100:.2f}%")
    m3.metric("Recall (fakes caught)", f"{summary['test_recall']*100:.2f}%")
    m4.metric("F1-score", f"{summary['test_f1']*100:.2f}%")

    st.markdown("**Model comparison**")
    st.dataframe(pd.read_csv("outputs/model_comparison.csv"), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.image("outputs/05_confusion_matrix.png", caption="Confusion matrix (test set)")
    c2.image("outputs/06_feature_importance.png", caption="Which feature matters most")
    c3, c4 = st.columns(2)
    c3.image("outputs/07_robustness.png", caption="Robustness: performance as input gets noisier")
    c4.image("outputs/04_model_comparison.png", caption="All models compared")
    st.image("outputs/03_pairplot.png", caption="Feature relationships by class")

# =====================================================================
# TAB 4: image lab
# =====================================================================
with tab4:
    st.subheader("Wavelet image lab (visual demo)")
    st.info(
        "This tab is a **visual demo only**. It shows the wavelet decomposition technique that "
        "banknote-authentication datasets are built on. It does NOT give a fake/genuine verdict, "
        "because the model above was trained on pre-extracted features, not raw photos. "
        "Photo-based detection with a CNN is listed as future work."
    )
    img_file = st.file_uploader("Upload a banknote photo", type=["png", "jpg", "jpeg"], key="img")
    if img_file is not None:
        try:
            import pywt

            img = Image.open(img_file).convert("L").resize((400, 400))
            arr = np.array(img, dtype=float)
            cA, (cH, cV, cD) = pywt.dwt2(arr, "haar")
            bands = [("Grayscale", arr), ("Approximation", cA), ("Horizontal detail", cH),
                     ("Vertical detail", cV), ("Diagonal detail", cD)]
            fig3, axes = plt.subplots(1, 5, figsize=(16, 3.6))
            for ax, (title, d) in zip(axes, bands):
                ax.imshow(d, cmap="gray")
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            st.pyplot(fig3)
            plt.close(fig3)

            stats_rows = []
            for title, d in bands[1:]:
                flat = d.flatten()
                stats_rows.append({"band": title, "variance": round(float(np.var(flat)), 2),
                                   "skewness": round(float(skew(flat)), 3),
                                   "kurtosis": round(float(kurtosis(flat)), 3)})
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
            st.caption("Statistics of each wavelet band. Not comparable to the dataset's feature scale.")
        except Exception as e:
            st.error(f"Could not process image: {e}")

st.divider()
st.caption("AI/ML Internship Final Project  |  Built with scikit-learn and Streamlit (all free tools)")
