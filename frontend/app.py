import os
import requests
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Dict, Any

st.set_page_config(
    page_title="Truth Scanner",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header   { visibility: hidden; }
    .stTextArea textarea        { border-radius: 10px; font-size: 0.95rem; }
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 2.5rem;
        transition: opacity 0.2s;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
_ROOT   = os.path.join(os.path.dirname(__file__), "..")


def _read_report(filename: str) -> str:
    """Return the content of a report file, or an empty string if absent."""
    path = os.path.join(_ROOT, filename)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


with st.sidebar:
    st.markdown("### Truth Scanner")
    st.caption("AI-powered fake news detection")
    st.divider()

    st.markdown(
        "**Model:** Soft-Voting Ensemble (5 classifiers)  \n"
        "**Features:** TF-IDF bigrams + Bag-of-Words  \n"
        "**NLP:** SpaCy lemmatisation  \n"
        "**Accuracy:** 99.64 % on ISOT dataset",
    )
    st.divider()

    st.info(
        "**Note:** Optimised for 2016-2017 political news. "
        "Topics from 2024+ may yield lower accuracy due to temporal data drift.",
    )
    st.divider()

    combined = (
        "TRUTH SCANNER — PROJECT REPORT\n"
        + "=" * 60 + "\n\n"
        + "=" * 60 + "\nEVALUATION RESULTS\n" + "=" * 60 + "\n"
        + _read_report("evaluation_results.txt")
        + "\n" + "=" * 60 + "\nROBUSTNESS REPORT\n" + "=" * 60 + "\n"
        + _read_report("robustness_report.md")
    )
    st.download_button(
        label="Download Full Report",
        data=combined,
        file_name="truth_scanner_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


_MAX_PAYLOAD = 50_000


def call_api(text: str) -> Optional[Dict[str, Any]]:
    """POST text to the prediction API and return the parsed JSON response."""
    try:
        r = requests.post(API_URL, json={"text": text}, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Backend server is currently unreachable. Please ensure the API is running.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may still be loading.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e.response.status_code} — {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def build_gauge(truth_prob: float) -> go.Figure:
    """Return a Plotly gauge figure coloured by the truth probability."""
    pct = truth_prob * 100

    if pct >= 60:
        color, label = "#22c55e", "Likely Real"
    elif pct <= 40:
        color, label = "#ef4444", "Likely Fake"
    else:
        color, label = "#f59e0b", "Uncertain"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40],  "color": "rgba(239,68,68,0.10)"},
                {"range": [40, 60],  "color": "rgba(245,158,11,0.10)"},
                {"range": [60, 100], "color": "rgba(34,197,94,0.10)"},
            ],
        },
        title={"text": f"Probability of Truth<br><b>{label}</b>", "font": {"size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=0),
        height=260,
    )
    return fig


st.title("Truth Scanner")
st.caption(
    "Paste any news article below. The ensemble model will classify it "
    "as Real or Fake based on its linguistic fingerprint.",
)
st.divider()

with st.container():
    text = st.text_area(
        label="News Article Text",
        height=200,
        placeholder="Paste the news article here...",
    )
    _, btn_col, _ = st.columns([2, 2, 2])
    with btn_col:
        run = st.button("Analyze", type="primary", use_container_width=True)

if run:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    elif len(text) > _MAX_PAYLOAD:
        st.warning(
            f"Input exceeds the {_MAX_PAYLOAD:,} character limit "
            f"({len(text):,} chars). Please trim the article."
        )
    else:
        with st.spinner("Analyzing..."):
            result = call_api(text)

        if result:
            prediction   = result.get("prediction", "ERROR")
            confidence   = result.get("confidence_score", 0.0)
            truth_prob   = confidence if prediction == "TRUE" else 1.0 - confidence
            is_uncertain = 0.40 <= truth_prob <= 0.60

            st.divider()

            col_verdict, col_gauge = st.columns(2, gap="large")

            with col_verdict:
                st.subheader("Verdict")
                if is_uncertain:
                    st.warning(
                        f"Uncertain — {truth_prob * 100:.1f}% probability of being real. "
                        "Please verify with additional sources.",
                    )
                elif prediction == "TRUE":
                    st.success(f"Real News — {confidence * 100:.1f}% confidence")
                else:
                    st.error(f"Fake News — {confidence * 100:.1f}% confidence")

                st.metric("Truth Probability", f"{truth_prob * 100:.1f}%")
                st.metric("Model Confidence",  f"{confidence * 100:.1f}%")
                st.info(
                    "Trained on 2016-2017 ISOT political news. "
                    "Accuracy may be lower for modern topics.",
                )

            with col_gauge:
                st.subheader("Probability of Truth")
                st.plotly_chart(
                    build_gauge(truth_prob),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            st.divider()
            tab_eval, tab_robust, tab_how = st.tabs([
                "Evaluation Metrics",
                "Robustness Report",
                "How It Works",
            ])

            with tab_eval:
                content = _read_report("evaluation_results.txt")
                st.code(content, language="text") if content else st.info(
                    "Run `python src/evaluate.py` to generate this report.",
                )

            with tab_robust:
                content = _read_report("robustness_report.md")
                st.code(content, language="text") if content else st.info(
                    "Run `python src/robustness_test.py` to generate this report.",
                )

            with tab_how:
                st.markdown("""
| Stage | Method |
|---|---|
| Cleaning | Lowercasing, URL/HTML strip, byline removal |
| NLP | SpaCy lemmatisation + stopword removal |
| Vectorisation | TF-IDF (bigrams) + CountVectorizer |
| Classifiers | MultinomialNB, Logistic Regression, LinearSVC, SGD, RandomForest |
| Ensemble | Soft-Voting (probability averaging) |
                """)
