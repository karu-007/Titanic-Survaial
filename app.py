import os
import pandas as pd
import streamlit as st
import plotly.express as px
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="Titanic Survival Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .card {
            background: #111827;
            border: 1px solid rgba(255,255,255,0.08);
            padding: 1rem 1.1rem;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .small-muted {
            color: #94a3b8;
            font-size: 0.9rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #94a3b8;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Constants ----------
MODEL_PATH = "catboost_titanic_model.cbm"
TRAIN_PATHS = ["data/train.csv", "train.csv"]

cabin_map = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
    "F": 5, "G": 6, "T": 7, "U": 8,
}
reverse_cabin_map = {v: k for k, v in cabin_map.items()}
embarked_map = {"S": 0, "C": 1, "Q": 2}
reverse_embarked_map = {v: k for k, v in embarked_map.items()}

features = [
    "Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "Cabin"
]


# ---------- Helpers ----------
def load_training_data() -> pd.DataFrame | None:
    for path in TRAIN_PATHS:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")
        df["Cabin"] = df["Cabin"].apply(lambda x: str(x)[0])
        df["Cabin"] = df["Cabin"].replace({"N": "U"})

    if "Sex" in df.columns:
        if df["Sex"].dtype == object:
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    if "Embarked" in df.columns and df["Embarked"].dtype == object:
        df["Embarked"] = df["Embarked"].map(embarked_map)

    if "Cabin" in df.columns and df["Cabin"].dtype == object:
        df["Cabin"] = df["Cabin"].map(cabin_map)

    return df


def make_input_df(
    pclass: int,
    sex: str,
    age: float,
    fare: float,
    sibsp: int,
    parch: int,
    embarked: str,
    cabin_letter: str,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [{
            "Pclass": pclass,
            "Sex": 0 if sex == "male" else 1,
            "Age": age,
            "Fare": fare,
            "SibSp": sibsp,
            "Parch": parch,
            "Embarked": embarked_map[embarked],
            "Cabin": cabin_map[cabin_letter],
        }]
    )
    return row[features]


@st.cache_resource
def load_model() -> CatBoostClassifier | None:
    if not os.path.exists(MODEL_PATH):
        return None
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def get_clean_train() -> pd.DataFrame | None:
    raw = load_training_data()
    if raw is None:
        return None
    return preprocess_dataframe(raw)


def metric_card(title: str, value: str, delta: str | None = None):
    delta_html = f'<div class="small-muted">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="card">
            <div class="small-muted">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 800; margin-top: 0.15rem;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">{title}</div>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Load assets ----------
model = load_model()
train_df = get_clean_train()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🚢 Titanic Inputs")
    st.caption("Enter passenger details and generate a prediction.")

    with st.form("prediction_form"):
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", min_value=0, max_value=80, value=28)
        fare = st.number_input("Fare", min_value=0.0, value=32.20, step=1.0)
        sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0, step=1)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])
        cabin_letter = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F", "G", "T", "U"], index=8)

        submitted = st.form_submit_button("Predict Survival", use_container_width=True)

    st.markdown("---")
    st.markdown("### App Notes")
    st.caption(
        "This dashboard uses the same preprocessing pattern as your training notebook and "
        "expects a saved CatBoost model file named `catboost_titanic_model.cbm`."
    )

# ---------- Prediction ----------
input_df = make_input_df(pclass, sex, age, fare, sibsp, parch, embarked, cabin_letter)

prediction = None
survival_prob = None
if model is not None:
    pred_raw = model.predict(input_df)
    prediction = int(pred_raw[0])
    proba = model.predict_proba(input_df)[0]
    survival_prob = float(proba[1])

# ---------- Header ----------
st.markdown('<div class="hero-title">Titanic Survival Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Interactive prediction dashboard with passenger inputs, model output, and survival trends from the Titanic dataset.</div>',
    unsafe_allow_html=True,
)

# ---------- Top metrics ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    if prediction is None:
        metric_card("Prediction", "Model Missing", "Save model first")
    else:
        metric_card("Prediction", "Survived" if prediction == 1 else "Did Not Survive")

with col2:
    if survival_prob is None:
        metric_card("Survival Probability", "--")
    else:
        metric_card("Survival Probability", f"{survival_prob:.1%}")

with col3:
    metric_card("Passenger Class", str(pclass), f"{sex.title()}, age {age}")

with col4:
    family_size = sibsp + parch + 1
    metric_card("Family Size", str(family_size), f"Fare ${fare:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Main dashboard section ----------
left, right = st.columns([1.2, 1])

with left:
    st.markdown("### Prediction Summary")

    if prediction is None:
        st.warning("Model file not found. Train and save your CatBoost model first.")
        st.code('model.save_model("catboost_titanic_model.cbm")')
    else:
        if prediction == 1:
            st.success("This passenger is predicted to survive.")
        else:
            st.error("This passenger is predicted not to survive.")

        st.progress(int((survival_prob or 0) * 100))
        st.write(f"**Survival probability:** {(survival_prob or 0):.2%}")
        st.write(f"**Non-survival probability:** {1 - (survival_prob or 0):.2%}")

        reasons = []
        if sex == "female":
            reasons.append("female passengers generally had higher survival rates")
        if pclass == 1:
            reasons.append("first-class passengers often had better survival outcomes")
        if age <= 12:
            reasons.append("children were more likely to be prioritized")
        if fare >= 75:
            reasons.append("higher fares often align with higher passenger class")
        if family_size == 1:
            reasons.append("travelling alone can affect evacuation patterns")

        if reasons:
            st.markdown("**Possible factors influencing the prediction**")
            for reason in reasons[:4]:
                st.write(f"- {reason.capitalize()}.")


with right:
    st.markdown("### Survival Percentage")
    
    if prediction is None:
        st.info("The pie chart will appear after the model is loaded and a prediction is generated.")
    else:
        chart_df = pd.DataFrame({
            "Outcome": ["Survive", "Not Survive"],
            "Probability": [survival_prob or 0, 1 - (survival_prob or 0)]
        })
        fig_pie = px.pie(
            chart_df,
            names="Outcome",
            values="Probability",
            hole=0.55,
            title="Prediction Probability Split"
        )
        fig_pie.update_traces(textinfo="label+percent")
        fig_pie.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font_color="#e5e7eb",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

