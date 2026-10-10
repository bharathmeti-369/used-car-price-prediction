

import base64
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download



# --------------------------------------------------
# 1. PAGE SETTINGS AND FILE PATHS
# --------------------------------------------------


BASE_DIR = Path(__file__).parent

MODEL_REPO_ID = "harathmeti-369/used-car-price-prediction"
MODEL_FILENAME = "used_car_price_model.pkl"

DATA_PATH = BASE_DIR / "cardekho_dataset.csv"
BACKGROUND_PATH = BASE_DIR / "green_rolls_royce.jpg"
BMW_PATH = BASE_DIR / "bmw_car.png"

FEATURES = [
    "brand",
    "model",
    "vehicle_age",
    "km_driven",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

st.set_page_config(
    page_title="CarPredictor | Used Car AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. LOAD YOUR EXISTING MODEL AND DATASET
# --------------------------------------------------


@st.cache_resource
def load_model():
    model_file = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
    )
    return joblib.load(model_file)



@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_dataset()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.info(
        "Check that app.py, used_car_model.pkl, and "
        "cardekho_dataset.csv are in the same folder."
    )
    st.stop()


required_columns = FEATURES + ["selling_price"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"Dataset is missing these columns: {missing_columns}")
    st.stop()


# --------------------------------------------------
# 3. DARK LUXURY DASHBOARD STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at top right,
            rgba(0, 100, 65, 0.22), transparent 40%),
            linear-gradient(135deg, #061018, #0a1720 55%, #07110f);
        color: #f0f5f7;
    }

 
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-color: #071019 !important;
}
   
     
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[role="radiogroup"] label,
[data-testid="stSidebar"] div[role="radiogroup"] p {
    color: #ffffff !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: transparent;
    border-radius: 8px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: #17352c;
}

    
    /* Your existing CSS above */

/* Make the main-page form labels white */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stTextArea label,
.stSlider label {
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Keep input boxes light grey */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #F1F3F6 !important;
}

/* Keep the text inside input boxes dark */
div[data-baseweb="select"] span,
div[data-baseweb="input"] input {
    color: #102033 !important;
}

h1, h2, h3 {
    color: #f5f8fa !important;
}

/* Your remaining CSS below */

    .brand-title {
        font-size: 2.7rem;
        font-weight: 850;
        line-height: 1.15;
        color: #ecfdf5;
    }

    .green-text {
        color: #20e6a0;
    }

    .subtext {
        color: #b0c2cb;
        font-size: 1.05rem;
    }

    .hero-card {
        padding: 28px;
        border-radius: 20px;
        border: 1px solid #20473d;
        background: linear-gradient(
            125deg, rgba(8, 34, 31, 0.94),
            rgba(12, 23, 36, 0.96)
        );
        margin-bottom: 20px;
    }

    
div[data-testid="stMetric"] {
    background: #10251f !important;
    border: 1px solid #20e6a0 !important;
    border-radius: 14px;
    padding: 16px;
    color: #ffffff !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
    color: #ffffff !important;
}

div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
}

    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #20e6a0;
        font-weight: 700;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"] {
        background: #12c98b;
        color: #03140e;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #101e29;
        border-color: #29414d;
    }

    .small-note {
        color: #9db1bc;
        font-size: 0.86rem;
    }

    .road-scene {
        height: 245px;
        position: relative;
        overflow: hidden;
        border-radius: 18px;
        margin: 15px 0 24px 0;
        border: 1px solid #304b4a;
        background:
            radial-gradient(ellipse at 50% 100%,
            #ff5b12 0%, #9d230c 18%,
            #17222a 43%, #081019 75%);
    }

    .road {
        position: absolute;
        bottom: -80px;
        left: 34%;
        width: 32%;
        height: 260px;
        transform: perspective(130px) rotateX(20deg);
        background: repeating-linear-gradient(
            to bottom,
            #53616a 0px, #53616a 20px,
            #19232b 20px, #19232b 42px
        );
        opacity: 0.9;
    }

    .road::after {
        content: "";
        position: absolute;
        top: 0;
        left: 49%;
        height: 100%;
        width: 4px;
        background: repeating-linear-gradient(
            to bottom,
            #f9e9a8 0px, #f9e9a8 18px,
            transparent 18px, transparent 38px
        );
    }

    .flames {
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse at 50% 90%,
            rgba(255, 45, 0, .82), transparent 35%),
            radial-gradient(ellipse at 22% 100%,
            rgba(255, 125, 0, .62), transparent 27%),
            radial-gradient(ellipse at 80% 100%,
            rgba(255, 75, 0, .65), transparent 30%);
        animation: firePulse 1s infinite alternate;
    }

    .moving-car {
        position: absolute;
        bottom: -70px;
        left: 50%;
        z-index: 3;
        font-size: 64px;
        transform: translateX(-50%);
        animation: driveUp 2.6s ease-in-out forwards;
        filter: drop-shadow(0 0 18px #ff7b28);
        white-space: nowrap;
    }

    .moving-car img {
        width: 180px;
        height: 100px;
        object-fit: contain;
    }

    .road-label {
        position: absolute;
        top: 18px;
        left: 22px;
        z-index: 4;
        color: #e8fff4;
        font-weight: 700;
    }

    @keyframes driveUp {
        0% {
            bottom: -90px;
            transform: translateX(-50%) scale(.45);
            opacity: .25;
        }
        55% {
            opacity: 1;
        }
        100% {
            bottom: 110%;
            transform: translateX(-50%) scale(1.15);
            opacity: 0;
        }
    }

    @keyframes firePulse {
        from { opacity: .55; }
        to { opacity: 1; }
    }

    @media (prefers-reduced-motion: reduce) {
        .moving-car, .flames {
            animation-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 4. HELPERS
# --------------------------------------------------

def image_as_data_uri(path):
    """Convert a local image into an embedded browser image."""
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except OSError:
        return None


def get_options(column):
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values) if values else ["Unknown"]


def get_default_number(column, fallback):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return fallback
    return float(values.median())


def make_car_form(prefix, title):
    """Create a car input form using the existing model's exact features."""
    st.subheader(title)

    left, right = st.columns(2)

    with left:
        brand = st.selectbox(
            "Brand", get_options("brand"), key=f"{prefix}_brand"
        )
        car_model = st.selectbox(
            "Model", get_options("model"), key=f"{prefix}_model"
        )
        vehicle_age = st.number_input(
            "Vehicle age (years)",
            min_value=0,
            max_value=50,
            value=max(0, min(50, int(get_default_number("vehicle_age", 5)))),
            key=f"{prefix}_age",
        )
        km_driven = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=5_000_000,
            value=max(
                0, min(5_000_000, int(get_default_number("km_driven", 50000)))
            ),
            step=1000,
            key=f"{prefix}_km",
        )
        seller_type = st.selectbox(
            "Seller type", get_options("seller_type"),
            key=f"{prefix}_seller"
        )
        fuel_type = st.selectbox(
            "Fuel type", get_options("fuel_type"), key=f"{prefix}_fuel"
        )

    with right:
        transmission_type = st.selectbox(
            "Transmission",
            get_options("transmission_type"),
            key=f"{prefix}_transmission",
        )
        mileage = st.number_input(
            "Mileage",
            min_value=0.0,
            max_value=1000.0,
            value=max(0.0, min(1000.0, get_default_number("mileage", 20.0))),
            step=0.5,
            key=f"{prefix}_mileage",
        )
        engine = st.number_input(
            "Engine capacity (cc)",
            min_value=0.0,
            max_value=10000.0,
            value=max(0.0, min(10000.0, get_default_number("engine", 1200))),
            step=50.0,
            key=f"{prefix}_engine",
        )
        max_power = st.number_input(
            "Maximum power",
            min_value=0.0,
            max_value=2000.0,
            value=max(
                0.0, min(2000.0, get_default_number("max_power", 85))
            ),
            step=5.0,
            key=f"{prefix}_power",
        )
        seat_values = sorted(
            pd.to_numeric(df["seats"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if not seat_values:
            seat_values = [5]

        default_seats = int(get_default_number("seats", 5))
        seat_index = (
            seat_values.index(default_seats)
            if default_seats in seat_values else 0
        )
        seats = st.selectbox(
            "Number of seats",
            seat_values,
            index=seat_index,
            key=f"{prefix}_seats",
        )

    return {
        "brand": brand,
        "model": car_model,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }


def predict_price(car):
    """Pass the same 11 feature columns expected by the trained model."""
    input_data = pd.DataFrame([car], columns=FEATURES)
    prediction = float(model.predict(input_data)[0])
    return max(0.0, prediction)


def record_prediction(car, price):
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.insert(
        0,
        {
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Brand": car["brand"],
            "Model": car["model"],
            "Age (years)": car["vehicle_age"],
            "Kilometres": car["km_driven"],
            "Predicted Price (₹)": round(price, 2),
        },
    )


def show_hero():
    background_uri = image_as_data_uri(BACKGROUND_PATH)

    if background_uri:
        st.markdown(
            f"""
            <div style="
                min-height:280px; border-radius:20px;
                border:1px solid #24564a; overflow:hidden;
                padding:28px;
                background:
                    linear-gradient(90deg,
                    rgba(3,12,18,.94),
                    rgba(3,12,18,.30)),
                    url('{background_uri}') center 55% / cover;
                display:flex; flex-direction:column;
                justify-content:center;">
                <div style="font-size:2.4rem;font-weight:850;">
                    Drive Your Dreams<br>
                    <span style="color:#20e6a0;">with Data.</span>
                </div>
                <p style="max-width:560px;color:#e0ecee;">
                    AI-powered used-car price estimation,
                    comparison, and model insights.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hero-card">
                <div class="brand-title">
                    Drive Your Dreams<br>
                    <span class="green-text">with Data.</span>
                </div>
                <p class="subtext">
                    AI-powered used-car price estimation,
                    comparison, and model insights.
                </p>
                <p class="small-note">
                    Add green_rolls_royce.jpg beside app.py
                    to use your luxury-car background.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# 5. LANDING PAGE AND ENTER ANIMATION
# --------------------------------------------------

if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown(
        """
        <div class="hero-card">
            <div class="brand-title">
                Used Car<br>
                <span class="green-text">Price Predictor</span>
            </div>
            <p class="subtext">
                Make smarter decisions with machine learning.
            </p>
            <p class="small-note">
                A student ML project powered by your trained model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="road-scene">
            <div class="flames"></div>
            <div class="road"></div>
            <div class="road-label">CARPREDICTOR • START YOUR ENGINE</div>
            <div class="moving-car">🏎️</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("ENTER THE EXPERIENCE  →", type="primary", use_container_width=True):
        st.session_state.entered = True
        st.session_state.show_drive_animation = True
        st.rerun()

    st.caption(
        "Tip: Add bmw_car.png to your project folder to use your own "
        "transparent car image in the driving animation."
    )
    st.stop()


# Show the transition animation after Enter is clicked.
if st.session_state.pop("show_drive_animation", False):
    bmw_uri = image_as_data_uri(BMW_PATH)

    if bmw_uri:
        car_html = f'<img src="{bmw_uri}" alt="BMW car">'
    else:
        car_html = "🏎️"

    st.markdown(
        f"""
        <div class="road-scene">
            <div class="flames"></div>
            <div class="road"></div>
            <div class="road-label">WELCOME TO CARPREDICTOR</div>
            <div class="moving-car">{car_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🟢 CarPredictor")
    st.caption("AI • AUTOMOTIVE • ANALYTICS")
    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Predict Price",
            "Compare Cars",
            "ML Insights",
            "Prediction History",
        ],
    )
    st.divider()
    st.caption("Built with Python + Streamlit")
    st.caption("Model: existing trained pipeline")


# --------------------------------------------------
# 7. HOME DASHBOARD
# --------------------------------------------------

if page == "Home":
    show_hero()

    st.subheader("Your AI car toolkit")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric("Input features", len(FEATURES))
    c3.metric(
        "Brands in dataset",
        f"{df['brand'].nunique():,}",
    )
    c4.metric(
        "Average listed price",
        f"₹ {df['selling_price'].median():,.0f}",
        help="Median price in the dataset, not a model score.",
    )

    st.markdown("### Explore the dashboard")

    a, b = st.columns(2)
    with a:
        st.markdown("#### 🚘 Predict a price")
        st.write("Enter a car's details and get an estimated selling price.")
        if st.button("Open Price Predictor", use_container_width=True):
            st.session_state.next_page = "Predict Price"
            st.rerun()

        st.markdown("#### 📊 Model insights")
        st.write("Explore dataset distributions and model feature importance.")
        if st.button("Open ML Insights", use_container_width=True):
            st.session_state.next_page = "ML Insights"
            st.rerun()

    with b:
        st.markdown("#### ⚖️ Compare cars")
        st.write("Compare the estimated prices of two different cars.")
        if st.button("Compare Two Cars", use_container_width=True):
            st.session_state.next_page = "Compare Cars"
            st.rerun()

        st.markdown("#### 🧾 Prediction history")
        st.write("Review your predictions from this browser session.")
        if st.button("View Prediction History", use_container_width=True):
            st.session_state.next_page = "Prediction History"
            st.rerun()


# Resolve a requested page from a Home dashboard shortcut.
if "next_page" in st.session_state:
    requested_page = st.session_state.pop("next_page")
    page = requested_page


# --------------------------------------------------
# 8. PRICE PREDICTION
# --------------------------------------------------

if page == "Predict Price":
    st.title("🚘 Used Car Price Prediction")
    st.write("Enter the car details to estimate its selling price.")

    with st.form("predict_form"):
        car = make_car_form("predict", "Car Details")
        submitted = st.form_submit_button(
            "Predict Selling Price", type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            price = predict_price(car)
            record_prediction(car, price)
            st.success("Prediction completed!")
            st.metric("Estimated Selling Price", f"₹ {price:,.0f}")
            st.caption(
                "This is a machine-learning estimate, not a guaranteed "
                "market price. Actual prices depend on condition, location, "
                "service history, and other factors."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 9. COMPARE TWO CARS
# --------------------------------------------------

elif page == "Compare Cars":
    st.title("⚖️ Compare Two Cars")
    st.write("Enter each car's specifications and compare model estimates.")

    left, right = st.columns(2)

    with left:
        car_a = make_car_form("car_a", "🚘 Car A")

    with right:
        car_b = make_car_form("car_b", "🚙 Car B")

    if st.button("Compare Cars", type="primary", use_container_width=True):
        try:
            price_a = predict_price(car_a)
            price_b = predict_price(car_b)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Car A estimate", f"₹ {price_a:,.0f}")
            col_b.metric("Car B estimate", f"₹ {price_b:,.0f}")
            col_c.metric("Difference", f"₹ {abs(price_a-price_b):,.0f}")

            comparison = pd.DataFrame({
                "Car": [
                    f"{car_a['brand']} {car_a['model']}",
                    f"{car_b['brand']} {car_b['model']}",
                ],
                "Estimated price (₹)": [price_a, price_b],
                "Age (years)": [
                    car_a["vehicle_age"], car_b["vehicle_age"]
                ],
                "Kilometres driven": [
                    car_a["km_driven"], car_b["km_driven"]
                ],
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            record_prediction(car_a, price_a)
            record_prediction(car_b, price_b)

            if price_a > price_b:
                st.info("Car A has the higher estimated price in this comparison.")
            elif price_b > price_a:
                st.info("Car B has the higher estimated price in this comparison.")
            else:
                st.info("Both cars have the same estimated price.")

        except Exception as error:
            st.error(f"Comparison failed: {error}")


# --------------------------------------------------
# 10. MACHINE LEARNING INSIGHTS
# --------------------------------------------------

elif page == "ML Insights":
    st.title("📊 Machine Learning Insights")
    st.write("Explore your dataset and, when available, model feature importance.")

    prices = pd.to_numeric(df["selling_price"], errors="coerce").dropna()
    ages = pd.to_numeric(df["vehicle_age"], errors="coerce").dropna()
    kilometres = pd.to_numeric(df["km_driven"], errors="coerce").dropna()

    a, b, c = st.columns(3)
    a.metric("Median selling price", f"₹ {prices.median():,.0f}")
    b.metric("Median vehicle age", f"{ages.median():.1f} years")
    c.metric("Median kilometres", f"{kilometres.median():,.0f} km")

    st.markdown("### Selling price distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(prices, bins=35)
    ax.set_xlabel("Selling price (₹)")
    ax.set_ylabel("Number of cars")
    ax.set_title("Distribution of used-car selling prices")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Average price by fuel type")

    fuel_summary = (
        df.assign(
            selling_price=pd.to_numeric(df["selling_price"], errors="coerce")
        )
        .groupby("fuel_type", dropna=False)["selling_price"]
        .median()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    fuel_summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Median selling price (₹)")
    ax.set_xlabel("Fuel type")
    ax.set_title("Median selling price by fuel type")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Model feature importance")

    try:
        fitted_model = model

        # Supports the Pipeline structure used in typical sklearn workflows.
        if hasattr(fitted_model, "named_steps"):
            final_estimator = list(fitted_model.named_steps.values())[-1]
            preprocessor = fitted_model.named_steps.get("preprocessor")
        else:
            final_estimator = fitted_model
            preprocessor = None

        if not hasattr(final_estimator, "feature_importances_"):
            st.info(
                "Feature importance is not exposed by the estimator saved "
                "in this model file. Predictions will still work."
            )
        elif preprocessor is None:
            st.info(
                "This model does not expose a fitted preprocessor. "
                "Showing the estimator's available feature importances."
            )
            importances = final_estimator.feature_importances_
            feature_names = [f"Feature {i+1}" for i in range(len(importances))]
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=False).head(15)

            st.bar_chart(importance_df.set_index("Feature"))
        else:
            feature_names = preprocessor.get_feature_names_out()
            importances = final_estimator.feature_importances_

            if len(feature_names) != len(importances):
                st.info(
                    "Feature names and importance values have different lengths; "
                    "skipping this chart to avoid a misleading display."
                )
            else:
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False).head(15)

                importance_df["Feature"] = (
                    importance_df["Feature"]
                    .str.replace("num__", "", regex=False)
                    .str.replace("cat__", "", regex=False)
                    .str.replace("remainder__", "", regex=False)
                )

                st.bar_chart(importance_df.set_index("Feature"))
                st.caption(
                    "Feature importance describes the trained model's relative "
                    "use of encoded features. It does not prove causation."
                )

    except Exception as error:
        st.info(f"Could not display feature importance: {error}")

    st.markdown("### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 11. PREDICTION HISTORY
# --------------------------------------------------

elif page == "Prediction History":
    st.title("🧾 Prediction History")
    st.write("Predictions are kept for the current Streamlit session.")

    history = st.session_state.get("history", [])

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History as CSV",
            data=csv_data,
            file_name="used_car_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "No predictions recorded yet. Open Predict Price or Compare Cars "
            "to generate your first prediction."
        )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "CarPredictor • Educational machine-learning project • "
    "Estimates are not guaranteed resale prices."
)
