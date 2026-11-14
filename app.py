# app.py  —  DeliverIQ · Delivery ETA Predictor
# Streamlit shell around one custom component (frontend/) that draws the whole UI.
# Flow: component sends inputs → build_input_df() → predict() → result back to component

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 0.  Page config + chrome
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeliverIQ — Delivery ETA Predictor",
    page_icon="🚚",
    layout="wide",
)

# Hide Streamlit's own header, footer, sidebar and menu so the component is the
# page. Selectors cover both the older class names and the newer data-testids.
st.markdown(
    """
    <style>
    header[data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"], #MainMenu, footer {
        display: none !important;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        background: #0B1220 !important;
    }
    .block-container, [data-testid="stAppViewBlockContainer"],
    [data-testid="stMainBlockContainer"] {
        padding: 0 !important;
        max-width: 100% !important;
    }
    [data-testid="stVerticalBlock"] { gap: 0 !important; }
    .element-container:has(style),
    [data-testid="stElementContainer"]:has(style) { display: none !important; }
    .stApp iframe { display: block; width: 100%; border: 0; }
    /* Keep the UI at full opacity while a prediction reruns the script. */
    .element-container, [data-testid="stElementContainer"] {
        opacity: 1 !important;
        transition: none !important;
    }
    [data-testid="stSpinner"], .stSpinner { padding: 24px; color: #B4C0D6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# 1.  Load artifacts  (cached — runs once at startup)
# ─────────────────────────────────────────────────────────────
MODEL_PATH      = Path("models/best_model.pkl")
METRICS_PATH    = Path("models/metrics.json")
FEAT_IMP_PATH   = Path("models/feature_importance.csv")


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    """Load the fitted sklearn Pipeline from disk."""
    if not path.exists():
        st.error(
            f"❌ `{path}` not found. Run NB-04 to completion first.",
            icon="🚨",
        )
        st.stop()
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    from sklearn.pipeline import Pipeline
    assert isinstance(pipeline, Pipeline), "Loaded object is not an sklearn Pipeline."
    return pipeline


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


model      = load_model(MODEL_PATH)
metrics    = load_metrics(METRICS_PATH)
feat_imp   = load_feature_importance(FEAT_IMP_PATH)

USE_LOG_TRANSFORM: bool = metrics.get("log_transform", True)


# ─────────────────────────────────────────────────────────────
# 2.  Helper — build_input_df  (mirrors src/predict.py)
# ─────────────────────────────────────────────────────────────
# Lookup tables used to derive hidden numeric features from
# the user's categorical selections (per src_module_reference).
# Low traffic means free-flowing roads, so it maps to the highest speed.
_AVG_SPEED = {"low": 60.0, "medium": 35.0, "high": 20.0}
_VISIBILITY = {"clear": 10.0, "rain": 4.0, "fog": 1.5}


def build_input_df(user_inputs: dict) -> pd.DataFrame:
    """
    Converts the Streamlit form dict to a single-row DataFrame
    with column names exactly matching the training feature schema.

    Derived columns:
        avg_speed_kmph  ← traffic_level lookup
        visibility_km   ← weather_type lookup
        is_rush_hour    ← hour in {7, 8, 17, 18}
        is_weekend      ← day_of_week >= 5
    """
    hour        = user_inputs["hour"]
    dow         = user_inputs["day_of_week"]
    traffic_lvl = user_inputs["traffic_level"]
    weather_t   = user_inputs["weather_type"]

    row = {
        # — numeric features —
        "distance_km":    user_inputs["distance_km"],
        "num_lanes":      user_inputs["num_lanes"],
        "num_signals":    user_inputs["num_signals"],
        "avg_speed_kmph": _AVG_SPEED[traffic_lvl],
        "temperature_c":  user_inputs.get("temperature_c", 28.0),
        "visibility_km":  _VISIBILITY[weather_t],
        "hour":           hour,
        "day_of_week":    dow,
        # — boolean features —
        "is_rush_hour":   int(hour in (7, 8, 17, 18)),
        "is_weekend":     int(dow >= 5),
        # — ordinal feature —
        "traffic_level":  traffic_lvl,
        # — categorical features —
        "road_type":      user_inputs["road_type"],
        "weather_type":   weather_t,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────
# 3.  Helper — predict  (mirrors src/predict.py)
# ─────────────────────────────────────────────────────────────
def predict(pipeline, input_df: pd.DataFrame, use_log_transform: bool = False) -> float:
    """
    Run inference.  If use_log_transform=True, the model was trained
    on log1p(travel_time_min); apply np.expm1() to back-transform.
    Clamps result to 0.0 to guard against negative predictions.
    """
    raw = pipeline.predict(input_df)[0]
    result = float(np.expm1(raw)) if use_log_transform else float(raw)
    return max(0.0, result)


# ─────────────────────────────────────────────────────────────
# 4.  Input validation
# ─────────────────────────────────────────────────────────────
# The component only ever sends values inside these bounds; anything else did
# not come from the UI and is dropped before it reaches the model.
_NUMERIC_BOUNDS = {
    "distance_km":   (float, 1.0, 100.0),
    "num_lanes":     (int,   1,   8),
    "num_signals":   (int,   0,   30),
    "hour":          (int,   0,   23),
    "day_of_week":   (int,   0,   6),
    "temperature_c": (float, -5.0, 50.0),
}
_CHOICES = {
    "traffic_level": tuple(_AVG_SPEED),
    "road_type":     ("highway", "urban", "residential"),
    "weather_type":  tuple(_VISIBILITY),
}


def clean_inputs(raw) -> dict | None:
    """Return the nine inputs with native types, or None if any is invalid."""
    if not isinstance(raw, dict):
        return None
    cleaned = {}
    for name, (kind, low, high) in _NUMERIC_BOUNDS.items():
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not np.isfinite(value) or not low <= value <= high:
            return None
        if kind is int and value != int(value):
            return None
        cleaned[name] = kind(value)
    for name, options in _CHOICES.items():
        if raw.get(name) not in options:
            return None
        cleaned[name] = raw[name]
    return cleaned


# ─────────────────────────────────────────────────────────────
# 5.  UI component  (frontend/ — both screens)
# ─────────────────────────────────────────────────────────────
_UI_KEY = "deliveriq_ui"
_RESULT_KEY = "deliveriq_result"      # last prediction, sent back to the component
_HANDLED_KEY = "deliveriq_handled"    # request_id of the last value acted on

deliveriq_ui = components.declare_component(
    _UI_KEY, path=str(Path(__file__).parent / "frontend")
)


def handle_request(value) -> bool:
    """
    Run the model for a {request_id, inputs} value the component sent.
    Returns True when a new prediction was stored in session state.
    Inputs are used for this one call and are not stored or logged.
    """
    if not isinstance(value, dict):
        return False
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or request_id == st.session_state.get(_HANDLED_KEY):
        return False
    st.session_state[_HANDLED_KEY] = request_id

    inputs = clean_inputs(value.get("inputs"))
    if inputs is None:
        return False

    input_df = build_input_df(inputs)
    pred_minutes = predict(model, input_df, use_log_transform=USE_LOG_TRANSFORM)
    st.session_state[_RESULT_KEY] = {
        "request_id":   request_id,
        "pred_minutes": pred_minutes,
        # to_json converts numpy scalars to plain JSON types
        "input_row":    json.loads(input_df.to_json(orient="records"))[0],
    }
    return True


# A value set in the browser is already in session state when the rerun starts,
# so the prediction is ready before the component is rendered in this run.
handle_request(st.session_state.get(_UI_KEY))

ui_value = deliveriq_ui(
    metrics=metrics,
    feature_importance=feat_imp.head(15)[["feature", "importance"]].to_dict("records"),
    result=st.session_state.get(_RESULT_KEY),
    key=_UI_KEY,
    default=None,
)

# Fallback for the case where the value only shows up as the return value.
if handle_request(ui_value):
    st.rerun()
