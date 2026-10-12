import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from security_engine import SecurityEngine


# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="Sentinel | Smart Home Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "security_log.csv"
EVIDENCE_DIR = BASE_DIR / "evidence"
MODEL_FILE = BASE_DIR / "yolo11n.pt"

EVIDENCE_DIR.mkdir(exist_ok=True)


# ==================================================
# DESIGN SYSTEM
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F4F7FC;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,252,0.95);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #172B4D;
        letter-spacing: -0.4px;
    }

    p, label {
        color: #53647C;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E8F2;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(30,55,90,0.03);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B;
    }

    [data-testid="stMetricValue"] {
        color: #172B4D;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
        border: 1px solid #DCE4F0;
    }

    div.stButton > button[kind="primary"] {
        background: #0f0f0f;
        border-color: #2859A6;
        color: white;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-color: #E1E8F2;
        border-radius: 14px;
    }

    .brand {
        font-size: 13px;
        font-weight: 700;
        color: #315FA8;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748B;
        font-size: 15px;
        margin-top: -8px;
    }

    .section-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background: #E5F6EC;
        color: #207347;
    }

    .status-off {
        background: #EDF1F7;
        color: #53647C;
    }

    .small-note {
        font-size: 13px;
        color: #718096;
    }

    hr {
        border-color: #E1E8F2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SHARED SECURITY ENGINE
# ==================================================

@st.cache_resource
def get_engine():
    return SecurityEngine(
        arduino_port="COM12",
        model_path=str(MODEL_FILE),
        confidence_threshold=0.60,
        confirmation_frames=4,
    )


engine = get_engine()


# ==================================================
# DATA HELPERS
# ==================================================

def load_logs():
    columns = [
        "timestamp",
        "event",
        "people_count",
        "confidence",
        "photo",
    ]

    if not LOG_FILE.exists():
        return pd.DataFrame(columns=columns)

    try:
        data = pd.read_csv(LOG_FILE)

        for column in columns:
            if column not in data.columns:
                data[column] = ""

        return data[columns]

    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(columns=columns)


def get_photos():
    return sorted(
        EVIDENCE_DIR.glob("*.jpg"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def get_engine_status():
    if not engine.running:
        return "Stopped"

    if engine.armed:
        return "Armed"

    return "Online"


# ==================================================
# HEADER
# ==================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">SENTINEL / HOME PROTECTION</div>',
        unsafe_allow_html=True,
    )
    st.title("Smart Home Security")
    st.markdown(
        '<div class="subtitle">'
        'A simple overview of your AI-powered home monitoring system.'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.write("")

    if engine.running:
        st.markdown(
            '<span class="status-pill">● ENGINE ONLINE</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-off">● ENGINE STOPPED</span>',
            unsafe_allow_html=True,
        )


st.write("")


# ==================================================
# CONTROLS
# ==================================================

with st.container(border=True):
    control_left, control_middle, control_right = st.columns([1, 1, 2])

    with control_left:
        if st.button(
            "Start monitoring",
            type="primary",
            use_container_width=True,
            disabled=engine.running,
        ):
            with st.spinner("Starting AI, webcam and Arduino..."):
                engine.start()

            st.rerun()

    with control_middle:
        if st.button(
            "Stop engine",
            use_container_width=True,
            disabled=not engine.running,
        ):
            engine.set_armed(False)
            engine.stop()
            st.rerun()

    with control_right:
        if not engine.running:
            st.caption("Start the engine before arming your security system.")
        elif engine.armed:
            st.error("Security is armed. Confirmed person detections can trigger the alarm.")
        else:
            st.success("Security is disarmed. The camera preview remains available.")


# ==================================================
# ARM / DISARM
# ==================================================

arm_left, arm_right, arm_note = st.columns([1, 1, 2])

with arm_left:
    if st.button(
        "🔒 Arm security",
        type="primary",
        use_container_width=True,
        disabled=(not engine.running or engine.armed),
    ):
        engine.set_armed(True)
        st.rerun()

with arm_right:
    if st.button(
        "🔓 Disarm security",
        use_container_width=True,
        disabled=(not engine.running or not engine.armed),
    ):
        engine.set_armed(False)
        st.rerun()

with arm_note:
    if engine.armed:
        st.markdown("**Active protection**")
        st.caption("AI-confirmed person detection can activate the buzzer.")
    else:
        st.markdown("**Standby mode**")
        st.caption("Arm the system before leaving the monitored area.")


st.write("")


# ==================================================
# OVERVIEW CARDS
# ==================================================

logs = load_logs()
photos = get_photos()

with engine.lock:
    current_people = engine.people_count if engine.running else 0
    current_confidence = engine.confidence if engine.running else 0.0
    confirmed = engine.confirmed_person if engine.running else False

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("System status", get_engine_status())

with m2:
    st.metric("People in current frame", current_people)

with m3:
    st.metric("Recorded alerts", len(logs))

with m4:
    st.metric("Evidence photos", len(photos))


st.write("")


# ==================================================
# CAMERA PANEL
# ==================================================

camera_column, info_column = st.columns([1.8, 1])

with camera_column:
    st.subheader("Live monitoring")
    st.caption("Webcam feed with AI detection annotations")

    @st.fragment(run_every="1s")
    def camera_panel():
        with engine.lock:
            frame = (
                engine.latest_frame.copy()
                if engine.latest_frame is not None
                else None
            )

            running = engine.running
            armed = engine.armed
            person_present = engine.person_detected
            confirmed_person = engine.confirmed_person
            count = engine.people_count
            confidence = engine.confidence

        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True,
            )
        else:
            st.info(
                "Your camera preview will appear here after monitoring starts."
            )

        if not running:
            st.caption("Camera is not running.")
        elif armed and confirmed_person:
            st.error("PERSON DETECTED — CHECK THE ALERT")
        elif armed:
            st.warning("ARMED · MONITORING")
        else:
            st.success("DISARMED · PREVIEW ACTIVE")

        st.caption(
            f"Person visible: {'Yes' if person_present else 'No'}"
            f"  ·  Confirmed: {'Yes' if confirmed_person else 'No'}"
            f"  ·  People: {count}"
            f"  ·  Confidence: {confidence:.1%}"
        )

    camera_panel()


with info_column:
    st.subheader("System overview")
    st.caption("Current operating state")

    with st.container(border=True):
        st.markdown("**AI person detection**")

        if engine.running:
            st.success("YOLO model is running")
            st.write(
                f"Current highest confidence: {current_confidence:.1%}"
            )
        else:
            st.info("Waiting for engine startup")

        st.divider()

        st.markdown("**Security mode**")

        if engine.armed:
            st.error("Armed")
            st.write(
                "Confirmed person detections can trigger an AI alarm."
            )
        else:
            st.info("Disarmed")
            st.write(
                "AI detection does not activate the buzzer in this mode."
            )

        st.divider()

        st.markdown("**Hardware and system messages**")

        if engine.last_error:
            st.error(engine.last_error)
        elif engine.running:
            st.caption("No reported engine errors.")
        else:
            st.caption(
                "Start monitoring to check the webcam and Arduino connection."
            )

    with st.container(border=True):
        st.markdown("**How protection works**")
        st.write(
            "1. Start monitoring.\n\n"
            "2. Arm before leaving.\n\n"
            "3. YOLO checks the camera frames.\n\n"
            "4. A confirmed detection can activate the buzzer.\n\n"
            "5. The system saves an evidence photo and a log entry."
        )


st.write("")


# ==================================================
# RECENT SECURITY ACTIVITY
# ==================================================

activity_left, activity_right = st.columns([1.5, 1])

with activity_left:
    st.subheader("Recent security activity")
    st.caption("Locally recorded detection events")

    logs = load_logs()

    if logs.empty:
        st.info(
            "No recorded alerts yet. Events will appear here "
            "after a confirmed detection while armed."
        )
    else:
        recent = logs.iloc[::-1].head(8).copy()

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download event history",
            data=logs.to_csv(index=False).encode("utf-8"),
            file_name="security_log.csv",
            mime="text/csv",
        )


with activity_right:
    st.subheader("Evidence archive")
    st.caption("Most recent saved photos")

    photos = get_photos()

    if not photos:
        st.info(
            "Evidence photos will appear here after an alert is recorded."
        )
    else:
        for photo in photos[:3]:
            with st.container(border=True):
                st.image(
                    str(photo),
                    use_container_width=True,
                )
                st.caption(photo.name)


# ==================================================
# FOOTER
# ==================================================

st.divider()

footer_left, footer_right = st.columns([3, 1])

with footer_left:
    st.caption(
        "SENTINEL · Smart Home Security · "
        "YOLO + OpenCV + Python + Arduino"
    )

with footer_right:
    st.caption("Local monitoring · Local evidence")