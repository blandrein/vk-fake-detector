import streamlit as st
import numpy as np
import joblib

from LASTExtractor import VKExtractor


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="VK Fake Detector",
    page_icon="🛡",
    layout="centered"
)


# =========================
# LOAD MODELS
# =========================

profile_model = joblib.load(
    "LASTprofile_model.pkl"
)

behavior_model = joblib.load(
    "LASTbehavior_model.pkl"
)

meta_model = joblib.load(
    "LASTmeta_model.pkl"
)


# =========================
# INIT EXTRACTOR
# =========================

VK_TOKEN = "vk1.a.Za-RRBVEk6DPMR5S_epMNERi70IPdc8-wmyKlrCTB0k6vX4J0O2mSaTYw-jn4Gbilb_g0wNDADx4NDV5YCii0mQWkO9mtbTVXjuNqFIValPxOdZfBFSPqFJbcKRNWZ7ffd4HAf0DpoHFXV7qZCrOa8d6uZ7YAA_pI2a0v8vPgAuucsfaT2ynbuvVE3MgD8vwNAjg3Cs4-JsnVrg9jtC_sA"

extractor = VKExtractor(VK_TOKEN)


# =========================
# FEATURES
# =========================

PROFILE_FEATURES = [

    "has_website",
    "gender",

    "has_nickname",
    "has_maiden_name",

    "has_interests",
    "has_books",
    "has_tv",
    "has_quotes",
    "has_about",
    "has_games",
    "has_movies",
    "has_activities",
    "has_music",

    "has_career",
    "has_military_service",
    "has_hometown",

    "has_universities",
    "has_schools",
    "has_relatives"
]

BEHAVIOR_FEATURES = [

    "posts_count",

    "avg_likes",
    "avg_comments",
    "avg_views",

    "avg_text_length",

    "links_ratio",
    "hashtags_ratio",

    "attachments_ratio",

    "reposts_ratio",

    "ads_ratio",

    "posting_frequency_days",

    "phone_numbers_ratio",

    "avg_text_uniqueness"
]


# =========================
# TITLE
# =========================

st.title("🛡 VK Fake Account Detector")

st.write(
    "Введите VK username или ID пользователя"
)


# =========================
# INPUT
# =========================

user_id = st.text_input(
    "VK username / ID"
)


# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze"):

    if not user_id:

        st.warning(
            "Введите VK ID"
        )

        st.stop()

    try:

        with st.spinner("Analyzing account..."):

            # =========================
            # EXTRACT FEATURES
            # =========================

            features = extractor.extract(
                user_id
            )

            # =========================
            # BUILD INPUTS
            # =========================

            X_profile = np.array([[

                features.get(f, 0)

                for f in PROFILE_FEATURES

            ]])

            X_behavior = np.array([[

                features.get(f, 0)

                for f in BEHAVIOR_FEATURES

            ]])

            # =========================
            # PROFILE MODEL
            # =========================

            prof_prob = profile_model.predict_proba(
                X_profile
            )[0][1]

            # =========================
            # BEHAVIOR MODEL
            # =========================

            beh_prob = behavior_model.predict_proba(
                X_behavior
            )[0][1]

            # =========================
            # META FEATURES
            # =========================

            X_meta = np.array([[

                prof_prob,

                beh_prob,

                abs(prof_prob - beh_prob),

                prof_prob * beh_prob

            ]])

            # =========================
            # META MODEL
            # =========================

            final_prob = meta_model.predict_proba(
                X_meta
            )[0][1]

        # =========================
        # RESULT
        # =========================

        st.divider()

        if final_prob >= 0.7:

            st.error(
                f"❌ FAKE ACCOUNT ({final_prob:.3f})"
            )

        elif final_prob >= 0.4:

            st.warning(
                f"⚠️ SUSPICIOUS ACCOUNT ({final_prob:.3f})"
            )

        else:

            st.success(
                f"✅ REAL ACCOUNT ({final_prob:.3f})"
            )

        # =========================
        # MODEL SCORES
        # =========================

        st.subheader("Model Scores")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Profile",
                f"{prof_prob:.3f}"
            )

        with col2:

            st.metric(
                "Behavior",
                f"{beh_prob:.3f}"
            )

        with col3:

            st.metric(
                "Final",
                f"{final_prob:.3f}"
            )

        # =========================
        # FEATURES
        # =========================

        with st.expander("Extracted features"):

            st.json(features)

    except Exception as e:

        st.error(
            f"Error: {e}"
        )