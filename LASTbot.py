import asyncio
import numpy as np
import joblib

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from LASTExtractor import VKExtractor


# =====================
# CONFIG
# =====================

TOKEN_TG = "8903765309:AAED0PHeCV3dbAXWIF917kzFCEPSjWoEjfc"

VK_TOKEN = "vk1.a.Za-RRBVEk6DPMR5S_epMNERi70IPdc8-wmyKlrCTB0k6vX4J0O2mSaTYw-jn4Gbilb_g0wNDADx4NDV5YCii0mQWkO9mtbTVXjuNqFIValPxOdZfBFSPqFJbcKRNWZ7ffd4HAf0DpoHFXV7qZCrOa8d6uZ7YAA_pI2a0v8vPgAuucsfaT2ynbuvVE3MgD8vwNAjg3Cs4-JsnVrg9jtC_sA"


# =====================
# INIT
# =====================

bot = Bot(token=TOKEN_TG)

dp = Dispatcher()

extractor = VKExtractor(VK_TOKEN)


# =====================
# LOAD MODELS
# =====================

profile_model = joblib.load(
    "LASTprofile_model.pkl"
)

behavior_model = joblib.load(
    "LASTbehavior_model.pkl"
)

meta_model = joblib.load(
    "LASTmeta_model.pkl"
)


# =====================
# FEATURES
# =====================

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


# =====================
# START COMMAND
# =====================

@dp.message(Command("start"))
async def start(message: types.Message):

    text = (

        "🛡 VK Fake Detector\n\n"

        "Отправь:\n"
        "- VK username\n"
        "- VK ID\n"
        "- короткую ссылку\n\n"

        "Пример:\n"
        "`durov`"
    )

    await message.answer(
        text,
        parse_mode="Markdown"
    )


# =====================
# MAIN ANALYSIS
# =====================

@dp.message()
async def analyze(message: types.Message):

    user_id = message.text.strip()

    wait_msg = await message.answer(
        "⏳ Анализирую аккаунт..."
    )

    try:

        # =====================
        # EXTRACT FEATURES
        # =====================

        features = extractor.extract(
            user_id
        )

        # =====================
        # BUILD MATRICES
        # =====================

        X_profile = np.array([[

            features.get(f, 0)

            for f in PROFILE_FEATURES

        ]])

        X_behavior = np.array([[

            features.get(f, 0)

            for f in BEHAVIOR_FEATURES

        ]])

        # =====================
        # PROFILE MODEL
        # =====================

        prof_prob = profile_model.predict_proba(
            X_profile
        )[0][1]

        # =====================
        # BEHAVIOR MODEL
        # =====================

        beh_prob = behavior_model.predict_proba(
            X_behavior
        )[0][1]

        # =====================
        # META FEATURES
        # =====================

        X_meta = np.array([[

            prof_prob,

            beh_prob,

            abs(prof_prob - beh_prob),

            prof_prob * beh_prob

        ]])

        # =====================
        # META MODEL
        # =====================

        final_prob = meta_model.predict_proba(
            X_meta
        )[0][1]

        # =====================
        # VERDICT
        # =====================

        if final_prob >= 0.7:

            verdict = "❌ FAKE ACCOUNT"

        elif final_prob >= 0.4:

            verdict = "⚠️ SUSPICIOUS ACCOUNT"

        else:

            verdict = "✅ REAL ACCOUNT"

        # =====================
        # RESPONSE
        # =====================

        result_text = (

            f"🧠 Результат анализа\n\n"

            f"📊 Profile score:\n"
            f"`{prof_prob:.3f}`\n\n"

            f"📈 Behavior score:\n"
            f"`{beh_prob:.3f}`\n\n"

            f"🎯 Final score:\n"
            f"`{final_prob:.3f}`\n\n"

            f"🏁 Verdict:\n"
            f"{verdict}"
        )

        await wait_msg.edit_text(
            result_text,
            parse_mode="Markdown"
        )

    except Exception as e:

        await wait_msg.edit_text(

            f"❌ Ошибка:\n"
            f"`{str(e)}`",

            parse_mode="Markdown"
        )


# =====================
# RUN
# =====================

async def main():

    print("Bot started...")

    await dp.start_polling(bot)


if __name__ == "__main__":

    asyncio.run(main())