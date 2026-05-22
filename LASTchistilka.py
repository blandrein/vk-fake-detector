import pandas as pd

# =========================
# LOAD
# =========================

df = pd.read_csv("dataset.csv")

print("BEFORE:", df.shape)

# =========================
# NORMALIZATION
# =========================

df = df.replace("Unknown", 0)

df = df.fillna(0)

# =========================
# REMOVE LEAKAGE FEATURES
# =========================

leakage_columns = [

    # SOCIAL LEAKAGE
    "subscribers_count",
    "is_verified",
    "is_blacklisted",
    "has_domain",
    "has_birth_date",
    "has_short_name",

    "has_photo",
    "has_status",

    "has_occupation",
    "occupation_type_university",
    "occupation_type_work",

    # SOCIAL STATUS
    "is_profile_closed",
    "access_to_closed_profile",
    "marital_status",

    # IDENTIFICATION
    "city",
    "has_first_name",
    "has_last_name",

    # API / SERVICE FIELDS
    "can_post_on_wall",
    "can_send_message",
    "can_add_as_friend",
    "can_invite_to_group",

    "has_mobile",
    "all_posts_visible",
    "audio_available",
    "is_confirmed",

    # NOISY FEATURES
    "avg_keywords",
    "has_personal_data"
]

# =========================
# DROP COLUMNS
# =========================

existing_cols = [

    c for c in leakage_columns
    if c in df.columns
]

df = df.drop(columns=existing_cols)

# =========================
# NUMERIC SAFETY
# =========================

for col in df.columns:

    if col != "target":

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# =========================
# FINAL NaN CLEAN
# =========================

df = df.fillna(0)

# =========================
# FINAL CHECK
# =========================

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass balance:")
print(df["target"].value_counts())

print("\nRemaining columns:")
print(df.columns.tolist())

# =========================
# SAVE
# =========================

df.to_csv(
    "LASTdataset.csv",
    index=False
)

print("\nAFTER:", df.shape)

print("Saved: LASTdataset.csv")