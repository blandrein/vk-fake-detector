import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("LASTdataset.csv")

print("Dataset shape:", df.shape)


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
# TARGET
# =========================

y = df["target"]

X_profile = df[PROFILE_FEATURES]

X_behavior = df[BEHAVIOR_FEATURES]


# =========================
# TRAIN / TEST SPLIT
# =========================

(
    X_prof_train,
    X_prof_test,

    X_beh_train,
    X_beh_test,

    y_train,
    y_test

) = train_test_split(

    X_profile,
    X_behavior,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

print("Train size:", len(y_train))
print("Test size:", len(y_test))


# =========================
# PROFILE MODEL
# =========================

profile_model = CatBoostClassifier(

    iterations=300,

    depth=4,

    learning_rate=0.03,

    l2_leaf_reg=8,

    loss_function="Logloss",

    eval_metric="AUC",

    verbose=100
)

print("\nTraining PROFILE model...")

profile_model.fit(

    X_prof_train,
    y_train
)


# =========================
# BEHAVIOR MODEL
# =========================

behavior_model = CatBoostClassifier(

    iterations=700,

    depth=6,

    learning_rate=0.03,

    l2_leaf_reg=5,

    loss_function="Logloss",

    eval_metric="AUC",

    verbose=100
)

print("\nTraining BEHAVIOR model...")

behavior_model.fit(

    X_beh_train,
    y_train
)


# =========================
# BASE PREDICTIONS
# =========================

prof_train_pred = profile_model.predict_proba(
    X_prof_train
)[:, 1]

prof_test_pred = profile_model.predict_proba(
    X_prof_test
)[:, 1]

beh_train_pred = behavior_model.predict_proba(
    X_beh_train
)[:, 1]

beh_test_pred = behavior_model.predict_proba(
    X_beh_test
)[:, 1]


# =========================
# META FEATURES
# =========================

X_meta_train = np.column_stack([

    prof_train_pred,

    beh_train_pred,

    np.abs(
        prof_train_pred - beh_train_pred
    ),

    prof_train_pred * beh_train_pred
])

X_meta_test = np.column_stack([

    prof_test_pred,

    beh_test_pred,

    np.abs(
        prof_test_pred - beh_test_pred
    ),

    prof_test_pred * beh_test_pred
])


# =========================
# META MODEL
# =========================

meta_model = CatBoostClassifier(

    iterations=300,

    depth=3,

    learning_rate=0.03,

    l2_leaf_reg=10,

    loss_function="Logloss",

    eval_metric="AUC",

    verbose=100
)

print("\nTraining META model...")

meta_model.fit(

    X_meta_train,
    y_train
)


# =========================
# EVALUATION
# =========================

profile_pred = profile_model.predict_proba(
    X_prof_test
)[:, 1]

behavior_pred = behavior_model.predict_proba(
    X_beh_test
)[:, 1]

final_pred = meta_model.predict_proba(
    X_meta_test
)[:, 1]


profile_auc = roc_auc_score(
    y_test,
    profile_pred
)

behavior_auc = roc_auc_score(
    y_test,
    behavior_pred
)

final_auc = roc_auc_score(
    y_test,
    final_pred
)


print("\n==============================")

print(
    "PROFILE AUC :",
    round(profile_auc, 4)
)

print(
    "BEHAVIOR AUC:",
    round(behavior_auc, 4)
)

print(
    "FINAL AUC   :",
    round(final_auc, 4)
)

print("==============================")


# =========================
# FEATURE IMPORTANCE
# =========================

print("\n=== PROFILE FEATURES ===")

profile_importance = pd.DataFrame({

    "feature": PROFILE_FEATURES,

    "importance":
        profile_model.get_feature_importance()

}).sort_values(

    "importance",
    ascending=False
)

print(profile_importance)


print("\n=== BEHAVIOR FEATURES ===")

behavior_importance = pd.DataFrame({

    "feature": BEHAVIOR_FEATURES,

    "importance":
        behavior_model.get_feature_importance()

}).sort_values(

    "importance",
    ascending=False
)

print(behavior_importance)


# =========================
# SAVE MODELS
# =========================

joblib.dump(
    profile_model,
    "LASTprofile_model.pkl"
)

joblib.dump(
    behavior_model,
    "LASTbehavior_model.pkl"
)

joblib.dump(
    meta_model,
    "LASTmeta_model.pkl"
)

print("\nModels saved successfully!")