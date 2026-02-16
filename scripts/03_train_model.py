"""
Entraînement des 3 modèles ML
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               RandomForestRegressor)
import joblib
import json
import os

print("🚀 Démarrage de l'entraînement...\n")

df      = pd.read_csv("data/processed/tickets_processed.csv")
results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

FEATURES = [
    "priorite_num", "nb_relances", "heure_creation", "jour_semaine",
    "est_weekend", "est_heure_creuse", "systeme_critique",
    "mois", "trimestre", "risk_score", "satisfaction",
]
X = df[FEATURES]

# ═══════════════════════════════════════════
# MODÈLE 1 — Classification Catégorie
# ═══════════════════════════════════════════
print("─" * 50)
print("📌 MODÈLE 1 : Classification Catégorie")
print("─" * 50)

y_cat = df["categorie_label"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

clf_cat = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    min_samples_split=5, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
cv_f1 = cross_val_score(clf_cat, X_tr, y_tr, cv=cv, scoring="f1_macro")
print(f"   CV F1-macro : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
clf_cat.fit(X_tr, y_tr)

feat_df = (pd.DataFrame({"feature": FEATURES, "importance": clf_cat.feature_importances_})
             .sort_values("importance", ascending=False))
print(f"\n   Top 5 features :\n{feat_df.head(5).to_string(index=False)}")

joblib.dump(clf_cat, "models/classifier_category.pkl")
print("   ✅ Sauvegardé → models/classifier_category.pkl")
results["categorie"] = {"cv_f1_macro": round(float(cv_f1.mean()), 4),
                         "cv_std":      round(float(cv_f1.std()),  4)}

# ═══════════════════════════════════════════
# MODÈLE 2 — Scoring de Risque
# ═══════════════════════════════════════════
print("\n" + "─" * 50)
print("📌 MODÈLE 2 : Scoring de Risque")
print("─" * 50)

FEATURES_R = FEATURES + ["categorie_label"]
X_r  = df[FEATURES_R]
y_r  = df["risk_label"]
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42, stratify=y_r
)

clf_risk = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1,
    max_depth=5, subsample=0.8, random_state=42,
)
cv_r = cross_val_score(clf_risk, X_tr_r, y_tr_r, cv=cv, scoring="f1_weighted")
print(f"   CV F1-weighted : {cv_r.mean():.4f} ± {cv_r.std():.4f}")
clf_risk.fit(X_tr_r, y_tr_r)
joblib.dump(clf_risk, "models/classifier_risk.pkl")
print("   ✅ Sauvegardé → models/classifier_risk.pkl")
results["risque"] = {"cv_f1_weighted": round(float(cv_r.mean()), 4),
                      "cv_std":         round(float(cv_r.std()),  4)}

# ═══════════════════════════════════════════
# MODÈLE 3 — Prédiction Délai
# ═══════════════════════════════════════════
print("\n" + "─" * 50)
print("📌 MODÈLE 3 : Prédiction Délai (heures)")
print("─" * 50)

y_d = df["log_resolution_heures"]
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X, y_d, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(
    n_estimators=200, max_depth=10,
    min_samples_split=5, random_state=42, n_jobs=-1,
)
cv_r2 = cross_val_score(reg, X_tr_d, y_tr_d, cv=5, scoring="r2")
print(f"   CV R² : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
reg.fit(X_tr_d, y_tr_d)
joblib.dump(reg, "models/regressor_delay.pkl")
print("   ✅ Sauvegardé → models/regressor_delay.pkl")
results["delai"] = {"cv_r2": round(float(cv_r2.mean()), 4),
                     "cv_std": round(float(cv_r2.std()), 4)}

with open("models/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "═" * 50)
print("✅ Entraînement terminé — tous les modèles sauvegardés")
print("═" * 50)