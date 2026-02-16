"""
Évaluation complète — métriques, matrices de confusion, rapport visuel
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score,
                              mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os

os.makedirs("docs", exist_ok=True)
print("🔍 Évaluation des modèles...\n")

# ── Chargement ────────────────────────────────────────────────────
df      = pd.read_csv("data/processed/tickets_processed.csv")
le_cat  = joblib.load("models/label_encoder_category.pkl")
le_risk = joblib.load("models/label_encoder_risk.pkl")

FEATURES = [
    "priorite_num", "nb_relances", "heure_creation", "jour_semaine",
    "est_weekend", "est_heure_creuse", "systeme_critique",
    "mois", "trimestre", "risk_score", "satisfaction",
]
X = df[FEATURES]

# ── Modèle 1 ──────────────────────────────────────────────────────
y_cat = df["categorie_label"]
_, X_te, _, y_te = train_test_split(X, y_cat, test_size=0.2,
                                    random_state=42, stratify=y_cat)
clf_cat   = joblib.load("models/classifier_category.pkl")
y_pred_c  = clf_cat.predict(X_te)

print("=" * 55)
print("MODÈLE 1 — CLASSIFICATION CATÉGORIE")
print("=" * 55)
print(f"Accuracy : {accuracy_score(y_te, y_pred_c):.4f}")
print(f"F1-macro : {f1_score(y_te, y_pred_c, average='macro'):.4f}")
print(classification_report(y_te, y_pred_c,
                             target_names=le_cat.classes_, digits=3))

# ── Modèle 2 ──────────────────────────────────────────────────────
FEATURES_R = FEATURES + ["categorie_label"]
X_r = df[FEATURES_R];  y_r = df["risk_label"]
_, X_te_r, _, y_te_r = train_test_split(X_r, y_r, test_size=0.2,
                                         random_state=42, stratify=y_r)
clf_risk  = joblib.load("models/classifier_risk.pkl")
y_pred_r  = clf_risk.predict(X_te_r)

print("=" * 55)
print("MODÈLE 2 — SCORING RISQUE")
print("=" * 55)
print(f"Accuracy   : {accuracy_score(y_te_r, y_pred_r):.4f}")
print(f"F1-weighted: {f1_score(y_te_r, y_pred_r, average='weighted'):.4f}")
print(classification_report(y_te_r, y_pred_r,
                             target_names=le_risk.classes_, digits=3))

# ── Modèle 3 ──────────────────────────────────────────────────────
y_d = df["log_resolution_heures"]
_, X_te_d, _, y_te_d = train_test_split(X, y_d, test_size=0.2, random_state=42)
reg        = joblib.load("models/regressor_delay.pkl")
y_pred_d   = reg.predict(X_te_d)

r2    = r2_score(y_te_d, y_pred_d)
rmse  = np.sqrt(mean_squared_error(y_te_d, y_pred_d))
y_h   = np.expm1(y_te_d);  yp_h  = np.expm1(y_pred_d)
rmse_h = np.sqrt(mean_squared_error(y_h, yp_h))
mae_h  = np.mean(np.abs(y_h - yp_h))

print("=" * 55)
print("MODÈLE 3 — PRÉDICTION DÉLAI")
print("=" * 55)
print(f"R²       : {r2:.4f}")
print(f"RMSE (h) : {rmse_h:.2f} heures")
print(f"MAE  (h) : {mae_h:.2f} heures")

# ── Rapport visuel ────────────────────────────────────────────────
print("\n📊 Génération du rapport visuel...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Rapport d'Évaluation — IA Ticket Intelligence",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Confusion matrix — catégorie
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(confusion_matrix(y_te, y_pred_c), annot=True, fmt="d",
            cmap="Blues", xticklabels=le_cat.classes_,
            yticklabels=le_cat.classes_, ax=ax1, cbar=False)
ax1.set_title("Confusion Matrix\nCatégorie", fontweight="bold")
ax1.set_xlabel("Prédit"); ax1.set_ylabel("Réel")
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)

# Confusion matrix — risque
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(confusion_matrix(y_te_r, y_pred_r), annot=True, fmt="d",
            cmap="Oranges", xticklabels=le_risk.classes_,
            yticklabels=le_risk.classes_, ax=ax2, cbar=False)
ax2.set_title("Confusion Matrix\nRisque", fontweight="bold")
ax2.set_xlabel("Prédit"); ax2.set_ylabel("Réel")
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)

# Feature importance
ax3 = fig.add_subplot(gs[0, 2])
fi = (pd.DataFrame({"feature": FEATURES,
                    "importance": clf_cat.feature_importances_})
       .sort_values("importance", ascending=True).tail(8))
ax3.barh(fi["feature"], fi["importance"], color="#2196F3")
ax3.set_title("Feature Importance\n(Classif. Catégorie)", fontweight="bold")

# Prédit vs réel
ax4 = fig.add_subplot(gs[1, 0])
idx = np.random.choice(len(y_h), min(300, len(y_h)), replace=False)
ax4.scatter(np.array(y_h)[idx], yp_h[idx], alpha=0.4, s=15, color="#4CAF50")
mv = max(np.array(y_h).max(), yp_h.max())
ax4.plot([0, mv], [0, mv], "r--", lw=1.5)
ax4.set_xlabel("Réel (h)"); ax4.set_ylabel("Prédit (h)")
ax4.set_title(f"Prédiction Délai\nR²={r2:.3f} RMSE={rmse_h:.1f}h", fontweight="bold")

# Distribution erreurs
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(yp_h - np.array(y_h), bins=40, color="#9C27B0", alpha=0.7, edgecolor="white")
ax5.axvline(0, color="red", ls="--", lw=1.5)
ax5.set_xlabel("Erreur (h)"); ax5.set_ylabel("Fréquence")
ax5.set_title("Distribution des Erreurs\n(Régression)", fontweight="bold")

# Tableau métriques
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
rows = [
    ("MODÈLE",    "MÉTRIQUE",       "SCORE"),
    ("─"*10,      "─"*14,           "─"*8),
    ("Catégorie", "Accuracy",       f"{accuracy_score(y_te, y_pred_c):.3f}"),
    ("Catégorie", "F1-macro",       f"{f1_score(y_te, y_pred_c, average='macro'):.3f}"),
    ("Risque",    "Accuracy",       f"{accuracy_score(y_te_r, y_pred_r):.3f}"),
    ("Risque",    "F1-weighted",    f"{f1_score(y_te_r, y_pred_r, average='weighted'):.3f}"),
    ("Délai",     "R²",             f"{r2:.3f}"),
    ("Délai",     "RMSE (h)",       f"{rmse_h:.1f}h"),
    ("Délai",     "MAE  (h)",       f"{mae_h:.1f}h"),
]
yp = 0.95
for r in rows:
    bold = "bold" if r[0] in ("MODÈLE", "─"*10) else "normal"
    for xi, txt in zip([0.0, 0.38, 0.76], r):
        ax6.text(xi, yp, txt, transform=ax6.transAxes,
                 fontsize=9, fontweight=bold)
    yp -= 0.09
ax6.set_title("Métriques Résumées", fontweight="bold")

plt.savefig("docs/evaluation_report.png", dpi=150, bbox_inches="tight")
print("   ✅ docs/evaluation_report.png sauvegardé")
plt.close()
print("\n✅ Évaluation terminée !")