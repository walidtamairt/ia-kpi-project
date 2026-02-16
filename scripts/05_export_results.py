"""
Export des résultats — CSV, JSON tickets critiques, KPI summary
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

os.makedirs("data/processed", exist_ok=True)

df      = pd.read_csv("data/processed/tickets_processed.csv")
le_cat  = joblib.load("models/label_encoder_category.pkl")
le_risk = joblib.load("models/label_encoder_risk.pkl")

FEATURES = [
    "priorite_num", "nb_relances", "heure_creation", "jour_semaine",
    "est_weekend", "est_heure_creuse", "systeme_critique",
    "mois", "trimestre", "risk_score", "satisfaction",
]
X        = df[FEATURES]
X_risk   = df[FEATURES + ["categorie_label"]]

clf_cat  = joblib.load("models/classifier_category.pkl")
clf_risk = joblib.load("models/classifier_risk.pkl")
reg      = joblib.load("models/regressor_delay.pkl")

df["categorie_predite"]   = le_cat.inverse_transform(clf_cat.predict(X))
df["risk_level_predit"]   = le_risk.inverse_transform(clf_risk.predict(X_risk))
df["delai_predit_heures"] = np.expm1(reg.predict(X)).round(1)

# ── Export 1 : CSV complet ────────────────────────────────────────
cols = [
    "ticket_id", "titre", "categorie", "categorie_predite",
    "priorite", "departement", "systeme", "agent_assigné",
    "risk_score", "risk_level", "risk_level_predit",
    "resolution_heures", "delai_predit_heures",
    "nb_relances", "statut", "satisfaction", "created_at",
]
df[cols].to_csv("data/processed/tickets_with_predictions.csv", index=False)
print(f"✅ CSV → data/processed/tickets_with_predictions.csv")

# ── Export 2 : JSON tickets critiques ─────────────────────────────
crit = (df[df["risk_level_predit"] == "Critique"]
        [["ticket_id","titre","categorie_predite","priorite",
          "departement","systeme","risk_score",
          "delai_predit_heures","agent_assigné","created_at"]]
        .sort_values("risk_score", ascending=False))

with open("data/processed/tickets_critiques.json", "w", encoding="utf-8") as f:
    json.dump({"generated_at": datetime.now().isoformat(),
               "total": len(crit),
               "tickets": crit.to_dict(orient="records")},
              f, ensure_ascii=False, indent=2)
print(f"✅ JSON → data/processed/tickets_critiques.json  ({len(crit)} critiques)")

# ── Export 3 : KPI summary ────────────────────────────────────────
kpis = {
    "generated_at":          datetime.now().isoformat(),
    "total_tickets":         len(df),
    "tickets_critiques":     int((df["risk_level_predit"] == "Critique").sum()),
    "tickets_eleves":        int((df["risk_level_predit"] == "Élevé").sum()),
    "delai_moyen_predit_h":  round(float(df["delai_predit_heures"].mean()), 1),
    "satisfaction_moyenne":  round(float(df["satisfaction"].mean()), 2),
    "taux_critique_pct":     round(float((df["risk_level_predit"] == "Critique").mean() * 100), 1),
    "repartition_categories": df["categorie_predite"].value_counts().to_dict(),
    "repartition_risques":    df["risk_level_predit"].value_counts().to_dict(),
    "top5_systemes_risque":   (df.groupby("systeme")["risk_score"]
                                 .mean().sort_values(ascending=False)
                                 .head(5).round(1).to_dict()),
}
with open("data/processed/kpi_summary.json", "w", encoding="utf-8") as f:
    json.dump(kpis, f, ensure_ascii=False, indent=2)
print(f"✅ KPI → data/processed/kpi_summary.json")

print(f"\n{'='*50}")
print("📊 RÉSUMÉ EXÉCUTIF")
print(f"{'='*50}")
print(f"Total analysés  : {kpis['total_tickets']}")
print(f"Critiques       : {kpis['tickets_critiques']} ({kpis['taux_critique_pct']}%)")
print(f"Délai moyen     : {kpis['delai_moyen_predit_h']} h")
print(f"Satisfaction    : {kpis['satisfaction_moyenne']}/5")