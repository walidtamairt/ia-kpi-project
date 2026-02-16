"""
Prétraitement — Nettoyage, normalisation, feature engineering, stockage SQL
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sqlite3
import joblib
import os

print("📂 Chargement des données brutes...")
df = pd.read_csv("data/raw/tickets_raw.csv")
print(f"   {len(df)} tickets — {df.shape[1]} colonnes")

# ── 1. Nettoyage ──────────────────────────────────────────────────
print("\n🧹 Nettoyage...")
missing = df.isnull().sum()
if missing.any():
    df["nb_relances"].fillna(0, inplace=True)
    df["satisfaction"].fillna(df["satisfaction"].median(), inplace=True)
    print(f"   Valeurs manquantes corrigées.")
else:
    print("   Aucune valeur manquante ✓")

dupes = df.duplicated(subset=["ticket_id"]).sum()
df.drop_duplicates(subset=["ticket_id"], inplace=True)
print(f"   Doublons supprimés : {dupes}")

df["created_at"]  = pd.to_datetime(df["created_at"])
df["resolved_at"] = pd.to_datetime(df["resolved_at"])

# ── 2. Feature Engineering ────────────────────────────────────────
print("\n⚙️  Feature engineering...")

df["mois"]             = df["created_at"].dt.month
df["trimestre"]        = df["created_at"].dt.quarter
df["est_weekend"]      = (df["jour_semaine"] >= 5).astype(int)
df["est_heure_creuse"] = ((df["heure_creation"] < 8) | (df["heure_creation"] > 18)).astype(int)

priority_order = {"Basse": 1, "Normale": 2, "Haute": 3, "Critique": 4}
df["priorite_num"] = df["priorite"].map(priority_order)

critical_sys = ["ERP SAP", "Serveur Web", "Base de données", "Active Directory"]
df["systeme_critique"]       = df["systeme"].isin(critical_sys).astype(int)
df["risk_priority_ratio"]    = df["risk_score"] / df["priorite_num"]
df["log_resolution_heures"]  = np.log1p(df["resolution_heures"])

print("   Features créées : mois, trimestre, est_weekend, est_heure_creuse,")
print("   priorite_num, systeme_critique, risk_priority_ratio, log_resolution_heures")

# ── 3. Label encoding des cibles ─────────────────────────────────
le_cat  = LabelEncoder()
le_risk = LabelEncoder()
df["categorie_label"] = le_cat.fit_transform(df["categorie"])
df["risk_label"]      = le_risk.fit_transform(df["risk_level"])

os.makedirs("models", exist_ok=True)
joblib.dump(le_cat,  "models/label_encoder_category.pkl")
joblib.dump(le_risk, "models/label_encoder_risk.pkl")
print("\n   Encodeurs sauvegardés dans models/")

# ── 4. Sauvegarde CSV ─────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/tickets_processed.csv", index=False)
print(f"\n✅ CSV sauvegardé → data/processed/tickets_processed.csv  {df.shape}")

# ── 5. Stockage SQLite ────────────────────────────────────────────
print("\n🗄️  Stockage SQLite...")
conn = sqlite3.connect("data/tickets.db")
df.to_sql("tickets", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_categorie  ON tickets(categorie)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_level ON tickets(risk_level)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tickets(created_at)")
conn.commit()

print("\n📊 Exemple requête SQL — risque moyen par département :")
query = """
    SELECT departement,
           COUNT(*)                      AS nb_tickets,
           ROUND(AVG(resolution_heures), 1) AS delai_moyen_h,
           ROUND(AVG(risk_score), 1)        AS risk_moyen
    FROM   tickets
    WHERE  risk_level = 'Critique'
    GROUP  BY departement
    ORDER  BY nb_tickets DESC
"""
print(pd.read_sql_query(query, conn).to_string(index=False))
conn.close()

print("\n✅ Prétraitement terminé !")