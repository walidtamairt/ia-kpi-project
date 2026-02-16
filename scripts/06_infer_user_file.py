import pandas as pd
import numpy as np
import joblib
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

REQUIRED_FEATURES = [
    "priorite_num", "nb_relances", "heure_creation", "jour_semaine",
    "est_weekend", "est_heure_creuse", "systeme_critique",
    "mois", "trimestre", "risk_score", "satisfaction",
]

PRIORITY_MAP     = {"Basse": 1, "Normale": 2, "Haute": 3, "Critique": 4}
CRITICAL_SYSTEMS = ["ERP SAP", "Serveur Web", "Base de données", "Active Directory"]


def load_file(path):
    ext = Path(path).suffix.lower()
    if ext == ".csv":   return pd.read_csv(path)
    if ext in (".xlsx", ".xls"): return pd.read_excel(path)
    if ext == ".json":  return pd.read_json(path)
    raise ValueError(f"Format non supporté : {ext}")


def engineer_features(df):
    df = df.copy()
    df["priorite_num"]     = df.get("priorite", pd.Series(["Normale"]*len(df))).map(PRIORITY_MAP).fillna(2)
    df["systeme_critique"] = df.get("systeme",  pd.Series(["Autre"]*len(df))).isin(CRITICAL_SYSTEMS).astype(int)
    df["est_weekend"]      = (df.get("jour_semaine",  pd.Series([1]*len(df))) >= 5).astype(int)
    df["est_heure_creuse"] = ((df.get("heure_creation", pd.Series([9]*len(df))) < 8) |
                              (df.get("heure_creation", pd.Series([9]*len(df))) > 18)).astype(int)
    defaults = {"nb_relances": 0, "mois": 6, "trimestre": 2,
                "risk_score": 40, "satisfaction": 3,
                "heure_creation": 9, "jour_semaine": 1}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            print(f"   ⚠️  '{col}' absent → valeur par défaut : {val}")
    return df


def run_inference(df):
    le_cat  = joblib.load("models/label_encoder_category.pkl")
    le_risk = joblib.load("models/label_encoder_risk.pkl")
    clf_cat = joblib.load("models/classifier_category.pkl")
    clf_rsk = joblib.load("models/classifier_risk.pkl")
    reg     = joblib.load("models/regressor_delay.pkl")

    X        = df[REQUIRED_FEATURES]
    cat_pred = clf_cat.predict(X)

    df["categorie_predite"]   = le_cat.inverse_transform(cat_pred)
    df["risk_level_predit"]   = le_risk.inverse_transform(
        clf_rsk.predict(np.hstack([X, cat_pred.reshape(-1, 1)]))
    )
    df["delai_predit_heures"] = np.expm1(reg.predict(X)).round(1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Chemin du fichier à analyser")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Analyse de : {Path(args.file).name}")
    print(f"{'='*50}\n")

    df = load_file(args.file)
    print(f"✅ {len(df)} lignes chargées")

    df = engineer_features(df)
    df = run_inference(df)

    os.makedirs("data/user_uploads", exist_ok=True)
    stem    = Path(args.file).stem
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"data/user_uploads/{stem}_predictions_{ts}.csv"

    df.to_csv(out_csv, index=False)
    print(f"\n✅ Résultats exportés → {out_csv}")

    print(f"\n📊 Résumé :")
    print(f"   Critiques : {(df['risk_level_predit']=='Critique').sum()}")
    print(f"   Délai moyen prédit : {df['delai_predit_heures'].mean():.1f} h")
    print(df["risk_level_predit"].value_counts().to_string())


if __name__ == "__main__":
    main()