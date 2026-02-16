"""
Dashboard Streamlit — IA Ticket Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
import os
from pathlib import Path

st.set_page_config(
    page_title="🎫 IA Ticket Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.risk-critique{color:#f44336;font-weight:bold}
.risk-eleve   {color:#FF9800;font-weight:bold}
.risk-moyen   {color:#FFC107;font-weight:bold}
.risk-faible  {color:#4CAF50;font-weight:bold}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/tickets_with_predictions.csv")
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df

@st.cache_data
def load_kpis():
    with open("data/processed/kpi_summary.json", encoding="utf-8") as f:
        return json.load(f)

try:
    df   = load_data()
    kpis = load_kpis()
except FileNotFoundError:
    st.error("⚠️ Données introuvables. Lancez d'abord les 5 scripts.")
    st.code(
        "python scripts/01_generate_data.py\n"
        "python scripts/02_preprocessing.py\n"
        "python scripts/03_train_model.py\n"
        "python scripts/04_evaluate_model.py\n"
        "python scripts/05_export_results.py"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("🔧 Filtres")

dept_sel = st.sidebar.selectbox(
    "Département", ["Tous"] + sorted(df["departement"].unique().tolist()))
risk_sel = st.sidebar.multiselect(
    "Niveau de risque", ["Critique","Élevé","Moyen","Faible"],
    default=["Critique","Élevé","Moyen","Faible"])
cat_sel = st.sidebar.selectbox(
    "Catégorie IA", ["Tous"] + sorted(df["categorie_predite"].unique().tolist()))

if "created_at" in df.columns:
    min_d = df["created_at"].min().date()
    max_d = df["created_at"].max().date()
    dates = st.sidebar.date_input("Période", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
else:
    dates = []

dff = df.copy()
if dept_sel != "Tous":
    dff = dff[dff["departement"] == dept_sel]
if risk_sel:
    dff = dff[dff["risk_level_predit"].isin(risk_sel)]
if cat_sel != "Tous":
    dff = dff[dff["categorie_predite"] == cat_sel]
if len(dates) == 2 and "created_at" in dff.columns:
    dff = dff[(dff["created_at"].dt.date >= dates[0]) &
              (dff["created_at"].dt.date <= dates[1])]

# ── Header ────────────────────────────────────────────────────────
st.title("🎫 IA Ticket Intelligence")
st.caption(f"{len(dff):,} tickets affichés après filtres")

# ── KPIs ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
nb_crit  = (dff["risk_level_predit"] == "Critique").sum()
pct_crit = nb_crit / len(dff) * 100 if len(dff) else 0

c1.metric("📋 Total",        f"{len(dff):,}")
c2.metric("🔴 Critiques",    f"{nb_crit}", f"{pct_crit:.1f}%", delta_color="inverse")
c3.metric("⏱️ Délai moy",   f"{dff['delai_predit_heures'].mean():.1f}h")
c4.metric("⭐ Satisfaction", f"{dff['satisfaction'].mean():.2f}/5")
if "statut" in dff.columns:
    c5.metric("🟡 En cours", f"{(dff['statut'] == 'En cours').sum():,}")

st.divider()

COLOR_RISK = {
    "Critique": "#f44336",
    "Élevé":    "#FF9800",
    "Moyen":    "#FFC107",
    "Faible":   "#4CAF50",
}

# ── Onglets ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Vue d'ensemble",
    "🔴 Risques",
    "📈 Tendances",
    "🤖 Prédictions IA",
    "📥 Export",
    "📂 Mes Données",
])

# ─ TAB 1 — Vue d'ensemble ─────────────────────────────────────────
with tab1:
    l, r = st.columns(2)

    with l:
        fig = px.pie(dff, names="categorie_predite",
                     title="Répartition par Catégorie (IA)",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        ds = (dff.groupby("departement")
                 .agg(nb=("ticket_id","count"), risk=("risk_score","mean"))
                 .reset_index()
                 .sort_values("nb", ascending=True))
        fig = px.bar(ds, x="nb", y="departement", orientation="h",
                     color="risk", color_continuous_scale="RdYlGn_r",
                     title="Tickets par Département (couleur = risque moyen)",
                     labels={"nb": "Nb tickets", "departement": ""})
        fig.update_layout(height=370, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap — uniquement si les colonnes existent
    if "jour_semaine" in dff.columns and "heure_creation" in dff.columns:
        st.subheader("📅 Heatmap Heure × Jour")
        hm = (dff.groupby(["jour_semaine", "heure_creation"])
                 .size().reset_index(name="n")
                 .pivot(index="jour_semaine", columns="heure_creation", values="n")
                 .fillna(0))
        jours = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        hm.index = [jours[i] for i in hm.index if i < 7]
        fig = px.imshow(hm, aspect="auto", color_continuous_scale="Blues",
                        labels={"x": "Heure", "y": "Jour", "color": "Tickets"})
        fig.update_layout(height=270)
        st.plotly_chart(fig, use_container_width=True)

# ─ TAB 2 — Risques ────────────────────────────────────────────────
with tab2:
    l, r = st.columns([1, 2])

    with l:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dff["risk_score"].mean(),
            delta={"reference": 50},
            title={"text": "Score de Risque Moyen"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2196F3"},
                "steps": [
                    {"range": [0,  25],  "color": "#E8F5E9"},
                    {"range": [25, 50],  "color": "#FFF9C4"},
                    {"range": [50, 75],  "color": "#FFE0B2"},
                    {"range": [75, 100], "color": "#FFEBEE"},
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": 75},
            }
        ))
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

    with r:
        rc = (dff.groupby(["categorie_predite", "risk_level_predit"])
                 .size().reset_index(name="n"))
        fig = px.bar(rc, x="categorie_predite", y="n",
                     color="risk_level_predit", barmode="stack",
                     title="Risques par Catégorie",
                     color_discrete_map=COLOR_RISK,
                     labels={"categorie_predite": "Catégorie", "n": "Tickets",
                             "risk_level_predit": "Risque"})
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 Top 20 Tickets Critiques")
    top_cols = [c for c in ["ticket_id","titre","categorie_predite","departement",
                             "systeme","risk_score","delai_predit_heures","agent_assigné"]
                if c in dff.columns]
    top = (dff[dff["risk_level_predit"] == "Critique"]
              .sort_values("risk_score", ascending=False)
              .head(20)[top_cols])
    st.dataframe(top.style.background_gradient(subset=["risk_score"], cmap="Reds"),
                 use_container_width=True, hide_index=True)

# ─ TAB 3 — Tendances ──────────────────────────────────────────────
with tab3:
    if "created_at" in dff.columns:
        dff["ym"] = dff["created_at"].dt.to_period("M").astype(str)
        mo = dff.groupby(["ym", "risk_level_predit"]).size().reset_index(name="n")
        fig = px.line(mo, x="ym", y="n", color="risk_level_predit",
                      title="Évolution mensuelle par niveau de risque",
                      color_discrete_map=COLOR_RISK, markers=True,
                      labels={"ym": "Mois", "n": "Tickets",
                              "risk_level_predit": "Risque"})
        fig.update_layout(height=370, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colonne 'created_at' absente — graphique tendances indisponible.")

    l, r = st.columns(2)
    with l:
        fig = px.box(dff, x="categorie_predite", y="delai_predit_heures",
                     color="categorie_predite",
                     title="Délai prédit par catégorie")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        s = dff.sample(min(500, len(dff)))
        fig = px.scatter(s, x="risk_score", y="delai_predit_heures",
                         color="risk_level_predit",
                         color_discrete_map=COLOR_RISK,
                         title="Risque vs Délai", opacity=0.55,
                         labels={"risk_score": "Score risque",
                                 "delai_predit_heures": "Délai (h)",
                                 "risk_level_predit": "Risque"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

# ─ TAB 4 — Prédictions IA ─────────────────────────────────────────
with tab4:
    st.subheader("🤖 Prédire sur un nouveau ticket")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_prio = st.selectbox("Priorité", ["Basse","Normale","Haute","Critique"])
            p_sys  = st.selectbox("Système", [
                "ERP SAP","CRM Salesforce","Messagerie","VPN","Imprimante",
                "Active Directory","Serveur Web","Base de données",
                "Application Mobile","Wi-Fi"])
        with c2:
            p_rel  = st.slider("Nb relances", 0, 10, 1)
            p_h    = st.slider("Heure création", 0, 23, 9)
        with c3:
            p_jour = st.selectbox("Jour", ["Lundi","Mardi","Mercredi","Jeudi",
                                            "Vendredi","Samedi","Dimanche"])
            p_sat  = st.slider("Satisfaction estimée", 1, 5, 3)
        run = st.form_submit_button("🔮 Prédire", use_container_width=True)

    if run:
        try:
            pmap = {"Basse":1,"Normale":2,"Haute":3,"Critique":4}
            jmap = {"Lundi":0,"Mardi":1,"Mercredi":2,"Jeudi":3,
                    "Vendredi":4,"Samedi":5,"Dimanche":6}
            csys = ["ERP SAP","Serveur Web","Base de données","Active Directory"]
            pn   = pmap[p_prio]
            jn   = jmap[p_jour]
            Xn   = np.array([[
                pn, p_rel, p_h, jn,
                int(jn >= 5),
                int(p_h < 8 or p_h > 18),
                int(p_sys in csys),
                6, 2,
                min(100, pn*20 + p_rel*5 + (15 if p_sys in csys else 0)),
                p_sat
            ]])

            clf_c = joblib.load("models/classifier_category.pkl")
            clf_r = joblib.load("models/classifier_risk.pkl")
            reg   = joblib.load("models/regressor_delay.pkl")
            lec   = joblib.load("models/label_encoder_category.pkl")
            ler   = joblib.load("models/label_encoder_risk.pkl")

            cat_p  = lec.inverse_transform(clf_c.predict(Xn))[0]
            cat_pr = clf_c.predict_proba(Xn)[0]
            Xr     = np.append(Xn, clf_c.predict(Xn)).reshape(1, -1)
            rsk_p  = ler.inverse_transform(clf_r.predict(Xr))[0]
            rsk_pr = clf_r.predict_proba(Xr)[0]
            del_p  = np.expm1(reg.predict(Xn))[0]

            st.success("✅ Prédiction calculée !")
            emoji = {"Critique":"🔴","Élevé":"🟠","Moyen":"🟡","Faible":"🟢"}
            r1, r2, r3 = st.columns(3)
            r1.metric("📂 Catégorie",     cat_p)
            r2.metric(f"{emoji.get(rsk_p,'⚪')} Risque", rsk_p)
            r3.metric("⏱️ Délai estimé", f"{del_p:.1f} h")

            l, r = st.columns(2)
            with l:
                fig = px.bar(x=lec.classes_, y=cat_pr,
                             title="Confiance — Catégorie",
                             labels={"x":"Catégorie","y":"Probabilité"})
                fig.update_layout(height=270, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with r:
                fig = px.bar(x=ler.classes_, y=rsk_pr,
                             color=ler.classes_,
                             color_discrete_map=COLOR_RISK,
                             title="Confiance — Risque",
                             labels={"x":"Risque","y":"Probabilité"})
                fig.update_layout(height=270, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur : {e}")

# ─ TAB 5 — Export ─────────────────────────────────────────────────
with tab5:
    l, r = st.columns(2)
    with l:
        st.download_button(
            "⬇️ CSV complet",
            dff.to_csv(index=False).encode("utf-8"),
            f"tickets_{pd.Timestamp.now():%Y%m%d}.csv",
            "text/csv",
            use_container_width=True,
        )
    with r:
        crit_json = (dff[dff["risk_level_predit"] == "Critique"]
                     .to_json(orient="records", force_ascii=False, indent=2))
        st.download_button(
            "⬇️ JSON Critiques",
            crit_json.encode("utf-8"),
            f"critiques_{pd.Timestamp.now():%Y%m%d}.json",
            "application/json",
            use_container_width=True,
        )

    st.subheader("🔍 Aperçu")
    st.dataframe(dff.head(100), use_container_width=True, hide_index=True)
    st.caption(f"100 / {len(dff)} tickets affichés")

# ─ TAB 6 — Mes Données ────────────────────────────────────────────
with tab6:
    st.subheader("📂 Analyser votre propre fichier")
    st.caption("Uploadez un fichier CSV, Excel ou JSON — les modèles IA analysent vos données.")

    uploaded = st.file_uploader(
        "Glissez votre fichier ici",
        type=["csv", "xlsx", "xls", "json"],
    )

    if uploaded is not None:

        ext = Path(uploaded.name).suffix.lower()
        if ext == ".csv":
            df_user = pd.read_csv(uploaded)
        elif ext in (".xlsx", ".xls"):
            df_user = pd.read_excel(uploaded)
        elif ext == ".json":
            df_user = pd.read_json(uploaded)

        st.success(f"✅ {uploaded.name} — {len(df_user)} lignes, {df_user.shape[1]} colonnes")

        with st.expander("👁️ Aperçu du fichier brut"):
            st.dataframe(df_user.head(10), use_container_width=True)

        st.divider()
        st.subheader("🗂️ Associer vos colonnes")
        st.caption("Si vos colonnes ont des noms différents, associez-les ici.")

        user_cols = ["(ignorer)"] + list(df_user.columns)
        NEEDED    = ["priorite","nb_relances","heure_creation",
                     "jour_semaine","systeme","mois",
                     "trimestre","risk_score","satisfaction"]
        mapping   = {}
        c1, c2    = st.columns(2)

        for i, field in enumerate(NEEDED):
            col         = c1 if i % 2 == 0 else c2
            default_idx = user_cols.index(field) if field in user_cols else 0
            mapping[field] = col.selectbox(
                f"**{field}**", options=user_cols,
                index=default_idx, key=f"map_{field}",
            )

        st.divider()
        if st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary"):

            with st.spinner("Analyse en cours..."):
                df_proc = df_user.copy()

                rename = {v: k for k, v in mapping.items()
                          if v != "(ignorer)" and v != k}
                df_proc.rename(columns=rename, inplace=True)

                PMAP = {"Basse":1,"Normale":2,"Haute":3,"Critique":4}
                CSYS = ["ERP SAP","Serveur Web","Base de données","Active Directory"]
                DEFS = {"nb_relances":0,"mois":6,"trimestre":2,
                        "risk_score":40,"satisfaction":3,
                        "heure_creation":9,"jour_semaine":1}

                for col, val in DEFS.items():
                    if col not in df_proc.columns:
                        df_proc[col] = val

                df_proc["priorite_num"] = (
                    df_proc["priorite"].map(PMAP).fillna(2)
                    if "priorite" in df_proc.columns
                    else 2
                )
                df_proc["systeme_critique"] = (
                    df_proc["systeme"].isin(CSYS).astype(int)
                    if "systeme" in df_proc.columns
                    else 0
                )
                df_proc["est_weekend"]      = (df_proc["jour_semaine"] >= 5).astype(int)
                df_proc["est_heure_creuse"] = (
                    (df_proc["heure_creation"] < 8) |
                    (df_proc["heure_creation"] > 18)
                ).astype(int)

                FEATURES = [
                    "priorite_num","nb_relances","heure_creation","jour_semaine",
                    "est_weekend","est_heure_creuse","systeme_critique",
                    "mois","trimestre","risk_score","satisfaction",
                ]

                try:
                    le_c  = joblib.load("models/label_encoder_category.pkl")
                    le_r  = joblib.load("models/label_encoder_risk.pkl")
                    clf_c = joblib.load("models/classifier_category.pkl")
                    clf_r = joblib.load("models/classifier_risk.pkl")
                    reg   = joblib.load("models/regressor_delay.pkl")

                    X        = df_proc[FEATURES]
                    cat_pred = clf_c.predict(X)

                    df_proc["categorie_predite"]   = le_c.inverse_transform(cat_pred)
                    df_proc["risk_level_predit"]   = le_r.inverse_transform(
                        clf_r.predict(np.hstack([X, cat_pred.reshape(-1, 1)]))
                    )
                    df_proc["delai_predit_heures"] = np.expm1(reg.predict(X)).round(1)

                    st.success("✅ Analyse terminée !")

                    k1, k2, k3, k4 = st.columns(4)
                    nb_c = (df_proc["risk_level_predit"] == "Critique").sum()
                    k1.metric("📋 Lignes analysées",   f"{len(df_proc):,}")
                    k2.metric("🔴 Critiques",          f"{nb_c}",
                              f"{nb_c/len(df_proc)*100:.1f}%", delta_color="inverse")
                    k3.metric("⏱️ Délai moyen prédit", f"{df_proc['delai_predit_heures'].mean():.1f} h")
                    k4.metric("📂 Catégorie dominante", df_proc["categorie_predite"].mode()[0])

                    g1, g2 = st.columns(2)
                    with g1:
                        fig = px.pie(df_proc, names="risk_level_predit",
                                     title="Répartition des risques",
                                     color="risk_level_predit",
                                     color_discrete_map=COLOR_RISK, hole=0.4)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    with g2:
                        fig = px.bar(
                            df_proc["categorie_predite"].value_counts().reset_index(),
                            x="categorie_predite", y="count",
                            title="Répartition des catégories",
                            color="categorie_predite",
                            labels={"categorie_predite":"Catégorie","count":"Nb"},
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    preview_cols = (
                        [c for c in ["ticket_id","titre"] if c in df_proc.columns] +
                        ["categorie_predite","risk_level_predit","delai_predit_heures"]
                    )
                    st.dataframe(df_proc[preview_cols],
                                 use_container_width=True, hide_index=True)

                    st.divider()
                    e1, e2 = st.columns(2)
                    with e1:
                        st.download_button(
                            "⬇️ Télécharger CSV",
                            df_proc.to_csv(index=False).encode("utf-8"),
                            f"{Path(uploaded.name).stem}_predictions.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                    with e2:
                        kpi_out = {
                            "source":      uploaded.name,
                            "total":       len(df_proc),
                            "critiques":   int(nb_c),
                            "delai_moyen": round(float(df_proc["delai_predit_heures"].mean()), 1),
                            "risques":     df_proc["risk_level_predit"].value_counts().to_dict(),
                            "categories":  df_proc["categorie_predite"].value_counts().to_dict(),
                        }
                        st.download_button(
                            "⬇️ Télécharger JSON",
                            json.dumps(kpi_out, ensure_ascii=False, indent=2).encode("utf-8"),
                            f"{Path(uploaded.name).stem}_kpi.json",
                            "application/json",
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("🤖 IA Ticket Intelligence · Python · Scikit-learn · Streamlit · Projet Stage Data Science")