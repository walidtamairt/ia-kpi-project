# 🎫 IA Ticket Intelligence

> Classification automatique de tickets IT, scoring de risques et prédiction de délais par Machine Learning.

---

## 🚀 Lancement en 2 commandes
```bash
pip install -r requirements.txt
python run_all.py
```

Dashboard disponible sur **http://localhost:8501**

---

## 🤖 Ce que fait l'application

Tu donnes des tickets IT → l'IA te dit automatiquement :

- 📂 **Ce que c'est** — Incident / Bug / Demande / Changement
- 🔴 **À quel point c'est urgent** — Faible / Moyen / Élevé / Critique  
- ⏱️ **Combien de temps ça va prendre** — en heures

---

## 🛠️ Stack
```
Python · Pandas · Scikit-learn · Streamlit · Plotly · SQLite · Git
```

---

## 📁 Structure
```
ia-ticket-intelligence/
├── run_all.py              ← Lance tout en 1 commande
├── scripts/                ← Pipeline ML (5 étapes)
├── dashboard/              ← Interface Streamlit
├── models/                 ← Modèles entraînés
├── data/                   ← Données brutes et traitées
└── docs/                   ← Documentation et rapports
```

---

## 📊 Dashboard — 6 onglets

| Onglet | Contenu |
|--------|---------|
| 📊 Vue d'ensemble | KPIs, répartition, heatmap activité |
| 🔴 Risques | Top 20 critiques, jauge risque moyen |
| 📈 Tendances | Évolution mensuelle, délais par catégorie |
| 🤖 Prédictions IA | Scorer un ticket en temps réel |
| 📥 Export | Télécharger CSV et JSON |
| 📂 Mes Données | Uploader son propre fichier et obtenir les prédictions |

---

## 📂 Utiliser vos propres données

**Via le dashboard** — onglet *Mes Données* :
1. Uploader votre fichier CSV / Excel / JSON
2. Mapper vos colonnes
3. Cliquer **Lancer l'analyse**
4. Télécharger les résultats

**Via le terminal** :
```bash
python scripts/06_infer_user_file.py --file mon_fichier.csv
```

---

## 👤 Auteur

**Walid TAMAIRT** — Stage Data Science / IA
