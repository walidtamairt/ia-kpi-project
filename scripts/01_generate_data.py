"""
Générateur de dataset simulé — Tickets IT
Génère ~2000 tickets réalistes avec features métier
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker('fr_FR')
np.random.seed(42)
random.seed(42)

N_TICKETS   = 2000
OUTPUT_PATH = "data/raw/tickets_raw.csv"

CATEGORIES  = ["Incident", "Bug", "Demande", "Changement"]
PRIORITIES  = ["Basse", "Normale", "Haute", "Critique"]
DEPARTMENTS = ["DSI", "Finance", "RH", "Commercial", "Logistique", "Direction"]
SYSTEMS     = [
    "ERP SAP", "CRM Salesforce", "Messagerie", "VPN", "Imprimante",
    "Active Directory", "Serveur Web", "Base de données",
    "Application Mobile", "Wi-Fi"
]
STATUS  = ["Résolu", "En cours", "Fermé"]
AGENTS  = [fake.name() for _ in range(15)]

TITLE_TEMPLATES = {
    "Incident":   ["Panne réseau sur {}",           "Serveur {} inaccessible",
                   "Coupure service {}",             "Interruption {} - Production impactée",
                   "Crash application {}"],
    "Bug":        ["Erreur calcul dans {}",          "Affichage incorrect sur {}",
                   "Bug authentification {}",        "Données corrompues sur {}",
                   "Exception non gérée dans {}"],
    "Demande":    ["Accès {} requis",                "Installation {} demandée",
                   "Nouveau compte {}",              "Mise à jour {} nécessaire",
                   "Formation {} souhaitée"],
    "Changement": ["Migration vers {}",              "Upgrade version {}",
                   "Déploiement patch {}",           "Configuration nouvelle {}",
                   "Remplacement matériel {}"],
}


def generate_resolution_hours(category, priority):
    base = {
        "Incident":   {"Basse": 24,  "Normale": 12, "Haute": 4,  "Critique": 1},
        "Bug":        {"Basse": 72,  "Normale": 48, "Haute": 24, "Critique": 8},
        "Demande":    {"Basse": 120, "Normale": 72, "Haute": 48, "Critique": 24},
        "Changement": {"Basse": 168, "Normale": 120,"Haute": 72, "Critique": 48},
    }
    mu    = base[category][priority]
    sigma = mu * 0.4
    return round(max(0.5, np.random.normal(mu, sigma)), 1)


def generate_risk_score(category, priority, nb_relances, system):
    priority_w  = {"Basse": 10, "Normale": 30, "Haute": 60, "Critique": 90}
    category_w  = {"Incident": 80, "Bug": 60, "Demande": 20, "Changement": 50}
    critical_sys = ["ERP SAP", "Serveur Web", "Base de données", "Active Directory"]
    base = (priority_w[priority]  * 0.4 +
            category_w[category]  * 0.3 +
            min(nb_relances * 5, 20) +
            (15 if system in critical_sys else 0))
    return int(np.clip(base + np.random.normal(0, 5), 0, 100))


def score_to_risk_level(score):
    if score < 25: return "Faible"
    if score < 50: return "Moyen"
    if score < 75: return "Élevé"
    return "Critique"


records    = []
start_date = datetime(2024, 1, 1)

for i in range(N_TICKETS):
    category  = random.choices(CATEGORIES,  weights=[35, 30, 25, 10])[0]
    priority  = random.choices(PRIORITIES,  weights=[20, 45, 25, 10])[0]
    dept      = random.choice(DEPARTMENTS)
    system    = random.choice(SYSTEMS)
    agent     = random.choice(AGENTS)

    created_at = start_date + timedelta(
        days=random.randint(0, 400),
        hours=random.randint(7, 19),
        minutes=random.randint(0, 59),
    )

    nb_relances      = max(0, int(np.random.poisson(1.5)))
    resolution_hours = generate_resolution_hours(category, priority)
    resolved_at      = created_at + timedelta(hours=resolution_hours)
    risk_score       = generate_risk_score(category, priority, nb_relances, system)

    records.append({
        "ticket_id":         f"TKT-{10000 + i}",
        "titre":             random.choice(TITLE_TEMPLATES[category]).format(system),
        "categorie":         category,
        "priorite":          priority,
        "departement":       dept,
        "systeme":           system,
        "agent_assigné":     agent,
        "nb_relances":       nb_relances,
        "heure_creation":    created_at.hour,
        "jour_semaine":      created_at.weekday(),
        "created_at":        created_at.isoformat(),
        "resolved_at":       resolved_at.isoformat(),
        "resolution_heures": resolution_hours,
        "risk_score":        risk_score,
        "risk_level":        score_to_risk_level(risk_score),
        "statut":            random.choices(STATUS, weights=[65, 20, 15])[0],
        "satisfaction":      random.choices([1,2,3,4,5], weights=[5,10,20,40,25])[0],
    })

df = pd.DataFrame(records)
os.makedirs("data/raw", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Dataset généré : {len(df)} tickets → {OUTPUT_PATH}")
print(f"\nRépartition catégories :\n{df['categorie'].value_counts().to_string()}")
print(f"\nRépartition risques :\n{df['risk_level'].value_counts().to_string()}")