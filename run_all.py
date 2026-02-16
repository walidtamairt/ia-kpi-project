"""
Lance tout le pipeline en une seule commande
"""
import subprocess
import sys
import os

scripts = [
    "scripts/01_generate_data.py",
    "scripts/02_preprocessing.py",
    "scripts/03_train_model.py",
    "scripts/04_evaluate_model.py",
    "scripts/05_export_results.py",
]

print("=" * 50)
print("  🚀 Lancement du pipeline complet")
print("=" * 50)

for script in scripts:
    print(f"\n▶️  {script} ...")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"\n❌ Erreur dans {script} — pipeline arrêté.")
        sys.exit(1)
    print(f"✅ {script} terminé")

print("\n" + "=" * 50)
print("  ✅ Pipeline terminé ! Lancement du dashboard...")
print("=" * 50 + "\n")

os.system(f"{sys.executable} -m streamlit run dashboard/app.py")
