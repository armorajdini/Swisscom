# src/config.py

DATA_PATH = "data/processed/agg_sr_dunning_month_bill.csv"
DATE_COL = "Datum (Month)"

# -----------------------------
# TARGET CONFIG
# -----------------------------

TARGET_MODE = "billing_clean"
# Optionen:
# - "billing_clean"
# - "billing_broad"

BILLING_SR_CODES = [
    "01 Offene RG/ Laufende Kosten",
    "01 Mutation",
    "02 Ohne Vertragsänderung",
    "14 Frage zu Versandart"
]

# -----------------------------
# SPLIT CONFIG
# -----------------------------

TRAIN_MONTHS = [
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01",
    "2025-01-01"
]

TEST_MONTHS = [
    "2025-02-01",
    "2025-03-01"
]
