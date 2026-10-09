import pandas as pd
import random

# -------------------------
# CONFIG
# -------------------------

NUM_DOCTORS = 15

# load data
shifts = pd.read_csv("../shifts_month/shifts_mar2026.csv")
unavail = pd.read_csv("../unavail_monthly_day/unavailability_mar2026.csv")

# map: doctor -> unavailable dates
doc_unavail = (
    unavail.groupby("doctor_id")["date"]
    .apply(set)
    .to_dict()
)

rows = []

# -------------------------
# GENERATE SHIFT-LEVEL UNAVAIL
# -------------------------

for d_id in range(1, NUM_DOCTORS + 1):

    # większość nie ma
    if random.random() < 0.6:
        continue

    k = random.randint(1, 3)

    # dostępne shifty (bez kolizji z day-unavailability)
    available_shifts = shifts[
        ~shifts["date"].isin(doc_unavail.get(d_id, set()))
    ]

    chosen = available_shifts.sample(
        n=min(k, len(available_shifts)),
        replace=False
    )

    for _, row in chosen.iterrows():
        rows.append([d_id, row["code"]])

# -------------------------
# SAVE
# -------------------------

df = pd.DataFrame(rows, columns=["doctor_id", "code"])
df.to_csv("shift_unavailability_mar2026.csv", index=False)

print("Generated shift_unavailability_mar2026.csv with", len(df), "rows")