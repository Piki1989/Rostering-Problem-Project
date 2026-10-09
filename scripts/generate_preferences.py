import pandas as pd
import random

# -------------------------
# CONFIG
# -------------------------

NUM_DOCTORS = 15  # dopasuj jak trzeba
WEEKENDS = ["Sat", "Sun"]

# probabilities
PREF_PROBS = {
    "weekday": {"dislike": 0.06, "like": 0.05},
    "weekend": {"dislike": 0.20, "like": 0.03}
}

# -------------------------
# LOAD SHIFTS
# -------------------------

shifts = pd.read_csv("../shifts_month/shifts_mar2026.csv")

rows = []

# -------------------------
# GENERATE PREFERENCES
# -------------------------

for _, row in shifts.iterrows():
    code = row["code"]
    weekday = row["weekday"]

    if weekday in WEEKENDS:
        probs = PREF_PROBS["weekend"]
    else:
        probs = PREF_PROBS["weekday"]

    for d in range(1, NUM_DOCTORS + 1):

        r = random.random()

        if r < probs["dislike"]:
            pref = "dislike"
        elif r < probs["dislike"] + probs["like"]:
            pref = "like"
        else:
            continue  # no preference → sparse dataset

        rows.append([d, code, pref])

# -------------------------
# SAVE
# -------------------------

df = pd.DataFrame(rows, columns=["doctor_id", "code", "preference"])
df.to_csv("preferences_mar2026.csv", index=False)

print("Generated preferences_mar2026.csv with", len(df), "rows")