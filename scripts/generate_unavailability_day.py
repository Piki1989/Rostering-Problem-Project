import pandas as pd
import random

# -------------------------
# CONFIG
# -------------------------

NUM_DOCTORS = 15
MIN_DAYS = 0
MAX_DAYS = 7

# load shifts to get all unique dates
shifts = pd.read_csv("../shifts_month/shifts_mar2026.csv")
all_dates = sorted(shifts["date"].unique())

# get weekday info
date_weekday = shifts.drop_duplicates("date")[["date", "weekday"]].set_index("date")["weekday"].to_dict()

# bias weights
def get_weight(date):
    wd = date_weekday[date]
    if wd in ["Sat", "Sun"]:
        return 1.5  # slightly more likely
    else:
        return 1.0

weights = [get_weight(d) for d in all_dates]

# -------------------------
# GENERATE
# -------------------------

rows = []

for d_id in range(1, NUM_DOCTORS + 1):

    k = random.randint(MIN_DAYS, MAX_DAYS)

    if k == 0:
        continue

    chosen_dates = set()

    # --- główny blok (urlop) ---
    if k >= 2:
        block_size = random.choice([2, 3])  # typowe krótkie urlopy
        block_size = min(block_size, k)

        start_idx = random.randint(0, len(all_dates) - block_size)
        block = all_dates[start_idx:start_idx + block_size]
        chosen_dates.update(block)

    # --- ewentualne pojedyncze dni ---
    remaining = k - len(chosen_dates)

    if remaining > 0:
        extra = random.choices(all_dates, weights=weights, k=remaining * 2)
        chosen_dates.update(extra)

    chosen_dates = list(chosen_dates)[:k]

    for d in chosen_dates:
        rows.append([d_id, d])

# -------------------------
# SAVE
# -------------------------

df = pd.DataFrame(rows, columns=["doctor_id", "date"])
df.to_csv("unavailability_mar2026.csv", index=False)

print("Generated unavailability_mar2026.csv with", len(df), "rows")