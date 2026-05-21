import pandas as pd
from datetime import datetime, timedelta

# -------------------------
# CONFIG
# -------------------------

# start = datetime(2026, 12, 1)
# end = datetime(2027, 1, 1)  # exclusive
start = datetime(2026, 3, 1)
end = datetime(2026, 4, 1)
# hard_days = {
#     "2026-12-24",
#     "2026-12-25",
#     "2026-12-26",
#     "2026-12-31"
# }

hard_days = {}

weekday_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

rows = []
shift_id = 1

def add_shift(code, dept, start_h, end_h, hours,
              min_staff, regular_staff, req_skill, date, weekday):
    global shift_id
    is_hard = 1 if date in hard_days else 0
    rows.append([
        shift_id, date, weekday, code, dept, start_h, end_h,
        hours, min_staff, regular_staff, req_skill, is_hard
    ])
    shift_id += 1

# -------------------------
# GENERATE DAYS
# -------------------------

current = start
dates = []
while current < end:
    dates.append(current)
    current += timedelta(days=1)

# -------------------------
# GENERATE SHIFTS
# -------------------------

for d in dates:
    date_str = d.strftime("%Y-%m-%d")
    wd = weekday_map[d.weekday()]

    # ===== WARD =====
    add_shift(f"WARD_M_{date_str}", "WARD", 7, 15, 8, 1, 3, "ward", date_str, wd)
    add_shift(f"WARD_E_{date_str}", "WARD", 15, 23, 8, 1, 3, "ward", date_str, wd)
    add_shift(f"WARD_N_{date_str}", "WARD", 23, 31, 8, 1, 2, "ward", date_str, wd)
    add_shift(f"WARD_24_{date_str}", "WARD", 7, 31, 24, 0, 1, "ward", date_str, wd)

    # ===== ICU =====
    add_shift(f"ICU_D_{date_str}", "ICU", 7, 19, 12, 1, 2, "icu", date_str, wd)
    add_shift(f"ICU_N_{date_str}", "ICU", 19, 31, 12, 1, 2, "icu", date_str, wd)
    add_shift(f"ICU_24_{date_str}", "ICU", 7, 31, 24, 0, 1, "icu", date_str, wd)

    # ===== OUTPATIENT CLINICS (WORKDAYS ONLY) =====
    if wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:

        # General
        add_shift(f"CL_GEN_{date_str}", "CLINIC", 7, 14, 7, 1, 1, "clinic_general", date_str, wd)

        # Arrhythmia Tue+Thu
        if wd in ["Tue", "Thu"]:
            add_shift(f"CL_ARR_{date_str}", "CLINIC", 7, 14, 7, 1, 1, "clinic_arrhythmia", date_str, wd)

        # Device Mon+Thu
        if wd in ["Mon", "Thu"]:
            add_shift(f"CL_DEV_{date_str}", "CLINIC", 7, 14, 7, 1, 1, "clinic_device", date_str, wd)

        # HF Mon+Wed
        if wd in ["Mon", "Wed"]:
            add_shift(f"CL_HF_{date_str}", "CLINIC", 7, 14, 7, 1, 1, "clinic_hf", date_str, wd)

        # HTN Fri
        if wd == "Fri":
            add_shift(f"CL_HTN_{date_str}", "CLINIC", 7, 14, 7, 1, 1, "clinic_htn", date_str, wd)

        # Afternoon Mon+Wed+Fri
        if wd in ["Mon", "Wed", "Fri"]:
            add_shift(f"CL_PM_{date_str}", "CLINIC", 14, 18, 4, 1, 1, "clinic_general", date_str, wd)

# -------------------------
# SAVE FILE
# -------------------------

df = pd.DataFrame(rows, columns=[
    "id", "date", "weekday", "code", "dept",
    "start_hour", "end_hour", "hours",
    "min_staff", "regular_staff", "required_skill",
    "is_hard_shift"
])

df.to_csv("sh_mar.csv", index=False, encoding="utf-8")

print("Generated sh_mar.csv with", len(df), "rows")
