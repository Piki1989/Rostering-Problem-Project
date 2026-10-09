# import pandas as pd
#
# df = pd.read_csv("shifts_dec_1.csv")
#
# total_min_staff = df["min_staff"].sum()
# print("Suma min_staff =", total_min_staff)

import pandas as pd

# wczytanie danych
df = pd.read_csv("shifts_45.csv")

# # obliczenie minimalnych godzin
df["min_hours"] = df["hours"] * df["min_staff"]
df["reg_hours"] = df["hours"] * df["regular_staff"]

total_min_staff = df["min_staff"].sum()
print("Suma min_staff =", total_min_staff)

total_min_hours = df["min_hours"].sum()
total_reg_hours = df["reg_hours"].sum()
#
print(f"Łączna minimalna liczba godzin: {total_min_hours}")
print(f"Łączna regularna liczba godzin: {total_reg_hours}")

# zmiany 24h
shifts_24h = df[df["hours"] == 24]
num_24h = len(shifts_24h)

# zmiany nocne (przykład: zaczynają się o 23 lub 19)
night_shifts = df[df["start_hour"].isin([23, 19])]
num_night = len(night_shifts)

# zmiany "hard"
hard_shifts = df[df["is_hard_shift"] == 1]
num_hard = len(hard_shifts)

# ======================
# WYNIKI
# ======================

print(f"Liczba zmian 24h: {num_24h}")
print(f"Liczba zmian nocnych: {num_night}")
print(f"Liczba zmian hard: {num_hard}")