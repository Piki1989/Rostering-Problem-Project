import pandas as pd

# wczytanie danych
df = pd.read_csv("preferences_mar2026.csv")

# ======================
# PODSTAWOWE STATYSTYKI
# ======================

likes = (df["preference"] == "like").sum()
dislikes = (df["preference"] == "dislike").sum()

print("=== OGÓLNE ===")
print("Like:", likes)
print("Dislike:", dislikes)
print("Razem:", len(df))


# ======================
# ŚREDNIA NA LEKARZA
# ======================

per_doctor = df.groupby("doctor_id")["preference"].value_counts().unstack(fill_value=0)

per_doctor["total"] = per_doctor.sum(axis=1)
per_doctor["like_ratio"] = per_doctor.get("like", 0) / per_doctor["total"]
per_doctor["dislike_ratio"] = per_doctor.get("dislike", 0) / per_doctor["total"]

print("\n=== ŚREDNIO NA LEKARZA ===")
print(per_doctor)

print("\nŚrednia liczba preferencji na lekarza:", per_doctor["total"].mean())
print("Średnia liczba like:", per_doctor.get("like", 0).mean())
print("Średnia liczba dislike:", per_doctor.get("dislike", 0).mean())


# ======================
# WYCIĄGANIE DATY
# ======================

# data jest na końcu code, np. WARD_M_2026-12-24
df["date"] = df["code"].str.extract(r'(\d{4}-\d{2}-\d{2})')
df["date"] = pd.to_datetime(df["date"])


# ======================
# OKRESY SPECJALNE
# ======================

# święta (24-26 grudnia)
# christmas = df[df["date"].dt.day.isin([24, 25, 26])]
#
# # sylwester (31 grudnia)
# new_year = df[df["date"].dt.day == 31]


# ======================
# FUNKCJA PODSUMOWANIA
# ======================

def summarize(df_subset, name):
    likes = (df_subset["preference"] == "like").sum()
    dislikes = (df_subset["preference"] == "dislike").sum()
    total = len(df_subset)

    ratio = dislikes / total if total > 0 else 0

    print(f"\n=== {name} ===")
    print("Like:", likes)
    print("Dislike:", dislikes)
    print("Razem:", total)
    print("Udział dislike:", round(ratio, 3))


# summarize(christmas, "ŚWIĘTA (24-26)")
# summarize(new_year, "SYLWESTER (31)")


# ======================
# BONUS: NATĘŻENIE NA LEKARZA
# ======================

def per_doctor_stats(df_subset, name):
    stats = df_subset.groupby("doctor_id")["preference"].value_counts().unstack(fill_value=0)
    stats["total"] = stats.sum(axis=1)

    print(f"\n=== {name} NA LEKARZA ===")
    print(stats)


# per_doctor_stats(christmas, "ŚWIĘTA")
# per_doctor_stats(new_year, "SYLWESTER")