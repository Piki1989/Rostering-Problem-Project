import numpy as np
import pandas as pd

df = pd.read_csv("doctors_with_cost.csv")

def scale_doctors_realistic(df, multiplier):
    dfs = []
    n = len(df)

    for i in range(multiplier):
        tmp = df.copy()

        tmp["id"] = tmp["id"] + i * n
        tmp["name"] = tmp["name"] + f"_g{i+1}"

        # 🔹 lekkie różnice w godzinach
        tmp["max_hours"] = tmp["max_hours"] + np.random.randint(-8, 9, size=len(tmp))

        # 🔹 lekkie różnice w kosztach
        tmp["hourly_cost"] = tmp["hourly_cost"] + np.random.randint(-20, 21, size=len(tmp))

        # 🔹 czasem ktoś nie bierze nocy
        mask = np.random.rand(len(tmp)) < 0.1
        tmp.loc[mask, "night_allowed"] = 0

        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)


# === 30 lekarzy ===
df_30 = scale_doctors_realistic(df, 2)

# === 45 lekarzy ===
df_45 = scale_doctors_realistic(df, 3)

df_30.to_csv("doctors_30.csv", index=False)
df_45.to_csv("doctors_45.csv", index=False)