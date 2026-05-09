import pandas as pd
from pathlib import Path

def load_all_summaries(results_dir):
    summaries = []

    for path in Path(results_dir).rglob("summary_*.csv"):
        df = pd.read_csv(path)
        df["run"] = path.parent.name
        summaries.append(df)

    return pd.concat(summaries, ignore_index=True)

def pareto_front(df, cost_col="total_cost", sat_col="avg_satisfaction"):
    pareto_mask = []

    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if j == i:
                continue
            if (
                other[cost_col] <= row[cost_col]
                and other[sat_col] >= row[sat_col]
                and (
                    other[cost_col] < row[cost_col]
                    or other[sat_col] > row[sat_col]
                )
            ):
                dominated = True
                break
        pareto_mask.append(not dominated)

    return df[pareto_mask]

def pareto_front_2d(df, x_col, y_col, minimize_x=True, maximize_y=True):
    mask = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue

            better_or_eq_x = other[x_col] <= row[x_col] if minimize_x else other[x_col] >= row[x_col]
            better_or_eq_y = other[y_col] >= row[y_col] if maximize_y else other[y_col] <= row[y_col]

            strictly_better_x = other[x_col] < row[x_col] if minimize_x else other[x_col] > row[x_col]
            strictly_better_y = other[y_col] > row[y_col] if maximize_y else other[y_col] < row[y_col]

            if better_or_eq_x and better_or_eq_y and (strictly_better_x or strictly_better_y):
                dominated = True
                break

        mask.append(not dominated)

    return df[mask]