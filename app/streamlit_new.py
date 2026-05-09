import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from streamlit_calendar import calendar
from datetime import datetime, timedelta
import altair as alt
from compute_global_metrics import *
from pareto_utils import *
import plotly.express as px
from altair_saver import save
import altair as alt


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

EXPORT_ROOT = ROOT_DIR / "exports"
EXPORT_ROOT.mkdir(exist_ok=True)

from model.model_ref import run_model_and_get_results
from model.emergency_replacement_new import find_replacement, precompute_replacements

def filter_active_doctors(data: dict) -> dict:
    stats = data["stats_df"]
    active_names = set(stats.loc[stats["TotalHours"] > 0, "Doctor"])

    filtered = data.copy()
    filtered["stats_df"] = stats[stats["Doctor"].isin(active_names)]

    filtered["schedule_df"] = data["schedule_df"][
        data["schedule_df"]["Doctor"].isin(active_names)
        | data["schedule_df"]["Doctor"].isna()
    ]

    return filtered


def schedule_to_calendar_events(schedule_df):
    events = []

    for _, row in schedule_df.iterrows():
        date = row["Date"]
        start_hour = int(row["StartHour"])
        end_hour_raw = int(row["EndHour"])

        start_dt = datetime.fromisoformat(f"{date} {start_hour:02d}:00")

        if end_hour_raw <= 24:
            end_dt = datetime.fromisoformat(f"{date} {end_hour_raw:02d}:00")
        else:
            end_dt = datetime.fromisoformat(
                f"{date} {(end_hour_raw - 24):02d}:00"
            ) + timedelta(days=1)

        if row["Missing"] == 1:
            title = f"{row['ShiftCode']} – Uncovered"
            color = "#e74c3c"
        else:
            title = f"{row['ShiftCode']} – {row['Doctor']}"
            color = "#2ecc71"

        events.append({
            "title": title,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "color": color,
        })

    return events

def compute_global_metrics(schedule_df: pd.DataFrame) -> dict:
    df = schedule_df.copy()

    total_missing_rows = int((df["Missing"] == 1).sum())

    if "ShiftCode" in df.columns and "Date" in df.columns:
        missing_shifts = df[df["Missing"] == 1][["Date", "ShiftCode"]].drop_duplicates()
        missing_shifts_cnt = int(len(missing_shifts))
    else:
        missing_shifts_cnt = None

    days_with_missing = int(df[df["Missing"] == 1]["Date"].nunique())

    covered_assignments = int((df["Missing"] == 0).sum())

    total_shifts = int(df[["Date", "ShiftCode"]].drop_duplicates().shape[0])

    missing_by_shift = (
        df.groupby(["Date", "ShiftCode"])["Missing"].sum().reset_index(name="MissingCnt")
    )
    fully_covered_shifts = int((missing_by_shift["MissingCnt"] == 0).sum())

    return {
        "TotalMissingRows": total_missing_rows,
        "MissingShifts": missing_shifts_cnt,
        "DaysWithMissing": days_with_missing,
        "CoveredAssignments": covered_assignments,
        "TotalShifts": total_shifts,
        "FullyCoveredShifts": fully_covered_shifts,
    }


def compute_delta(before: dict, after: dict, key: str):
    b = before.get(key)
    a = after.get(key)
    if b is None or a is None:
        return None
    diff = a - b
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff}"

def chart_work_hours(stats_df: pd.DataFrame, title: str):
    df = stats_df.copy()

    bars = (
        alt.Chart(df)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("Doctor:N", sort="-y", title="Doctor"),
            y=alt.Y("TotalHours:Q", title="Worked hours"),
            tooltip=["Doctor", "TotalHours", "MaxHours"]
        )
    )

    limits = (
        alt.Chart(df)
        .mark_rule(color="crimson", strokeWidth=2)
        .encode(
            x="Doctor:N",
            y="MaxHours:Q",
            tooltip=["Doctor", "MaxHours"]
        )
    )

    return (bars + limits).properties(
        title=title,
        height=350
    )


def compute_weekly_workload(schedule_df: pd.DataFrame):
    df = schedule_df.copy()
    df = df[df["Doctor"].notna()]
    df["Date"] = pd.to_datetime(df["Date"])

    start_date = df["Date"].min()
    start_monday = start_date - pd.Timedelta(days=start_date.weekday())

    df["LocalWeek"] = df["Date"].apply(
        lambda d: int((d - start_monday).days // 7)
    )

    weekly = (
        df
        .groupby(["Doctor", "LocalWeek"])
        .agg(Hours=("Hours", "sum"))
        .reset_index()
    )

    return weekly

def chart_weekly_work_hours(
    weekly_df: pd.DataFrame,
    weekly_limits: dict,
    doctors_df: pd.DataFrame,
    week: int,
    title_prefix: str
):
    name_to_id = {row["name"]: row["id"] for _, row in doctors_df.iterrows()}

    df = weekly_df[weekly_df["LocalWeek"] == week].copy()
    df["DoctorID"] = df["Doctor"].map(name_to_id)

    df["WeeklyLimit"] = df.apply(
        lambda r: weekly_limits.get(r["DoctorID"], {}).get(week, 0),
        axis=1
    )

    bars = (
        alt.Chart(df)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("Doctor:N", sort="-y"),
            y=alt.Y("Hours:Q", title="Worked hours (week)"),
            tooltip=["Doctor", "Hours", "WeeklyLimit"]
        )
    )

    limits = (
        alt.Chart(df)
        .mark_rule(color="crimson", strokeWidth=2)
        .encode(
            x="Doctor:N",
            y="WeeklyLimit:Q"
        )
    )

    return (bars + limits).properties(
        title=f"{title_prefix} – Week {week + 1}",
        height=350
    )


def chart_night_and_24h(stats_df: pd.DataFrame, title_prefix: str):
    df = stats_df.copy()

    col1, col2 = st.columns(2)

    with col1:
        st.altair_chart(
            alt.Chart(df)
            .mark_bar(color="#5B8FF9")
            .encode(
                x=alt.X("Doctor:N", sort="-y"),
                y=alt.Y("NightShifts:Q", title="Night shifts"),
                tooltip=["Doctor", "NightShifts"]
            )
            .properties(title=f"{title_prefix} – Night shifts", height=300),
            width="stretch"
        )

    with col2:
        st.altair_chart(
            alt.Chart(df)
            .mark_bar(color="#F6BD16")
            .encode(
                x=alt.X("Doctor:N", sort="-y"),
                y=alt.Y("TwentyFourCount:Q", title="24h shifts"),
                tooltip=["Doctor", "TwentyFourCount"]
            )
            .properties(title=f"{title_prefix} – 24h shifts", height=300),
            width="stretch"
        )

def chart_department_load(stats_df: pd.DataFrame, title: str):
    df = stats_df.copy()

    df_melt = df.melt(
        id_vars=["Doctor"],
        value_vars=["WardCount", "ICUCount", "ClinicCount"],
        var_name="Department",
        value_name="Count"
    )

    color_scale = alt.Scale(
        domain=["WardCount", "ICUCount", "ClinicCount"],
        range=["#E45756", "#4C78A8", "#72B7B2"]
    )

    chart = (
        alt.Chart(df_melt)
        .mark_bar()
        .encode(
            x=alt.X("Doctor:N", sort=None),
            y=alt.Y("Count:Q"),
            color=alt.Color("Department:N", scale=color_scale, title="Department"),
            tooltip=["Doctor", "Department", "Count"]
        )
        .properties(title=title, height=350)
    )

    return chart

def chart_preferences(stats_df: pd.DataFrame, title: str):
    df = stats_df.copy()
    df["PreferenceScore"] = df["like_satisfied"] - df["dislike_violated"]

    df_melt = df.melt(
        id_vars=["Doctor"],
        value_vars=["like_satisfied", "dislike_violated"],
        var_name="Type",
        value_name="Count"
    )

    bars = (
        alt.Chart(df_melt)
        .mark_bar()
        .encode(
            x=alt.X("Doctor:N", sort=None),
            y=alt.Y("Count:Q"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["like_satisfied", "dislike_violated"],
                    range=["#4CAF50", "#E53935"]
                ),
                title="Preference"
            ),
            tooltip=["Doctor", "Type", "Count"]
        )
        .properties(height=300)
    )

    score = (
        alt.Chart(df)
        .mark_text(dy=-5)
        .encode(
            x="Doctor:N",
            y="like_satisfied:Q",
            text=alt.Text("PreferenceScore:Q")
        )
    )

    return bars.properties(title=title)


def chart_fairness(stats_df: pd.DataFrame, title: str):
    df = stats_df.copy()

    max_n = int(df["NightShifts"].max())
    min_n = int(df["NightShifts"].min())
    spread = max_n - min_n

    st.write(
        f"Night shift range: **{min_n} – {max_n}**, "
        f"spread = **{spread}**"
    )

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Doctor:N", sort=None),
            y=alt.Y("NightShifts:Q", title="Number of night shifts"),
            tooltip=["Doctor", "NightShifts", "TwentyFourCount"]
        )
    )

    rules = alt.Chart(
        pd.DataFrame({"y": [min_n, max_n]})
    ).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")

    return (bars + rules).properties(height=300, title=title)

def chart_satisfaction_vs_hours(stats_df, title):
    return (
        alt.Chart(stats_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("TotalHours:Q", title="Worked hours"),
            y=alt.Y("S_pct:Q", title="Satisfaction (%)"),
            color=alt.Color("Role:N", legend=None),
            tooltip=["Doctor", "TotalHours", "S_pct"]
        )
        .properties(title=title, height=350)
    )

def chart_satisfaction_components(stats_df, doctor_name):
    row = stats_df[stats_df["Doctor"] == doctor_name]

    comp = row[
        [
            "S_likes",
            "S_dislikes",
            "S_night",
            "S_24h",
            "S_hard",
            "S_weekend",
            "S_overwork",
        ]
    ].melt(var_name="Component", value_name="Score")

    return (
        alt.Chart(comp)
        .mark_bar()
        .encode(
            x=alt.X("Component:N", title="Component"),
            y=alt.Y("Score:Q", title="Contribution to satisfaction"),
            color=alt.condition(
                alt.datum.Score > 0,
                alt.value("#4CAF50"),
                alt.value("#E53935"),
            ),
            tooltip=["Component", "Score"]
        )
        .properties(
            title=f"Satisfaction components – {doctor_name}",
            height=300
        )
    )


def render_optout_overwork_analysis(
    stats_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    doctors_df: pd.DataFrame,
    weekly_limits: dict
):
    st.subheader("Weekly Overwork Analysis (Opt-Out Doctors)")

    if doctors_df is None or doctors_df.empty:
        st.error("Missing doctors_df data – cannot determine opt-out status")
        return

    if schedule_df.empty:
        st.info("No schedule data available")
        return

    name_to_id = {row["name"]: row["id"] for _, row in doctors_df.iterrows()}
    optout_ids = set(doctors_df[doctors_df["opt_out"] == 1]["id"])

    sched = schedule_df.copy()
    sched = sched[sched["Doctor"].notna()]
    sched["Date"] = pd.to_datetime(sched["Date"])

    if sched.empty:
        st.info("No doctor assignments – only slack entries")
        return

    start_date = sched["Date"].min()
    start_monday = start_date - pd.Timedelta(days=start_date.weekday())

    sched["LocalWeek"] = sched["Date"].apply(
        lambda d: int((d - start_monday).days // 7)
    )

    weekly = (
        sched
        .groupby(["Doctor", "LocalWeek"])
        .agg(
            Hours=("Hours", "sum"),
            NightWeek=("ShiftCode", lambda s: s.str.contains("_N_").sum()),
            TwentyFourWeek=("Hours", lambda h: (h == 24).sum())
        )
        .reset_index()
    )

    weekly["DoctorID"] = weekly["Doctor"].map(name_to_id)
    weekly["IsOptOut"] = weekly["DoctorID"].isin(optout_ids)

    weekly = weekly[weekly["IsOptOut"]]

    if weekly.empty:
        st.info("No opt-out doctors in the schedule")
        return

    def weekly_limit_for_row(row):
        d = row["DoctorID"]
        w = row["LocalWeek"]
        return weekly_limits.get(d, {}).get(w, 0)

    weekly["WeeklyLimit"] = weekly.apply(weekly_limit_for_row, axis=1)

    weekly["WeeklyOver90"] = (
        (weekly["WeeklyLimit"] > 0) &
        (weekly["Hours"] >= 0.9 * weekly["WeeklyLimit"])
    )

    weekly["WeeklyMany24h"] = weekly["TwentyFourWeek"] >= 2
    weekly["WeeklyManyNights"] = weekly["NightWeek"] >= 3

    agg = (
        weekly.groupby("Doctor")
        .agg(
            WeeksOver90=("WeeklyOver90", "sum"),
            WeeksMany24=("WeeklyMany24h", "sum"),
            WeeksManyNights=("WeeklyManyNights", "sum"),
            TotalWeeks=("LocalWeek", "nunique"),
        )
        .reset_index()
    )

    agg["OptOutOverworked"] = (
        (agg["WeeksOver90"] == agg["TotalWeeks"]) &
        (
            (agg["WeeksMany24"] >= 2) |
            (agg["WeeksManyNights"] >= 2)
        )
    )

    bar = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("Doctor:N", sort="-y"),
            y=alt.Y("WeeksOver90:Q", title="Weeks exceeding 90% of weekly limit"),
            color=alt.condition(
                "datum.OptOutOverworked",
                alt.value("crimson"),
                alt.value("steelblue")
            ),
            tooltip=[
                "Doctor",
                "WeeksOver90",
                "WeeksMany24",
                "WeeksManyNights",
                "TotalWeeks",
                "OptOutOverworked"
            ]
        )
        .properties(height=320)
    )

    st.altair_chart(bar, width="stretch")

    st.subheader("Opt-out doctors requiring intervention")

    critical = agg[agg["OptOutOverworked"]]

    if critical.empty:
        st.success("No opt-out doctors classified as overworked")
    else:
        st.dataframe(
            critical[
                [
                    "Doctor",
                    "WeeksOver90",
                    "WeeksMany24",
                    "WeeksManyNights",
                    "TotalWeeks"
                ]
            ],
            width="stretch"
        )



def render_absences(
    doctors: pd.DataFrame,
    shifts: pd.DataFrame,
    unavail_day: pd.DataFrame,
    unavail_shift: pd.DataFrame
):
    st.header("Doctor Unavailability")

    id_to_name = {row["id"]: row["name"] for _, row in doctors.iterrows()}
    code_to_id = {row["code"]: row["id"] for _, row in shifts.iterrows()}

    st.subheader("Unavailability Overview")

    un_day_vis = unavail_day.copy()
    un_day_vis["Type"] = "DAY"
    un_day_vis["Shift"] = un_day_vis["date"]
    un_day_vis["Doctor"] = un_day_vis["doctor_id"].map(id_to_name)

    un_shift_vis = unavail_shift.copy()
    un_shift_vis["Type"] = "SHIFT"
    un_shift_vis["Shift"] = un_shift_vis["code"]
    un_shift_vis["Doctor"] = un_shift_vis["doctor_id"].map(id_to_name)

    un_all = pd.concat(
        [
            un_day_vis[["Doctor", "Shift", "Type"]],
            un_shift_vis[["Doctor", "Shift", "Type"]],
        ],
        ignore_index=True,
    )

    heatmap_un = (
        alt.Chart(un_all)
        .mark_rect()
        .encode(
            y=alt.Y("Doctor:N", sort=None, title="Doctor"),
            x=alt.X("Shift:N", title="Day / Shift"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["DAY", "SHIFT"],
                    range=["#E53935", "#FFB300"],
                ),
                title="Unavailability type",
            ),
            tooltip=["Doctor", "Shift", "Type"],
        )
        .properties(height=350)
    )

    st.altair_chart(heatmap_un, width="stretch")

    st.header("Daily unavailability")

    all_dates = sorted(shifts["date"].unique())

    grid = []
    for _, doc in doctors.iterrows():
        for d in all_dates:
            grid.append(
                {
                    "Doctor": doc["name"],
                    "DoctorID": doc["id"],
                    "Date": d,
                    "Absent": 0,
                }
            )

    day_df = pd.DataFrame(grid)

    for _, row in unavail_day.iterrows():
        day_df.loc[
            (day_df["DoctorID"] == row["doctor_id"])
            & (day_df["Date"] == row["date"]),
            "Absent",
        ] = 1

    heatmap_days = (
        alt.Chart(day_df)
        .mark_rect()
        .encode(
            y=alt.Y("Doctor:N", title="Doctor"),
            x=alt.X("Date:N", title="Date", sort=all_dates),
            color=alt.Color(
                "Absent:N",
                scale=alt.Scale(domain=[0, 1], range=["#6ABF4B", "#E53935"]),
                title="Absent?",
            ),
            tooltip=["Doctor", "Date", "Absent"],
        )
        .properties(height=350)
    )

    st.altair_chart(heatmap_days, width="stretch")

    st.header("Shift-level unavailability")

    with st.expander("Show details (day / shift)", expanded=False):

        selected_date = st.selectbox(
            "Select day",
            all_dates,
            format_func=lambda d: str(d)
        )

        shifts_today = shifts[shifts["date"] == selected_date]

        grid = []
        for _, doc in doctors.iterrows():
            for _, sh in shifts_today.iterrows():
                grid.append(
                    {
                        "Doctor": doc["name"],
                        "DoctorID": doc["id"],
                        "ShiftID": sh["id"],
                        "ShiftLabel": sh["code"],
                        "Status": "Available",
                    }
                )

        df_day = pd.DataFrame(grid)

        absent_full = unavail_day[
            unavail_day["date"] == selected_date
            ]["doctor_id"].tolist()

        df_day.loc[
            df_day["DoctorID"].isin(absent_full),
            "Status"
        ] = "DayAbsent"

        un_shift = unavail_shift.copy()
        un_shift["ShiftID"] = un_shift["code"].map(code_to_id)

        un_shift_today = un_shift[
            un_shift["code"].isin(shifts_today["code"])
        ]

        for _, row in un_shift_today.iterrows():
            df_day.loc[
                (df_day["DoctorID"] == row["doctor_id"])
                & (df_day["ShiftID"] == row["ShiftID"]),
                "Status",
            ] = "ShiftAbsent"

        heatmap_shifts = (
            alt.Chart(df_day)
            .mark_rect()
            .encode(
                y=alt.Y("Doctor:N", title="Doctor"),
                x=alt.X("ShiftLabel:N", title="Shift"),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Available", "ShiftAbsent", "DayAbsent"],
                        range=["#6ABF4B", "#E53935", "#FFB300"],
                    ),
                    title="Status",
                ),
                tooltip=["Doctor", "ShiftLabel", "Status"],
            )
            .properties(height=300)
        )

        st.altair_chart(heatmap_shifts, width="stretch")




st.set_page_config(
    page_title="Hospital Rostering – Lexicographic Optimization",
    layout="wide"
)

st.title("Hospital Rostering – Lexicographic Optimization")

st.markdown("""
Comparison of two scenarios:
- **Baseline** – without hiring new doctors
- **Hire allowed** – hiring candidates is permitted
""")

@st.cache_data(show_spinner=False)
def run_model():
    return run_model_and_get_results()

with st.spinner("Running optimization model (this may take a few minutes)..."):
    results = run_model()

    baseline = results["baseline"]
    reserve_df = pd.read_csv("data/reserve/reserve_doctors.csv")
    candidates_df = pd.read_csv("data/doctors_to_hire/doctors_to_hire.csv")

    replacements = precompute_replacements(
        results=baseline,
        reserve_doctors_df=reserve_df,
        candidates_df=candidates_df
    )

current_run = results.get("run")

baseline = results["baseline"]
hire_allowed = results.get("hire_allowed")

RESULTS_DIR = ROOT_DIR / "results" / "mar_2026"

if RESULTS_DIR.exists():
    try:
        pareto_source_df = load_all_summaries(RESULTS_DIR)

        if current_run is not None and not pareto_source_df.empty:
            pareto_source_df["run"] = pareto_source_df["run"].astype(str).str.strip()
            current_run = str(current_run).strip()

            pareto_source_df["is_current"] = (
                pareto_source_df["run"].str.contains(current_run)
            )
        else:
            pareto_source_df["is_current"] = False

        if current_run is not None:
            run_export_dir = EXPORT_ROOT / str(current_run)
            run_export_dir.mkdir(exist_ok=True)
        else:
            run_export_dir = EXPORT_ROOT / "manual_run"
            run_export_dir.mkdir(exist_ok=True)
        export_flag_file = run_export_dir / "_export_done.txt"


        required_cols = {"total_cost", "avg_satisfaction", "hospital_efficiency"}

        if not required_cols.issubset(pareto_source_df.columns):
            st.warning(
                "Summary files do not contain the required columns "
                "(total_cost, avg_satisfaction, hospital_efficiency)."
            )
            pareto_df = pd.DataFrame()

        if len(pareto_source_df) >= 2:
            pareto_df = pareto_front(pareto_source_df)
        else:
            pareto_df = pd.DataFrame()

    except Exception as e:
        pareto_source_df = pd.DataFrame()
        pareto_df = pd.DataFrame()
        st.warning(f"Pareto analysis not available: {e}")
else:
    pareto_source_df = pd.DataFrame()
    pareto_df = pd.DataFrame()

baseline_ui = filter_active_doctors(baseline)

if hire_allowed is not None:
    hire_ui = filter_active_doctors(hire_allowed)

scenarios = {"Baseline (no hiring)": baseline_ui}
if hire_allowed is not None:
    scenarios["Hire allowed"] = hire_ui


baseline_metrics = compute_global_metrics(scenarios["Baseline (no hiring)"]["schedule_df"])
if "Hire allowed" in scenarios:
    hire_metrics = compute_global_metrics(scenarios["Hire allowed"]["schedule_df"])

st.header("Key Metrics")

baseline_eff = compute_hospital_efficiency_v2(baseline)
if hire_allowed is not None:
    hire_eff = compute_hospital_efficiency_v2(hire_allowed)


def components_df(eff):
    return (
        pd.DataFrame.from_dict(
            eff["components"],
            orient="index",
            columns=["Score (0–1)"]
        )
        .assign(Score_pct=lambda df: (df["Score (0–1)"] * 100).round(1))
    )

st.metric("Hospital Efficiency (Baseline)", baseline_eff["HospitalEfficiencyScore"])


st.metric(
    "Hospital Efficiency (Baseline)",
    baseline_eff["HospitalEfficiencyScore"]
)

st.subheader("Efficiency Components – Baseline")
st.dataframe(components_df(baseline_eff), width="stretch")


if hire_allowed is not None:
    st.metric(
        "Hospital Efficiency (Hiring allowed)",
        hire_eff["HospitalEfficiencyScore"],
        delta=f"{hire_eff['HospitalEfficiencyScore'] - baseline_eff['HospitalEfficiencyScore']:+.1f}"
    )

    st.subheader("Efficiency Components – Hiring allowed")
    st.dataframe(components_df(hire_eff), width="stretch")
else:
    st.info("All shifts were covered without the need for additional hiring.")


if hire_allowed is None:
    col1 = st.container()
    with col1:
        st.subheader("Without hiring new doctors")
        st.metric("Objective", baseline["objective"])
        st.metric("Total slack", baseline["slack"])
        st.metric("Doctors hired", len(baseline["hired_df"]))

        st.divider()
        st.metric("Days with shortages", baseline_metrics["DaysWithMissing"])
        st.metric("Uncovered shifts", baseline_metrics["MissingShifts"])
        st.metric("Fully covered shifts", baseline_metrics["FullyCoveredShifts"])

        st.metric(
            "Total labor cost",
            f"{baseline['total_cost']:,} PLN"
        )
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Without hiring new doctors")
        st.metric("Objective", baseline["objective"])
        st.metric("Total slack", baseline["slack"])
        st.metric("Doctors hired", len(baseline["hired_df"]))

        st.divider()
        st.metric("Days with shortages", baseline_metrics["DaysWithMissing"])
        st.metric("Uncovered shifts", baseline_metrics["MissingShifts"])
        st.metric("Fully covered shifts", baseline_metrics["FullyCoveredShifts"])

        st.metric(
            "Total labor cost",
            f"{baseline['total_cost']:,} PLN"
        )

    with col2:
        st.subheader("Hiring allowed")
        st.metric("Objective", hire_allowed["objective"])
        st.metric("Total slack", hire_allowed["slack"])
        st.metric("Doctors hired", len(hire_allowed["hired_df"]))

        st.divider()
        st.metric(
            "Days with shortages",
            hire_metrics["DaysWithMissing"],
            delta=compute_delta(baseline_metrics, hire_metrics, "DaysWithMissing")
        )
        st.metric(
            "Missing shifts",
            hire_metrics["MissingShifts"],
            delta=compute_delta(baseline_metrics, hire_metrics, "MissingShifts")
        )
        st.metric(
            "Fully covered shifts",
            hire_metrics["FullyCoveredShifts"],
            delta=compute_delta(baseline_metrics, hire_metrics, "FullyCoveredShifts")
        )

        st.metric(
            "Cost difference",
            f"{hire_allowed['total_cost'] - baseline['total_cost']:+,} PLN"
        )

st.header("Scenario Comparison")

summary_rows = []
for name, data in scenarios.items():
    summary_rows.append({
        "Scenario": name,
        "Objective": data["objective"],
        "Slack": data["slack"],
        "Hired doctors": len(data["hired_df"]),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, width="stretch")

st.header("Cost vs Satisfaction")

if pareto_source_df.empty or pareto_df.empty:
    st.info("Insufficient data to compute the Pareto front.")
else:
    st.caption(
        "Each point represents one experimental run. "
        "The Pareto front highlights non-dominated solutions "
        "(lower cost, higher satisfaction)."
    )

    base = (
        alt.Chart(pareto_source_df)
        .mark_circle(size=80, opacity=0.35)
        .encode(
            x=alt.X("total_cost:Q", title="Total cost"),
            y=alt.Y("avg_satisfaction:Q", title="Average satisfaction (%)"),
            color=alt.Color("allow_hire:N", title="Hiring allowed"),
            tooltip=["run", "total_cost", "avg_satisfaction", "allow_hire"]
        )
    )

    pareto_points = (
        alt.Chart(pareto_df)
        .mark_circle(size=150, color="crimson")
        .encode(
            x="total_cost:Q",
            y="avg_satisfaction:Q",
            tooltip=["run", "total_cost", "avg_satisfaction"]
        )
    )

    current_point = (
        alt.Chart(pareto_source_df[pareto_source_df["is_current"]])
        .mark_circle(size=320, color="green")
        .encode(
            x="total_cost:Q",
            y="avg_satisfaction:Q"
        )
    )

    pareto_line_df = pareto_df.sort_values("total_cost").copy()
    pareto_line = (
        alt.Chart(pareto_line_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("total_cost:Q", title="Total cost"),
            y=alt.Y("avg_satisfaction:Q", title="Average satisfaction (%)")
        )
    )

    st.altair_chart((base + pareto_line + pareto_points + current_point).properties(height=400), width="stretch")

    cost_sat_chart = (base + pareto_line + pareto_points + current_point)

st.header("Satisfaction vs Efficiency")

df2 = pareto_source_df.dropna(subset=["avg_satisfaction", "hospital_efficiency", "total_cost"])

chart_se = (
    alt.Chart(df2)
    .mark_circle(size=90, opacity=0.6)
    .encode(
        x=alt.X("avg_satisfaction:Q", title="Average satisfaction (%)"),
        y=alt.Y("hospital_efficiency:Q", title="Hospital efficiency (0–100)"),
        color=alt.Color("allow_hire:N", title="Hire allowed"),
        tooltip=["run", "avg_satisfaction", "hospital_efficiency", "total_cost", "allow_hire"]
    )
    .properties(height=420)
)

current_point_se = (
    alt.Chart(df2[df2["is_current"]])
    .mark_circle(size=320, color="green")
    .encode(
        x="avg_satisfaction:Q",
        y="hospital_efficiency:Q"
    )
)

pareto_se = pareto_front_2d(df2, "avg_satisfaction", "hospital_efficiency", minimize_x=False, maximize_y=True)
pareto_se_line_df = pareto_se.sort_values(["avg_satisfaction", "hospital_efficiency"]).copy()

frontier_se = (
    alt.Chart(pareto_se_line_df)
    .mark_line(strokeWidth=3)  # bez stepów
    .encode(x="avg_satisfaction:Q", y="hospital_efficiency:Q")
)

pareto_se_points = (
    alt.Chart(pareto_se)
    .mark_circle(size=150, color="crimson")
    .encode(x="avg_satisfaction:Q", y="hospital_efficiency:Q", tooltip=["run","avg_satisfaction","hospital_efficiency"])
)

st.altair_chart((chart_se + frontier_se + pareto_se_points + current_point_se).properties(height=420),
                width="stretch")
frontier_se_chart = (chart_se + frontier_se + pareto_se_points + current_point_se)


st.header("Cost vs Efficiency")

chart_ce = (
    alt.Chart(df2)
    .mark_circle(size=90, opacity=0.6)
    .encode(
        x=alt.X("total_cost:Q", title="Total cost"),
        y=alt.Y("hospital_efficiency:Q", title="Hospital efficiency (0–100)"),
        color=alt.Color("allow_hire:N", title="Hire allowed"),
        tooltip=["run", "total_cost", "hospital_efficiency", "avg_satisfaction", "allow_hire"]
    )
    .properties(height=420)
)

current_point_ce = (
    alt.Chart(df2[df2["is_current"]])
    .mark_circle(size=320, color="green")
    .encode(
        x="total_cost:Q",
        y="hospital_efficiency:Q"
    )
)

pareto_ce = pareto_front_2d(df2, "total_cost", "hospital_efficiency", minimize_x=True, maximize_y=True)
pareto_ce_line_df = pareto_ce.sort_values(["total_cost", "hospital_efficiency"]).copy()

frontier_ce = (
    alt.Chart(pareto_ce_line_df)
    .mark_line(strokeWidth=3, interpolate="step-after")
    .encode(x="total_cost:Q", y="hospital_efficiency:Q")
)

pareto_ce_points = (
    alt.Chart(pareto_ce)
    .mark_circle(size=150, color="crimson")
    .encode(x="total_cost:Q", y="hospital_efficiency:Q", tooltip=["run","total_cost","hospital_efficiency"])
)

st.altair_chart((chart_ce + frontier_ce + pareto_ce_points + current_point_ce).properties(height=420),
                width="stretch")
frontier_ce_chart = (chart_ce + frontier_ce + pareto_ce_points + current_point_ce)


st.header("3D: Cost vs Satisfaction vs Efficiency")

df3 = pareto_source_df.dropna(subset=["total_cost", "avg_satisfaction", "hospital_efficiency"])

fig3d = px.scatter_3d(
    df3,
    x="total_cost",
    y="avg_satisfaction",
    z="hospital_efficiency",
    color="allow_hire",
    hover_name="run",
    hover_data={
        "total_cost": True,
        "avg_satisfaction": True,
        "hospital_efficiency": True,
        "allow_hire": True,
    },
)

fig3d.update_layout(
    height=650,
    scene=dict(
        xaxis_title="Total cost",
        yaxis_title="Avg satisfaction (%)",
        zaxis_title="Efficiency (0–100)",
    ),
    margin=dict(l=0, r=0, b=0, t=30),
)

df3_current = df3[df3["run"] == current_run]
if not df3_current.empty:
    fig3d.add_scatter3d(
        x=df3_current["total_cost"],
        y=df3_current["avg_satisfaction"],
        z=df3_current["hospital_efficiency"],
        mode="markers+text",
        text=["Current run"] * len(df3_current),
        textposition="top center",
        marker=dict(size=10, color="green"),
        showlegend=False
    )

st.plotly_chart(fig3d, use_container_width=True)

if not export_flag_file.exists():

    # save(cost_sat_chart, run_export_dir / "fig1_cost_vs_satisfaction.svg")
    # save(frontier_ce_chart, run_export_dir / "fig2_cost_vs_efficiency.svg")
    # save(frontier_se_chart, run_export_dir / "fig3_satisfaction_vs_efficiency.svg")
    cost_sat_chart.save(run_export_dir / "fig1_cost_vs_satisfaction.svg")
    frontier_ce_chart.save(run_export_dir / "fig2_cost_vs_efficiency.svg")
    frontier_se_chart.save(run_export_dir / "fig3_satisfaction_vs_efficiency.svg")
    fig3d.write_image(run_export_dir / "fig4_3d.png", scale=3)
    pareto_source_df.to_csv(run_export_dir / "all_runs_snapshot.csv", index=False)
    pareto_df.to_csv(run_export_dir / "pareto_front.csv", index=False)
    with open(run_export_dir / "run_summary.txt", "w") as f:
        f.write(f"Run: {current_run}\n")
        f.write(f"Total cost: {baseline['total_cost']}\n")
        f.write(f"Avg satisfaction: {baseline['stats_df']['S_pct'].mean():.2f}\n")
        f.write(f"Efficiency: {baseline_eff['HospitalEfficiencyScore']}\n")

    export_flag_file.write_text("export completed")

    st.info(f"Figures exported to {run_export_dir}")

st.header("Hired Doctors")

if hire_allowed is None:
    st.info("The hiring scenario was not executed, as the baseline solution covers all shifts.")
else:
    if hire_allowed["hired_df"].empty:
        st.info("No doctors were hired.")
    else:
        st.dataframe(hire_allowed["hired_df"], width="stretch")


st.header("Results")

tabs = st.tabs(list(scenarios.keys()))
for tab, (name, data) in zip(tabs, scenarios.items()):
    with tab:
        st.dataframe(data["stats_df"], width="stretch")

st.header("Calendar View")

scenario = st.radio(
    "Select scenario",
    list(scenarios.keys()),
    horizontal=True
)

cal_df = scenarios[scenario]["schedule_df"]

cal_df["Date"] = cal_df["Date"].dt.strftime("%Y-%m-%d")
events = schedule_to_calendar_events(cal_df)

calendar_options = {
    "initialView": "dayGridMonth",
    "initialDate": str(pd.to_datetime(cal_df["Date"]).min().date()),
    "height": 700,
}

calendar(
    events=events,
    options=calendar_options
)

st.header("Doctor Statistics")

tabs = st.tabs(list(scenarios.keys()))

for tab, (name, data) in zip(tabs, scenarios.items()):
    with tab:
        st.subheader(f"{name} – Workload & Duties")

        st.altair_chart(
            chart_work_hours(data["stats_df"], f"{name}: Worked hours vs limit"),
            width="stretch"
        )

        weekly_df = compute_weekly_workload(data["schedule_df"])
        if weekly_df.empty:
            st.info("No weekly data available")
            continue

        available_weeks = sorted(weekly_df["LocalWeek"].unique())
        selected_week = st.selectbox(
            "Select week",
            available_weeks,
            format_func=lambda w: f"Week {w + 1}",
            key=f"{name}_week_selector"
        )

        st.altair_chart(
            chart_weekly_work_hours(
                weekly_df,
                data["weekly_limits"],
                data["doctors"],
                selected_week,
                title_prefix=name
            ),
            width="stretch"
        )

        chart_night_and_24h(data["stats_df"], name)
        st.altair_chart(chart_department_load(data["stats_df"], f"{name}: Department workload"), width="stretch")
        st.altair_chart(chart_preferences(data["stats_df"], f"{name}: Doctor preferences"), width="stretch")
        st.altair_chart(chart_fairness(data["stats_df"], f"{name}: Fairness"), width="stretch")

        render_optout_overwork_analysis(
            data["stats_df"],
            data["schedule_df"],
            data["doctors"],
            data["weekly_limits"]
        )

        render_absences(
            data["doctors"],
            data["shifts"],
            data["unavail_day"],
            data["unavail_shift"],
        )

st.header("Doctor Satisfaction")

cols = st.columns(len(scenarios))

for col, (name, data) in zip(cols, scenarios.items()):
    with col:
        st.subheader(name)

        st.metric(
            "Average satisfaction (%)",
            f"{data['stats_df']['S_pct'].mean():.1f}%"
        )

        st.metric(
            "Minimum satisfaction (%)",
            f"{data['stats_df']['S_pct'].min():.1f}%"
        )

        rank_df = (
            data["stats_df"]
            .sort_values("S_pct", ascending=False)
            [["Doctor", "Role", "S_pct", "TotalHours"]]
        )

        st.dataframe(rank_df, width="stretch")

        st.altair_chart(
            chart_satisfaction_vs_hours(
                data["stats_df"],
                f"Satisfaction vs workload ({name})"
            ),
            width="stretch"
        )

        selected_doc = st.selectbox(
            "Select doctor",
            data["stats_df"]["Doctor"].tolist(),
            key=f"sat_{name}"
        )

        st.altair_chart(
            chart_satisfaction_components(
                data["stats_df"],
                selected_doc
            ),
            width="stretch"
        )

with st.expander("Solver statistics"):
    for name, data in scenarios.items():
        st.subheader(f"{name} solver stats")
        st.dataframe(data["solver_stats_df"], width="stretch")


st.header("Emergency Replacement Analysis")

replacement_keys = list(replacements.keys())

selected_key = st.selectbox(
    "Select shift + doctor (simulated sickness)",
    replacement_keys
)

res = replacements[selected_key]

info = res["shift_info"]

st.markdown(f"""
### Shift details
- **Date:** {info.get('date')}
- **Department:** {info.get('dept')}
- **Hours:** {info.get('start')}–{info.get('end')} ({info.get('hours')}h)
- **Required skill:** {info.get('required_skill', '—')}
""")

if res["fallback_message"]:
    st.warning(res["fallback_message"])

candidates_df = pd.DataFrame(res["candidates"])

if not candidates_df.empty:

    # === STATUS COLUMN ===
    def get_status(r):
        if r["source"] == "reserve":
            return "Reserve (call required)"

        # standardowa logika
        if r["rest_ok"] and r["skill_ok"]:
            return "OK"

        return "Violation"


    candidates_df["status"] = candidates_df.apply(get_status, axis=1)

    # === SORT ===
    candidates_df = candidates_df.sort_values("score")

    st.subheader("Candidates ranking")

    st.dataframe(
        candidates_df[
            [
                "name",
                "source",
                "score",
                "extra_cost",
                "workload_pct",
                "hard_shifts",
                "status"
            ]
        ],
        width="stretch"
    )

    # === BAR CHART SCORE ===
    st.subheader("Candidate scores")

    chart = (
        alt.Chart(candidates_df.head(10))
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Score (lower is better)"),
            y=alt.Y("name:N", sort="-x"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=["OK", "Violation", "Reserve (call required)"],
                    range=["#2ecc71", "#e74c3c", "#f39c12"]
                )
            ),
            tooltip=["name", "score", "extra_cost"]
        )
        .properties(height=350)
    )

    st.altair_chart(chart, width="stretch")

    # === BEST ===
    if res["recommendation"]:
        best = res["recommendation"]

        msg = (
            f"Recommended: **{best['name']}** "
            f"({best['source']}) | cost +{best['extra_cost']} PLN"
        )

        if best["source"] == "reserve":
            st.warning("⚠ Reserve doctor selected — confirmation required")

        st.success(msg)

        if best["warnings"]:
            for w in best["warnings"]:
                st.warning(w)

    # === WARNINGS EXPANDER ===
    with st.expander("⚠ Candidate warnings"):
        for c in res["candidates"]:
            if c["warnings"]:
                st.write(f"**{c['name']}**:")
                for w in c["warnings"]:
                    st.write(f"- {w}")


if "score_components" in candidates_df.columns:
    st.subheader("Score breakdown")

    comp_rows = []
    for _, row in candidates_df.head(5).iterrows():
        sc = row.get("score_components", {})
        if isinstance(sc, dict):
            for k, v in sc.items():
                comp_rows.append({
                    "Doctor": row["name"],
                    "Component": k,
                    "Value": v
                })

    comp_df = pd.DataFrame(comp_rows)

    if not comp_df.empty:
        chart2 = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x="Doctor:N",
                y="Value:Q",
                color="Component:N",
                tooltip=["Doctor", "Component", "Value"]
            )
            .properties(height=350)
        )

        st.altair_chart(chart2, width="stretch")


st.success("Optimization finished successfully.")
