import streamlit as st
import sys
import os
import altair as alt
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from model.cp_sat_model import run_model_and_get_results, run_with_one_extra_doctor


st.title("HARMONOGRAM DYŻURÓW")

# === READ DATA FILES ===
doctors = pd.read_csv(os.path.join(project_root, "data/doctors/doctors2.csv"))
shifts = pd.read_csv(os.path.join(project_root, "data/shifts/shifts3.csv"))
unavail_day = pd.read_csv(os.path.join(project_root, "data/unavailabilities_day/unavailabilities_day_2.csv"))
unavail_shift = pd.read_csv(os.path.join(project_root, "data/unavailabilities_shift/unavailabilities_shift.csv"))


# === DAYS OF WEEK ===
POLISH_DAYS = {
    "Mon": "Poniedziałek",
    "Tue": "Wtorek",
    "Wed": "Środa",
    "Thu": "Czwartek",
    "Fri": "Piątek",
    "Sat": "Sobota",
    "Sun": "Niedziela"
}

# === FUNCTION TO SHOW RESULTS AND STATISTICS ===
def render_all_charts(stats_df, solver_stats_df, doctors, shifts, unavail_day, unavail_shift, title_prefix=""):
    # === Godziny pracy na osobę ===
    st.subheader("Liczba godzin pracy na lekarza")

    df_hours = stats_df[["Doctor", "TotalHours", "MaxHours"]]
    bars = (
        alt.Chart(df_hours)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("Doctor", sort="-y"),
            y=alt.Y("TotalHours", title="Godziny pracy"),
            tooltip=["Doctor", "TotalHours", "MaxHours"]
        )
    )
    limits = (
        alt.Chart(df_hours)
        .mark_rule(color="pink", strokeWidth=2)
        .encode(
            x="Doctor",
            y="MaxHours",
            tooltip=["Doctor", "MaxHours"]
        )
    )
    chart = (bars + limits).properties(height=350)
    st.altair_chart(chart, use_container_width=True)


    # ===== LICZBA DYŻURÓW NOCNYCH =====
    print("Liczba dyżurów nocnych (nocne + 24h)")
    df_night_shifts = stats_df[["Doctor", "NightShifts"]]
    chart_shifts = (
        alt.Chart(df_night_shifts)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("Doctor", sort="-y"),
            y=alt.Y("NightShifts", title="Liczba zmian nocnych"),
            tooltip=["Doctor", "NightShifts"]
        )
    )
    st.altair_chart(chart_shifts, use_container_width=True)

    # ===== LICZBA ZMIAN 24H =====
    print("Liczba zmian 24-godzinnych")
    df_24_shifts = stats_df[["Doctor", "TwentyFourCount"]]
    chart_24_shifts = (
        alt.Chart(df_24_shifts)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("Doctor", sort="-y"),
            y=alt.Y("TwentyFourCount", title="Liczba zmian 24-godzinnych"),
            tooltip=["Doctor", "TwentyFourCount"]
        )
    )
    st.altair_chart(chart_24_shifts, use_container_width=True)

    # ===== OBCIĄŻENIE ODDZIAŁ / ICU / PORADNIA =====
    st.subheader("D. Obciążenie pracą (oddział / OIOM / poradnia)")

    df_load = stats_df[["Doctor", "WardCount", "ICUCount", "ClinicCount"]]

    df_melt = df_load.melt(id_vars=["Doctor"],
                           value_vars=["WardCount", "ICUCount", "ClinicCount"],
                           var_name="Department",
                           value_name="Count")
    color_scale = alt.Scale(
        domain=["WardCount", "ICUCount", "ClinicCount"],
        range=["#df7680", "#008a80", "#68295c"]
    )
    stacked = (
        alt.Chart(df_melt)
        .mark_bar()
        .encode(
            x=alt.X("Doctor", sort=None),
            y="Count",
            color=alt.Color("Department", scale=color_scale, title="Oddział"),
            tooltip=["Doctor", "Department", "Count"]
        )
        .properties(height=350)
    )

    st.altair_chart(stacked, use_container_width=True)

    # ===== SPEŁNIONE PREFERENCJE =====
    st.subheader("Preferencje lekarzy — spełnione like / naruszone dislike")

    df_pref = stats_df[["Doctor", "like_satisfied", "dislike_violated"]]

    df_pref_melt = df_pref.melt(
        id_vars=["Doctor"],
        value_vars=["like_satisfied", "dislike_violated"],
        var_name="PreferenceType",
        value_name="Count"
    )

    pref_chart = (
        alt.Chart(df_pref_melt)
        .mark_bar()
        .encode(
            x=alt.X("Doctor:N", sort=None),
            y=alt.Y("Count:Q"),
            color=alt.Color("PreferenceType:N",
                            scale=alt.Scale(
                                domain=["like_satisfied", "dislike_violated"],
                                range=["#4CAF50", "#E53935"]  # zielony / czerwony
                            )),
            column=alt.Column("PreferenceType:N", header=alt.Header(title="")),
            tooltip=["Doctor", "PreferenceType", "Count"]
        )
        .properties(height=300)
    )

    st.altair_chart(pref_chart, use_container_width=True)


    # ===== FAIRNESS =====
    st.header("Fairness")

    st.subheader("Dyżury nocne – rozkład na lekarzy")

    max_nights = int(solver_stats_df["max_nights"].iloc[0])
    min_nights = int(solver_stats_df["min_nights"].iloc[0])
    spread = int(solver_stats_df["spread"].iloc[0])


    st.write(
        f"Zakres dyżurów nocnych: **{min_nights}–{max_nights}**, "
        f"różnica (spread) = **{spread}**"
    )

    base = alt.Chart(stats_df).encode(
        x=alt.X("Doctor:N", sort=None)
    )

    bars = base.mark_bar().encode(
        y=alt.Y("NightShifts:Q", title="Liczba dyżurów nocnych"),
        tooltip=["Doctor", "NightShifts", "TwentyFourCount"]
    )

    rules_df = pd.DataFrame({
        "y": [min_nights, max_nights],
        "label": ["min_nights", "max_nights"]
    })

    rules = (
        alt.Chart(rules_df)
        .mark_rule(strokeDash=[4, 4])
        .encode(
            y="y:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["min_nights", "max_nights"],
                    range=["#4CAF50", "#E53935"]
                ),
                title="Granice"
            )
        )
    )

    fairness_chart = (bars + rules).properties(height=300)

    st.altair_chart(fairness_chart, use_container_width=True)

    # ===== POTENCJALNIE PRZEPRACOWANI =====
    st.subheader("Lekarze potencjalnie przepracowani")

    cols = ["Doctor", "TotalHours", "MaxHours", "NightShifts", "TwentyFourCount"]
    problematic = stats_df.copy()
    problematic["OverLimit90"] = (
        (problematic["MaxHours"] > 48) &
        (problematic["TotalHours"] >= 0.9 * problematic["MaxHours"])
    )
    problematic["Many24h"] = problematic["TwentyFourCount"] >= 2
    problematic["ManyNights"] = problematic["NightShifts"] >= 3

    problematic["AnyIssue"] = (
        problematic["OverLimit90"] |
        problematic["Many24h"] |
        problematic["ManyNights"]
    )

    bar = (
        alt.Chart(problematic)
        .mark_bar()
        .encode(
            x=alt.X("Doctor:N", sort="-y"),
            y=alt.Y("TotalHours:Q", title="Liczba godzin w tygodniu"),
            color=alt.condition(
                "datum.AnyIssue",
                alt.value("crimson"),
                alt.value("steelblue"),
            ),
            tooltip=[
                "Doctor",
                "TotalHours",
                "MaxHours",
                "NightShifts",
                "TwentyFourCount",
                "OverLimit90",
                "Many24h",
                "ManyNights",
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(bar, use_container_width=True)

    risk_df = problematic.melt(
        id_vars=["Doctor"],
        value_vars=["OverLimit90", "Many24h", "ManyNights"],
        var_name="RiskType",
        value_name="Flag",
    )

    risk_df = risk_df[risk_df["Flag"]]

    heatmap = (
        alt.Chart(risk_df)
        .mark_rect(stroke='white', strokeWidth=2)
        .encode(
            y=alt.Y("Doctor:N", sort=None),
            x=alt.X("RiskType:N", title="Typ problemu"),
            color=alt.Color(
                "RiskType:N",
                title="Typ problemu",
                scale=alt.Scale(
                    domain=["OverLimit90", "Many24h", "ManyNights"],
                    range=["green", "purple", "yellow"]
                ),
            ),
            tooltip=["Doctor", "RiskType"],
        )
        .properties(height=300)
    )

    st.subheader("Mapa problemów (kto ma jaki typ obciążenia)")
    st.altair_chart(heatmap, use_container_width=True)

    st.subheader("Tabela – szczegóły dla lekarzy z co najmniej jednym problemem")

    problematic_filtered = problematic[problematic["AnyIssue"]]

    st.dataframe(
        problematic_filtered[
            ["Doctor", "TotalHours", "MaxHours",
             "NightShifts", "TwentyFourCount",
             "OverLimit90", "Many24h", "ManyNights"]
        ].sort_values(by="TotalHours", ascending=False)
    )


    st.header("Nieobecności lekarzy")

    id_to_name = {row["id"]: row["name"] for _, row in doctors.iterrows()}
    code_to_id = {row["code"]: row["id"] for _, row in shifts.iterrows()}

    st.subheader("Grafik nieobecności lekarzy")

    # ===== Przygotowanie danych =====

    # Mapowanie day → kolejność dni
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # 1. Nieobecności całodniowe
    un_day_vis = unavail_day.copy()
    un_day_vis["Type"] = "DAY"
    un_day_vis["Shift"] = un_day_vis["day"]
    un_day_vis["Doctor"] = un_day_vis["doctor_id"].map(id_to_name)

    # 2. Nieobecności na konkretne zmiany
    un_shift_vis = unavail_shift.copy()
    un_shift_vis["Type"] = "SHIFT"
    un_shift_vis["Shift"] = un_shift_vis["code"]
    un_shift_vis["Doctor"] = un_shift_vis["doctor_id"].map(id_to_name)

    # Połączenie
    un_all = pd.concat([un_day_vis[["Doctor", "Shift", "Type"]],
                        un_shift_vis[["Doctor", "Shift", "Type"]]],
                       ignore_index=True)

    heatmap_un = (
        alt.Chart(un_all)
        .mark_rect()
        .encode(
            y=alt.Y("Doctor:N", sort=None, title="Lekarz"),
            x=alt.X("Shift:N", sort=day_order, title="Dzień / Zmiana"),
            color=alt.Color("Type:N",
                            scale=alt.Scale(
                                domain=["DAY", "SHIFT"],
                                range=["#E53935", "#FFB300"]
                            ),
                            title="Rodzaj nieobecności"),
            tooltip=["Doctor", "Shift", "Type"]
        )
        .properties(height=350)
    )

    st.altair_chart(heatmap_un, use_container_width=True)

    st.header("Nieobecności — poziom dzienny")

    id_to_name = {row["id"]: row["name"] for _, row in doctors.iterrows()}

    # Przygotowanie danych dziennych
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    day_grid = []
    for _, doc in doctors.iterrows():
        for day in days:
            day_grid.append({
                "Doctor": doc["name"],
                "DoctorID": doc["id"],
                "Day": day,
                "Absent": 0
            })

    day_df = pd.DataFrame(day_grid)

    # Nieobecności całodniowe
    for _, row in unavail_day.iterrows():
        doc = row["doctor_id"]
        day = row["day"]
        day_df.loc[(day_df["DoctorID"] == doc) & (day_df["Day"] == day), "Absent"] = 1

    # Heatmapa — dni
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    heatmap_days = (
        alt.Chart(day_df)
        .mark_rect()
        .encode(
            y=alt.Y("Doctor:N", title="Lekarz"),
            x=alt.X("Day:N", title="Dzień tygodnia", sort=days),
            color=alt.Color(
                "Absent:N",
                scale=alt.Scale(
                    domain=[0, 1],
                    range=["#6ABF4B", "#E53935"]  # zielony / czerwony
                ),
                title="Status"
            ),
            tooltip=["Doctor", "Day", "Absent"]
        )
        .properties(height=350)
    )

    st.altair_chart(heatmap_days, use_container_width=True)


    st.header("Nieobecności na poziomie zmian")

    id_to_name = {row["id"]: row["name"] for _, row in doctors.iterrows()}
    code_to_shift_id = {row["code"]: row["id"] for _, row in shifts.iterrows()}

    shifts_by_day = {day: [] for day in days}
    for _, sh in shifts.iterrows():
        shifts_by_day[sh["day"]].append(sh)

    un_shift = unavail_shift.copy()
    un_shift["ShiftID"] = un_shift["code"].map(code_to_shift_id)
    un_shift["Doctor"] = un_shift["doctor_id"].map(id_to_name)

    for day in days:
        st.subheader(f"{day} — dostępność na zmiany")

        shifts_today = shifts_by_day[day]
        shift_labels = [f"{sh['code']}" for sh in shifts_today]

        grid = []
        for _, doc in doctors.iterrows():
            for sh in shifts_today:
                grid.append({
                    "Doctor": doc["name"],
                    "DoctorID": doc["id"],
                    "ShiftID": sh["id"],
                    "ShiftLabel": sh["code"],
                    "Status": "Available"
                })

        df_day = pd.DataFrame(grid)

        day_unavailable_docs = unavail_day[unavail_day["day"] == day]["doctor_id"].tolist()
        df_day.loc[df_day["DoctorID"].isin(day_unavailable_docs), "Status"] = "DayAbsent"

        for _, row in un_shift[un_shift["ShiftID"].isin(df_day["ShiftID"])].iterrows():
            df_day.loc[
                (df_day["DoctorID"] == row["doctor_id"]) &
                (df_day["ShiftID"] == row["ShiftID"]),
                "Status"
            ] = "ShiftAbsent"

        # Heatmapa dla zmian dla każdego dnia
        heatmap_shifts = (
            alt.Chart(df_day)
            .mark_rect()
            .encode(
                y=alt.Y("Doctor:N", title="Lekarz"),
                x=alt.X("ShiftLabel:N", title="Zmiana"),
                color=alt.Color(
                    "Status:N",
                    scale=alt.Scale(
                        domain=["Available", "ShiftAbsent", "DayAbsent"],
                        range=["#6ABF4B", "#E53935", "#FFB300"]
                    ),
                    title="Status"
                ),
                tooltip=["Doctor", "ShiftLabel", "Status"]
            )
            .properties(height=300)
        )

        st.altair_chart(heatmap_shifts, use_container_width=True)


# === RUN MODEL ===
result = run_with_one_extra_doctor()
status = result["status"]
st.write("Status:", status)
schedule_before = result["schedule_before"]
stats_before = result["stats_before"]
solver_stats_before = result["solver_stats_before"]

# === IT WAS NECESSARY TO HIRE A NEW DOCTOR ===
if result["added"]:
    schedule_after = result["schedule_after"]
    stats_after = result["stats_after"]
    solver_stats_after = result["solver_stats_after"]
    tab1, tab2 = st.tabs(["PRZED dodaniem lekarza", "PO dodaniu lekarza"])

    with tab1:
        st.header("Harmonogram PRZED")
        st.dataframe(schedule_before)

        st.header("Wykresy i statystyki PRZED")
        render_all_charts(stats_before, solver_stats_before, doctors, shifts, unavail_day, unavail_shift)


    with tab2:
        new_doc = result["new_doctor"]
        with st.container(border=True):
            st.subheader("Dodano nowego lekarza do rozwiązania")

            st.write(f"**Imię i nazwisko:** {new_doc['name']}")
            st.write(f"**Rola:** {new_doc['role']}")
            st.write(f"**Umiejętności:** {new_doc['skills']}")
            if 'salary' in new_doc:
                st.write(f"**Koszt/h:** {new_doc['salary']} zł")
            if 'improvement' in new_doc:
                st.write(f"**Poprawa slack:** {new_doc['improvement']} braków → po nim mniejsza liczba braków")

            st.info("Lekarz został automatycznie wybrany jako najlepszy dostępny kandydat.")
        st.header("Harmonogram PO dodaniu lekarza")
        st.dataframe(schedule_after)

        st.header("Wykresy i statystyki PO")
        render_all_charts(stats_after, solver_stats_after, doctors, shifts, unavail_day, unavail_shift, "PO")

# === SOLVER FOUND A SOLUTION WITHOUT HIRING A NEW DOCTOR ===
else:
    st.dataframe(schedule_before)
    st.header("Wykresy i statystyki")
    render_all_charts(stats_before, solver_stats_before, doctors, shifts, unavail_day, unavail_shift, "Wykresy i statystyki")