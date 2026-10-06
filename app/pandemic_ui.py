"""
Streamlit UI component for the Pandemic Mode tab.
"""

import streamlit as st
import pandas as pd
import altair as alt
from streamlit_calendar import calendar
from datetime import datetime, timedelta

from pandemic_mode import (
    PandemicConfig,
    bubble_summary_df,
    bubble_contact_matrix,
    simulate_quarantine,
    find_replacement_pandemic,
    suggest_num_bubbles,
    QuarantineEvent,
    simulate_quarantine_v2,
)

# CALENDAR

def schedule_to_pandemic_calendar_events(
    schedule_df: pd.DataFrame,
    bubble_assignment: dict,
    name_to_id: dict,
    contact_depts: list[str],
) -> list[dict]:
    """
    Builds streamlit_calendar events colored by bubble assignment.

    Contact shifts (WARD/ICU) use full bubble colors.
    Non-contact shifts (CLINIC) use muted colors.
    Uncovered shifts are shown in red.
    """
    BUBBLE_COLORS = {
        0: "#3b82f6",  # blue
        1: "#10b981",  # green
        2: "#f59e0b",  # orange
        3: "#ec4899",  # pink
        4: "#8b5cf6",  # purple
    }
    BUBBLE_COLORS_MUTED = {
        0: "#93c5fd",
        1: "#6ee7b7",
        2: "#fcd34d",
        3: "#f9a8d4",
        4: "#c4b5fd",
    }

    events = []
    contact_set = set(contact_depts)

    for _, row in schedule_df.iterrows():
        date = row["Date"]
        if hasattr(date, "strftime"):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)[:10]

        start_hour = int(row["StartHour"])
        end_hour_raw = int(row["EndHour"])

        start_dt = datetime.fromisoformat(f"{date_str} {start_hour:02d}:00")
        if end_hour_raw <= 24:
            end_dt = datetime.fromisoformat(f"{date_str} {end_hour_raw:02d}:00")
        else:
            end_dt = datetime.fromisoformat(
                f"{date_str} {(end_hour_raw - 24):02d}:00"
            ) + timedelta(days=1)

        if row["Missing"] == 1:
            # title = f"{row['ShiftCode']} – UNCOVERED"
            short_code = row["ShiftCode"]
            if "_20" in short_code:
                short_code = short_code.split("_20")[0]
            title = f"{short_code} – UNCOVERED"
            color = "#e74c3c"
        else:
            doc_name = row["Doctor"]
            doc_id = name_to_id.get(doc_name)
            bubble = bubble_assignment.get(doc_id)
            is_contact = row["Dept"] in contact_set

            if bubble is None:
                color = "#9ca3af"
            elif is_contact:
                color = BUBBLE_COLORS.get(bubble, "#6b7280")
            else:
                color = BUBBLE_COLORS_MUTED.get(bubble, "#d1d5db")

            bubble_label = f"B{bubble}" if bubble is not None else "?"
            short_code = row["ShiftCode"]
            if "_20" in short_code:
                short_code = short_code.split("_20")[0]
            short_name = doc_name
            if doc_name and doc_name.lower().startswith("doctor"):
                short_name = "D" + doc_name[6:]

            title = f"[{bubble_label}] {short_code} {short_name}"

            # title = f"[{bubble_label}] {row['ShiftCode']} – {doc_name}"

        events.append({
            "title": title,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "color": color,
        })

    return events


def quarantine_calendar_events(quarantine_result):

    events = []

    affected = quarantine_result.get(
        "affected_assignments",
        pd.DataFrame()
    )

    for _, row in affected.iterrows():

        date_str = pd.to_datetime(
            row["Date"]
        ).strftime("%Y-%m-%d")

        events.append({
            "title": f"❌ {row['ShiftCode']}",
            "start": f"{date_str}T08:00:00",
            "end": f"{date_str}T20:00:00",
            "color": "#c0392b",
        })

    reserve_df = quarantine_result.get(
        "reserve_assignments",
        pd.DataFrame()
    )

    for _, row in reserve_df.iterrows():

        events.append({
            "title": f"🟠 RESERVE {row['ShiftCode']}",
            "start": f"{row['Date']}T08:00:00",
            "end": f"{row['Date']}T20:00:00",
            "color": "#f39c12",
        })

    reassignment_df = quarantine_result.get(
        "reassignments",
        pd.DataFrame()
    )

    for _, row in reassignment_df.iterrows():

        events.append({
            "title": f"🟢 REDISTR {row['ShiftCode']}",
            "start": f"{row['Date']}T08:00:00",
            "end": f"{row['Date']}T20:00:00",
            "color": "#2ecc71",
        })

    residual_df = quarantine_result.get(
        "residual_uncovered",
        pd.DataFrame()
    )

    for _, row in residual_df.iterrows():

        events.append({
            "title": f"⚫ UNCOVERED {row['ShiftCode']}",
            "start": f"{row['Date']}T08:00:00",
            "end": f"{row['Date']}T20:00:00",
            "color": "#2d3436",
        })

    return events

# MAIN RENDER

def render_pandemic_tab(
    results: dict,
    doctors: pd.DataFrame,
    id_to_name: dict,
    id_to_role: dict,
    reserve_doctors_df: pd.DataFrame | None = None,
):
    st.header("Pandemic Mode")

    st.markdown("""
    Epidemic/pandemic scheduling mode. Doctors are divided into **isolated bubble teams**
    that never share contact shifts (WARD/ICU). If one bubble is quarantined,
    only that group is affected — the hospital continues operating with the remaining bubbles.

    > **Contact definition:** sharing the same shift in WARD or ICU department.
    > Clinic (outpatient) shifts are excluded — separate rooms, lower transmission risk.
    """)

    pandemic_data = results.get("pandemic")

    if pandemic_data is None:
        st.warning(
            "Pandemic mode was not enabled during optimization."
        )
        return

    config = pandemic_data["config"]
    bubble_assignment = pandemic_data["bubble_assignment"]

    num_bubbles = config.num_bubbles
    contact_depts = config.contact_depts

    # =====================================================================
    # ADAPTIVE BUBBLE SIZING
    # =====================================================================
    st.subheader("Adaptive Bubble Sizing")

    clinic_pool_count = sum(
        1 for b in bubble_assignment.values()
        if b == -1
    )

    bubble_doctors_count = len(doctors) - clinic_pool_count

    suggestion = suggest_num_bubbles(
        n_doctors=bubble_doctors_count,
        target_loss_pct=config.target_loss_pct,
        min_bubble_size=config.min_bubble_size,
        max_bubbles=10,
        min_bubbles=2,
    )

    col_s1, col_s2, col_s3 = st.columns(3)

    col_s1.metric(
        "Current bubbles",
        config.num_bubbles
    )

    col_s2.metric(
        "Recommended bubbles",
        suggestion["recommended"]
    )

    col_s3.metric(
        "Expected loss per quarantined bubble",
        f"{suggestion['loss_pct']}%"
    )

    if suggestion["recommended"] == config.num_bubbles:
        st.success(
            "The current number of bubbles matches the adaptive recommendation."
        )
    else:
        st.warning(
            f"The adaptive recommendation is {suggestion['recommended']} bubbles, "
            f"but the current schedule uses {config.num_bubbles}. "
            "Consider adjusting the number of bubbles before optimization."
        )

    if suggestion.get("warning"):
        st.info(suggestion["warning"])

    with st.expander("How is the recommended bubble count calculated?"):
        st.markdown(f"""
        The recommendation balances two goals:

        1. **Limit the workforce loss if one bubble is quarantined.**  
           The target loss is currently set to **{config.target_loss_pct * 100:.0f}%**.

        2. **Keep bubbles large enough to remain operational.**  
           The minimum bubble size is currently set to **{config.min_bubble_size} doctors**.

        Internally, the function computes:

        - minimum bubbles needed by risk limit: `ceil(1 / target_loss_pct)`
        - maximum feasible bubbles by size: `number_of_doctors // min_bubble_size`
        - final recommendation: the smaller of those two values, capped by `max_bubbles`

        In this schedule:

        - doctors considered for bubbles: **{bubble_doctors_count}**
        - doctors in CLINIC pool: **{clinic_pool_count}**
        - recommended bubbles: **{suggestion["recommended"]}**
        - expected loss if one bubble is quarantined: **{suggestion["loss_pct"]}%**
        """)

    # =====================================================================
    # BUBBLE ASSIGNMENT
    # =====================================================================
    st.subheader("Bubble Assignment")

    validation = pandemic_data.get("validation")

    if validation is not None:
        st.subheader("Bubble Validation")

        is_safe = validation.get("is_safe", True)
        issues = validation.get("issues", [])
        validation_df = validation.get("validation_df")

        if is_safe and not issues:
            st.success("All bubbles satisfy the minimum requirements.")
        elif is_safe:
            st.info(
                f"The bubbles are safe, but {len(issues)} warnings were detected."
            )
        else:
            errors = [i for i in issues if i["severity"] == "ERROR"]
            st.error(
                f"{len(errors)} critical bubble issues detected. "
                "The schedule may be infeasible or require significant slack. "
                "Consider changing the bubble assignment or reducing the number of bubbles."
            )

        if validation_df is not None and not validation_df.empty:
            def color_status(val):
                if val == "OK":
                    return "background-color: #d4edda; color: #155724"
                elif val == "PROBLEM":
                    return "background-color: #f8d7da; color: #721c24"
                return ""

            styled = validation_df.style.map(
                color_status, subset=["Status"]
            )
            st.dataframe(styled, width="stretch")

        if issues:
            with st.expander(f"Detected issues ({len(issues)})"):
                for issue in issues:
                    icon = {
                        "ERROR": "🔴",
                        "WARNING": "🟡",
                        "INFO": "🔵",
                    }.get(issue["severity"], "•")
                    st.write(f"{icon} **{issue['severity']}** — {issue['message']}")

    st.success(
        "This schedule was optimized with pandemic constraints enabled."
    )

    name_to_id = {v: k for k, v in id_to_name.items()}

    summary = bubble_summary_df(bubble_assignment, id_to_name, id_to_role, num_bubbles)

    col_b = st.columns(num_bubbles)
    for b in range(num_bubbles):
        b_df = summary[summary["Bubble"] == b]
        with col_b[b]:
            spec_count = (b_df["Role"].isin(["specialist", "icu_specialist"])).sum()
            st.metric(
                f"Bubble {b}",
                f"{len(b_df)} doctors",
                delta=f"{spec_count} specialists",
                delta_color="normal"
            )

    with st.expander("Show full bubble assignment", expanded=False):

        def highlight_bubble(row):

            colors = [
                "#dbeafe",
                "#dcfce7",
                "#fef9c3",
                "#fce7f3",
                "#ede9fe",
            ]

            try:
                color = colors[int(row["Bubble"]) % len(colors)]
            except (ValueError, TypeError):
                color = "#f3f4f6"

            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            summary.style.apply(highlight_bubble, axis=1),
            width="stretch",
            height=400,
        )

    st.subheader("Contact Validation")
    st.caption(
        "A valid pandemic schedule should show **zero cross-bubble contacts** "
        "in contact departments (WARD/ICU). Any violations below indicate "
        "the current schedule was built without pandemic constraints."
    )

    contact_matrix = bubble_contact_matrix(
        results["schedule_df"],
        bubble_assignment,
        name_to_id,
        contact_depts,
    )

    if contact_matrix.empty:
        st.success("No cross-bubble contacts detected in contact departments.")
    else:
        total_violations = int(contact_matrix["ContactCount"].sum())
        st.error(
            f"{total_violations} emergency cross-bubble contacts detected."
        )

        st.caption(
            "These violations were allowed only because the solver "
            "used emergency relaxation variables to preserve feasibility."
        )

        chart_violations = (
            alt.Chart(contact_matrix)
            .mark_rect()
            .encode(
                x=alt.X("Bubble_A:O", title="Bubble A"),
                y=alt.Y("Bubble_B:O", title="Bubble B"),
                color=alt.Color(
                    "ContactCount:Q",
                    scale=alt.Scale(scheme="reds"),
                    title="Contact events"
                ),
                facet=alt.Facet("Dept:N", columns=2),
                tooltip=["Bubble_A", "Bubble_B", "Dept", "ContactCount"]
            )
            .properties(width=180, height=180)
        )

        st.altair_chart(chart_violations, use_container_width=True)

        with st.expander("Raw contact events"):
            st.dataframe(contact_matrix, width="stretch")

    pandemic_bundle = results.get("pandemic_bundle")

    if pandemic_bundle is not None:
        violations = pandemic_bundle.get(
            "crossbubble_violations",
            []
        )

        st.metric(
            "Emergency cross-bubble violations",
            len(violations)
        )

    # =====================================================================
    # CLINIC POOL
    # =====================================================================

    clinic_pool_ids = [
        d for d, b in bubble_assignment.items()
        if b == -1
    ]

    if clinic_pool_ids:
        st.subheader("Doctors Outside Bubble System (CLINIC Pool)")

        clinic_pool_df = pd.DataFrame({
            "Doctor": [
                id_to_name.get(d, f"Doctor {d}")
                for d in clinic_pool_ids
            ],
            "Role": [
                id_to_role.get(d, "")
                for d in clinic_pool_ids
            ]
        })

        st.info(
            f"{len(clinic_pool_ids)} doctor(s) work exclusively in outpatient clinics "
            "and are not assigned to pandemic bubbles."
        )

        st.dataframe(
            clinic_pool_df,
            width="stretch",
            hide_index=True,
        )

        with st.expander("Why are these doctors excluded from bubbles?"):
            st.markdown("""
            Clinic-only physicians do not participate in WARD/ICU contact shifts.

            Therefore:
            - they are not included in bubble isolation,
            - they do not generate cross-bubble contacts,
            - quarantine is handled individually.
            """)

    # =====================================================================
    # CALENDAR
    # =====================================================================
    st.subheader("Pandemic Schedule – Bubble Calendar View")

    st.caption(
        "Pandemic schedule calendar colored by bubble assignment. "
        "Full colors = contact shifts (WARD/ICU), muted colors = "
        "non-contact shifts (CLINIC), red = uncovered, gray = unassigned."
    )

    legend_cols = st.columns(num_bubbles + 1)
    BUBBLE_LEGEND_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
    for b in range(num_bubbles):
        with legend_cols[b]:
            st.markdown(
                f"<div style='background-color:{BUBBLE_LEGEND_COLORS[b % 5]};"
                f"padding:8px;border-radius:5px;text-align:center;color:white;"
                f"font-weight:bold;'>Bubble {b}</div>",
                unsafe_allow_html=True
            )
    with legend_cols[num_bubbles]:
        st.markdown(
            "<div style='background-color:#e74c3c;padding:8px;border-radius:5px;"
            "text-align:center;color:white;font-weight:bold;'>Uncovered</div>",
            unsafe_allow_html=True
        )

    pandemic_events = schedule_to_pandemic_calendar_events(
        schedule_df=results["schedule_df"],
        bubble_assignment=bubble_assignment,
        name_to_id=name_to_id,
        contact_depts=contact_depts,
    )

    view_options = {
        "Month": "dayGridMonth",
        "Week": "timeGridWeek",
        "Day": "timeGridDay",
        "List (most readable)": "listWeek",
    }
    selected_view_label = st.selectbox(
        "Calendar view",
        list(view_options.keys()),
        index=0,
        key="pandemic_calendar_view",
        help="List view shows full doctor names and shift codes without truncation."
    )
    selected_view = view_options[selected_view_label]

    pandemic_cal_options = {
        "initialView": selected_view,
        "initialDate": str(
            pd.to_datetime(results["schedule_df"]["Date"]).min().date()
        ),
        "height": 700,
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay,listWeek",
        },
        "dayMaxEvents": False,
        "eventDisplay": "block",
        "eventClassNames": "compact-event",
    }

    st.markdown("""
    <style>
    .fc-event-title {
        font-size: 0.75em !important;
        line-height: 1.1 !important;
    }
    .fc-list-event-title {
        font-size: 0.9em !important;
    }
    .fc-event {
        padding: 1px 3px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    calendar(
        events=pandemic_events,
        options=pandemic_cal_options,
        key=f"pandemic_calendar_{selected_view}",
    )

    # pandemic_cal_options = {
    #     "initialView": "dayGridMonth",
    #     "initialDate": str(
    #         pd.to_datetime(results["schedule_df"]["Date"]).min().date()
    #     ),
    #     "height": 700,
    # }
    #
    # calendar(
    #     events=pandemic_events,
    #     options=pandemic_cal_options,
    #     key="pandemic_calendar",
    # )

    # =====================================================================
    # QUARANTINE SIMULATION
    # =====================================================================
    st.subheader("Quarantine Simulation")

    st.markdown(
        "Simulate what happens when a bubble, a single doctor, or selected doctors "
        "are quarantined for a chosen period. The analysis uses the current schedule "
        "and re-optimizes coverage within the quarantine window."
    )

    quarantine_type = st.radio(
        "Quarantine target",
        [
            "Bubble",
            "Single doctor",
            "Multiple doctors",
        ],
        horizontal=True,
    )

    start_date = st.date_input(
        "Quarantine start date",
        value=pd.to_datetime(
            results["schedule_df"]["Date"]
        ).min()
    )

    duration_days = st.slider(
        "Duration (days)",
        min_value=1,
        max_value=30,
        value=7,
    )

    doctor_names = sorted(
        list(id_to_name.values())
    )

    event = None
    quarantine_bubble = None

    if quarantine_type == "Bubble":
        quarantine_bubble = st.selectbox(
            "Select bubble",
            options=list(range(num_bubbles)),
            format_func=lambda b:
            f"Bubble {b} ({sum(1 for bb in bubble_assignment.values() if bb == b)} doctors)"
        )

        event = QuarantineEvent(
            target_type="bubble",
            target=quarantine_bubble,
            start_date=start_date.strftime("%Y-%m-%d"),
            duration_days=duration_days,
        )

    elif quarantine_type == "Single doctor":

        selected_doctor = st.selectbox(
            "Doctor",
            doctor_names,
        )

        event = QuarantineEvent(
            target_type="doctor",
            target=selected_doctor,
            start_date=start_date.strftime("%Y-%m-%d"),
            duration_days=duration_days,
        )

    else:

        selected_doctors = st.multiselect(
            "Doctors",
            doctor_names,
        )

        if selected_doctors:
            event = QuarantineEvent(
                target_type="doctors",
                target=selected_doctors,
                start_date=start_date.strftime("%Y-%m-%d"),
                duration_days=duration_days,
            )

    if event is not None:

        quarantine_result = simulate_quarantine_v2(
            results=results,
            events=[event],
            bubble_assignment=bubble_assignment,
            id_to_name=id_to_name,
            config=config,
            reserve_doctors_df=reserve_doctors_df,
        )

    else:
        st.info("Select at least one doctor.")
        return

    # Mini CP-SAT solver diagnostics
    st.subheader("Mini CP-SAT Solver")

    col_solver1, col_solver2, col_solver3 = st.columns(3)

    col_solver1.metric(
        "Status",
        quarantine_result.get("status", "UNKNOWN")
    )

    col_solver2.metric(
        "Solve time",
        f"{quarantine_result.get('solve_time', 0.0):.2f}s"
    )

    objective_value = quarantine_result.get("objective")
    col_solver3.metric(
        "Objective",
        "—" if objective_value is None else f"{objective_value:.0f}"
    )

    with st.expander("Quarantined doctors"):

        for date_str, doctors_list in quarantine_result[
            "quarantined_doctors_by_date"
        ].items():
            st.write(
                f"**{date_str}**: "
                + ", ".join(doctors_list)
            )


    severity = quarantine_result["severity"]

    col_sev1, col_sev2 = st.columns([1, 3])

    col_sev1.metric(
        "Pandemic Risk",
        severity
    )

    severity_color = {
        "LOW": "success",
        "MEDIUM": "warning",
        "HIGH": "error",
    }[quarantine_result["severity"]]

    getattr(st, severity_color)(quarantine_result["recommendation"])

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    all_quarantined = set()

    for doctors_list in quarantine_result[
        "quarantined_doctors_by_date"
    ].values():
        all_quarantined.update(doctors_list)

    col_q1.metric(
        "Doctors quarantined",
        len(all_quarantined)
    )
    col_q2.metric(
        "Shifts lost",
        len(quarantine_result["affected_assignments"])
    )
    col_q3.metric("Coverage impact", f"{quarantine_result['coverage_impact_pct']}%")
    col_q4.metric("Days affected", quarantine_result["days_affected"])

    if quarantine_result["dept_impact"]:
        dept_df = pd.DataFrame(
            list(quarantine_result["dept_impact"].items()),
            columns=["Department", "Shifts lost"]
        )
        dept_chart = (
            alt.Chart(dept_df)
            .mark_bar(color="#e74c3c")
            .encode(
                x=alt.X("Department:N"),
                y=alt.Y("Shifts lost:Q"),
                tooltip=["Department", "Shifts lost"]
            )
            .properties(height=250, title="Shifts lost by department")
        )
        st.altair_chart(dept_chart, use_container_width=True)

    with st.expander("Affected assignments"):

        if quarantine_result["affected_assignments"].empty:
            st.info("No assignments affected.")
        else:
            st.dataframe(
                quarantine_result["affected_assignments"],
                width="stretch"
            )


    st.subheader("Quarantine Scenario Calendar")

    scenario_events = quarantine_calendar_events(
        quarantine_result
    )

    calendar(
        events=scenario_events,
        options={
            "initialView": "dayGridMonth",
            "height": 600,
        },
        key="quarantine_scenario_calendar",
    )

    legend_cols = st.columns(4)

    with legend_cols[0]:
        st.markdown("🔴 Quarantined shifts")

    with legend_cols[1]:
        st.markdown("🟠 Reserve coverage")

    with legend_cols[2]:
        st.markdown("🟢 Internal redistribution")

    with legend_cols[3]:
        st.markdown("⚫ Still uncovered")

    # =====================================================================
    # RESERVE ACTIVATION
    # =====================================================================
    st.subheader("Reserve Activation")

    st.caption(
        "Reserve activation provides an operational feasibility estimate only. "
        "Final assignments require manual validation of labor-law constraints "
        "(e.g., 11-hour rest periods, weekly workload limits)."
    )

    reserve_available = reserve_doctors_df is not None and not reserve_doctors_df.empty

    if not reserve_available:
        st.warning("No reserve physician pool available — reserve activation impossible.")
    else:
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)

        col_r1.metric(
            "Covered by reserve",
            quarantine_result.get("covered_by_reserve", 0),
        )

        col_r2.metric(
            "Internal redistribution",
            quarantine_result.get("reassigned_count", 0),
        )

        col_r3.metric(
            "Residual uncovered",
            quarantine_result.get("residual_gap", 0),
        )

        col_r4.metric(
            "Reserve cost",
            f"{quarantine_result.get('reserve_total_cost', 0):,} PLN"
        )

        if quarantine_result["residual_gap"] == 0:
            st.success(
                "Reserve staff covers all lost shifts. "
                "The hospital can continue full operations."
            )
        elif quarantine_result["residual_gap"] < quarantine_result["covered_by_reserve"]:
            st.info(
                f"Reserve staff covers most losses. "
                f"{quarantine_result['residual_gap']} shifts remain uncovered — "
                f"consider temporarily closing some outpatient clinics "
                f"or reducing staffing demand."
            )

        else:
            st.error(
                f"Even with reserve staff, most shifts cannot be covered. "
                f"Operational restructuring required: clinic closures, "
                f"cancellation of elective procedures, escalation to hospital management."
            )

        if not quarantine_result["reserve_assignments"].empty:
            with st.expander(f"Reserve assignments ({quarantine_result['covered_by_reserve']} shifts)"):
                st.dataframe(
                    quarantine_result["reserve_assignments"],
                    width="stretch"
                )

        if not quarantine_result.get("reassignments", pd.DataFrame()).empty:
            with st.expander(
                    f"Internal reassignments ({quarantine_result.get('reassigned_count', 0)} shifts)"
            ):
                st.dataframe(
                    quarantine_result["reassignments"],
                    width="stretch"
                )

        if quarantine_result["residual_gap"] > 0:
            with st.expander(f"Still uncovered ({quarantine_result['residual_gap']} shifts)"):
                st.dataframe(
                    quarantine_result["residual_uncovered"],
                    width="stretch"
                )

    st.subheader("Bubble Capacity Overview")

    bubble_stats = []
    for b in range(num_bubbles):
        b_doctors = [d for d, bb in bubble_assignment.items() if bb == b]
        b_names = [id_to_name[d] for d in b_doctors if d in id_to_name]

        b_shifts = results["schedule_df"][
            results["schedule_df"]["Doctor"].isin(b_names) &
            (results["schedule_df"]["Missing"] == 0)
        ]

        spec_count = sum(
            1 for d in b_doctors
            if id_to_role.get(d, "") in {"specialist", "icu_specialist"}
        )

        bubble_stats.append({
            "Bubble": f"Bubble {b}",
            "Doctors": len(b_doctors),
            "Specialists": spec_count,
            "Total shift-hours": int(b_shifts["Hours"].sum()) if not b_shifts.empty else 0,
            "Quarantined": b == quarantine_bubble,
        })

    bubble_stats_df = pd.DataFrame(bubble_stats)

    bubble_bar = (
        alt.Chart(bubble_stats_df)
        .mark_bar()
        .encode(
            x=alt.X("Bubble:N"),
            y=alt.Y("Total shift-hours:Q"),
            color=alt.condition(
                alt.datum.Quarantined,
                alt.value("#e74c3c"),
                alt.value("#4C78A8")
            ),
            tooltip=["Bubble", "Doctors", "Specialists", "Total shift-hours"]
        )
        .properties(height=300, title="Shift-hours by bubble (red = quarantined)")
    )
    st.altair_chart(bubble_bar, use_container_width=True)

    # =====================================================================
    # PANDEMIC EMERGENCY REPLACEMENT
    # =====================================================================
    st.subheader("Emergency Replacement (Pandemic Mode)")

    st.markdown(
        "On WARD/ICU shifts the replacement must be in the **same bubble as the doctors "
        "remaining on that shift**. If nobody else is on the shift, other bubbles are allowed "
        "but ranked last. Doctors quarantined in the simulation above are excluded. "
        "New hires (candidates) are disabled. Reserve staff can be used with confirmation."
    )

    st.caption(
        "All labour-law checks (11h rest, weekly limit, nights, working days, "
        "unavailability, mentor) are the same as in the standard replacement search."
    )

    assigned_shifts = results["schedule_df"].dropna(subset=["Doctor"])
    shift_options = [
        f"{row['ShiftCode']}__{row['Doctor']}"
        for _, row in assigned_shifts[["ShiftCode", "Doctor"]]
        .drop_duplicates()
        .iterrows()
    ]

    if not shift_options:
        st.info("No assigned shifts available for replacement simulation.")
        return

    selected = st.selectbox(
        "Select shift + sick doctor",
        shift_options,
        key="pandemic_replacement_selector",
        format_func=lambda k: k.replace("__", " → sick: "),
    )

    shift_code_sel, sick_doctor_sel = selected.split("__", 1)

    sick_id = name_to_id.get(sick_doctor_sel)
    sick_bubble_disp = bubble_assignment.get(sick_id, "?")

    st.markdown(
        f"**Sick doctor:** {sick_doctor_sel} | "
        f"**Bubble:** {sick_bubble_disp} | "
        f"**Shift:** {shift_code_sel}"
    )

    rep_result = find_replacement_pandemic(
        results=results,
        sick_doctor=sick_doctor_sel,
        shift_code=shift_code_sel,
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
        reserve_doctors_df=reserve_doctors_df,
        quarantined_by_date=quarantine_result["quarantined_doctors_by_date"],
    )

    if rep_result["fallback_message"]:
        st.error(rep_result["fallback_message"])

    if rep_result["recommendation"]:
        best = rep_result["recommendation"]
        is_reserve = best["source"] == "reserve"
        bubble_label = (
            "Reserve — confirmation needed"
            if is_reserve
            else f"Bubble {best.get('bubble', '?')}"
        )

        msg = (
            f"Recommended: **{best['name']}** "
            f"({bubble_label}) "
            f"| Cost +{best['extra_cost']} PLN"
        )
        if is_reserve:
            st.warning(msg)
        else:
            st.success(msg)

    shift_info_disp = rep_result["shift_info"]
    st.markdown(
        f"**Date:** {shift_info_disp.get('date')} | "
        f"**Dept:** {shift_info_disp.get('dept')} | "
        f"**Hours:** {shift_info_disp.get('start')}–{shift_info_disp.get('end')} | "
        f"**Required skill:** {shift_info_disp.get('required_skill', '—')} | "
        f"**Contact dept:** {'Yes' if rep_result.get('is_contact_dept') else 'No'}"
    )

    candidates_df_rep = pd.DataFrame(rep_result["candidates"])

    if not candidates_df_rep.empty:
        def get_status(r):
            if r["source"] == "reserve":
                return "Reserve"
            if r["rest_ok"] and r["skill_ok"]:
                return "OK – other bubble" if r.get("other_bubble") else "OK – same bubble"
            return "Violation"

        candidates_df_rep["status"] = candidates_df_rep.apply(get_status, axis=1)
        candidates_df_rep = candidates_df_rep.sort_values("score")

        display_cols = ["name", "source", "bubble", "score", "extra_cost", "workload_pct", "hard_shifts", "status"]
        display_cols = [c for c in display_cols if c in candidates_df_rep.columns]

        st.dataframe(candidates_df_rep[display_cols], width="stretch")

        chart_rep = (
            alt.Chart(candidates_df_rep.head(10))
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title="Score (lower = better)"),
                y=alt.Y("name:N", sort="-x"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["OK – same bubble", "OK – other bubble", "Reserve", "Violation"],
                        range=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
                    )
                ),
                tooltip=["name", "source", "bubble", "score", "extra_cost"]
            )
            .properties(height=300, title="Replacement candidates ranking")
        )
        st.altair_chart(chart_rep, use_container_width=True)
