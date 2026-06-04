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

            styled = validation_df.style.applymap(
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
            colors = ["#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3", "#ede9fe"]
            color = colors[int(row["Bubble"]) % len(colors)]
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
        "Simulate what happens if one bubble is quarantined due to exposure. "
        "This analysis uses the **current schedule** and removes the quarantined bubble."
    )

    quarantine_bubble = st.selectbox(
        "Select bubble to quarantine",
        options=list(range(num_bubbles)),
        format_func=lambda b: f"Bubble {b} ({sum(1 for bb in bubble_assignment.values() if bb == b)} doctors)"
    )

    from pandemic_mode import simulate_quarantine_with_reserve

    quarantine_result = simulate_quarantine_with_reserve(
        results=results,
        quarantine_bubble=quarantine_bubble,
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
        reserve_doctors_df=reserve_doctors_df,
    )

    severity_color = {
        "LOW": "success",
        "MEDIUM": "warning",
        "HIGH": "error",
    }[quarantine_result["severity"]]

    getattr(st, severity_color)(quarantine_result["recommendation"])

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Doctors quarantined", len(quarantine_result["quarantined_doctors"]))
    col_q2.metric("Shifts lost", len(quarantine_result["uncovered_shifts"]))
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

    with st.expander(f"Quarantined doctors (Bubble {quarantine_bubble})"):
        st.write(", ".join(quarantine_result["quarantined_doctors"]))

    with st.expander("Uncovered shifts after quarantine"):
        if quarantine_result["uncovered_shifts"].empty:
            st.info("No shifts become uncovered.")
        else:
            st.dataframe(quarantine_result["uncovered_shifts"], width="stretch")

    # =====================================================================
    # RESERVE ACTIVATION
    # =====================================================================
    st.subheader("Reserve Activation")

    st.caption(
        "Reserve activation provides an operational feasibility estimate only. "
        "Final assignments require manual validation of labor-law constraints "
        "(e.g., 11-hour rest periods, weekly workload limits)."
    )

    if not quarantine_result.get("reserve_available"):
        st.warning("No reserve physician pool available — reserve activation impossible.")
    else:
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric(
            "Covered by reserve",
            quarantine_result["covered_by_reserve"],
            delta=f"-{quarantine_result['covered_by_reserve']} from gap",
            delta_color="inverse"
        )
        col_r2.metric(
            "Residual uncovered",
            quarantine_result["residual_gap"],
        )
        col_r3.metric(
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
        "In pandemic mode, replacements are restricted to **same-bubble doctors**. "
        "New hires (candidates) are disabled. Reserve staff can be used with confirmation."
    )

    st.caption(
        "Replacement search is restricted to the same bubble whenever possible."
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
                return "OK – same bubble"
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
                        domain=["OK – same bubble", "Reserve", "Violation"],
                        range=["#2ecc71", "#f39c12", "#e74c3c"]
                    )
                ),
                tooltip=["name", "source", "bubble", "score", "extra_cost"]
            )
            .properties(height=300, title="Replacement candidates ranking")
        )
        st.altair_chart(chart_rep, use_container_width=True)
