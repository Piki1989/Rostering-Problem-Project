"""
pandemic_ui.py
==============
Streamlit UI component for the Pandemic Mode tab.

Add to your main app.py:

    from pandemic_ui import render_pandemic_tab

    with st.tabs(["...", "🦠 Pandemic Mode"])[pandemic_tab_index]:
        render_pandemic_tab(results, doctors, id_to_name, id_to_role)

Expected inputs:
    results      : run_scenario() output (baseline or hire_allowed)
    doctors      : doctors DataFrame
    id_to_name   : {doctor_id -> doctor_name}
    id_to_role   : {doctor_id -> role}
"""

import streamlit as st
import pandas as pd
import altair as alt

from pandemic_mode import (
    PandemicConfig,
    greedy_bubble_assignment,
    bubble_summary_df,
    bubble_contact_matrix,
    simulate_quarantine,
    find_replacement_pandemic,
)


# ---------------------------------------------------------------------------
# MAIN RENDER
# ---------------------------------------------------------------------------

def render_pandemic_tab(
    results: dict,
    doctors: pd.DataFrame,
    id_to_name: dict,
    id_to_role: dict,
    reserve_doctors_df: pd.DataFrame | None = None,
):
    st.header("🦠 Pandemic Mode")

    st.markdown("""
    Epidemic/pandemic scheduling mode. Doctors are divided into **isolated bubble teams**
    that never share contact shifts (WARD/ICU). If one bubble is quarantined,
    only that group is affected — the hospital continues operating with the remaining bubbles.

    > **Contact definition:** sharing the same shift in WARD or ICU department.
    > Clinic (outpatient) shifts are excluded — separate rooms, lower transmission risk.
    """)

    # =====================================================================
    # SECTION 1: CONFIGURATION
    # =====================================================================
    st.subheader("⚙️ Bubble Configuration")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        num_bubbles = st.slider(
            "Number of bubbles",
            min_value=2, max_value=5, value=3,
            help="Typically 2–4. More bubbles = smaller teams, more resilience but tighter staffing."
        )

    with col_cfg2:
        contact_depts = st.multiselect(
            "Contact departments",
            options=["WARD", "ICU", "CLINIC"],
            default=["WARD", "ICU"],
            help="Departments where sharing a shift = infection contact."
        )

    with col_cfg3:
        min_spec = st.number_input(
            "Min specialists per bubble",
            min_value=1, max_value=5, value=1,
            help="Each bubble must contain at least this many specialists to be clinically viable."
        )

    config = PandemicConfig(
        num_bubbles=num_bubbles,
        contact_depts=contact_depts,
        bubble_assignment="auto",
        min_specialists_per_bubble=min_spec,
    )

    # =====================================================================
    # SECTION 2: BUBBLE ASSIGNMENT
    # =====================================================================
    st.subheader("👥 Bubble Assignment")

    name_to_id = {v: k for k, v in id_to_name.items()}

    @st.cache_data(show_spinner="Assigning doctors to bubbles...")
    def get_bubble_assignment(_doctors, _num_bubbles, _min_spec, _config_hash):
        return greedy_bubble_assignment(
            doctors_df=_doctors,
            num_bubbles=_num_bubbles,
            min_specialists_per_bubble=_min_spec,
        )

    config_hash = f"{num_bubbles}_{min_spec}_{','.join(sorted(contact_depts))}"
    bubble_assignment = get_bubble_assignment(doctors, num_bubbles, min_spec, config_hash)

    summary = bubble_summary_df(bubble_assignment, id_to_name, id_to_role, num_bubbles)

    # Bubble size stats
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
        # Color coding per bubble
        def highlight_bubble(row):
            colors = ["#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3", "#ede9fe"]
            color = colors[int(row["Bubble"]) % len(colors)]
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            summary.style.apply(highlight_bubble, axis=1),
            width="stretch",
            height=400,
        )

    # Contact validation heatmap
    st.subheader("🔍 Contact Validation")
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
        st.success("✅ No cross-bubble contacts detected in contact departments.")
    else:
        total_violations = int(contact_matrix["ContactCount"].sum())
        st.error(
            f"⚠️ {total_violations} cross-bubble contact events detected. "
            "This schedule was optimized **without** pandemic constraints. "
            "Re-run the solver with pandemic mode enabled to fix this."
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

    st.info(
        "💡 To schedule **with** pandemic constraints, pass the `PandemicConfig` "
        "to `add_pandemic_constraints()` in `build_model()`. "
        "This will enforce bubble isolation at the solver level."
    )

    # =====================================================================
    # SECTION 3: QUARANTINE SIMULATION
    # =====================================================================
    st.subheader("🔴 Quarantine Simulation")

    st.markdown(
        "Simulate what happens if one bubble is quarantined due to exposure. "
        "This analysis uses the **current schedule** and removes the quarantined bubble."
    )

    quarantine_bubble = st.selectbox(
        "Select bubble to quarantine",
        options=list(range(num_bubbles)),
        format_func=lambda b: f"Bubble {b} ({sum(1 for bb in bubble_assignment.values() if bb == b)} doctors)"
    )

    quarantine_result = simulate_quarantine(
        results=results,
        quarantine_bubble=quarantine_bubble,
        bubble_assignment=bubble_assignment,
        id_to_name=id_to_name,
        config=config,
    )

    # Severity color
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

    # Department breakdown
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

    # Remaining bubble capacity comparison
    st.subheader("📊 Bubble Capacity Overview")

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
    # SECTION 4: PANDEMIC EMERGENCY REPLACEMENT
    # =====================================================================
    st.subheader("🚨 Emergency Replacement (Pandemic Mode)")

    st.markdown(
        "In pandemic mode, replacements are restricted to **same-bubble doctors**. "
        "New hires (candidates) are disabled. Reserve staff can be used with confirmation."
    )

    # Build shift options: only assigned shifts
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
            f"✅ Recommended: **{best['name']}** "
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
        f"**Contact dept:** {'Yes 🔴' if rep_result.get('is_contact_dept') else 'No 🟢'}"
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

    # =====================================================================
    # SECTION 5: HOW TO ENABLE IN SOLVER
    # =====================================================================
    with st.expander("📖 How to enable pandemic constraints in the solver", expanded=False):
        st.code("""
# In model_ref.py, inside build_model(), after all hard constraints:

from pandemic_mode import PandemicConfig, add_pandemic_constraints, greedy_bubble_assignment

# 1. Define config
pandemic_config = PandemicConfig(
    num_bubbles=3,
    contact_depts=["WARD", "ICU"],
    bubble_assignment="static",   # use greedy pre-assignment
    min_specialists_per_bubble=1,
)

# 2. Pre-assign bubbles (fast, no extra solver variables)
bubble_assignment = greedy_bubble_assignment(
    doctors_df=pd.concat([doctors, candidates]),
    num_bubbles=pandemic_config.num_bubbles,
    min_specialists_per_bubble=pandemic_config.min_specialists_per_bubble,
)

# 3. Add constraints
pandemic_vars = add_pandemic_constraints(
    model=model,
    x=x,
    D_all=D_all,
    S=S,
    shifts=shifts,
    shift_idx=shift_idx,
    config=pandemic_config,
    id_to_role=id_to_role,
    static_assignment=bubble_assignment,
)

# 4. Store in bundle for later use
bundle["pandemic_vars"] = pandemic_vars
bundle["bubble_assignment"] = bubble_assignment
        """, language="python")

        st.markdown("""
**Performance note:** Using `bubble_assignment="static"` with `greedy_bubble_assignment()`
adds only O(D×S) pairwise constraints and has minimal impact on solver runtime.

The `auto` mode (solver assigns bubbles) adds O(D×B + S×D×B) variables and is
significantly slower — use only for small instances (<20 doctors).
        """)