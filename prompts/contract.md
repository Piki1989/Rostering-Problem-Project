The model is a monthly hospital doctor rostering problem.
- Doctors (employed doctors and candidates who may be hired) are assigned to shifts.
- Shifts belong to departments (WARD, ICU, CLINIC) and last from 4 to 24 hours.
- Shift times are given in hours from the start of the planning horizon. A shift may end after midnight: an end of 31 means 07:00 the next day.

The function receives `m`, an object with the following fields. Treat every field except `m.model` and the output fields as read-only.

| Field | Meaning |
|---|---|
| `m.model` | `cp_model.CpModel` to which variables and constraints are added |
| `m.x[(d, s)]` | BoolVar: doctor `d` works shift `s` (for every `d` in `D_all`, `s` in `S`) |
| `m.hire[c]` | BoolVar: candidate `c` is hired |
| `m.ctx.D`, `m.ctx.C`, `m.ctx.D_all`, `m.ctx.S` | lists: employed doctors, candidates, both (`D` then `C`), shifts |
| `m.ctx.id_to_role[d]` | role of a doctor: `specialist`, `icu_specialist`, `resident`, `outpatient`, ... |
| `m.p.date_list` | sorted list of the days of the horizon (`"YYYY-MM-DD"` strings) |
| `m.p.days_to_shifts[t]` | list of shifts starting on day `t` |
| `m.p.shift_date[s]` | day on which shift `s` starts |
| `m.p.abs_start[s]`, `m.p.abs_end[s]` | start / end of shift `s` in hours from the start of the horizon |
| `m.p.hours[s]` | length of shift `s` in hours |
| `m.p.dept[s]` | department of shift `s` |
| `m.p.req_skill[s]` | skill required by shift `s` |
| `m.p.min_staff[s]`, `m.p.regular_staff[s]` | minimum and regular number of doctors on shift `s` |
| `m.p.night_shifts` | list of night shifts (night shifts and 24 h shifts) |
| `m.p.night_shifts_by_date[t]` | night shifts starting on day `t` |
| `m.p.twentyfour_shifts` | list of 24 h shifts |
| `m.p.weekend_shifts` | list of shifts on Saturdays and Sundays |
| `m.p.hard_shifts` | list of shifts marked as hard |
| `m.p.weeks` | list of week numbers (calendar weeks starting on Monday) |
| `m.p.shift_week[s]` | week of shift `s` |
| `m.p.skill_map[d]` | list of skills of doctor `d` |
| `m.p.night_allowed[d]`, `m.p.twentyfour_allowed[d]` | 1 if `d` may work night / 24 h shifts, else 0 (dict over `D_all`, employed doctors first) |
| `m.p.specialists` | employed doctors with role `specialist` or `icu_specialist` |
| `m.p.needs_mentor` | doctors who may only work together with a specialist |
| `m.p.opt_out_doctors` | employed doctors who opted out of the working-time limit |
| `m.p.regular_doctors` | the other employed doctors |
| `m.p.weekly_regular_lookup` | set of doctors that get the "regular" weekly target (use as given) |
| `m.p.adjusted_max_hours[d]` | monthly hour limit `M[d]` |
| `m.p.weekly_limits[d][w]` | weekly hour limit `L[d,w]` |
| `m.p.weekly_available_days[d][w]` | number of days of week `w` on which `d` is available |
| `m.p.hourly_cost_map[d]` | hourly cost of `d` (integer) |
| `m.p.unavailable_days` | list of `(d, day)` on which `d` cannot work at all |
| `m.p.unavailable_shifts` | list of `(d, s)`: single shifts `d` cannot work |
| `m.p.preferences` | list of `(d, s, kind)` with kind `"like"` or `"dislike"` |
| `m.pandemic_config.contact_depts` | pandemic mode: departments where working together counts as contact |
| `m.bubble_assignment[d]` | pandemic mode: bubble of `d`; `-1` means outside the bubble system |
| `m.targets` | targets of the compromise objective: `cost`, `max_over_budget_pct`, `spread`, `weekend_spread`, `hard_spread`, `underwork`, `weekly_underwork`, `pref`, `scale` |

Output fields filled by other equations (available only when listed as already available):
- `m.slacks[s]`
- `m.slacks_o[s]`
- `m.worked_hours[d]`
- `m.doctor_cost[d]`
- `m.weekly_hours[d][w]`
- `m.pref_terms`
- `m.spread`, `m.weekend_spread`, `m.hard_spread`, `m.ratio_spread`
- `m.underwork[d]`
- `m.weekly_underwork[d][w]`
- `m.pandemic_bundle`

Specialists are the doctors whose role is `specialist` or `icu_specialist`.
