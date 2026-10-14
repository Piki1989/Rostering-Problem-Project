# Rostering model - mathematical specification

This document describes exactly the model built by `rostering/model.py` and solved by
`rostering/solve.py`. It is the reference for the equations: `tests/test_model_equivalence.py`
checks that the code still produces the same model as the original implementation
(`model/model_ref.py`, commit `a2610f5`).

Notation: `⌊·⌋` is integer truncation (`int()` in Python), `|X|` the size of a set.
Every equation is a separate function in `rostering/equations.py`; their ids (H1, C1, D1, F1, U1,
O1, ...) and contracts are listed in `docs/EQUATIONS.md`.

## 1. Sets

| Symbol | Meaning | Source |
|---|---|---|
| `D` | employed doctors | `doctors.csv` |
| `C` | candidates that may be hired | `doctors_to_hire.csv` |
| `A = D ∪ C` | all doctors | |
| `S` | shifts of the month | `shifts_*.csv` |
| `T` | days of the horizon (sorted dates of `S`) | |
| `S_t` | shifts starting on day `t` | |
| `W` | weeks; week of a day = `⌊(day - Monday of the first week) / 7⌋` | |
| `S_w` | shifts in week `w` | |
| `N ⊆ S` | night shifts: code contains `_N_` or 24 h shifts | |
| `N_t` | night shifts on day `t` | |
| `F ⊆ S` | 24 h shifts (`hours = 24`) | |
| `WE ⊆ S` | weekend shifts (Saturday, Sunday) | |
| `HS ⊆ S` | hard shifts (`is_hard_shift = 1`) | |
| `SP` | specialists: role ∈ {`specialist`, `icu_specialist`} | |
| `MN` | doctors needing a mentor (`needs_mentor = 1`, from `D` and `C`) | |
| `OO ⊆ D` | opt-out doctors (`opt_out = 1`) | |
| `R ⊆ D` | regular doctors (`opt_out = 0`) | |

## 2. Parameters

| Symbol | Meaning |
|---|---|
| `a_s, e_s` | start / end of shift `s` in hours from the start of the horizon: `24·index(day) + start_hour` (end hours after midnight are stored as e.g. 31 = 07:00 next day) |
| `h_s` | hours of shift `s` |
| `min_s, reg_s` | minimum and regular staffing of shift `s` |
| `skill_s` | skill required by `s`; `skills_d` skills of doctor `d` |
| `maxh_d` | weekly hour limit of `d` (candidates without the column: 48) |
| `night_d, tf_d` | night work / 24 h shifts allowed (candidates without the column: 1) |
| `rate_d` | hourly cost |
| `U_d^day, U_d^shift` | unavailable days / shifts (candidates: none) |
| `avail_d` | days in `T` not in `U_d^day` |
| `M_d = ⌊maxh_d · avail_d / 7⌋` | monthly hour limit (0 if `avail_d = 0`) |
| `avail_dw` | available days of `d` in week `w` |
| `L_dw = ⌊maxh_d · avail_dw / 7⌋` | weekly hour limit (0 if `avail_dw = 0`) |
| `pref_ds ∈ {like, dislike}` | preferences |

## 3. Variables

| Variable | Domain | Meaning |
|---|---|---|
| `x_ds` | {0,1} | doctor `d ∈ A` works shift `s` |
| `y_c` | {0,1} | candidate `c` is hired |
| `work_dt` | {0,1} | `d` works on day `t` (for H8) |
| `u_s` | [0, min_s] | understaffing of `s` |
| `o_s` | [0, reg_s] | "overstaffing slack" of `s` (see known issues) |
| `H_d` | [0, M_d] | hours worked in the month |
| `cost_d` | [0, 10⁷] | cost of `d` |
| `E_c` | [0, 2000] | effective hours of a candidate (`H_c · y_c`) |
| `WH_dw` | [0, L_dw] | hours worked in week `w` |
| `n_d, we_d, hd_d` | [0, 100] | night / weekend / hard shift counts |
| `spread_N, spread_WE, spread_H` | [0, 100] | max - min of these counts |
| `r_d` | [0, 2000] | workload ratio of opt-out doctors (per mille) |
| `UW_d` | [0, M_d] | monthly underwork of regular doctors |
| `WU_dw` | [0, L_dw] | weekly underwork |
| `v_{d1,d2,s}` | {0,1} | pandemic: cross-bubble contact |
| `Z` | [0, 10·SCALE] | Chebyshev objective (phase 3) |
| `ob` | [0, ⌊p·B⌋] | cost over budget (phase 3) |

## 4. Hard constraints

| Id | Constraint |
|---|---|
| H1 skills | `x_ds = 0` if `skill_s ∉ skills_d` |
| H2 one shift per day | `Σ_{s∈S_t} x_ds ≤ 1` ∀ d, t |
| H3 rest | `x_ds1 + x_ds2 ≤ 1` for all `s1 ≠ s2` with `e_s1 < a_s2 < e_s1 + 11` |
| H4 weekly hours | `Σ_{s∈S_w} h_s x_ds ≤ L_dw` ∀ d, w |
| H5 mentor | `x_ds ≤ Σ_{k ∈ SP ∩ D} x_ks` ∀ d ∈ MN, s |
| H6 consecutive nights | `Σ_{s ∈ N_t ∪ N_t+1 ∪ N_t+2} x_ds ≤ 2` ∀ d, t (windows of 3 days) |
| H7 day off after night | `Σ_{s∈N_t} x_ds + Σ_{s∈S_t+1} x_ds ≤ 1` ∀ d, t |
| H8 working days | `work_dt ≤ Σ_{s∈S_t} x_ds ≤ |S_t| · work_dt`, `Σ_{t ∈ block} work_dt ≤ 6` for every block of 7 consecutive days starting at the first day of the horizon |
| H9 24 h eligibility | `x_ds = 0` ∀ s ∈ F if `tf_d = 0` |
| H9a pandemic | `x_ds = 0` ∀ d, s ∈ F |
| H10 unavailability | `x_ds = 0` for `s ∈ S_t, t ∈ U_d^day` and for `s ∈ U_d^shift` |
| H11 hiring | `x_cs ≤ y_c` ∀ c ∈ C, s |
| H12 night eligibility | `x_ds = 0` ∀ s ∈ N if `night_d = 0` |
| HP1 pandemic isolation | for every contact shift `s` (dept ∈ {WARD, ICU}) and every pair `d1 < d2` (order of `A`) from different bubbles, both outside the CLINIC pool: `x_d1s + x_d2s ≤ 1 + v_{d1,d2,s}` |
| Hiring off | `y_c = 0` ∀ c in the baseline scenario (`allow_hire = False`) |

## 5. Coverage and definitions

* Coverage: `Σ_{d∈A} x_ds + u_s ≥ min_s` and `Σ_{d∈D} x_ds + o_s ≤ reg_s`.
* Slack weights `w_s`: 1, in pandemic mode ICU 100, WARD 50, CLINIC 1, other 10.
* Hours: `H_d = Σ_s h_s x_ds`, `WH_dw = Σ_{s∈S_w} h_s x_ds`.
* Cost: `cost_d = rate_d · H_d` (d ∈ D); `E_c = H_c · y_c`, `cost_c = rate_c · E_c` (c ∈ C).
* Preferences: `P = Σ_{like} (-x_ds) + Σ_{dislike} x_ds`.
* Fairness:
  * `n_d = Σ_{s∈N} x_ds` for specialists in `A` with `night_d = 1`;
  * `we_d = Σ_{s∈WE} x_ds` and `hd_d = Σ_{s∈HS} x_ds` for specialists in `D`;
  * `spread_X = max_d X_d - min_d X_d`.
* Workload ratio (built, not used in any objective): `r_d · M_d ∈ [1000 H_d - 50, 1000 H_d + 50]` for `d ∈ OO`.
* Monthly underwork: `UW_d ≥ ⌊0.95 M_d⌋ - H_d`, `UW_d ≥ 0` for `d ∈ R`.
* Weekly underwork: `WU_dw ≥ target_dw - WH_dw`, `WU_dw ≥ 0` for all `d ∈ A`, `w ∈ W`.
  * `target_dw = ⌊0.95 L_dw⌋` if `d ∈ R*`.
  * Otherwise `target_dw = ⌊0.95 · 48 · avail_dw / 7⌋`.
  * See known issue 1 for `R*`.

## 6. Objectives (lexicographic)

The phases are solved one after another; the optimum (or best solution found within the time limit)
of each phase is fixed with an equality constraint before the next phase.

1. **Understaffing**: minimise `Σ_s w_s u_s`, then add `Σ_s w_s u_s = best`.
1b. **Isolation** (pandemic, if any HP1 pair exists): minimise `Σ v`, then fix.
2. **Hiring**: minimise `Σ_c y_c`, then fix.
3. **Compromise** (min-max / Chebyshev): minimise `Z` subject to

```
Z ≥ 0
ob ≥ Σ_d cost_d - B,  ob ≥ 0,  ob ∈ [0, ⌊p·B⌋]
Z ≥ ob                       · ⌊SCALE / ⌊p·B⌋⌋
Z ≥ (spread_N  - 4)          · ⌊SCALE / |N|⌋
Z ≥ (spread_WE - 3)          · ⌊SCALE / |WE|⌋
Z ≥ (spread_H  - 2)          · ⌊SCALE / |HS|⌋
Z ≥ (Σ_d UW_d - 0)           · ⌊SCALE / Σ_{d∈R} ⌊0.95 M_d⌋⌋
Z ≥ (Σ_{d,w} WU_dw - 0)      · ⌊SCALE / Σ_{d∈A,w} ⌊0.95 L_dw⌋⌋
Z ≥ (P - (-300))             · ⌊SCALE / |preferences|⌋
```

with `SCALE = 10⁶`, budget `B = 540 000`, `p = 0.05` (`ObjectiveTargets` in `rostering/config.py`).
A term whose normaliser is 0 gets the coefficient 0. In the original code this was done only for
`|HS|`; the other terms crashed with `ZeroDivisionError`.

Pandemic mode also sets `linearization_level = 2` of the solver.

The single-phase experiment (`solve_single_phase_comparison`) minimises
`10⁶ Σ u_s + 10⁴ Σ y_c + Z` with `B = 1 620 000`, `p = 0.15` and the preference target `-200`.

## 7. Satisfaction score (reporting only)

These scores are computed from the solution and are not part of the optimisation.

* `S_pct` (`compute_satisfaction_v2`), per doctor, clipped to [0, 100]:

  ```
  util = H_d / Σ_w L_dw
  S_pct = 82 - 12·util + 6·√likes + 5·min(comfort_weeks, 4) + 6·min(free_weekends, 4) + 4·min(free_blocks, 6)
          - 7·excess_24h² - 5·excess_night - 4·excess_weekend - 3·excess_hard - 8·dislikes
  ```

  * `excess_X = max(0, count_X - limit_X)`.
  * The limits are the per-role standards scaled by `max(0.2, √util)`.
  * `comfort_weeks`: weeks with 70-90 % of `L_dw`.
  * `free_weekends`: ISO weeks without a weekend shift.
  * `free_blocks`: runs of ≥ 2 days off.
* `S_raw` is a simpler additive score with the same ingredients. It is reported, but `avg_satisfaction` uses `S_pct`.

## 8. Known issues (kept on purpose - the equations are unchanged)

1. **Weekly underwork uses the wrong doctors as "regular".** The original code tested
   `d in regular_doctors` on a pandas Series, which checks the row *index*, not the doctor id.
   * Effect in `doctors_with_cost.csv`: the regular doctors are ids {1, 7, 8, 9, 15}, but the target uses ids {6, 7, 8, 14}.
   * The code reproduces this explicitly (`Params.weekly_regular_lookup`).
2. **The overstaffing slack does not relax anything.**
   * `o_s ≥ 0` is not penalised, so `Σ_{d∈D} x_ds ≤ reg_s - o_s` is simply the hard cap `Σ_{d∈D} x_ds ≤ reg_s`.
   * Candidates are not counted in the cap.
   * `o_s` has the same name (`slack_<s>`) as `u_s`.
3. **The budget is a hard cap.** The upper bound of `ob` makes `Σ cost_d ≤ B · (1 + p)` a hard
   constraint in phase 3. If the cost cannot stay below the cap, phase 3 is infeasible.
   * Example: datasets `h30` and `h45` (30/45 doctors) with the default `B = 540 000`.
   * The original code then silently returned and saved a meaningless result (status INFEASIBLE,
     slack 163, cost 9, satisfaction 0 on `h30`).
   * `solve.py` now stops with an explicit error. Larger hospitals need their own `ObjectiveTargets.cost`.
4. **H8 is not "35 h weekly rest".**
   * The original comment said so, but the constraint limits working days to 6 per 7-day block counted from the first day of the horizon.
   * H4 and the weekly underwork use calendar weeks (from Monday) instead.
5. **The weekly target can exceed the weekly limit.** For doctors outside `R*`, `target_dw` uses a
   48 h base. When `0.95 · 48 > maxh_d`, `target_dw` can exceed `L_dw`, and the bound `WU_dw ≤ L_dw` turns into
   the hard constraint `WH_dw ≥ target_dw - L_dw`. These cases are logged at DEBUG level.
6. **Unhired candidates still get a weekly underwork term.** It is a constant, so it does not change
   the optimum, but it shifts the normalised value.
7. **The night-shift fairness group includes candidates; weekend and hard-shift fairness do not.**
   Unhired candidates have 0 night shifts, so in the baseline the minimum is always 0. `spread_N`
   is then simply the largest number of night shifts of a specialist.
8. **H3 does not catch overlapping shifts** (`a_s2 ≤ e_s1`). These are excluded by H2 and H7.
9. **The workload ratio (F4) restricts hours although no objective uses it.** `r_d · M_d` must lie
   within ±50 of `1000 H_d` for an integer `r_d`. When `M_d > 100`, the interval is narrower than 1,
   so many values of `H_d` have no such `r_d`.
   * Only about 40 % of the possible monthly hours are allowed for opt-out doctors.
   * Example: with `M_d = 248`, `H_d` cannot be 162-169.
   * This affects 8 of the 14 doctors in `doctors_with_cost.csv`.
10. **Specialists who "need a mentor" are their own mentors.** In `doctors_with_cost.csv` doctors
    6, 7 and 8 are specialists with `needs_mentor = 1`. H5 counts the doctor itself among the
    specialists on the shift, so the requirement never binds for them.
