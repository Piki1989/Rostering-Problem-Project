# Set s3_spec - full specification in natural language

Every `## <ID>` section is the description inserted into the prompt of that equation.
- It states the exact rules, the variable domains and where the outputs are stored, but contains no code.
- `python -m rostering.prompting prompts --set s3_spec` shows the resulting prompts.

## H1
A doctor may only work a shift if the skill required by the shift is one of the doctor's skills.

## H2
Every doctor works at most one shift per day (a shift belongs to the day on which it starts).

## H3
A doctor needs at least 11 hours of rest between the end of one shift and the start of a later shift
that the doctor also works. Use the absolute start and end times. The rule only concerns a pair of
shifts where the second one starts strictly after the first one ends; shifts that overlap or start
exactly when the other ends are not restricted by this rule.

## H4
In every week, the total number of hours a doctor works (sum of the lengths of the shifts of that
week) must not exceed the doctor's weekly hour limit for that week.

## H5
A doctor who needs a mentor may only work a shift if at least one specialist from `m.p.specialists`
works the same shift.

## H6
In every window of three consecutive days of the horizon (consecutive entries of the day list), a
doctor works at most two night shifts.

## H7
A doctor who works a night shift that starts on some day must not work any shift that starts on the
next day of the horizon. Only apply this to pairs of consecutive days where the first day has night
shifts and the next day has shifts.

For such a pair of days, state it as one inequality per doctor: the number of night shifts the
doctor works on the first day plus the number of shifts the doctor works on the next day is at
most 1.

## H8
Split the days of the horizon into consecutive blocks of 7 days, starting with the first day (the
last block may be shorter). In every block, a doctor works on at most 6 different days.

## H9
Doctors who are not allowed to work 24-hour shifts must not work any 24-hour shift.

## H9a
In pandemic mode, no doctor works any 24-hour shift.

## H10a
A doctor does not work any shift that starts on a day on which the doctor is unavailable.

## H10b
A doctor does not work the single shifts for which the doctor is unavailable.

## H11
A candidate may only work shifts if the candidate is hired.

## H12
Doctors who are not allowed to work at night must not work any night shift.

## H13
In the baseline scenario no candidate is hired.

## HP1
Pandemic mode: doctors are split into isolated bubbles. Consider every shift whose department is
one of the contact departments. For every such shift, consider every unordered pair of different
doctors with different bubbles. Skip doctors that have no bubble or whose bubble is -1.

For every such shift and pair, create a Boolean violation variable that must be 1 if both doctors
work the shift (it may be 0 otherwise). Store all violation variables in
`m.pandemic_bundle = {"crossbubble_violations": [...]}`.

## C1
Every shift should have at least its minimum number of doctors (employed doctors and candidates
together). For every shift create an integer understaffing variable with domain from 0 to the
minimum staffing of the shift. The number of doctors working the shift plus the understaffing
variable must be at least the minimum staffing. Store it in `m.slacks[s]`.

## C2
The number of employed doctors (not candidates) working a shift must not exceed the regular
staffing of the shift. Formulate it with an integer slack variable with domain from 0 to the
regular staffing: the number of employed doctors plus the slack is at most the regular staffing.
Store the slack in `m.slacks_o[s]`.

## D1
For every doctor, define an integer variable for the hours worked in the month: the sum of the
lengths of the shifts the doctor works. Its domain is 0 to the doctor's monthly hour limit. Store
it in `m.worked_hours[d]`.

## D2a
For every employed doctor, define an integer cost variable with domain 0 to 10,000,000. It equals
the worked hours times the doctor's hourly cost. Store it in `m.doctor_cost[d]`.

## D2b
For every candidate, define the effective hours as an integer variable with domain 0 to 2000. It
equals the worked hours times the hire variable (a product of two variables).

Then define an integer cost variable with domain 0 to 10,000,000, equal to the effective hours
times the candidate's hourly cost. Store the cost in `m.doctor_cost[c]`.

## D3
For every doctor and week, define an integer variable for the hours worked in that week: the sum of
the lengths of the shifts of that week the doctor works. Its domain is 0 to the doctor's weekly
hour limit. Store it in `m.weekly_hours[d][w]` (a dict of dicts).

## S1
Preference score of the schedule: -1 for every "like" preference whose shift the doctor works, and
+1 for every "dislike" preference whose shift the doctor works. Store the score as a list of linear
terms (one per preference) in `m.pref_terms`; the score is their sum.

## F1
Night-shift fairness. For every doctor in `m.p.night_allowed` (employed doctors and candidates)
who is a specialist and allowed to work at night, define an integer variable with domain 0 to 100
for the number of night shifts the doctor works. Candidates count too, even when they are not
hired (they then have 0 night shifts).

Define integer variables (domain 0 to 100) that are exactly equal to the maximum and to the minimum
of these counts; equality, not only bounds. Store the spread, max minus min, as an integer variable
with domain 0 to 100 in `m.spread`.

## F2
Weekend fairness. For every employed specialist (not candidates), define an integer variable with
domain 0 to 100 for the number of weekend shifts the doctor works.

Define integer variables (domain 0 to 100) that are exactly equal to the maximum and to the minimum
of these counts. Store the spread, max minus min, as an integer variable with domain 0 to 100 in
`m.weekend_spread`.

## F3
Hard-shift fairness. For every employed specialist (not candidates), define an integer variable
with domain 0 to 100 for the number of hard shifts the doctor works.

Define integer variables (domain 0 to 100) that are exactly equal to the maximum and to the minimum
of these counts. Store the spread, max minus min, as an integer variable with domain 0 to 100 in
`m.hard_spread`.

## F4
Workload ratio of opt-out doctors, in per mille of their monthly limit. For every opt-out doctor,
define an integer ratio variable with domain 0 to 2000. The ratio times the monthly hour limit must
lie between 1000 times the worked hours minus 50 and 1000 times the worked hours plus 50
(inclusive).

Define integer variables (domain 0 to 2000) exactly equal to the maximum and the minimum of the
ratios. Store their difference as an integer variable with domain 0 to 2000 in `m.ratio_spread`.
Without opt-out doctors, the maximum, the minimum and the spread are 0.

## U1
Monthly underwork of regular doctors. The target is 95 % of the monthly hour limit, truncated to an
integer. For a candidate in the list, the target is multiplied by the hire variable.

For every doctor in `m.p.regular_doctors`, define an integer variable with domain 0 to the monthly
hour limit. It must be at least the target minus the worked hours, and at least 0. Store it in
`m.underwork[d]`.

## U2
Weekly underwork of every doctor (`m.ctx.D_all`) in every week. The weekly target depends on the
doctor:
- Doctors in `m.p.weekly_regular_lookup`: 95 % of the weekly hour limit.
- All other doctors: 95 % of 48 hours times the doctor's available days in that week divided by 7.

In both cases the target is truncated to an integer.

Define an integer variable with domain 0 to the weekly hour limit. It must be at least the target
minus the weekly worked hours, and at least 0. Store it in `m.weekly_underwork[d][w]` (a dict of
dicts).

## O1
Return the objective of phase 1: the weighted sum of the understaffing variables.

The weight of a shift is 1, except in pandemic mode (when `m.pandemic_config` is not None):
- ICU: 100
- WARD: 50
- CLINIC: 1
- any other department: 10

## O1b
Return the objective of phase 1b in pandemic mode: the sum of all cross-bubble violation variables.

## O2
Return the objective of phase 2: the number of hired candidates.

## O3
Return the variable `Z` of the compromise objective of phase 3 (to be minimised), after adding its
constraints. Let `t = m.targets` and `SCALE = t.scale`. All divisions are integer divisions
(floor).

Create the variables:
- `Z`: integer with domain 0 to 10 times SCALE, and require Z >= 0.
- Budget: `max_over = int(t.cost * t.max_over_budget_pct)`.
- `over`: integer with domain 0 to max_over, with over >= (total cost of all doctors) - t.cost and over >= 0.

Add these constraints:
- Budget: if max_over > 0, Z >= over * (SCALE // max_over).
- Spreads:
  - Z >= (night spread - t.spread) * c(number of night shifts).
  - Z >= (weekend spread - t.weekend_spread) * c(number of weekend shifts).
  - Z >= (hard spread - t.hard_spread) * c(number of hard shifts).
- Monthly underwork: Z >= (sum of monthly underwork - t.underwork) * c(sum over regular doctors of int(0.95 * monthly limit)).
- Weekly underwork: Z >= (sum of weekly underwork - t.weekly_underwork) * c(sum over all doctors and weeks of int(0.95 * weekly limit)).
- Preferences: Z >= (preference score - t.pref) * c(number of preferences).

Here `c(n) = SCALE // n`, or 0 when n is 0.
