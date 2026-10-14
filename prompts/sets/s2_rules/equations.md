# Set s2_rules - the exact rules, without the CP-SAT encoding

Who is concerned, what is counted and the numbers of the rules are given. The variable domains,
the encoding of the constraints and the formulas of the objectives are not.

## H1
A doctor may only work a shift if the skill required by the shift is one of the doctor's skills.

## H2
Every doctor works at most one shift per day; a shift belongs to the day on which it starts.

## H3
Between the end of a shift and the start of a later shift worked by the same doctor there must be
at least 11 hours, measured with the absolute times. Shifts that overlap or follow each other
without any gap are not concerned.

## H4
The hours a doctor works in a calendar week must not exceed the doctor's limit for that week.

## H5
A doctor who needs a mentor may only work a shift together with at least one specialist from
`m.p.specialists`. A doctor who is a specialist counts as that specialist.

## H6
Any three consecutive days of the horizon contain at most two night shifts of the same doctor.

## H7
A doctor who works a night shift starting on one day works no shift starting on the next day of
the horizon.

## H8
Counting in blocks of 7 days from the first day of the horizon, a doctor works on at most 6 days of
each block.

## H9
Doctors not allowed to work 24-hour shifts do not work them.

## H9a
In pandemic mode nobody works a 24-hour shift.

## H10a
A doctor works no shift starting on one of the doctor's unavailable days.

## H10b
A doctor does not work the shifts listed as unavailable for that doctor.

## H11
A candidate works only if hired.

## H12
Doctors not allowed to work at night do not work night shifts.

## H13
No candidate is hired.

## HP1
Pandemic mode: on shifts of the contact departments, every pair of doctors from different bubbles
working together is a contact. Doctors without a bubble or in bubble -1 are outside the system.
Represent every possible contact by a Boolean variable that is forced to 1 when the contact
happens, and store the list in `m.pandemic_bundle["crossbubble_violations"]`.

## C1
A shift should have at least its minimum staffing, counting employed doctors and candidates. The
missing number of doctors per shift is a variable that phase 1 minimises.

## C2
The number of employed doctors on a shift (candidates are not counted) must not exceed its regular
staffing; formulate it with a non-negative slack variable.

## D1
Worked hours of a doctor in the month = sum of the lengths of the worked shifts; they may not
exceed the monthly limit.

## D2a
Cost of an employed doctor = worked hours times the hourly cost.

## D2b
Cost of a candidate = hourly cost times the worked hours, counted only if the candidate is hired.

## D3
Worked hours of a doctor in every week = sum of the lengths of the worked shifts of that week; they
may not exceed the weekly limit.

## S1
The preference score adds -1 for every "like" shift that is worked and +1 for every "dislike" shift
that is worked.

## F1
Among the specialists allowed to work at night, the spread is the largest minus the smallest number
of night shifts worked. The group includes employed doctors and candidates, and candidates count
even when they are not hired (with 0 night shifts).

## F2
Among the employed specialists, the spread is the largest minus the smallest number of weekend
shifts worked.

## F3
Among the employed specialists, the spread is the largest minus the smallest number of hard shifts
worked.

## F4
For every opt-out doctor, the load ratio is the worked hours per mille of the monthly limit, as an
integer, with a tolerance of 50 on the scaled value. The output is the largest minus the smallest
ratio (0 without opt-out doctors).

## U1
For every regular doctor, the monthly underwork is how many hours are missing to 95 % of the
monthly limit (truncated to an integer, never negative). For a candidate the target only applies
if hired.

## U2
For every doctor and week, the weekly underwork is how many hours are missing to the weekly target
(never negative, at most the weekly limit). The target is 95 % of the weekly limit for the doctors
in `m.p.weekly_regular_lookup`. For everybody else it is 95 % of a 48-hour week scaled by the
available days of that week. Both targets are truncated to an integer.

## O1
Weighted understaffing over all shifts. The weight is 1. In pandemic mode the weight depends on the
department: ICU 100, WARD 50, CLINIC 1, other 10.

## O1b
Number of cross-bubble contacts.

## O2
Number of hired candidates.

## O3
Minimise the largest normalised deviation of six criteria from their targets in `m.targets`. Each
deviation is scaled by `m.targets.scale` divided (integer division) by the largest possible value
of the criterion, and the result may not be negative. The criteria:
- The budget overrun above `cost`. Its scale is the allowed overrun `cost * max_over_budget_pct`, which is also the largest overrun allowed.
- The night, weekend and hard-shift spreads, scaled by the number of night, weekend and hard shifts.
- The total monthly underwork, scaled by the sum of the monthly targets of the regular doctors.
- The total weekly underwork, scaled by the sum of 95 % of all weekly limits.
- The preference score, scaled by the number of preferences.
