# Set s1_intent - the intent in the words of a hospital planner

One or two sentences per equation: what the rule is for, without groups of doctors, exact numbers
of the encoding, variable domains or formulas. The model has to infer the details.

## H1
Doctors can only be scheduled for shifts they are qualified for.

## H2
Nobody works more than one shift a day.

## H3
Doctors need at least 11 hours of rest between two shifts.

## H4
Respect every doctor's weekly working-hour limit.

## H5
Doctors who need supervision must always have a specialist on the same shift.

## H6
No more than two night shifts within three consecutive days.

## H7
After a night shift the doctor gets the next day off.

## H8
At most six working days per week.

## H9
Only doctors who are allowed to do 24-hour shifts get them.

## H9a
During a pandemic there are no 24-hour shifts.

## H10a
Respect the days on which doctors are on leave or otherwise absent.

## H10b
Respect the individual shifts that doctors cannot work.

## H11
Candidates only work if we hire them.

## H12
Only doctors who are allowed to work at night get night shifts.

## H13
In the baseline scenario we do not hire anyone.

## HP1
In a pandemic the doctors are split into isolated bubbles. Doctors from different bubbles should
not work together in contact departments; record every such contact so that it can be minimised.

## C1
Every shift needs its minimum number of doctors; measure how many are missing so that it can be
minimised instead of making the problem infeasible.

## C2
Do not put more employed doctors on a shift than its regular staffing.

## D1
Keep track of how many hours each doctor works in the month.

## D2a
Keep track of the salary cost of every employed doctor.

## D2b
Keep track of the cost of every candidate - a candidate is only paid if hired.

## D3
Keep track of how many hours each doctor works in each week.

## S1
Score how well the schedule respects the doctors' likes and dislikes (lower is better).

## F1
Night shifts should be shared fairly among the specialists; measure the difference between the
busiest and the least busy one.

## F2
Weekend shifts should be shared fairly among the employed specialists; measure the difference
between the busiest and the least busy one.

## F3
Hard shifts should be shared fairly among the employed specialists; measure the difference between
the busiest and the least busy one.

## F4
Measure how evenly the opt-out doctors are loaded relative to their monthly limits.

## U1
Measure how far the regular doctors fall short of a full month of work.

## U2
Measure how far every doctor falls short of full employment in every week.

## O1
Phase 1 goal: minimise understaffing; in a pandemic intensive care and wards matter more than
clinics.

## O1b
Phase 1b goal (pandemic): minimise the contacts between bubbles.

## O2
Phase 2 goal: hire as few candidates as possible.

## O3
Phase 3 goal: a balanced compromise between staying within the budget, fairness, underwork and
preferences - minimise the worst normalised deviation from its target (the targets are in
`m.targets`).
