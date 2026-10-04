# Combined controller V1 rejected

Four selection, steering-memory and real-camera/reset tests pass. Both embedded
sources match their frozen parents except public class renames. Independent
static review found no concrete source blocker; this did not certify driving.

Fresh mandatory source-frozen result: track1 DNF at0.310714 progress,3contacts;
track2 24.68s,track3 18.98s,track4 21.28s, all three withzero contacts.
The10–13s target is unmet and the source is not adopted. No holdout was opened.

Combining fast clear road with an accepted committed hazard reference fails
the required coverage even when both separate sources finish allfour. This
rejects the simple selector hypothesis; constituent laps are not combined
into a synthetic claim. An exact first-contact camera replay investigates
controller handoff and route tracking before any new selector is proposed.
