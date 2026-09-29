# Three-class pixel student TRAIN driving: REJECT

On matched TRAIN tracks 1–3 × seeds 5020–5023, the selected stable pixel control validly finished 9/12 and the learned hold/boost/coast student 8/12. The student made 1,519 learned boost and 64 learned coast actions with zero invalid actions. Finished-lap medians were 23.96 s control and 22.51 s student, but completion takes priority. The student had 14 collisions and 2.8 damage versus control 7 collisions and 1.4 damage. Candidate-only failures were 1:5020, 2:5021 and 3:5020; control-only finishes were not enough to offset the completion count. The preregistered endpoint failed. Formal outcome `REJECT`: rule compliance `UNKNOWN`, mechanism activation `PASS`, competitive outcome `FAIL`, no release. These seeds are consumed TRAIN and were not used to change the model. No TUNE or external action occurred.

The classifier learned and expressed coasting, but positive boost remained dominant. This is evidence that the new coast class alone did not preserve completion, not evidence that a particular coast threshold would fix it. The training checkpoint is retained for diagnosis and is not a submission candidate.

Frozen source and checkpoint: `tmp/haic2-coast-class-driving-train-frozen-20260929/`. The source SHA-256 entries in the run manifest refer to that snapshot.
