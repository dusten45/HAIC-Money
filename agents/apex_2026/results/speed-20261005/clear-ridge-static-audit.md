# Clear ridge static review

Frozen V1 source SHA256 `86230b49efa177b28eafd8f62190fc2260e3203ad832046b371e23ce03e41d2f`
uses the ego-connected component for the inherited confidence road, but its
distance-field ridge uses every road-colored pixel. Its dense chord check
covers the ridge itself, not the gap from the actual ego pose to the ridge
start. The ridge walk also has no monotonic forward-distance requirement.

A read-only 84×84 synthetic frame with one ego road strip in columns 39:47,
one separate broader strip in columns 48:80, and one grass column between
them selects a V1 ridge beginning at metric `(8,3)`, 8.54 m from the ego pose.
The ridge then reverses to forward coordinate −4.18 m. All sampled chords
have distance-field clearance at least 4.33 m, so V1 emits steer +0.24 in
`ridge` mode; the confidence controller emits +0.003 on the same frame. This
establishes a concrete V1 selection error, without attributing its observed
track-4 failure to that frame.

The current V2 depth-support candidate, source SHA256
`e5ce693a4b9ec00d7ecbab8abf71d4a2c350ab95b4f816ae2b6547be91cfc00b`,
rejects this same ridge and exactly retains the confidence action. V2's four
mandatory finishes are empirical evidence only; this probe does not certify
physical clearance, tire behavior, or all disconnected road scenes. No
episode or holdout was run for this review.
