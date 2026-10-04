# Clear-road parametric ridge with preserved support and hazard decisions

The explicit confidence source reproduces all four initial graph finishes but
still averages 15–24 seconds and finishes only 12/16 extra cells. Its row
representation misplaces nearly horizontal curves by about 2.5 m. The frozen
NumPy ridge reduces selected errors below 1 m but applying it to every scene
loses obstacle finishes.

This root experiment combines those verified observations in a new standalone
source: copy the confidence controller exactly, and use parametric ridge
pursuit only on supported clear road. Current circles or an active pass retain
the confidence controller's exact decisions. Missing support also retains its
bounded recovery, which rescued four required finishes. Ridge segment samples
must stay in the parsed distance field; no sampled geometry is a dynamic or
full hull safety certificate.

On selected clear road, steering/target memory is computed once from its
pre-action state. The inherited tyre/propulsion rules apply to the new path,
with the independently tested projected emitted-steering gas reserve. No
simulator state, seed, external import or new holdout is used. Regressions
precede implementation, followed by one source-frozen four-cell benchmark.
