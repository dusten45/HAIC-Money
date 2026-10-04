# Prospective single correction: preserve a transported pass constraint

Exact early-circle track2 replay sees its real circle onframes94/96 but misses
95/97 because antialiased core aspect is0.625, just below unchanged0.65gate.
Transported memory remains valid for those one-frame misses; no global detector
relaxation is proposed here.

Atmiss95 the corridor optimizer drops remembered-center separation from base
pass3.70m to2.73m. Atmiss97 it drops3.72m to2.40m, below body+circle2.8m.
Its reference curvature becomes−.00534/m, causing feedforward−.0173 plus yaw
feedback−.0250 =countersteer−.0424. The original remembered basepass curvature
+.01487/m would instead emit+.0460steer. The path is lost before yaw feedback;
feedback alone does not explain the countersteer.

The optimizer constrains only currently detected circles even after the base
pass routine has transported a remembered circle. Preserve that single valid
transported circle as an additional optimizer constraint whenever pass_side
is active and pass_missing>0. Invoke base `_PathReference._route` once, and use
its updated pass_x/pass_y. Preserve memory expiry, road limits, ellipse margin,
fallback, optimizer, detectors and controls. Test savedmiss95/97 RED before
implementation; visible-circle/expired-memory/clear/reset decisions stayexact.
One fresh four mandatory benchmark follows focused GREEN tests; no holdout.
