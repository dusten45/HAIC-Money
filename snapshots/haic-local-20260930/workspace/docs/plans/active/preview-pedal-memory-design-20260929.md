# Preview pedal continuity repair

User strategy remains preview acceleration and short braking. R1 PIVOT evidence: 30matchedepisodes control6/6,distance1/6,time0/6,steering5/6,response0/6. Do not abandon preview because of the first failure. Revise its state architecture, preserving R1 source and stableZIP.

R1 response track1/38300 accelerates atstep131 when target37.2 rebounds62.9 withsteer-.307, before geometry loss. At139 geometry loss returns inheritedgas.416 andsteer0. Distance arm similarly loses a curvature restriction before turn settles. No collision in either failure. Falsifiable hypothesis: maintaining observed restrictions across image changes prevents premature acceleration.

Create separate preview_pedal_memory_runtime.py from R1. Retain four independent parent pedal mechanisms (distance,timebudget,steeringbudget,response) and control. Apply common continuity repair: propagate curve/obstacle constraints byHUDspeed*.08, retainuntil3distanceunits behind orage10 decisions, never memorize horizon; keep pedal planning during atmost4missinggeometrydecisions ifmemoryexists; latch lowtarget<55 and release only after3successive aligned near-road frames (error<3,abssteer<.2,steerdelta<.08). Memory target may decrease before release. This deliberately combines three continuity changes; results cannot assign causality to just one. Log agedmemory,grace,latch,release counters and source of bound.

5arms x6 consumedTRAINcells tracks1–3/seeds38300,38302 =30episodes. Identical control andfirst10prefix.2CPU2GiB1200s,1200decisions. Acceptance6/6plusfasterthanstablemedian; no release without broader comparison/exactZIP. Do not label these fresh. Standing local authorization covers design and execution; record newplanhash/separateevents. No external action or baseline replacement. Further diagnosis required if this repair fails.

Execution: save R1 unchanged, add runtime+actorroute, register exact source hash, read-only review, approved CLIrun, audit results/control replay and integrate three gates. No parameter sweep or unrelatedtests.
