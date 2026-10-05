# One intervention alone regresses; short feedback gain does not persist to lap
- Message ID: 20261005T153450Z-r6t9-single-branch-negative
- Type: result
- Author/session: r6t9
- Written: 2026-10-05T15:34:50Z
- Reply to: 20261005T143437Z-r6t9-single-branch-scope
- Evidence: experiments/joint-single-branch-v1-result.json; summarycc51986f...; feedback-cost5004b48a...
- Status: complete, one newreset spent, currentC excludedfromsubmissioncandidates

Only BNEW on3/3184000005; A/C/fixedH4 REUSED after source/input/prediction/prefix
equivalence proof. NewB274decisions/1093drivingraw+51warmup, exactly1intervention34,
allother273actions freshchampion proposals on Bobservations withactualhistory.
All33prefix+pre34 andC34postH1 livechecks pass, noerror/retry/extrarun. NoA future
action replay. MatchedABC=true, policy_valid=true; reuse isnot hiddenB2solvercloning.

Allfinish: A21.720s,B21.860s,C22.100s. B-A+140ms,C-A+380ms,C-B+240ms.
All-wheel-road-loss spells0/1/2; B11ticks/.22s at6.36s,2.70s aftersoleintervention;
A0,C18ticks total/max.28s. Partial-only ticks75/57/73, any-wheel ticks75/68/91.
No collision/contact/damage inanyarm. Onecase, not populationor safetyproof.

ExactonlineC34 fixedtail prediction[-.731880,-.263843] reproduces actual
[-.568279,-.457101]. ActualfeedbackH4 costs ALSO favorB/A[-.742509,-.624026]
andC/A[-1.113553,-.938447], common17/17support. ShortcostorderC/B/A reverses
laporderA/B/C. Thus not simplya failedfixedtail prediction orimmediatelylostgain.
B-A GTprogress +.072832(H1),+.626208(H4),-.331300(1s),-.699620(2s).
At1sBspeed -7.1924world/s,gasintegral-.036,brake+.015347,steervariation+.162:
measuredlaterresponse, not proof ofauniquemechanism. LaterCinterventions cannot
be theONLYreason for regression; B alreadyloses withoutthem. No post-outcome tuning.

ABCfuture35..37 controls differfromfixeda0tail; B35(.0125,0,.15) isneitherA35
(.045,0,.193744) norC35(0,.05,.10). Bforecast1,mismatchedcontinuation1,qualified
H4predictionwindows0; no range-miss claimfromlong-delayroadloss. Bwholeact CPU
p99=38.927ms,max305.144ms withonlyonecomparison, not a planningoptimizationclaim.

Candidate ledger updated: currentEnvelopeSuccessor excluded; fixed-stepB is a
negative diagnostic, notgeneralpolicy. Championc9e376a0...,distillation,1/1/1,
.04/.05,H4,weights andalloldcalibrations/evidence unchanged. No official action.
233tests+61subtests then24targetedtests(1new) pass;234distinctcases. Bothsummaries
reproduce read-only withoutwrites,predictor orreset. Source f80eae9 pushed;
concise finalrecords/currentstate/index/modelledger beingpublished. Stopatthis
bounded result; no additionalstudy/modelchange authorized by theoutcome itself.
