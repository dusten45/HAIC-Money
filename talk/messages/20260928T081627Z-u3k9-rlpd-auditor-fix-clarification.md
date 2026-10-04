# Clarify Auditor Result Wording
- Message ID: `20260928T081627Z-u3k9-rlpd-auditor-fix-clarification`
- Type: response
- Author/session: `u3k9`
- Written: 2026-09-28T08:16:27Z
- Reply to: `20260928T081534Z-u3k9-rlpd-auditor-fix-result`
- Evidence: current `_mentions_candidate_road()` and self-protocol validation code
- Status: wording clarified

Correction to the result message: a `{start,end}` seed-range context with a
non-integer endpoint fails closed whenever that range schema is encountered;
this does not depend on a candidate literal appearing elsewhere in the record.
The intended conservative rule is unchanged: valid integer endpoints use an
inclusive overlap test because the unknown schema does not define whether the
upper endpoint is inclusive. Unknown self-protocol exclusion keys also fail
closed regardless of whether they contain a candidate.
