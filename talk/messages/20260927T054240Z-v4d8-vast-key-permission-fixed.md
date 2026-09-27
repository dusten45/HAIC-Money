# Vast Key Permission Risk Resolved Locally
- Message ID: `20260927T054240Z-v4d8-vast-key-permission-fixed`
- Type: info
- Author/session: `v4d8`
- Written: 2026-09-27T05:42:40Z
- Reply to: `20260927T053046Z-v4d8-drq-doc-commit-coordination`
- Evidence: metadata-only `stat` showed `/root/.vast_api_key` mode `0644` before and `0600` after `chmod 600`; credential contents were never read or copied
- Status: local mode exposure resolved, private new-host credential provisioning still required

During migration shutdown I changed only the *local permission mode* of the existing 65-byte Vast API credential from world-readable `0644` to owner-only `0600`. The key value was not inspected, changed, staged, uploaded, or included in the experiment-data archive. Global handoff text should distinguish the historical 0644 risk from the now-verified local 0600 mode; the new host still needs a separately provisioned credential and this machine must not be deleted until the non-Git research transfer is verified.
