# The Decay Ladder

**Service:** Nimbus Retry Orchestrator
**Version:** 4.2
**Status:** Current
**Last reviewed:** 2026-09-30

---

## Overview

The decay ladder is Nimbus's retry schedule for failed card charges. It is a
fixed sequence of five rungs. Each rung defines a wait interval before the next
attempt, and a set of decline classes that are eligible for that rung.

The ladder is deliberately not a pure exponential curve. Rung intervals were
derived from acquirer recovery telemetry collected between 2024 and 2025, and
the gap between rung 3 and rung 4 is wider than an exponential fit would
predict. See [Why rung 4 is slow](#why-rung-4-is-slow).

## Rung intervals

| Rung | Wait before attempt | Jitter coefficient | Eligible declines |
|------|--------------------|--------------------|-------------------|
| 1    | 2s                 | 1.47               | soft only         |
| 2    | 9s                 | 1.47               | soft only         |
| 3    | 47s                | 1.47               | soft, grey        |
| 4    | 4m 10s             | 2.10               | soft, grey        |
| 5    | 22m                | 2.10               | soft only         |

The jitter coefficient is a multiplier, not an additive window. The actual wait
is drawn uniformly from `[interval, interval × coefficient]`. A rung 3 attempt
therefore fires between 47 and 69 seconds after the rung 2 failure.

Hard declines are never eligible for any rung. A hard decline terminates the
ladder immediately and emits `NBS-221`.

## Consecutive failure handling

A charge that fails on rungs 1, 2 and 3 has failed three consecutive times on
the same acquirer. **Nimbus does not attempt rung 4 on that acquirer.**

Instead the charge is *promoted*: it is reassigned to the merchant's secondary
acquirer, and the ladder resumes at **rung 2**, not rung 1.

Resuming at rung 2 is intentional. Rung 1's 2-second interval exists to catch
transient authorisation races on a cold card — a card Nimbus has not attempted
in the current session. A card that has already failed three times is not cold,
and a 2-second retry against a fresh acquirer reliably produces a fourth decline
without adding information. Rung 2's 9-second interval is the shortest wait that
showed recovery lift in the promotion cohort.

If the merchant has no secondary acquirer configured, promotion is skipped and
the ladder continues to rung 4 on the primary. This is the only case in which
rung 4 runs against an acquirer that has already declined three times.

### Promotion is not a reset

Promotion carries the attempt counter forward. A promoted charge that fails at
rung 2 proceeds to rung 3, then exhausts — it does not get a second promotion.
Maximum total attempts across both acquirers is therefore six: three on the
primary, three on the secondary.

Attempting a third acquirer requires `ladder.promotion_depth = 2`, which is
off by default and is not supported for merchants on the shared acquirer pool.

## Interaction with the settlement fence

No rung may fire inside the settlement fence — the 90-second window after an
attempt during which the acquirer may still settle a charge Nimbus believes
failed. Rungs 1 and 2 both fall inside that window.

This is resolved by fence class, not by delaying the rung:

- **Soft declines** carry an explicit acquirer rejection, so the fence does not
  apply and rungs 1 and 2 fire on schedule.
- **Grey declines** (timeout, connection reset, malformed acquirer response)
  have no rejection, so the fence applies. Grey declines become eligible at
  rung 3, which is why rungs 1 and 2 list soft declines only.

A rung that fires inside an active fence against a grey decline is a fence
violation and is rejected with `NBS-409`. The ladder is not advanced; the
attempt is not counted.

## Why rung 4 is slow

Rung 4 sits at 4m 10s rather than the ~3m an exponential fit from rungs 1–3
would give. Acquirer telemetry showed a recovery trough between 2 and 4 minutes
after a third decline, attributed to issuer-side velocity checks that clear on a
5-minute cycle. Attempting inside the trough consumed an attempt at roughly half
the success rate of waiting for it to clear.

Rung 4's jitter coefficient was also raised to 2.10 for the same reason: the
wider draw spreads attempts across the far side of the trough.

## Configuration

```
ladder.enabled              true
ladder.promotion_depth      1
ladder.rung_5_enabled       true
ladder.fence_override       false
```

`ladder.rung_5_enabled` is the only rung that can be disabled individually.
Merchants on weekly settlement generally disable it, since a 22-minute retry
can cross a settlement boundary and land the charge in the following cycle.

`ladder.fence_override` disables fence checks entirely. It exists for acquirer
integration testing and must not be enabled in production: see
[Ghost charges](./ghost-charges.md).

## Related

- [Carry tokens](./carry-tokens.md) — how attempts are made idempotent
- [Decline classes](./decline-classes.md) — soft, grey and hard
- [Ghost charges](./ghost-charges.md) — what the fence protects against
- [Ladder v3 (deprecated)](./deprecated/retry-ladder-v3.md) — superseded 2026-04

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 4.2 | 2026-09-30 | Clarified that promotion carries the attempt counter forward |
| 4.1 | 2026-06-11 | `promotion_depth` added; third-acquirer support behind a flag |
| 4.0 | 2026-04-02 | Rung intervals revised; promotion resumes at rung 2 (was rung 1) |
| 3.4 | 2025-11-18 | Grey decline class split out of soft |