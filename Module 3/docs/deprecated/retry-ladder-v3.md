# The Decay Ladder (v3, deprecated)

**Service:** Nimbus Retry Orchestrator
**Version:** 3.4
**Status:** DEPRECATED, superseded by v4.0 on 2026-04-02
**Last reviewed:** 2025-11-18

---

> This version is kept for audit purposes only. Merchants were migrated to v4
> in April 2026. See [The Decay Ladder](../retry-ladder.md) for current
> behaviour.

## Overview

The decay ladder is Nimbus's retry schedule for failed card charges. It is a
fixed sequence of five rungs following an exponential curve.

## Rung intervals

| Rung | Wait before attempt | Jitter coefficient | Eligible declines |
|------|--------------------|--------------------|-------------------|
| 1    | 2s                 | 1.47               | soft only         |
| 2    | 8s                 | 1.47               | soft only         |
| 3    | 32s                | 1.47               | soft, grey        |
| 4    | 2m 8s              | 1.47               | soft, grey        |
| 5    | 8m 32s             | 1.47               | soft only         |

## Consecutive failure handling

A charge that fails on rungs 1, 2 and 3 has failed three consecutive times on
the same acquirer. Nimbus does not attempt rung 4 on that acquirer.

Instead the charge is *promoted*: it is reassigned to the merchant's secondary
acquirer, and the ladder restarts at **rung 1**.

Promotion resets the attempt counter, so a promoted charge receives a full set
of five rungs on the secondary acquirer.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 3.4 | 2025-11-18 | Grey decline class split out of soft |
| 3.0 | 2025-03-07 | Promotion to secondary acquirer introduced |
