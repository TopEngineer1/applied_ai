# Ghost Charges

**Service:** Nimbus Retry Orchestrator
**Version:** 1.6
**Status:** Current
**Last reviewed:** 2026-07-22

---

## What a ghost charge is

A ghost charge is a charge the acquirer settled after Nimbus had already
recorded it as failed. If Nimbus retries it, the customer is billed twice: once
by the settled "failed" attempt and once by the retry.

Ghost charges almost always start with a grey decline. The request timed out or
the connection dropped, Nimbus saw no approval, and meanwhile the acquirer
approved and settled the charge.

## The settlement fence

The settlement fence is a 90-second window that opens after every attempt.
During the fence, the acquirer may still settle an attempt that Nimbus believes
failed.

- **Soft declines** carry an explicit rejection, so they are known not to have
  settled. The fence does not apply.
- **Grey declines** have no rejection. The fence applies, and no further attempt
  may be sent until it closes.

An attempt sent inside an active fence for a grey decline is rejected with
`NBS-409`. The attempt is not counted.

## Detection

Nimbus reconciles against acquirer settlement files every 15 minutes. A charge
marked failed that appears as settled in a file is flagged as a suspected ghost
charge, and an automatic refund is queued if a duplicate successful attempt
exists for the same carry token.

## Disabling the fence

`ladder.fence_override = true` turns off fence checks. It exists only for
acquirer integration testing. In 2025-Q3 a merchant enabled it in production
and produced 1,140 duplicate charges in under an hour. The flag now requires
platform-team approval to set on any production merchant.

## Related

- [Decline classes](./decline-classes.md)
- [Carry tokens](./carry-tokens.md)
- [Refunds](./refunds.md)
