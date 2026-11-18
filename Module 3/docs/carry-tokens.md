# Carry Tokens

**Service:** Nimbus Retry Orchestrator
**Version:** 3.0
**Status:** Current
**Last reviewed:** 2026-09-02

---

## Overview

A carry token is the idempotency key Nimbus attaches to every attempt of a
charge. All attempts for the same charge share one carry token, so if two
attempts both reach the acquirer, the acquirer settles at most one of them.

## Format

```
ct_<merchant_id>_<order_id>_<sequence>
```

The sequence starts at `0` and is only incremented when the merchant creates a
new charge for the same order, for example after the customer updates their
card. It is never incremented by Nimbus itself.

## Lifetime

- A carry token is created when the charge is first submitted.
- It is persisted before the first attempt is sent, so it survives a process
  restart or a worker crash mid-attempt.
- It expires 72 hours after the charge reaches a final state.

Generating the token inside the attempt loop, instead of before it, was the root
cause of incident INC-2207: each restarted worker created a fresh token and the
acquirer treated every attempt as a new charge.

## Across acquirers

When a charge moves to a different acquirer, the carry token is sent unchanged.
Acquirers do not share idempotency stores, so the carry token alone does not
prevent a duplicate across two acquirers. That protection comes from the
settlement fence. See [Ghost charges](./ghost-charges.md).

## Related

- [Decline classes](./decline-classes.md)
- [Ghost charges](./ghost-charges.md)
