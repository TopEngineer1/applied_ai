# Decline Classes

**Service:** Nimbus Retry Orchestrator
**Version:** 2.3
**Status:** Current
**Last reviewed:** 2026-08-14

---

## Overview

Every failed card attempt is sorted into one of three decline classes before
Nimbus decides what to do next. The class is assigned from the acquirer's
response code, or from the absence of one.

| Class | Meaning | Retry eligible |
|-------|---------|----------------|
| Soft  | The acquirer explicitly rejected the attempt, but the rejection may clear | Yes |
| Grey  | No usable rejection came back, so the outcome is unknown | Yes, with restrictions |
| Hard  | The acquirer rejected the attempt permanently | Never |

## Soft declines

A soft decline carries an explicit rejection that is expected to clear on its
own. Common examples:

- `51` insufficient funds
- `05` do not honour (generic issuer refusal)
- `91` issuer or switch unavailable

Because the acquirer has told Nimbus the attempt did not go through, a soft
decline can be retried without risk of a duplicate charge.

## Grey declines

A grey decline is any attempt where Nimbus cannot be sure what happened:

- request timeout
- connection reset before a response arrived
- a malformed or unparseable acquirer response

The acquirer may still have approved the charge. Grey declines are therefore
subject to the settlement fence. See [Ghost charges](./ghost-charges.md).

Grey was split out of soft in version 3.4 of the ladder (2025-11). Before that,
timeouts were retried as if they were explicit rejections.

## Hard declines

A hard decline means retrying cannot succeed:

- `14` invalid card number
- `41` / `43` card reported lost or stolen
- `54` expired card
- account closed

Nimbus never retries a hard decline. The attempt is closed, the merchant is
notified, and error `NBS-221` is emitted.

## Overriding a class

Merchants cannot reclassify response codes. Requests to treat a specific hard
decline as soft are handled by the payments platform team and require a risk
review.

## Related

- [Carry tokens](./carry-tokens.md)
- [Ghost charges](./ghost-charges.md)
