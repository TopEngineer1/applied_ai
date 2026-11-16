# Webhooks

**Service:** Nimbus Payments API
**Version:** 1.9
**Status:** Current
**Last reviewed:** 2026-08-05

---

## Overview

Nimbus sends webhook events to a merchant's endpoint when something changes on
a charge, so merchants don't need to poll the API.

## Events

| Event | Sent when |
|-------|-----------|
| `charge.succeeded` | An attempt is approved |
| `charge.failed` | The charge reaches a final failed state |
| `charge.attempt_failed` | A single attempt fails and another is scheduled |
| `refund.created` | A refund is requested |
| `refund.completed` | A refund is confirmed by the card network |

`charge.attempt_failed` is off by default because it can be noisy. Enable it in
the dashboard under **Developers → Webhooks**.

## Delivery

Each event is delivered with a `Nimbus-Signature` header: an HMAC-SHA256 of the
request body using the endpoint's signing secret. Always verify it before
trusting the payload.

If your endpoint does not respond with a `2xx` status within 10 seconds, the
delivery is marked failed and redelivered up to 8 times over 24 hours. After
that the event is dropped and appears in the dashboard as undelivered.

## Ordering

Events are not guaranteed to arrive in order. Use the `created_at` field and the
charge's current state from the API, not arrival order, to decide what happened.

## Related

- [Refunds](./refunds.md)
- [Rate limits](./rate-limits.md)
