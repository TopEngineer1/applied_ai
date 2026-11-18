# Rate Limits

**Service:** Nimbus Payments API
**Version:** 1.4
**Status:** Current
**Last reviewed:** 2026-05-19

---

## Overview

Nimbus limits how many API requests each merchant can make, to protect shared
acquirer capacity.

| Endpoint group | Limit |
|----------------|-------|
| Create charge | 100 requests per second |
| Read endpoints (`GET`) | 300 requests per second |
| Refunds | 25 requests per second |

Limits apply per merchant account, across all API keys.

## When you hit a limit

Requests over the limit return HTTP `429` with error `NBS-429`, and a
`Retry-After` header giving the number of seconds to wait.

Client libraries back off automatically. If you call the API directly, wait at
least the `Retry-After` value before sending again, and add random jitter so
your workers don't all resume at the same moment.

Requests rejected with `429` are never processed, so it is always safe to send
them again with the same carry token.

## Raising a limit

Merchants with sustained traffic above these limits can request an increase from
their account manager. Increases for the refunds group require a fraud review.

## Related

- [Webhooks](./webhooks.md)
- [Carry tokens](./carry-tokens.md)
