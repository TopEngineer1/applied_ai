# Refunds

**Service:** Nimbus Payments API
**Version:** 2.1
**Status:** Current
**Last reviewed:** 2026-06-30

---

## Overview

Refunds return money from a settled charge to the customer's card. A charge can
be refunded in full or in several partial refunds, up to the original amount.

## Creating a refund

```
POST /v2/charges/{charge_id}/refunds
{
  "amount": 1500,
  "reason": "requested_by_customer"
}
```

`amount` is in the smallest currency unit. Omit it to refund the full remaining
balance.

Valid reasons: `requested_by_customer`, `duplicate`, `fraudulent`,
`ghost_charge`. The `ghost_charge` reason is set automatically by reconciliation
and cannot be used in API requests.

## Timing

- Refunds on charges settled less than 24 hours ago are processed as voids and
  usually disappear from the customer's statement within a day.
- Older charges are refunded through the card network and take 5 to 10 business
  days to appear.

## Limits

- Refunds are allowed for 180 days after settlement.
- A charge can have at most 20 partial refunds.
- Refunding more than the remaining balance returns `NBS-512`.

## Related

- [Ghost charges](./ghost-charges.md)
- [Webhooks](./webhooks.md)
