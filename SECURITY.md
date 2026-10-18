# Security policy

## Reporting a vulnerability

Please report a vulnerability privately, through GitHub's private vulnerability reporting:
**Security** tab of this repository, then **Report a vulnerability**
(<https://github.com/w2ur/midas-core/security/advisories/new>).

Do not open a public issue or pull request for it. Include the version (`pip show midas-core`),
the command or input that triggers it, and what you observed.

This is a one-person project: expect an acknowledgement within a few days and a fix or a
written assessment after that, not a bug bounty.

## Scope

midas-core is a paper-trading framework. It never holds broker credentials and never moves real
money. In scope: anything that lets an agent-authored input (an order, a cancel request, a
roster file) read, write or delete outside the data directory, bypass a broker safety rail, or
act on another agent's orders or portfolio.

## Trust boundary of the paper broker

The paper broker trusts the identities in the files it reads (`agent_id`, `order_id` and a
cancel's `request_id`) only as far as shape checks: each must be a string in the minted alphabet
(letters, digits, `_`, `-`) and `request_id` must start with `cnl_`, so none can name a path or
collide with an order id. It does not authenticate who wrote a line, and it does not reject an
`agent_id` missing from the roster (that agent gets the default safety rails).

**The driver must mint `agent_id`, `order_id` and `request_id` itself** and never copy one out of
agent output. Deployments that let an agent author its own identities, for example a
bring-your-own-agent desk that writes the outbox directly, give that agent the ability to act as
another one, and that is outside what this project defends.

midas-core is a one-way mirror of [`w2ur/midas`](https://github.com/w2ur/midas), so a fix lands
there first and reaches this repository on the next sync.
