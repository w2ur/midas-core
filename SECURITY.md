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

midas-core is a one-way mirror of [`w2ur/midas`](https://github.com/w2ur/midas), so a fix lands
there first and reaches this repository on the next sync.
