# Why Candor exists instead of a rental

Someone else's answer to "remember things and answer questions" is an account,
an API key, a meter running per token, your facts on their disks, and a model
update that can quietly change what your system believes. That is rental in
every sense that matters: you never own the memory, the behavior, or the bill.

Candor is the opposite proposition, kept small on purpose:

- **One static binary.** No server, no daemon, no account, no network.
  It runs from an empty directory and answers identically.
- **Your facts stay yours.** Witnessed rows live in your files (Candor) or
  inside your binary (SCandor). Nothing phones home because there is nothing
  to phone home *with* — no socket code, no telemetry, no update channel.
- **Marked answers.** REMEMBERED means witnessed, with source. GUESSED means
  inferred, with probabilities. A rental endpoint never tells you which of
  its answers it actually knows; Candor cannot tell you otherwise.
- **Retrainable by you.** New facts are JSONL rows and a command, not a
  support ticket. `--init-model` continues from where you were.

## What the license is for

Candor is source-available under the Business Source License 1.1, Saphira
terms (see LICENSE, NOTICE). The license exists for one specific threat:
someone taking this code, hosting it, and selling *access to your own
question-answering* back to you by the month. That is the rental this
project exists to destroy, and the license forbids exactly that commercial
capture while leaving genuine use alone.

Read it plainly: run it yourself, run it for your team, embed it, study it,
fork it. Do not wrap it in a meter and sell the meter. We do not do rental —
neither of our product, nor of anyone's memory.
