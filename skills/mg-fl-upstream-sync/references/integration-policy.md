# Integration Policy

Use target files as the starting point for every both-changed path. Replay approved fork behavior at symbol or block granularity. Never replace a target file wholesale merely because the fork version is marked P0.

For every delta ID record its approved action, target symbols affected, implementation commit, invariant, observing test, and result. Stop when implementation reveals a dependency or semantic choice absent from the classifier decision.

Treat upstream backports carefully: drop an old patch only with commit, patch, or semantic evidence that the target covers it. If FL later modified that area, replay only the later FL delta.

Treat a newly discovered dependency or conflict shape as a coverage question before treating it as an implementation problem. If the active domain method has no rule, invariant, or observing test for it, record a skill gap and follow `skill-gap-protocol.md`. Resume only after a permitted one-off disposition or a skill update plus rerun.
