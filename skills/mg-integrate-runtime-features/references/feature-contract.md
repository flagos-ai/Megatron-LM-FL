# Runtime Feature Contract

Model each feature as an observable lifecycle:

1. configuration or argument;
2. construction, import, or initialization;
3. dispatch, routing, or scheduling;
4. runtime execution or communication;
5. output, checkpoint, metrics, or state compatibility;
6. observing unit, functional, E2E, or external hardware test.

Not every feature requires every stage, but missing configuration/dispatch/output stages need a reason. No feature is complete without execution semantics and observation evidence.

Keep feature identity independent of release, PR, vendor, and path. Use stable source group IDs as candidates only; human review may split/merge them with evidence.
