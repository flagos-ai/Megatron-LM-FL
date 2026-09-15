# Override Audit Artifact Contract

Bind the audit to exact full refs and record tool inputs. Sort registry rows by method key, vendor, target, and implementation.

For every row require target, method key, vendor, implementation, registration source/line, target kind and signatures at available refs, implementation kind/signature, mismatch reasons, disposition, owner, and tests.

Keep extracted facts immutable. Put human exceptions in a separate decisions artifact keyed by registry identity. Reject decisions that change identity or cite only import success, equal counts, or a clean merge.

Acceptance requires registry count closure, unique identities, dynamically discovered vendor closure, zero unresolved target/implementation paths, reviewed signature/default/binding differences, and explicit external owners for blocked runtime tests.
