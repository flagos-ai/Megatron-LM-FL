# Override Compatibility Contract

Use `(method_key, vendor)` as registration identity. Derive `method_key` with the runtime registry rule and retain full dotted target and implementation paths as evidence.

Compare:

- function, async function, method, class, and callable-class kind;
- positional-only, positional, varargs, keyword-only, kwargs, annotations, defaults, and return annotation;
- staticmethod/classmethod behavior and binding of `self` or `cls`;
- target existence at sync-tree, fork, and target refs;
- implementation existence at the fork ref and planned integrated tree;
- registry duplication, vendor normalization, lazy import resolution, and default fallback;
- class inheritance and constructor compatibility;
- parameter forwarding and return shape through the whole call chain.

Classify each row as `compatible`, `review-signature`, `missing-target`, `missing-implementation`, `duplicate-registration`, `unsupported-dynamic-registration`, or `manual`. Do not label a signature mismatch compatible without a reviewed adapter and focused test.
