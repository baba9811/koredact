# koredact Working Agreements

## Architecture

- One Cargo package producing a Rust library and a Python extension from the same source. Do not split into a workspace for internal modularity.
- The crate is a single linear pipeline, not a set of independent domains. Treat it as one capability with stages, and keep the source grouped by those stages rather than by file count.
- The shape is: shared leaves at the bottom, a directory per pipeline stage group, a facade that composes them, and a binding layer on top. Shared value and error types stay flat; a directory that would hold one small file is not a boundary.
- Add a module or a directory for a responsibility that exists. Do not scaffold stages, traits, or extension points for features that are not implemented.
- Moving a file into a directory later leaves import paths unchanged, so nest when a stage actually grows, not in advance.

## Dependency Direction

- One way only: shared leaves ← stage groups ← facade ← bindings. Nothing imports the binding layer, and the shared leaves import nothing from the crate.
- Stage groups do not depend on each other. The facade is the only place that composes them.
- The Python wrapper stays a thin surface over the facade. Logic belongs in Rust.

## Visibility

- The public surface is the facade, the shared value and error types, and the pieces that are a published contract. Everything else is crate-private.
- Prefer private, then crate-visible, then public. Public is a deliberate decision about what callers may depend on, not a default.
- Narrow visibility is what makes internal moves non-breaking, and it is also what lets dead-code analysis find unused items.
- A published path is a contract independent of where its source lives. Re-export it from a stable location so that regrouping the source never renames it.

## Behavior Contracts

- The decoder is a port of a Python reference and must stay byte-identical to it. Changing its rules requires a new decoder version and a regenerated vector fixture.
- Span coordinates are character offsets, matching the reference tokenizer's convention.
- The regex backstop is opt-in and never overrides a typed model span that survived type filtering. Type filtering runs on each source before the merge, so restricting to a type cannot lose one of its spans to a filtered-out span of another type. Keep that order.
- Preserve masking output, span semantics, and the model bundle layout unless a change explicitly asks otherwise.

## Building and Testing

- The default feature set links the ONNX runtime statically; the test and wheel paths load it dynamically instead. Select the dynamic feature when running the Rust suite, or the link step fails.
- Verify both sides after a change: the Rust suite, and the Python smoke test against the tiny fixture bundle through a locally built wheel.
- Keep structural moves and behavior changes in separate commits, and run both suites after each.

## Versioning and Release

- Semantic versioning. The Rust and Python manifests carry the same number and always move together.
- While on `0.x`, a moved or removed public Rust path is a minor bump. Take the highest bump the change set requires.
- Never reuse or lower a published version. Yanking is the only remedy on the index, so settle the number before tagging.
- The internal remote is the origin; the public one is the build and publish mirror. Pushing a release tag runs the wheel workflow, which publishes through trusted publishing.
- Do not cut a tag only to exercise the pipeline. A release needs content a user would want.

## Writing

- Korean comments use the telegraphic noun-ending style. No Markdown emphasis, headings, or decorative bullets inside a comment; backticks around identifiers, flags, paths, and literal values are fine, and a leading warning mark for a real hazard is fine. English comments are left as they are.
- A comment states the invariant that holds now and what breaks if it is violated, never how the code came to be.
