# Security Policy

## Supported versions

Only the latest release line receives fixes. A security fix ships as a patch
release of the current minor (for example, a report against 0.3.0 is fixed in
0.3.x); earlier minors are not patched. Check the
[releases page](https://github.com/wlazlod/treecf/releases) or PyPI for the
current version.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's
[Report a vulnerability](https://github.com/wlazlod/treecf/security/advisories/new)
form. Do not open a public issue for security problems. You will get an
acknowledgement within a few days. There is no bug bounty.

## Scope

treecf's runtime is numpy plus a bundled Rust extension; it makes no network
requests. The only file I/O is explicit and caller-directed: `BatchResult.save`
/ `BatchResult.load` (JSON), certificates written and read as JSON, and model
dump files read by the parsers. Serialized artifacts are plain JSON; the
package never unpickles. Parsing of untrusted model dumps **is** in scope for
reports — a crafted dump must fail cleanly, never execute code or corrupt
memory. The build, the wheels, and the release pipeline are also in scope.

The Rust crate is compiled with `#![forbid(unsafe_code)]`: the extension
contains no `unsafe` blocks, and a change introducing one fails the build.
