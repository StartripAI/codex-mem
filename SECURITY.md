# Security Policy

## Scope

This repository handles local memory records and launch assets. Sensitive data handling is mandatory.

## Data Handling Rules

- Do not commit real API keys, tokens, credentials, or private keys.
- Do not commit personal machine paths, personal account IDs, or personal contact info.
- Use sanitized demo data for all recordings and screenshots.

## Built-in Controls

- Dual-tag privacy policy via `--privacy-tag`
- Block write tags: `no_mem`, `block`, `skip`, `secret_block`
- Private visibility tags: `private`, `sensitive`, `secret`
- Redaction tags: `redact`, `mask`, `sensitive`, `secret`
- Session export supports anonymization by default (`export-session`)

## Reporting a Security Issue

1. Do not open a public issue with secrets included.
2. Prepare minimal reproduction with redacted values.
3. Contact maintainers privately and include:
   - impact summary
   - affected files/commands
   - mitigation proposal

## Secret Rotation Guidance

If a credential is exposed:
1. Revoke immediately at provider side.
2. Rotate and redeploy.
3. Purge history if needed and force-push sanitized history.
4. Record postmortem in internal incident log.
