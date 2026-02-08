# First-Screen Conversion Instrumentation Spec

Define measurable signals for README first-screen effectiveness.

## Objective

Increase conversion from repository visits to actionable user steps.

## Primary Funnel

1. README view
2. Click Quick Start command block
3. Run `init`
4. Run `mem-search`
5. Run `smoketest`

## Proxy Metrics (manual/semi-automated)

- CTR on README section links:
  - Quick Start
  - Feature Tour
  - Release Notes
- Ratio of users reaching smoke-test command
- Number of installs from release cycle
- Social post click-through into repo

## Experiment Dimensions

- Badge order/layout
- GIF ordering
- 30-second value path placement
- comparison table depth
- CTA wording in first 300 lines

## Measurement Cadence

- Baseline weekly
- Compare before/after each README major revision
- Record snapshots in release notes and internal sheet

## Reporting Format

| Date | Version | Variant | QuickStart CTR | SmokeTest CTR | Notes |
|---|---|---|---:|---:|---|
| YYYY-MM-DD | vX.Y.Z | A | - | - | - |
