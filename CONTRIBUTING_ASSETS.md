# Contributing Asset Media

Contributions for GIFs, screenshots, and launch visuals must follow this guide.

## Folder Rules

- GIF sources: `Assets/LaunchKit/gif/source/`
- Final GIF: `Assets/LaunchKit/gif/export/`
- Posters: `Assets/LaunchKit/gif/posters/`
- Screenshots final: `Assets/LaunchKit/screenshots/final/`

## Naming Rules

- GIF: `gif_<seq>_<feature>_v<rev>.gif`
- Poster: same stem with `.png`
- Screenshot: `ss_<surface>_<state>_<seq>.png`

## Quality Rules

- GIF width: 1200px
- GIF size: <= 8MB
- GIF duration: 0.2s - 25s
- No personal data or secrets in pixels/text

## Validation

Run before PR:

```bash
python Scripts/validate_assets.py --check-readme --strict
```

## PR Checklist

- [ ] Added/updated shotlist if new GIFs
- [ ] Added poster for each GIF
- [ ] Updated README links if media changed
- [ ] Passed asset validation
