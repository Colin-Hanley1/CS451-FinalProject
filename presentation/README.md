# Presentation

10-slide deck styled to match the almanac frontend. Self-contained HTML —
no build step, no dependencies, no server required.

## Use

```bash
open presentation/slides.html            # presenter mode
```

Or double-click `slides.html`. Works in any modern browser.

### Keyboard
- **→ / Space / PageDown** — next slide
- **← / PageUp** — previous slide
- **Home / End** — first / last slide
- **P** — print-to-PDF dialog

### Export to PDF
Press **P** or use the browser's print dialog → "Save as PDF". The print
stylesheet renders one slide per page, hides the on-screen controls, and
preserves all formatting.

## Slide outline (~55 s each for 10-min talk)

| #  | Title                        | Visual                                  |
| -- | ---------------------------- | --------------------------------------- |
| 1  | Title                        | Masthead type                           |
| 2  | The Question                 | Pull-quote only                         |
| 3  | The Data                     | Target-distribution histogram           |
| 4  | Exploratory Analysis         | 3-up small multiples (pick, position, recruit) |
| 5  | Feature Engineering          | Feature-correlation bar chart           |
| 6  | Modeling                     | Stage list, temporal-split explainer    |
| 7  | Results                      | Results table + model-comparison chart  |
| 8  | What Matters (SHAP)          | SHAP beeswarm summary plot              |
| 9  | Archetypes                   | PCA scatter + VORP-by-cluster bars      |
| 10 | 2026 Big Board + Live Demo   | Top-5 projections → open the dashboard  |
| 11 | Limitations & Future Work    | R² ceiling, SOS, era drift, Q&A cue     |

All figures live in `presentation/figures/` (copied from `../figures/` so
the folder is portable as a standalone zip).

## Demo cue on slide 9

The demo is the dashboard (`docs/index.html`). Open it in a second browser
tab before the talk starts. Rehearsed demo flow:
1. Show the hero callout + ticker
2. Scroll to the 2026 big board — click Cameron Boozer → drawer opens with
   SHAP drivers + historical comps
3. Scroll to Build-a-Prospect — drag the WS/40 slider from 0.10 to 0.30
   and narrate how the projection jumps
4. Scroll to the archetype scatter — hover a point, click, drawer opens

Total demo should be ~2 minutes (inside the 10-minute budget).
