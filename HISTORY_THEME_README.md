# Statomix history-report theme overlay

This overlay repairs the interactive history HTML renderer and applies a
shared Statomix visual theme to HTML and SVG history reports. The visual
language is inspired by the supplied OpenAI landing-page reference: warm
neutral surfaces, near-black editorial typography, generous spacing, rounded
cards, and a bright lime accent. It is an original Statomix theme, not a copy
of OpenAI assets or source code.

The HTML renderer is replaced with a complete, readable template so a malformed
manual edit cannot leave a blank page. Dataset headings use their display label,
and edge labels are drawn after nodes with opaque label pills so they remain
visible on dense graphs.

## Apply

From the Statomix repository:

```bash
unzip -o /workspace/MoriartyG3/python_packages/statomix_history_theme_overlay.zip
uv run --active black \
    src/statomix/history/renderers/html.py \
    src/statomix/history/renderers/svg.py \
    src/statomix/reporting/html_theme.py \
    tests/test_history_html_theme.py
uv run --active ruff check src tests
uv run --active pytest -q \
    tests/test_history_html_theme.py \
    tests/test_project_history.py
```

Then regenerate the history report with `project.create_history_report(...)`.
The existing project store is read-only input; the report files are regenerated
outside the store.
