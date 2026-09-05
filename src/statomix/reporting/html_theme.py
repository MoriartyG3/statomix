"""Shared visual language for Statomix HTML reports."""

from __future__ import annotations

STATOMIX_HTML_THEME_NAME = "statomix-openai-inspired"

STATOMIX_HTML_CSS = r"""
:root {
  --stx-page: #f4f1ea;
  --stx-surface: #fffdf8;
  --stx-surface-strong: #ffffff;
  --stx-ink: #171714;
  --stx-muted: #6d6a60;
  --stx-line: #d9d5ca;
  --stx-line-strong: #bdb7a8;
  --stx-accent: #d8ff4f;
  --stx-accent-ink: #283300;
  --stx-blue: #dcecff;
  --stx-blue-ink: #17406b;
  --stx-danger: #d94a3a;
  --stx-shadow: 0 16px 44px rgba(32, 31, 25, 0.08);
}
* { box-sizing: border-box; }
html, body { min-height: 100%; }
body {
  margin: 0;
  background: var(--stx-page);
  color: var(--stx-ink);
  font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input, select { font: inherit; }
header {
  padding: 30px clamp(20px, 4vw, 64px) 28px;
  background: var(--stx-ink);
  color: var(--stx-surface);
  border-bottom: 5px solid var(--stx-accent);
}
header h1 {
  max-width: 1100px;
  margin: 0;
  font: 600 clamp(26px, 4vw, 48px)/1.03 Georgia, "Times New Roman", serif;
  letter-spacing: -0.03em;
}
header p { margin: 12px 0 0; color: #d8d4c8; }
.eyebrow {
  margin-bottom: 12px;
  color: var(--stx-accent);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  padding: 14px clamp(16px, 3vw, 36px);
  background: var(--stx-surface);
  border-bottom: 1px solid var(--stx-line);
}
.controls label { display: flex; align-items: center; gap: 7px; color: var(--stx-muted); font-weight: 650; }
select, input[type="search"] {
  min-height: 34px;
  border: 1px solid var(--stx-line-strong);
  border-radius: 999px;
  padding: 7px 12px;
  background: var(--stx-surface-strong);
  color: var(--stx-ink);
}
.controls input[type="checkbox"] { accent-color: var(--stx-ink); }
.layout { display: grid; grid-template-columns: minmax(600px, 1fr) 390px; height: calc(100vh - 166px); min-height: 540px; }
.canvas { overflow: auto; padding: clamp(14px, 2vw, 26px); }
.side { overflow: auto; padding: clamp(16px, 2vw, 28px); background: var(--stx-surface); border-left: 1px solid var(--stx-line); }
#graph { min-width: 1400px; background: var(--stx-surface-strong); border: 1px solid var(--stx-line); border-radius: 18px; box-shadow: var(--stx-shadow); }
.lane { fill: #fbfaf5; stroke: var(--stx-line); }
.lane-title { font-size: 14px; font-weight: 800; fill: var(--stx-ink); }
.stage-title { font-size: 12px; font-weight: 800; fill: var(--stx-muted); letter-spacing: .04em; text-transform: uppercase; }
.edge { stroke: #8c8a81; stroke-width: 1.7; fill: none; }
.edge-label-bg { fill: var(--stx-surface-strong); stroke: var(--stx-line); stroke-width: 1; }
.edge-label { font-size: 10px; font-weight: 700; fill: var(--stx-muted); }
.node { cursor: pointer; outline: none; }
.node rect { stroke: var(--stx-surface-strong); stroke-width: 2; }
.node text { fill: var(--stx-surface-strong); font-size: 11px; font-weight: 750; pointer-events: none; }
.node:hover rect, .node.selected rect, .node:focus rect { stroke: var(--stx-ink); stroke-width: 3; }
h2 { margin: 0 0 12px; font-size: 17px; letter-spacing: -0.01em; }
hr { margin: 24px 0; border: 0; border-top: 1px solid var(--stx-line); }
.badge { display: inline-block; margin: 2px 4px 2px 0; padding: 4px 9px; border-radius: 999px; background: var(--stx-blue); color: var(--stx-blue-ink); font-size: 12px; font-weight: 750; }
.warning { margin: 9px 0; padding: 10px 12px; border-left: 4px solid #e3a51b; background: #fff7d6; border-radius: 8px; }
.warning.error { border-color: var(--stx-danger); background: #fff0ed; }
pre { white-space: pre-wrap; word-break: break-word; padding: 12px; border: 1px solid var(--stx-line); border-radius: 10px; background: #f7f4ed; font-size: 12px; }
.empty { color: var(--stx-muted); }
@media (max-width: 1050px) {
  .layout { grid-template-columns: 1fr; height: auto; }
  .side { min-height: 360px; border-top: 1px solid var(--stx-line); border-left: 0; }
}
"""
