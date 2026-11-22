"""One self-contained HTML page for a tool's state: cards for a person, JSON for an agent.

Contract (a Rust tool can copy it; `vigil` is the intended second user):
- `page(title, cards, snapshot)` returns one document: inline CSS, no network, no script but the
  data block, `<script id="agentpace-state" type="application/json">` holding `snapshot`.
- The snapshot is exactly what the tool's `--json` prints; the page never embeds a different view.
- Card fragments are HTML the caller built; every untrusted string in them goes through `esc`.
- `main` keeps 6rem under the last card so T3's mobile navigation does not cover it.
- Colours come from the host theme's variables (T3's `--card`, `--border`, `--chart-n`, ...) with a
  fallback set, light and dark. `html` and `body` get no background and no side padding: the host
  frame supplies them. Text sits on card surfaces, so a dark fallback stays legible on a white
  canvas when the page is opened outside the host.

Mating part: `site.json_block` is the one escape for data inside a script element.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

from agentpace.site import json_block

DATA_ID = "agentpace-state"

# Own variables point at the host's, so the dark fallback never overrides a host theme.
_CSS = """
:root{
  --s-card:var(--card,#fff);--s-border:var(--border,#d9d9de);--s-text:var(--foreground,#18181b);
  --s-muted:var(--muted-foreground,#62626b);--s-radius:var(--radius,10px);
  --s-font:var(--font-sans,system-ui,-apple-system,"Segoe UI",sans-serif);
  --s-c1:var(--chart-1,#2563eb);--s-c2:var(--chart-2,#0d9488);--s-c3:var(--chart-3,#7c3aed);
  --s-c4:var(--chart-4,#d97706);--s-c5:var(--chart-5,#db2777);--s-c6:var(--chart-6,#65a30d);
  --s-warn:var(--chart-4,#b45309);
}
@media (prefers-color-scheme: dark){
  :root{
    --s-card:var(--card,#1c1c21);--s-border:var(--border,#34343c);--s-text:var(--foreground,#f4f4f5);
    --s-muted:var(--muted-foreground,#a1a1aa);
    --s-c1:var(--chart-1,#60a5fa);--s-c2:var(--chart-2,#2dd4bf);--s-c3:var(--chart-3,#a78bfa);
    --s-c4:var(--chart-4,#fbbf24);--s-c5:var(--chart-5,#f472b6);--s-c6:var(--chart-6,#a3e635);
    --s-warn:var(--chart-4,#fbbf24);
  }
}
body{margin:0;font-family:var(--s-font);color:var(--s-text);line-height:1.45;font-size:14.5px}
main{display:grid;gap:.75rem;padding-block:.75rem;padding-block-end:6rem;min-width:0}
.card{background:var(--s-card);color:var(--s-text);border:1px solid var(--s-border);
  border-radius:var(--s-radius);padding:1rem;min-width:0;overflow-wrap:anywhere}
.item{border:1px solid var(--s-border);border-radius:var(--s-radius);padding:.75rem;min-width:0}
.cards{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr))}
h1,h2,h3,p{margin:0}
h1{font-size:1.1rem;font-weight:600}
h2{font-size:.95rem;font-weight:600;margin-bottom:.5rem}
h3{font-size:.95rem;font-weight:600}
.muted{color:var(--s-muted);font-size:.85rem}
.note{font-size:.88rem;margin-top:.3rem}
.figures{display:flex;flex-wrap:wrap;gap:.5rem 1.75rem;margin:.6rem 0 .4rem}
.figure{display:grid}
.figure span{color:var(--s-muted);font-size:.8rem}
.figure strong{font-size:clamp(1.6rem,8vw,2.6rem);line-height:1.1;font-weight:650;font-variant-numeric:tabular-nums}
.figure.lead strong{color:var(--s-c1)}
.banner{border-left:4px solid var(--s-warn);padding:.4rem .75rem;margin-top:.6rem;font-weight:500}
.head{display:flex;flex-wrap:wrap;gap:.35rem .6rem;align-items:baseline;justify-content:space-between}
.badges{display:flex;flex-wrap:wrap;gap:.3rem;margin:.4rem 0}
.badge{border:1px solid var(--s-border);border-radius:999px;padding:.05rem .55rem;font-size:.78rem;color:var(--s-muted)}
.badge.accent{border-color:var(--s-c1);color:var(--s-c1)}
.badge.warn{border-color:var(--s-warn);color:var(--s-warn)}
.facts{display:grid;grid-template-columns:auto 1fr;gap:.15rem .9rem;margin:.5rem 0 0;font-size:.88rem}
.facts dt{color:var(--s-muted)}.facts dd{margin:0;font-variant-numeric:tabular-nums}
[data-status="done"],[data-status="removed"]{border-left:4px solid var(--s-border)}
.badge.crit{border-color:var(--s-warn);color:var(--s-warn)}
.timeline{display:grid;grid-template-columns:3.25rem 1fr;gap:.75rem}
.rail{position:relative;border-right:1px dashed var(--s-border)}
.rail .tick{position:absolute;left:0;font-size:.7rem;color:var(--s-muted);transform:translateY(-50%)}
.rail .bar{position:absolute;width:4px;min-height:2px;border-radius:2px}
.rail .ghost{position:absolute;width:4px;min-height:2px;border-radius:2px;opacity:.35;background-image:repeating-linear-gradient(180deg,currentColor 0 4px,transparent 4px 7px)}
.timeline .basis{grid-column:1/-1}
.tasks{display:grid;gap:.6rem;min-width:0}
.task{border-left:4px solid var(--c)}
.notes{margin:.6rem 0 0;padding:0;list-style:none;display:grid;gap:.35rem}
.notes li{font-size:.86rem;padding:.35rem .6rem;border-radius:calc(var(--s-radius) - 4px);background:color-mix(in srgb,var(--c) 10%,transparent)}
.notes b{font-weight:600}
.notes .badge{margin-left:.25rem}
.whatif{display:grid;gap:.4rem}
.row{display:grid;grid-template-columns:minmax(8rem,1fr) auto;gap:.5rem;font-size:.9rem;align-items:baseline}
.row strong{font-variant-numeric:tabular-nums}
.gain{color:var(--s-c2);font-weight:600}
.groups{display:grid;gap:.6rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,15rem),1fr))}
.group{border:1px solid var(--s-border);border-radius:var(--s-radius);padding:.75rem;min-width:0}
.cov{display:grid;grid-template-columns:2.2rem 1fr 3.2rem;gap:.25rem .5rem;align-items:center;font-size:.8rem;margin-top:.5rem}
.track{position:relative;height:8px;border-radius:4px;background:color-mix(in srgb,var(--s-border) 60%,transparent)}
.fill{position:absolute;inset:0 auto 0 0;border-radius:4px;background:var(--s-c1)}
.fill.low{background:var(--s-warn)}
.target{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--s-text);opacity:.6}
details>summary{cursor:pointer;font-weight:600;padding:.25rem 0}
details>.cards{margin-top:.6rem}
ul{margin:.25rem 0 0;padding-left:1.1rem}
"""


def esc(value: Any) -> str:
    """Text for an element or a quoted attribute."""
    return escape(str(value), quote=True)


def page(title: str, cards: Sequence[str], snapshot: Any) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}</style></head>\n"
        f"<body><main>\n{chr(10).join(cards)}\n</main>\n"
        f'<script id="{DATA_ID}" type="application/json">{json_block(snapshot)}</script>\n'
        "</body></html>\n"
    )


def write(target: str | Path, text: str) -> None:
    """Write the page to `target` and nowhere else. A path that cannot be written is a usage error."""
    path = Path(target)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot write --html {path}: {error.strerror or error}") from error
