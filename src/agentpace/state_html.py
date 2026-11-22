"""One self-contained HTML page for a tool's state: cards for a person, JSON for an agent.

Contract (a Rust tool can copy it; `vigil` is the intended second user):
- `page(title, cards, snapshot)` returns one document: inline CSS, no network, no script but the
  data block, `<script id="agentpace-state" type="application/json">` holding `snapshot`.
- The snapshot is exactly what the tool's `--json` prints; the page never embeds a different view.
- Card fragments are HTML the caller built; every untrusted string in them goes through `esc`.
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
body{margin:0;font-family:var(--s-font);color:var(--s-text);line-height:1.45}
main{display:grid;gap:.75rem;padding-block:.75rem;min-width:0}
.card{background:var(--s-card);color:var(--s-text);border:1px solid var(--s-border);
  border-radius:var(--s-radius);padding:1rem;min-width:0;overflow-wrap:anywhere}
.item{border:1px solid var(--s-border);border-radius:var(--s-radius);padding:.75rem;min-width:0}
.cards{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,16rem),1fr))}
h1,h2,h3,p{margin:0}
h1{font-size:1.1rem;font-weight:600}
h2{font-size:.95rem;font-weight:600;margin-bottom:.5rem}
h3{font-size:.95rem;font-weight:600}
.muted{color:var(--s-muted);font-size:.85rem}
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
[data-status="running"]{border-left:4px solid var(--s-c2)}
[data-status="todo"]{border-left:4px solid var(--s-c1)}
[data-status="done"],[data-status="removed"]{border-left:4px solid var(--s-border)}
details>summary{cursor:pointer;font-weight:600;padding:.25rem 0}
details>.cards{margin-top:.6rem}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.3rem .6rem .3rem 0;border-bottom:1px solid var(--s-border);vertical-align:top}
th{color:var(--s-muted);font-weight:500;white-space:nowrap}
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
