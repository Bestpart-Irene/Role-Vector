"""Zotero integration — push found papers into a SHARED group library, read what teammates shared,
and export BibTeX for the paper. Pure stdlib (urllib), no extra deps.

Set in .env (gitignored):
    ZOTERO_API_KEY   from https://www.zotero.org/settings/keys  (give it group read/write)
    ZOTERO_GROUP_ID  numeric id of your shared group (zotero.org/groups/<id>/...)  ← shared w/ teammate
    ZOTERO_USER_ID   (alternative) your personal library id, if not using a group

CLI:
    python -m rolevec.zotero list           # show items in the (shared) library, incl. teammate's
    python -m rolevec.zotero export [path]   # write BibTeX (default templates/refs.bib)
    python -m rolevec.zotero add-survey      # push this project's survey papers into the library
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://api.zotero.org"


def _cfg() -> tuple[str, str]:
    key = os.environ.get("ZOTERO_API_KEY")
    if not key:
        raise SystemExit("set ZOTERO_API_KEY in .env (https://www.zotero.org/settings/keys)")
    gid, uid = os.environ.get("ZOTERO_GROUP_ID"), os.environ.get("ZOTERO_USER_ID")
    if gid:
        return key, f"groups/{gid}"      # shared group library (teammate sees the same)
    if uid:
        return key, f"users/{uid}"       # personal library
    raise SystemExit("set ZOTERO_GROUP_ID (shared group) or ZOTERO_USER_ID in .env")


def _req(method: str, path: str, key: str, data=None, query=None, raw=False):
    url = f"{API}/{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Zotero-API-Key": key, "Zotero-API-Version": "3"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode()
    return text if raw else (json.loads(text) if text else {})


def _to_item(p: dict) -> dict:
    item = {
        "itemType": "preprint",
        "title": p["title"],
        "url": p.get("url", ""),
        "abstractNote": p.get("abstract", ""),
    }
    if p.get("arxiv"):
        item["repository"] = "arXiv"
        item["archiveID"] = f"arXiv:{p['arxiv']}"
    if p.get("authors"):
        item["creators"] = [{"creatorType": "author", "name": a} for a in p["authors"]]
    return item


def add_papers(papers: list[dict]) -> dict:
    """Push papers (each: {title, url?, arxiv?, authors?, abstract?}) into the shared library."""
    key, prefix = _cfg()
    return _req("POST", f"{prefix}/items", key, data=[_to_item(p) for p in papers])


def list_items(limit: int = 100) -> list[dict]:
    """List top-level items in the (shared) library — includes whatever teammates added."""
    key, prefix = _cfg()
    data = _req("GET", f"{prefix}/items/top", key, query={"limit": limit, "format": "json"})
    out = []
    for it in data:
        d, meta = it.get("data", {}), it.get("meta", {})
        out.append({
            "title": d.get("title", "(untitled)"),
            "type": d.get("itemType", ""),
            "added_by": meta.get("createdByUser", {}).get("username", "—"),
            "url": d.get("url", ""),
        })
    return out


def export_bibtex(path: str = "templates/refs.bib") -> str:
    """Export the (shared) library to BibTeX — the source of truth for the paper once set up."""
    key, prefix = _cfg()
    text = _req("GET", f"{prefix}/items/top", key, query={"limit": 100, "format": "bibtex"}, raw=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def survey_papers() -> list[dict]:
    """The papers from docs/auto-research-survey.md, ready to push into the shared library."""
    return [
        {"title": "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
         "arxiv": "2408.06292", "url": "https://sakana.ai/ai-scientist/"},
        {"title": "Agent Laboratory: Using LLM Agents as Research Assistants",
         "arxiv": "2501.04227", "authors": ["Samuel Schmidgall et al."]},
        {"title": "AgentRxiv: Towards Collaborative Autonomous Research", "arxiv": "2503.18102"},
        {"title": "AI-Researcher: Autonomous Scientific Innovation", "arxiv": "2505.18705"},
        {"title": "From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery",
         "arxiv": "2508.14111"},
        {"title": "The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems",
         "arxiv": "2509.08713"},
        {"title": "Designing Role Vectors to Improve LLM Inference Behaviour", "arxiv": "2502.12055"},
        {"title": "Steering LLM Interactions Using Persona Vectors",
         "url": "https://openreview.net/forum?id=HpUDi5Pe8S"},
        {"title": "NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals",
         "url": "https://openreview.net/forum?id=MxbEiFRf39"},
        {"title": "TransformerLens: Mechanistic Interpretability of Language Models",
         "url": "https://github.com/TransformerLensOrg/TransformerLens"},
    ]


def _main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "list"
    if cmd == "list":
        items = list_items()
        print(f"{len(items)} items in library:")
        for it in items:
            print(f"  [{it['added_by']:>12}] ({it['type']}) {it['title'][:70]}")
    elif cmd == "export":
        path = argv[1] if len(argv) > 1 else "templates/refs.bib"
        print("wrote", export_bibtex(path))
    elif cmd == "add-survey":
        res = add_papers(survey_papers())
        print(f"added: {len(res.get('successful', {}))}  failed: {len(res.get('failed', {}))}")
    else:
        raise SystemExit("usage: python -m rolevec.zotero list|export [path]|add-survey")


if __name__ == "__main__":
    _main()
