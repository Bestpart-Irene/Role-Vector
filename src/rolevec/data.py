"""Load roles + questions from data/*.yaml into light dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Role:
    id: str
    name: str
    prompt: str
    domain_family: str | None = None
    reported: bool = True
    contrast_group: str | None = None


@dataclass(frozen=True)
class Question:
    text: str
    family: str
    in_domain_role: str


def load_roles(path: Path) -> tuple[Role, list[Role]]:
    """Return (baseline_role, reported_roles)."""
    d = yaml.safe_load(path.read_text())
    b = d["baseline"]
    baseline = Role(id=b["id"], name=b["name"], prompt=b["prompt"], reported=False)
    roles = [
        Role(
            id=r["id"], name=r["name"], prompt=r["prompt"],
            domain_family=r.get("domain_family"),
            contrast_group=r.get("contrast_group"),
        )
        for r in d["roles"]
    ]
    return baseline, roles


def load_questions(path: Path) -> list[Question]:
    d = yaml.safe_load(path.read_text())
    out = []
    for fam in d["families"]:
        for q in fam["questions"]:
            out.append(Question(text=q, family=fam["id"], in_domain_role=fam["in_domain_role"]))
    return out
