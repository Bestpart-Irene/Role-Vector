# Zotero — shared group library for the team

Goal: one shared reference library you and your teammate both see; `lit-scan` pushes found papers
into it; `paper-draft` pulls `refs.bib` out of it.

## 1. Create the shared group (one person, once)
1. Go to https://www.zotero.org/groups/new
2. Name it (e.g. `role-vector`). Type: **Private Membership** (only invited members) or
   **Public, Closed Membership** (visible, but only members edit). Either is fine for sharing.
3. Create → **Members Settings → Send More Invitations** → invite your teammate by Zotero username/email.
4. Teammate accepts → you both now see the group's library (in the Zotero desktop app under *Group Libraries*).

## 2. Get the IDs/keys (each member, for API access)
- **Group ID:** open the group page; the URL is `zotero.org/groups/<GROUP_ID>/...` → that number is `ZOTERO_GROUP_ID`.
- **API key:** https://www.zotero.org/settings/keys → *Create new private key* →
  check **"Allow library access"** and, under *Default Group Permissions* (or per-group), **Read/Write**.
  Copy the key → `ZOTERO_API_KEY`.

Put both in `.env` (gitignored — never commit):
```
ZOTERO_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
ZOTERO_GROUP_ID=1234567
```

## 3. Use it
```bash
set -a && . ./.env && set +a              # load creds

# push this project's survey papers into the shared group:
PYTHONPATH=src python -m rolevec.zotero add-survey

# see everything in the shared library — including what your teammate added:
PYTHONPATH=src python -m rolevec.zotero list

# export BibTeX for the paper (writes templates/refs.bib):
PYTHONPATH=src python -m rolevec.zotero export
```

## 4. How it ties into the project
- **`lit-scan`** → calls `rolevec.zotero.add_papers(...)` to file newly found role/persona-vector papers
  into the shared group (so the whole team's reading list accumulates in one place).
- **Better BibTeX** (Zotero plugin) or `rolevec.zotero export` → keeps `templates/refs.bib` in sync →
  `paper-draft` cites them in the LaTeX (`\bibliography{refs}`).
- **Sharing:** because the library is a *group* library, every member reads/writes the same items —
  your teammate's shared papers show up in `zotero list` and in the exported `refs.bib`.

## Notes
- Personal-only (no sharing): set `ZOTERO_USER_ID` instead of `ZOTERO_GROUP_ID` (find it on the keys page).
- The API key is per-user; each teammate makes their own. The group ID is shared.
- arXiv IDs in pushed items let Zotero auto-enrich full metadata (authors, abstract, date).
