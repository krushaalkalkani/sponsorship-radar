---
title: H-1B Sponsorship Radar
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Which US employers actually sponsor H-1B, from real DOL data
---

# H-1B Sponsorship Radar

Answers one question fast, from real government data: **will this company sponsor me?**

Built for F-1/OPT students. The FY2027 cap closed with ~211,600 registrations for 85,000
slots and no second lottery, so the next shot is FY2028 (~March 2027). Between now and then
the highest-leverage move is targeting employers who *actually* sponsor — which is public
record, just scattered across DOL disclosure files nobody wants to open.

## What it does

**Company check** — type a name, get a verdict (`sponsors at scale` / `regularly` /
`rarely` / `renewals only`), volume, recency, the roles they sponsor, where those jobs are,
salary bands, and whether they also file green cards.

**Target list** — role + state + salary floor → a ranked list of employers to actually apply
to, with staffing firms filterable out and cap-exempt employers filterable in.

**Ask** — natural language over the whole dataset, answered by a LangGraph agent that
queries the data and shows which tools it called.

**Your profile** — set your target role, location, salary floor and whether you missed
the cap once; every shortlist is pre-filtered and every agent answer is tailored to it.
Stored in your browser only.

## The distinction that makes it useful

Most H-1B lookup sites report *total filings*. That number is misleading. A DOL filing is
either:

- **an outside hire** (`NEW_EMPLOYMENT` + `CHANGE_EMPLOYER`) — a job someone not already
  at the company can get, or
- **a renewal** (`CONTINUED_EMPLOYMENT` + `AMENDED_PETITION`) — an existing employee's
  extension, which is worth nothing to an applicant.

An employer with 4,000 renewals and 3 outside hires looks like a top sponsor on every other
site and is a dead end in practice. This tool leads with outside hires everywhere.

**When you need to apply.** Cap-subject H-1B jobs start on Oct 1, because the petition
is filed in April after the March lottery. So the share of an employer's jobs that begin
on Oct 1 tells you whether they run candidates through the lottery — and therefore
whether you need an offer from them by January. The data separates cleanly:
cap-exempt employers show **3%** Oct-1 starts, cap-subject ones **24%**. Every company
page states which track the employer is on and when to be in their pipeline.

Two other signals it surfaces that matter and are rarely shown:

- **Cap-exempt** (universities, hospitals) — they sponsor H-1B *outside the March lottery,
  year round*. If you missed the cap, this is the single most actionable filter here.
  Inferred from NAICS 61/622; it's a heuristic, not a legal determination.
- **PERM green-card filings** — whether an employer takes people past H-1B to permanent
  residency, i.e. whether the job is a path or a dead end.

## Honest limits

- An **LCA is not a visa.** It is step one of sponsoring. Employers file more LCAs than
  people they hire, so position counts overstate actual hires. This is intent-to-sponsor
  data, and the app says so.
- LCA approval rate is ~99% and means nothing — the real filter is the USCIS lottery and
  petition, which is not in this dataset.
- **FY2026 contains Q1 only** (Oct–Dec 2025). The app labels it; don't compare it to a
  full year.
- **Only certified filings count** toward sponsorship signals. Denied and withdrawn
  filings are reported separately as a denial rate. This matters more than it sounds:
  before the fix, an employer whose filings were 100% denied ranked 9th in the country
  by outside hires.
- Employers are keyed on normalized legal name. `Amazon.com Services LLC` and
  `Amazon Web Services, Inc.` stay separate because they genuinely are. Search surfaces
  related entities rather than silently merging them. (FEIN looked like a better key until
  it turned out to group Amazon with an unrelated corporation and to lump every SUNY campus
  and the NY Dept of Health under one number.)

## Data

| Source | Coverage | Rows |
|---|---|---|
| DOL LCA (H-1B/E-3/H-1B1) disclosure | FY2023 Q1 – FY2026 Q1 | 1,885,316 cases |
| DOL PERM (green card) disclosure | FY2024 – FY2026 Q1 | 257,472 cases |

Employer rollups are computed from every case. The per-case tables that ship in the
container are windowed to FY2025+ to keep the image small (see `ingest/pack.py`).

Two things worth knowing if you extend this, both of which cost me time:

1. The DOL LCA "Q4" file is **not** cumulative — it contains that quarter only. The PERM
   Q4 file **is** cumulative. You need all four quarterly LCA files per fiscal year.
2. Every xlsx has ~450k trailing blank rows. DuckDB's `read_xlsx` defaults to
   `stop_at_empty=true`, which is correct here — but if you turn it off to be safe, you
   get 79% phantom rows and every aggregate silently changes.

## Architecture

```
DOL xlsx ──► ingest/build.py   per-case fact tables (DuckDB)
             ingest/rollup.py  employer rollups + 0–100 score
             ingest/embed.py   profile cards ──► int8 vector store
                                    │
FastAPI ──► app/db.py  (SQL + fuzzy + vector retrieval)
        └─► app/agent.py  LangGraph: agent ⇄ tools ⇄ END
                tools: lookup_company · shortlist_sponsors · compare_companies
                       semantic_search · query_data (read-only SQL)
```

The agent is tool-bound and told never to state a number it didn't get from a tool, which
is what stops it inventing sponsorship statistics. The read-only SQL tool is the escape
hatch for questions the fixed tools don't cover ("who grew the most FY24→FY25").

**Provider-agnostic**: whichever key is present wins — `OPENAI_API_KEY`,
`GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`. Override the model with `RADAR_MODEL`.

**Retrieval is hybrid on purpose.** Structured filters (role/state/wage) go to SQL because
they're exact; open-ended descriptions ("biotech startups in San Diego") go to the vector
store. At ~53k employer cards a brute-force int8 cosine scan takes ~5ms, so there is no ANN
index — it would add a dependency and a build step to save nothing.

## Scoring

`score` is a transparent 0–100 blend, defined in `ingest/rollup.py`:

| Weight | Signal |
|---|---|
| 35 | recent outside-hire volume (log-scaled) |
| 25 | recent filing volume (log-scaled) |
| 15 | outside-hire ratio vs renewals |
| 10 | recency of last filing |
| 8 | certified green-card cases |
| 7 | cap-exempt |
| −15 | willful violator |

It's a ranking heuristic, not a probability. Don't read it as odds.

## Run it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./scripts/rebuild.sh          # downloads ~1.4GB, builds everything (~30 min)
echo "OPENAI_API_KEY=sk-..." > .env
./scripts/run.sh              # http://127.0.0.1:8000
./.venv/bin/python scripts/validate.py   # 24 sanity checks on the build
```

The data endpoints need no API key. Only `/api/ask` does, and it fails with a clear
message rather than breaking the page.

`ingest/embed.py --cards-only` rebuilds the profile-card table without re-running the
~25 minute embed; it refuses if the card text actually changed (verified by hash).

## API

| Endpoint | |
|---|---|
| `GET /api/company?name=Stripe` | full sponsorship profile |
| `GET /api/shortlist?role=Data+Scientist&state=MA&min_wage=120000&exclude_staffing=true` | ranked target list |
| `GET /api/semantic?q=biotech+startups+that+sponsor` | vector search |
| `POST /api/ask {"question": "..."}` | agent |
| `GET /api/stats` | dataset coverage |

## Deploy

Runtime footprint is ~231MB resident, so the free tier of most hosts works.
Semantic search calls the OpenAI embeddings API for the query rather than loading
a local model — that choice is what keeps it under 512MB.

### Render (recommended)

`render.yaml` + `Dockerfile` are ready. Render needs to be able to fetch the repo:

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
```

or connect your GitHub account in the Render dashboard to grant access to a
private repo. Then create the service from the repo and set `OPENAI_API_KEY` in
**Environment** (never commit it).

Free-tier services sleep after ~15 min idle; the first request after that takes
~50s to wake.

### Hugging Face Spaces

`scripts/deploy_hf.py` is written and working, but **Docker and Gradio Spaces now
require a PRO subscription** — only Static Spaces are free, and a static Space
cannot run this backend. If you have PRO:

```bash
export HF_TOKEN=hf_...
python scripts/deploy_hf.py --space <user>/sponsorship-radar
```

It uploads over HF's HTTP API, so git-lfs isn't needed for the ~85MB database,
and sets the LLM key as a Space secret.

### Data

Baked into the image (85MB DuckDB + 18MB vectors, read-only) — no database
service to run. To refresh, re-run `scripts/rebuild.sh` and redeploy.

## Rebuilding after a data refresh

```bash
./scripts/download.sh              # new quarters land ~6 weeks after quarter end
./.venv/bin/python ingest/build.py
./.venv/bin/python ingest/rollup.py
./.venv/bin/python ingest/embed_openai.py   # ~8 min, ~$0.20, resumable
./.venv/bin/python ingest/embed.py --cards-only   # verifies cards match vectors
./.venv/bin/python ingest/pack.py
./.venv/bin/python scripts/validate.py
```

The `--cards-only` check refuses to proceed if any card text drifted from what
was embedded — worth keeping, since it caught a real determinism bug where tied
`string_agg` ordering silently changed 10,070 cards between identical runs.
