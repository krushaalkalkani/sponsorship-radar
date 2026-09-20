"""Data access for Sponsorship Radar.

Every function here is a read-only view over the DuckDB build. These double as
the agent's tools, so each returns plain JSON-able dicts with the numbers the
answer must cite.
"""
import duckdb, os, re, numpy as np, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "build", "radar.duckdb")
VEC = os.path.join(ROOT, "data", "build", "vectors.npz")

_local = threading.local()
CURRENT_FY, LATEST_QUARTER = 2026, "FY2026 Q1 (Oct-Dec 2025)"


def con():
    if not hasattr(_local, "c"):
        _local.c = duckdb.connect(DB, read_only=True)
    return _local.c


SUFFIXES = ["INCORPORATED", "CORPORATION", "COMPANY", "LIMITED", "HOLDINGS", "GROUP",
            "INC", "LLC", "LTD", "CORP", "CO", "LP", "LLP", "PLLC", "PC", "PA", "NA", "USA", "US"]


def normalize(name):
    s = re.sub(r"[^A-Z0-9 &]", " ", (name or "").upper())
    s = re.sub(r"\s+", " ", s).strip()
    for suf in SUFFIXES:
        s = re.sub(r" %s$" % suf, "", s).strip()
    return s


def _rows(sql, params=None):
    cur = con().execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------- lookup
def find_employers(name, limit=8):
    """Fuzzy company-name lookup. Returns candidates, best first."""
    key = normalize(name)
    if not key:
        return []
    rows = _rows("""
        SELECT employer_key, display_name, hq_city, hq_state, lca_total,
               recent_outside, score, last_fy
        FROM employers
        WHERE employer_key = ? OR employer_key LIKE ? OR display_name ILIKE ?
        ORDER BY (employer_key = ?) DESC, lca_total DESC LIMIT ?""",
        [key, key + "%", "%" + name.strip() + "%", key, limit * 4])
    try:
        from rapidfuzz import fuzz
        rows.sort(key=lambda r: (-fuzz.token_set_ratio(key, r["employer_key"]), -(r["lca_total"] or 0)))
    except ImportError:
        pass
    return rows[:limit]


def employer_profile(name_or_key):
    """Full sponsorship profile for one employer: the 'will they sponsor me' answer."""
    cand = find_employers(name_or_key, limit=1)
    if not cand:
        return {"found": False, "query": name_or_key}
    key = cand[0]["employer_key"]
    e = _rows("SELECT * FROM employers WHERE employer_key = ?", [key])[0]
    e["by_year"] = _rows("""SELECT fy_year, filings, certified, denied, positions,
        outside_hires, renewals, round(median_wage) median_wage
        FROM employer_year WHERE employer_key=? ORDER BY fy_year""", [key])
    e["top_roles"] = _rows("""SELECT soc_title, example_title, filings, outside_hires,
        round(median_wage) median_wage, last_year FROM employer_role
        WHERE employer_key=? ORDER BY filings DESC LIMIT 10""", [key])
    e["top_locations"] = _rows("""SELECT city, state, filings, outside_hires,
        round(median_wage) median_wage FROM employer_location
        WHERE employer_key=? ORDER BY filings DESC LIMIT 10""", [key])
    c = _rows("SELECT card FROM employer_card WHERE employer_key=?", [key])
    e["card"] = c[0]["card"] if c else None
    e["found"] = True
    e["also_matching"] = [r for r in find_employers(name_or_key, limit=6)[1:]]
    return e


# ---------------------------------------------------------------- search
def search_employers(role=None, state=None, city=None, min_wage=None, cap_exempt=None,
                     exclude_dependent=False, min_outside_hires=1, active_since=2025,
                     sort="score", limit=25):
    """Structured shortlist: which employers sponsor <role> in <state> above <wage>.

    Params are appended in the same order the placeholders appear in the SQL text,
    since DuckDB binds positionally.
    """
    params, joins, extra = [], "", ""

    if role:
        like = "%" + role.strip() + "%"
        joins += (" JOIN (SELECT employer_key, sum(filings) rf, sum(outside_hires) ro,"
                  " median(median_wage) rw, max(example_title) ex FROM employer_role"
                  " WHERE soc_title ILIKE ? OR example_title ILIKE ? GROUP BY 1) r"
                  " ON r.employer_key = e.employer_key")
        params += [like, like]
        extra += (", r.rf role_filings, r.ro role_outside_hires,"
                  " round(r.rw) role_median_wage, r.ex example_title")

    if state or city:
        conds, lp = [], []
        if state:
            conds.append("state = ?"); lp.append(state.strip().upper())
        if city:
            conds.append("city ILIKE ?"); lp.append("%" + city.strip() + "%")
        joins += (" JOIN (SELECT employer_key, sum(filings) lf, sum(outside_hires) lo"
                  " FROM employer_location WHERE %s GROUP BY 1) g"
                  " ON g.employer_key = e.employer_key" % " AND ".join(conds))
        params += lp
        extra += ", g.lf location_filings, g.lo location_outside_hires"

    where = ["e.last_fy >= ?", "coalesce(e.recent_outside,0) >= ?"]
    params += [active_since, min_outside_hires]
    if min_wage:
        where.append("coalesce(e.wage_median,0) >= ?"); params.append(min_wage)
    if cap_exempt is True:
        where.append("e.cap_exempt_likely")
    elif cap_exempt is False:
        where.append("NOT e.cap_exempt_likely")
    if exclude_dependent:
        where.append("coalesce(e.h1b_dependent,'No') <> 'Yes'")

    order = {"score": "e.score DESC", "volume": "e.recent_outside DESC",
             "wage": "e.wage_median DESC NULLS LAST",
             "recent": "e.last_fy DESC, e.score DESC"}.get(sort, "e.score DESC")

    sql = ("SELECT e.display_name, e.employer_key, e.hq_city, e.hq_state,"
           " e.lca_total, e.recent_filings, e.recent_outside, round(e.wage_median) wage_median,"
           " e.perm_certified, e.cap_exempt_likely, e.h1b_dependent, e.willful_violator,"
           " e.last_fy, e.score" + extra +
           " FROM employers e" + joins + " WHERE " + " AND ".join(where) +
           " ORDER BY " + order + " LIMIT ?")
    params.append(limit)
    return _rows(sql, params)


def compare_employers(names):
    return [employer_profile(n) for n in names]


# ---------------------------------------------------------------- semantic
class VectorStore:
    """int8-quantized cosine store over employer profile cards."""
    def __init__(self):
        self.ok = False
        try:
            z = np.load(VEC, allow_pickle=False)
            self.v = z["vecs"].astype(np.float32) / 127.0
            self.keys = z["keys"]
            self.model = None
            self.ok = True
        except Exception as e:
            self.err = str(e)

    def _embed(self, text):
        if self.model is None:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        q = np.array(list(self.model.embed([text]))[0], dtype=np.float32)
        return q / (np.linalg.norm(q) + 1e-9)

    def search(self, text, k=10):
        if not self.ok:
            return []
        sims = self.v @ self._embed(text)
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(str(self.keys[i]), float(sims[i])) for i in idx]


_store = None
def store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def semantic_search(query, k=10):
    """Meaning-based search over employer profile cards (the RAG retrieval step)."""
    hits = store().search(query, k)
    if not hits:
        return []
    keys = [h[0] for h in hits]
    ph = ",".join("?" * len(keys))
    rows = _rows("""SELECT e.display_name, e.employer_key, e.hq_city, e.hq_state,
        e.recent_outside, round(e.wage_median) wage_median, e.perm_certified,
        e.cap_exempt_likely, e.h1b_dependent, e.score, c.card
        FROM employers e LEFT JOIN employer_card c USING (employer_key)
        WHERE e.employer_key IN (%s)""" % ph, keys)
    sim = dict(hits)
    for r in rows:
        r["similarity"] = round(sim.get(r["employer_key"], 0), 3)
    rows.sort(key=lambda r: -r["similarity"])
    return rows


# ---------------------------------------------------------------- raw SQL
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|install|load|pragma)\b", re.I)

def run_sql(sql, limit=200):
    """Read-only SQL escape hatch so the agent can answer questions the fixed tools don't cover."""
    if FORBIDDEN.search(sql):
        return {"error": "read-only: only SELECT queries are allowed"}
    if not sql.strip().lower().startswith(("select", "with")):
        return {"error": "query must start with SELECT or WITH"}
    try:
        cur = con().execute("SELECT * FROM (%s) LIMIT %d" % (sql.rstrip("; "), limit))
        cols = [d[0] for d in cur.description]
        return {"columns": cols, "rows": [dict(zip(cols, r)) for r in cur.fetchall()]}
    except Exception as e:
        return {"error": str(e)}


def schema():
    out = []
    for t in ("employers", "employer_year", "employer_role", "employer_location", "lca", "perm"):
        cols = con().execute("DESCRIBE %s" % t).fetchall()
        out.append("%s(%s)" % (t, ", ".join("%s %s" % (c[0], c[1]) for c in cols)))
    return "\n".join(out)


def stats():
    r = _rows("""SELECT (SELECT count(*) FROM lca) lca_cases,
        (SELECT count(*) FROM perm) perm_cases,
        (SELECT count(*) FROM employers) employers,
        (SELECT count(*) FROM employers WHERE last_fy>=2025 AND recent_outside>=5) actionable,
        (SELECT max(decision_date) FROM lca) latest""")[0]
    r["latest_quarter"] = LATEST_QUARTER
    return r
