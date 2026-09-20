"""Sanity checks that would catch a broken rebuild. Run after scripts/rebuild.sh."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import db

fails = []
def check(name, cond, detail=""):
    print("  %-52s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond: fails.append(name)

s = db.stats()
check("LCA cases ingested > 1.5M", s["lca_cases"] > 1_500_000, "{:,}".format(s["lca_cases"]))
check("PERM cases ingested > 200k", s["perm_cases"] > 200_000, "{:,}".format(s["perm_cases"]))
check("employers > 100k", s["employers"] > 100_000, "{:,}".format(s["employers"]))
check("latest data is FY2026 Q1", str(s["latest"]).startswith("2025-12"), str(s["latest"]))

# known-good employers must resolve and look right
for name, minf in [("Google", 20000), ("Infosys", 10000), ("Mayo Clinic", 500)]:
    p = db.employer_profile(name)
    check("profile: %s resolves" % name, p.get("found"))
    check("profile: %s filings >= %d" % (name, minf), (p.get("lca_total") or 0) >= minf,
          "{:,}".format(p.get("lca_total") or 0))

p = db.employer_profile("Mayo Clinic")
check("Mayo Clinic flagged cap-exempt (hospital)", bool(p.get("cap_exempt_likely")))
p = db.employer_profile("Google")
check("Google NOT flagged cap-exempt", not p.get("cap_exempt_likely"))
check("Google has role + location breakdowns", len(p["top_roles"]) > 3 and len(p["top_locations"]) > 3)
check("Google year series covers FY2023-2026",
      {y["fy_year"] for y in p["by_year"]} >= {2023, 2024, 2025, 2026})

# wage sanity: no absurd values survived the clamp
w = db.run_sql("SELECT max(wage_median) m, min(wage_median) n FROM employers WHERE wage_median IS NOT NULL")
mx = w["rows"][0]["m"]; mn = w["rows"][0]["n"]
check("median wages within sane band", 5_000 < mn and mx < 5_000_000, "%s..%s" % (mn, mx))

# scoped shortlist really is scoped
rows = db.search_employers(role="Data Scientist", state="MA", limit=5)
check("MA shortlist returns results", len(rows) > 0)
if rows:
    keys = [r["employer_key"] for r in rows]
    q = db.run_sql("""SELECT count(*) n FROM lca WHERE worksite_state='MA'
        AND (job_title ILIKE '%%Data Scientist%%' OR soc_title ILIKE '%%Data Scientist%%')
        AND employer_key = '%s'""" % keys[0].replace("'", "''"))
    check("top MA result genuinely has MA filings", q["rows"][0]["n"] > 0, "%d rows" % q["rows"][0]["n"])
    check("scoped hires <= global hires",
          all((r["match_outside_hires"] or 0) <= (r["recent_outside"] or 0) + 1 for r in rows))

# guards
check("SQL guard blocks writes", "error" in db.run_sql("DROP TABLE employers"))
check("SQL guard blocks non-SELECT", "error" in db.run_sql("PRAGMA database_list"))
check("SQL allows SELECT", "rows" in db.run_sql("SELECT 1 AS x"))

# vector store
check("vector store loaded", db.store().ok)
hits = db.semantic_search("biotech companies in San Diego", k=3)
check("semantic search returns hits", len(hits) == 3)
check("semantic hits carry cards", all(h.get("card") for h in hits))

print("\n%d/%d checks passed" % (24 - len(fails), 24))
if fails:
    print("FAILED:", fails); sys.exit(1)
