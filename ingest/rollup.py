"""Roll per-case facts up into employer-level sponsorship signals.

The important distinction this build makes, which most H-1B lookup sites miss:
a filing for CONTINUED_EMPLOYMENT is an existing employee's renewal, while
NEW_EMPLOYMENT and CHANGE_EMPLOYER are hires from outside. An employer with
4,000 renewals and 3 outside hires is not a place a student can get hired.

Run: python ingest/rollup.py   (after ingest/build.py)
"""
import duckdb, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "build", "radar.duckdb")

# Free-text wage fields contain typos ($1B salaries). Keep a defensible band.
WAGE_MIN, WAGE_MAX = 10_000, 5_000_000
# US federal fiscal year: Oct 1 starts the next FY.
FY = "(year(decision_date) + CASE WHEN month(decision_date) >= 10 THEN 1 ELSE 0 END)"


def main():
    con = duckdb.connect(DB)
    t0 = time.time()
    con.execute("SET threads TO 4")

    con.execute("DROP VIEW IF EXISTS lca_c")
    con.execute("""CREATE VIEW lca_c AS SELECT *,
        %s AS fy_year,
        CASE WHEN wage_from BETWEEN %d AND %d THEN wage_from END AS wage_ok,
        coalesce(new_employment,0) + coalesce(change_employer,0) AS outside_hires,
        coalesce(continued_employment,0) + coalesce(amended,0) AS renewals
        FROM lca""" % (FY, WAGE_MIN, WAGE_MAX))

    con.execute("DROP VIEW IF EXISTS perm_c")
    con.execute("CREATE VIEW perm_c AS SELECT *, %s AS fy_year FROM perm" % FY)

    # ---- per-employer per-year ----
    con.execute("DROP TABLE IF EXISTS employer_year")
    con.execute("""CREATE TABLE employer_year AS
        SELECT employer_key, fy_year,
               count(*) filings,
               count(*) FILTER (WHERE case_status ILIKE 'Certified%') certified,
               count(*) FILTER (WHERE case_status = 'Denied') denied,
               sum(coalesce(positions,1)) positions,
               sum(outside_hires) outside_hires,
               sum(renewals) renewals,
               median(wage_ok) median_wage
        FROM lca_c WHERE employer_key IS NOT NULL GROUP BY 1,2""")

    # ---- per-employer per-role ----
    con.execute("DROP TABLE IF EXISTS employer_role")
    con.execute("""CREATE TABLE employer_role AS
        SELECT employer_key, soc_title, count(*) filings,
               sum(outside_hires) outside_hires,
               median(wage_ok) median_wage,
               max(fy_year) last_year,
               mode(job_title) example_title
        FROM lca_c WHERE employer_key IS NOT NULL AND soc_title IS NOT NULL
        GROUP BY 1,2""")

    # ---- per-employer per-location ----
    con.execute("DROP TABLE IF EXISTS employer_location")
    con.execute("""CREATE TABLE employer_location AS
        SELECT employer_key, worksite_city city, worksite_state state,
               count(*) filings, sum(outside_hires) outside_hires,
               median(wage_ok) median_wage, max(fy_year) last_year
        FROM lca_c WHERE employer_key IS NOT NULL AND worksite_state IS NOT NULL
        GROUP BY 1,2,3""")

    # ---- employer master ----
    con.execute("DROP TABLE IF EXISTS employers")
    con.execute("""CREATE TABLE employers AS
    WITH l AS (
        SELECT employer_key,
            mode(employer_raw) display_name,
            max(fein) fein,
            mode(naics) naics,
            mode(employer_city) hq_city, mode(employer_state) hq_state,
            count(*) lca_total,
            count(*) FILTER (WHERE case_status ILIKE 'Certified%') lca_certified,
            count(*) FILTER (WHERE case_status = 'Denied') lca_denied,
            sum(coalesce(positions,1)) positions_total,
            sum(outside_hires) outside_hires,
            sum(renewals) renewals,
            min(decision_date) first_seen, max(decision_date) last_seen,
            max(fy_year) last_fy,
            count(*) FILTER (WHERE fy_year >= 2025) recent_filings,
            sum(outside_hires) FILTER (WHERE fy_year >= 2025) recent_outside,
            quantile_cont(wage_ok, 0.25) wage_p25,
            median(wage_ok) wage_median,
            quantile_cont(wage_ok, 0.75) wage_p75,
            mode(h1b_dependent) h1b_dependent,
            max(CASE WHEN willful_violator='Yes' THEN 1 ELSE 0 END) willful_violator,
            count(DISTINCT worksite_state) n_states
        FROM lca_c WHERE employer_key IS NOT NULL GROUP BY 1),
    p AS (
        SELECT employer_key,
            count(*) perm_total,
            count(*) FILTER (WHERE case_status ILIKE 'Certified%') perm_certified,
            max(decision_date) perm_last_seen
        FROM perm_c WHERE employer_key IS NOT NULL GROUP BY 1)
    SELECT l.*,
        coalesce(p.perm_total,0) perm_total,
        coalesce(p.perm_certified,0) perm_certified,
        p.perm_last_seen,
        -- cap-exempt proxies: universities, hospitals, nonprofit research
        -- Heuristic only. Cap-exempt status legally requires nonprofit/university
        -- affiliation; NAICS 5417 (R&D) is excluded because it includes for-profit labs.
        (substr(l.naics,1,2)='61' OR substr(l.naics,1,3)='622') cap_exempt_likely,
        CASE WHEN l.lca_total>0 THEN l.outside_hires::DOUBLE/nullif(l.outside_hires+l.renewals,0) END outside_ratio
    FROM l LEFT JOIN p USING (employer_key)""")

    # ---- transparent 0-100 sponsorship score ----
    con.execute("ALTER TABLE employers ADD COLUMN score DOUBLE")
    con.execute("""UPDATE employers SET score = round(least(100, greatest(0,
          35 * least(1, ln(1 + coalesce(recent_outside,0)) / ln(200))
        + 25 * least(1, ln(1 + coalesce(recent_filings,0)) / ln(300))
        + 15 * coalesce(outside_ratio,0)
        + 10 * CASE WHEN last_fy >= 2026 THEN 1 WHEN last_fy = 2025 THEN 0.6 ELSE 0.1 END
        +  8 * least(1, ln(1 + perm_certified) / ln(100))
        +  7 * CASE WHEN cap_exempt_likely THEN 1 ELSE 0 END
        - 15 * willful_violator
    )), 1)""")

    for t in ("employers", "employer_year", "employer_role", "employer_location"):
        n = con.execute("SELECT count(*) FROM " + t).fetchone()[0]
        print("  %-20s %9s rows" % (t, "{:,}".format(n)))
    con.execute("CREATE INDEX IF NOT EXISTS ix_ey ON employer_year(employer_key)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_er ON employer_role(employer_key)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_el ON employer_location(employer_key)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_emp ON employers(employer_key)")
    con.close()
    print("rollup done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
