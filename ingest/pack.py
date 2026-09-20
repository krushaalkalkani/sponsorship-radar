"""Pack the full build into a slim, read-only serving database.

Two size decisions, both measured rather than assumed:

  * No indexes. DuckDB's ART indexes cost ~118MB here and bought nothing -
    point lookups on 117k employers are ~3ms from a plain columnar scan.
  * Per-case rows only from FY2025 on. The rollup tables (employer_year /
    _role / _location) still cover all four fiscal years, so trend questions
    are unaffected; only ad-hoc per-case SQL is windowed. Keeping every year
    costs 117MB, which would break GitHub's 100MB file limit and make the
    container image needlessly fat.

Result: ~78MB, small enough to commit and ship inside the image, so the
deployment needs no database service and no artifact hosting.

Run: python ingest/pack.py    Out: data/build/radar_serve.duckdb
"""
import duckdb, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "build", "radar.duckdb")
DST = os.path.join(ROOT, "data", "build", "radar_serve.duckdb")
PERCASE_FROM_FY = 2025
FY = "(year(decision_date) + CASE WHEN month(decision_date) >= 10 THEN 1 ELSE 0 END)"


def main():
    if os.path.exists(DST):
        os.remove(DST)
    con = duckdb.connect(DST)
    con.execute("ATTACH '%s' AS srcdb (READ_ONLY)" % SRC)

    for t in ("employers", "employer_year", "employer_role", "employer_location",
              "employer_card", "employer_month"):
        con.execute("CREATE TABLE %s AS SELECT * FROM srcdb.%s" % (t, t))

    con.execute("""CREATE TABLE lca AS SELECT
        employer_key, case_status, visa_class, decision_date, %s fy_year,
        job_title, soc_title, worksite_city, worksite_state,
        positions, new_employment, continued_employment, change_employer,
        wage_from, pw_level
        FROM srcdb.lca WHERE employer_key IS NOT NULL AND %s >= %d"""
        % (FY, FY, PERCASE_FROM_FY))

    con.execute("""CREATE TABLE perm AS SELECT
        employer_key, case_status, decision_date, %s fy_year,
        job_title, soc_title, worksite_city, worksite_state, wage_from
        FROM srcdb.perm WHERE employer_key IS NOT NULL AND %s >= %d"""
        % (FY, FY, PERCASE_FROM_FY))

    for t in ("employers", "employer_year", "employer_role", "employer_location",
              "employer_card", "employer_month", "lca", "perm"):
        n = con.execute("SELECT count(*) FROM %s" % t).fetchone()[0]
        print("  %-20s %10s rows" % (t, "{:,}".format(n)))

    # Record true source coverage: the employer rollups were computed from every
    # case, not just the per-case window that ships. Reporting the window count
    # as "filings analyzed" would understate the dataset by ~3x.
    con.execute("""CREATE TABLE meta AS SELECT
        (SELECT count(*) FROM srcdb.lca) AS lca_cases_total,
        (SELECT count(*) FROM srcdb.perm) AS perm_cases_total,
        (SELECT count(*) FROM lca) AS lca_cases_percase,
        (SELECT count(*) FROM perm) AS perm_cases_percase,
        (SELECT min(decision_date) FROM srcdb.lca) AS first_date,
        (SELECT max(decision_date) FROM srcdb.lca) AS last_date,
        %d AS percase_from_fy""" % PERCASE_FROM_FY)

    con.execute("DETACH srcdb")
    con.close()
    print("serving db: %.1f MB -> %s" % (os.path.getsize(DST) / 1e6, DST))


if __name__ == "__main__":
    main()
