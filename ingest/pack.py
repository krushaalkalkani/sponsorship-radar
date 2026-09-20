"""Pack the full build into a slim, read-only serving database.

The full radar.duckdb keeps every raw disclosure column (~270MB) because the
ingest needs them. The deployed app only needs the rollups plus a trimmed
per-case table for ad-hoc SQL, which is small enough to commit and ship inside
the container image - no database service, no artifact hosting.

Run: python ingest/pack.py    Out: data/build/radar_serve.duckdb
"""
import duckdb, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "build", "radar.duckdb")
DST = os.path.join(ROOT, "data", "build", "radar_serve.duckdb")


def main():
    if os.path.exists(DST):
        os.remove(DST)
    con = duckdb.connect(DST)
    con.execute("ATTACH '%s' AS full (READ_ONLY)" % SRC)

    for t in ("employers", "employer_year", "employer_role", "employer_location", "employer_card"):
        con.execute("CREATE TABLE %s AS SELECT * FROM full.%s" % (t, t))

    # Trimmed per-case tables: the columns the agent's SQL tool actually reasons
    # over. Drops names/addresses/attorney/POC fields, which is most of the bulk.
    con.execute("""CREATE TABLE lca AS SELECT
        employer_key, case_status, visa_class, decision_date,
        (year(decision_date) + CASE WHEN month(decision_date) >= 10 THEN 1 ELSE 0 END) fy_year,
        job_title, soc_title, worksite_city, worksite_state,
        positions, new_employment, continued_employment, change_employer,
        wage_from, pw_level, h1b_dependent
        FROM full.lca WHERE employer_key IS NOT NULL""")

    con.execute("""CREATE TABLE perm AS SELECT
        employer_key, case_status, decision_date,
        (year(decision_date) + CASE WHEN month(decision_date) >= 10 THEN 1 ELSE 0 END) fy_year,
        job_title, soc_title, worksite_city, worksite_state, wage_from
        FROM full.perm WHERE employer_key IS NOT NULL""")

    for t, c in [("employer_year", "employer_key"), ("employer_role", "employer_key"),
                 ("employer_location", "employer_key"), ("employers", "employer_key"),
                 ("employer_card", "employer_key"), ("lca", "employer_key"),
                 ("perm", "employer_key")]:
        con.execute("CREATE INDEX ix_%s ON %s(%s)" % (t, t, c))

    con.execute("DETACH full")
    con.close()
    print("serving db: %.1f MB -> %s" % (os.path.getsize(DST) / 1e6, DST))


if __name__ == "__main__":
    main()
