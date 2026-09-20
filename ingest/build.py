"""Build the Sponsorship Radar fact tables from raw DOL disclosure files.

DOL renames columns between fiscal years (EMPLOYER_NAME -> EMP_BUSINESS_NAME,
PW_SOC_CODE -> PWD_SOC_CODE, ...), so every field is resolved through pick()
against the actual columns present in each file rather than assumed.

Run:  python ingest/build.py     Out: data/build/radar.duckdb
"""
import duckdb, os, glob, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "build", "radar.duckdb")

XLDATE = "(DATE '1899-12-30' + CAST(TRY_CAST({c} AS DOUBLE) AS INTEGER))"
SUFFIXES = ["INCORPORATED", "CORPORATION", "COMPANY", "LIMITED", "HOLDINGS", "GROUP",
            "INC", "LLC", "LTD", "CORP", "CO", "LP", "LLP", "PLLC", "PC", "PA", "NA", "USA", "US"]


def norm_sql(col):
    """Normalize an employer name into a stable join key."""
    s = "upper(trim(%s))" % col
    s = "regexp_replace(%s, '[^A-Z0-9 &]', ' ', 'g')" % s
    s = "regexp_replace(%s, ' +', ' ', 'g')" % s
    for suf in SUFFIXES:
        s = "trim(regexp_replace(%s, ' %s$', '', 'g'))" % (s, suf)
    return "nullif(trim(%s), '')" % s


def annualize(w, unit):
    if w == "NULL":
        return "NULL"
    u = "upper(trim(%s))" % unit if unit != "NULL" else "'YEAR'"
    return """CASE %s
        WHEN 'YEAR' THEN TRY_CAST(%s AS DOUBLE)
        WHEN 'HOUR' THEN TRY_CAST(%s AS DOUBLE)*2080
        WHEN 'MONTH' THEN TRY_CAST(%s AS DOUBLE)*12
        WHEN 'WEEK' THEN TRY_CAST(%s AS DOUBLE)*52
        WHEN 'BI-WEEKLY' THEN TRY_CAST(%s AS DOUBLE)*26
        ELSE NULL END""" % (u, w, w, w, w, w)


def src(path):
    return "read_xlsx('%s', all_varchar=true)" % path


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    con = duckdb.connect(OUT)
    con.execute("INSTALL excel; LOAD excel;")
    t0 = time.time()

    def cols(path):
        return {r[0] for r in con.execute("DESCRIBE SELECT * FROM %s" % src(path)).fetchall()}

    def pick(have, *names):
        """First column name that exists in this file, else SQL NULL."""
        for n in names:
            if n in have:
                return n
        return "NULL"

    con.execute("""CREATE TABLE lca(
        fy VARCHAR, case_number VARCHAR, case_status VARCHAR, visa_class VARCHAR,
        received_date DATE, decision_date DATE,
        employer_raw VARCHAR, employer_key VARCHAR, fein VARCHAR, naics VARCHAR,
        employer_city VARCHAR, employer_state VARCHAR,
        job_title VARCHAR, soc_code VARCHAR, soc_title VARCHAR,
        worksite_city VARCHAR, worksite_state VARCHAR,
        positions INTEGER, new_employment INTEGER, continued_employment INTEGER,
        change_employer INTEGER, amended INTEGER,
        wage_from DOUBLE, wage_to DOUBLE, prevailing_wage DOUBLE, pw_level VARCHAR,
        full_time VARCHAR, h1b_dependent VARCHAR, willful_violator VARCHAR)""")

    for path in sorted(glob.glob(os.path.join(RAW, "lca_*.xlsx"))):
        fy = os.path.basename(path)[4:-5]
        c = cols(path)
        name = pick(c, "EMPLOYER_NAME")
        q = """INSERT INTO lca SELECT
            '%s', CASE_NUMBER, CASE_STATUS, %s,
            %s, %s,
            trim(%s), %s, %s, %s,
            upper(trim(%s)), upper(trim(%s)),
            trim(%s), %s, trim(%s),
            upper(trim(%s)), upper(trim(%s)),
            TRY_CAST(%s AS INTEGER), TRY_CAST(%s AS INTEGER),
            TRY_CAST(%s AS INTEGER), TRY_CAST(%s AS INTEGER), TRY_CAST(%s AS INTEGER),
            %s, %s, %s, %s, %s, %s, %s
            FROM %s WHERE CASE_NUMBER IS NOT NULL""" % (
            fy, pick(c, "VISA_CLASS"),
            XLDATE.format(c=pick(c, "RECEIVED_DATE")), XLDATE.format(c=pick(c, "DECISION_DATE")),
            name, norm_sql(name), pick(c, "EMPLOYER_FEIN"), pick(c, "NAICS_CODE"),
            pick(c, "EMPLOYER_CITY"), pick(c, "EMPLOYER_STATE"),
            pick(c, "JOB_TITLE"), pick(c, "SOC_CODE"), pick(c, "SOC_TITLE"),
            pick(c, "WORKSITE_CITY"), pick(c, "WORKSITE_STATE"),
            pick(c, "TOTAL_WORKER_POSITIONS"), pick(c, "NEW_EMPLOYMENT"),
            pick(c, "CONTINUED_EMPLOYMENT"), pick(c, "CHANGE_EMPLOYER"), pick(c, "AMENDED_PETITION"),
            annualize(pick(c, "WAGE_RATE_OF_PAY_FROM"), pick(c, "WAGE_UNIT_OF_PAY")),
            annualize(pick(c, "WAGE_RATE_OF_PAY_TO"), pick(c, "WAGE_UNIT_OF_PAY")),
            annualize(pick(c, "PREVAILING_WAGE"), pick(c, "PW_UNIT_OF_PAY")),
            pick(c, "PW_WAGE_LEVEL"), pick(c, "FULL_TIME_POSITION"),
            pick(c, "H_1B_DEPENDENT"), pick(c, "WILLFUL_VIOLATOR"), src(path))
        con.execute(q)
        n = con.execute("SELECT count(*) FROM lca WHERE fy=?", [fy]).fetchone()[0]
        print("  LCA %-9s %9s rows  (%.0fs)" % (fy, "{:,}".format(n), time.time() - t0))

    con.execute("""CREATE TABLE perm(
        fy VARCHAR, case_number VARCHAR, case_status VARCHAR, decision_date DATE,
        employer_raw VARCHAR, employer_key VARCHAR, fein VARCHAR, naics VARCHAR,
        job_title VARCHAR, soc_code VARCHAR, soc_title VARCHAR,
        worksite_city VARCHAR, worksite_state VARCHAR, wage_from DOUBLE)""")

    for path in sorted(glob.glob(os.path.join(RAW, "perm_*.xlsx"))):
        fy = os.path.basename(path)[5:-5]
        c = cols(path)
        name = pick(c, "EMP_BUSINESS_NAME", "EMPLOYER_NAME")
        q = """INSERT INTO perm SELECT
            '%s', CASE_NUMBER, CASE_STATUS, %s,
            trim(%s), %s, %s, %s,
            trim(%s), %s, trim(%s),
            upper(trim(%s)), upper(trim(%s)), %s
            FROM %s WHERE CASE_NUMBER IS NOT NULL""" % (
            fy, XLDATE.format(c=pick(c, "DECISION_DATE")),
            name, norm_sql(name), pick(c, "EMP_FEIN", "EMPLOYER_FEIN"),
            pick(c, "EMP_NAICS", "NAICS_CODE"),
            pick(c, "JOB_TITLE"), pick(c, "PWD_SOC_CODE", "PW_SOC_CODE"),
            pick(c, "PWD_SOC_TITLE", "PW_SOC_TITLE"),
            pick(c, "PRIMARY_WORKSITE_CITY", "WORKSITE_CITY"),
            pick(c, "PRIMARY_WORKSITE_STATE", "WORKSITE_STATE"),
            annualize(pick(c, "JOB_OPP_WAGE_FROM", "WAGE_OFFER_FROM"),
                      pick(c, "JOB_OPP_WAGE_PER", "WAGE_OFFER_UNIT_OF_PAY")), src(path))
        con.execute(q)
        n = con.execute("SELECT count(*) FROM perm WHERE fy=?", [fy]).fetchone()[0]
        print("  PERM %-9s %9s rows  (%.0fs)" % (fy, "{:,}".format(n), time.time() - t0))

    con.close()
    print("\nfact tables built in %.0fs -> %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
