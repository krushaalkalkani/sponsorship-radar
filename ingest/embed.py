"""Build the RAG layer: one natural-language profile card per employer, embedded.

Cards are written back into DuckDB (so the agent can cite them verbatim) and the
vectors go to a compact int8-quantized numpy store. At ~53k vectors a brute-force
cosine scan is ~5ms, which is faster than the overhead of an ANN index and keeps
the deploy free of a vector-DB service.

Run: python ingest/embed.py    (after ingest/rollup.py)
"""
import duckdb, numpy as np, os, time, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "build", "radar.duckdb")
VEC = os.path.join(ROOT, "data", "build", "vectors.npz")
VEC_OPENAI = os.path.join(ROOT, "data", "build", "vectors_openai.npz")
MODEL = "BAAI/bge-small-en-v1.5"          # 384-dim, ~130MB ONNX, no torch


def _h(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def money(x):
    return "$%s" % format(int(x), ",") if x and x == x else "unknown"


def build_cards(con):
    """One paragraph per employer, written the way a student would ask about it."""
    rows = con.execute("""
        SELECT e.employer_key, e.display_name, e.hq_city, e.hq_state, e.naics,
               e.lca_total, e.recent_filings, e.recent_outside, e.outside_ratio,
               e.wage_median, e.perm_certified, e.cap_exempt_likely,
               e.h1b_dependent, e.willful_violator, e.last_fy, e.score,
               (SELECT string_agg(soc_title, '; ' ORDER BY filings DESC, soc_title)
                  FROM (SELECT soc_title, filings FROM employer_role r
                        WHERE r.employer_key=e.employer_key
                        ORDER BY filings DESC, soc_title LIMIT 6)) roles,
               (SELECT string_agg(city || ', ' || state, '; ' ORDER BY filings DESC, city, state)
                  FROM (SELECT city, state, filings FROM employer_location l
                        WHERE l.employer_key=e.employer_key
                        ORDER BY filings DESC, city, state LIMIT 5)) locs
        FROM employers e
        WHERE e.last_fy >= 2025 AND coalesce(e.recent_outside,0) >= 1
        ORDER BY e.score DESC, e.employer_key""").fetchall()

    cards = []
    for (k, name, city, st, naics, tot, rf, ro, ratio, wage, perm, capx,
         dep, wv, lastfy, score, roles, locs) in rows:
        p = ["%s%s is an employer that has sponsored H-1B workers." %
             (name, " (%s, %s)" % (city, st) if city and st else "")]
        p.append("It filed %s H-1B labor condition applications in total, %s of them since FY2025."
                 % (format(tot, ","), format(rf or 0, ",")))
        p.append("About %s worker positions in recent filings were new hires from outside the company "
                 "rather than renewals of existing staff." % format(int(ro or 0), ","))
        if roles: p.append("Roles sponsored: %s." % roles)
        if locs:  p.append("Work locations: %s." % locs)
        p.append("Median offered salary is %s." % money(wage))
        if perm: p.append("It has also had %s permanent-residency (green card) cases certified." % format(perm, ","))
        else:    p.append("It has no certified green card (PERM) cases in this dataset.")
        if capx: p.append("As a university or hospital it is likely cap-exempt, meaning it can sponsor H-1B outside the annual lottery.")
        if dep == "Yes": p.append("It is flagged H-1B dependent, typical of IT staffing and consulting firms.")
        if wv:   p.append("It has been flagged as a willful violator of H-1B rules.")
        p.append("Most recent filing activity: fiscal year %s. Sponsorship score %s out of 100." % (lastfy, score))
        cards.append((k, " ".join(p)))
    return cards


def main(cards_only=False):
    """cards_only re-creates the employer_card table and verifies it still lines up
    with the existing vectors, skipping the ~25 min embed. Safe only when the card
    text is unchanged - the key check below enforces that."""
    con = duckdb.connect(DB)
    t0 = time.time()
    cards = build_cards(con)
    print("  built %s profile cards (%.0fs)" % (format(len(cards), ","), time.time() - t0))
    print("  example:\n    %s\n" % cards[0][1][:400])

    con.execute("DROP TABLE IF EXISTS employer_card")
    con.execute("CREATE TABLE employer_card(employer_key VARCHAR, card VARCHAR)")
    con.executemany("INSERT INTO employer_card VALUES (?,?)", cards)

    if cards_only:
        # verify against whichever vector file is present
        vp = VEC if os.path.exists(VEC) else VEC_OPENAI
        if not os.path.exists(vp):
            con.close()
            raise SystemExit("no vector file to verify against")
        print("  verifying against %s" % os.path.basename(vp))
        z = np.load(vp, allow_pickle=False)
        old_keys = [str(k) for k in z["keys"]]
        new_by_key = {k: t for k, t in cards}
        if set(old_keys) != set(new_by_key):
            con.close()
            raise SystemExit("employer set changed (%d -> %d); run a full embed"
                             % (len(old_keys), len(new_by_key)))
        if "hashes" in z.files:
            drift = [k for k, h in zip(old_keys, z["hashes"])
                     if _h(new_by_key[k]) != str(h)]
            if drift:
                con.close()
                raise SystemExit("%d card texts changed (e.g. %s); run a full embed"
                                 % (len(drift), drift[0]))
            print("  verified all %s card texts unchanged" % format(len(old_keys), ","))
        else:
            print("  NOTE: vectors predate hash tracking; set match only. "
                  "Run a full embed if card wording changed.")
        con.close()
        print("  cards rebuilt against %s existing vectors (%.0fs)"
              % (format(len(old_keys), ","), time.time() - t0))
        return

    from fastembed import TextEmbedding
    print("  embedding with %s ..." % MODEL)
    model = TextEmbedding(model_name=MODEL)
    texts = [c for _, c in cards]
    vecs = np.array(list(model.embed(texts, batch_size=256)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)   # int8 quantize
    np.savez_compressed(VEC, vecs=q,
                        keys=np.array([k for k, _ in cards]),
                        hashes=np.array([_h(t) for _, t in cards]))
    con.close()
    print("  %s vectors dim=%d -> %s (%.1f MB) in %.0fs"
          % (format(len(q), ","), q.shape[1], VEC, os.path.getsize(VEC) / 1e6, time.time() - t0))


if __name__ == "__main__":
    import sys
    main(cards_only="--cards-only" in sys.argv)
