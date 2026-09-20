"""Embed employer profile cards with the OpenAI embeddings API.

Why a second backend alongside fastembed: the local ONNX model costs ~440MB of
resident memory at query time, which does not fit a 512MB free-tier container.
Embedding the *query* through the API instead keeps the server at ~190MB. The
53k card vectors are precomputed here once (~$0.20) and shipped as int8.

Run: python ingest/embed_openai.py      Out: data/build/vectors_openai.npz
"""
import duckdb, numpy as np, os, sys, time, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "build", "radar.duckdb")
OUT = os.path.join(ROOT, "data", "build", "vectors_openai.npz")
MODEL = "text-embedding-3-small"
DIMS = 512            # 3-small supports truncation; 512 keeps the store at ~27MB
BATCH = 512

sys.path.insert(0, os.path.join(ROOT, "ingest"))
from embed import build_cards, _h            # reuse the exact card text


def main():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=key)

    con = duckdb.connect(DB, read_only=True)
    cards = build_cards(con)
    con.close()
    print("  %s cards to embed with %s (dims=%d)" % (format(len(cards), ","), MODEL, DIMS))

    # Resume: a dropped connection partway through a 52k-card run shouldn't cost
    # the whole pass, so completed batches are checkpointed to disk.
    ckpt = OUT + ".partial.npy"
    vecs = list(np.load(ckpt)) if os.path.exists(ckpt) else []
    if vecs:
        print("  resuming from checkpoint at %s" % format(len(vecs), ","))

    t0 = time.time()
    for i in range(len(vecs), len(cards), BATCH):
        chunk = [t for _, t in cards[i:i + BATCH]]
        for attempt in range(6):
            try:
                r = client.embeddings.create(model=MODEL, input=chunk, dimensions=DIMS)
                break
            except Exception as e:
                if attempt == 5:
                    np.save(ckpt, np.array(vecs, dtype=np.float32))
                    raise SystemExit("\nfailed after retries: %s\ncheckpoint saved at %s "
                                     "- rerun to resume" % (str(e)[:200], ckpt))
                wait = 2 ** attempt
                print("\n  %s; retry %d in %ds" % (type(e).__name__, attempt + 1, wait))
                time.sleep(wait)
        vecs.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        done = min(i + BATCH, len(cards))
        if (done // BATCH) % 10 == 0:
            np.save(ckpt, np.array(vecs, dtype=np.float32))
        print("\r  %s/%s  (%.0fs)" % (format(done, ","), format(len(cards), ","),
                                      time.time() - t0), end="", flush=True)
    print()

    v = np.array(vecs, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    q = np.clip(np.round(v * 127), -127, 127).astype(np.int8)
    np.savez_compressed(OUT, vecs=q,
                        keys=np.array([k for k, _ in cards]),
                        hashes=np.array([_h(t) for _, t in cards]),
                        backend=np.array("openai:%s:%d" % (MODEL, DIMS)))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("  %s vectors dim=%d -> %s (%.1f MB) in %.0fs"
          % (format(len(q), ","), q.shape[1], OUT, os.path.getsize(OUT) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
