"""Deploy Sponsorship Radar to a Hugging Face Space (Docker SDK).

Uploads over the HTTP API, so git-lfs is not required even though the DuckDB
file is ~79MB.

  export HF_TOKEN=hf_...            # needs write permission; Docker Spaces require HF PRO
  python scripts/deploy_hf.py --space krushalkalkani/sponsorship-radar

Set the LLM key as a Space *secret* afterwards (or pass --openai-key):
  Settings -> Variables and secrets -> New secret -> OPENAI_API_KEY
"""
import argparse, os, sys
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything the running container needs. Raw xlsx and the full build DB stay local.
INCLUDE = [
    "Dockerfile", "README.md", "requirements.txt",
    "app/__init__.py", "app/main.py", "app/db.py", "app/agent.py",
    "app/static/index.html",
    "ingest/build.py", "ingest/rollup.py", "ingest/embed.py", "ingest/pack.py",
    "ingest/sources.txt",
    "scripts/download.sh", "scripts/rebuild.sh", "scripts/run.sh",
    "scripts/validate.py", "scripts/deploy_hf.py",
    "data/build/radar_serve.duckdb", "data/build/vectors.npz",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. username/sponsorship-radar")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"),
                    help="stored as a Space secret, not committed")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set. Create one at https://huggingface.co/settings/tokens "
                 "with 'write' permission.")

    api = HfApi(token=token)
    who = api.whoami()
    print("authenticated as:", who.get("name"))

    missing = [f for f in INCLUDE if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        sys.exit("missing files (run scripts/rebuild.sh first): %s" % missing)

    api.create_repo(repo_id=args.space, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    print("space ready:", args.space)

    if args.openai_key:
        api.add_space_secret(repo_id=args.space, key="OPENAI_API_KEY",
                             value=args.openai_key)
        print("secret OPENAI_API_KEY set")

    total = sum(os.path.getsize(os.path.join(ROOT, f)) for f in INCLUDE)
    print("uploading %d files (%.0f MB) ..." % (len(INCLUDE), total / 1e6))
    api.upload_folder(
        repo_id=args.space, repo_type="space", folder_path=ROOT,
        allow_patterns=INCLUDE,
        commit_message="Deploy H-1B Sponsorship Radar",
    )
    url = "https://huggingface.co/spaces/%s" % args.space
    print("\ndone ->", url)
    print("build logs:", url + "?logs=build")


if __name__ == "__main__":
    main()
