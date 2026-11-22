"""Download the confirmed raw data files (docs/data_sources.md) anonymously and verify sha256.

Files go to data/raw/<domain>/<repo path>. Every repo is pinned to the commit seen at download time; the
expected sha256 comes from the Git LFS metadata of that commit. Results go to data/raw/manifest.json.

Usage: python scripts/prep_download.py --domains web,math,code,eval --code-files 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"
STARCODER = "data/starcoder/v1-decon-100_to_20k-2star-top_token_030/documents"


def planned_files(api: HfApi, domains: list[str], code_files: int) -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    if "web" in domains:
        plan += [("web", "HuggingFaceFW/fineweb-edu", f"sample/10BT/{i:03d}_00000.parquet") for i in range(2)]
    if "math" in domains:
        plan += [("math", "HuggingFaceTB/finemath", f"finemath-4plus/train-{i:05d}-of-00064.parquet")
                 for i in range(2)]
    if "code" in domains:
        plan += [("code", "allenai/olmo-mix-1124", f"{STARCODER}/python-{i:04d}.json.gz")
                 for i in range(code_files)]
    if "eval" in domains:
        plan.append(("eval", "EleutherAI/lambada_openai", "en/test/en.parquet"))
        blimp = sorted(f for f in api.list_repo_files("nyu-mll/blimp", repo_type="dataset", token=False)
                       if f.endswith(".parquet"))
        plan += [("eval", "nyu-mll/blimp", f) for f in blimp]
    return plan


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="web,math,code,eval")
    ap.add_argument("--code-files", type=int, default=1)
    args = ap.parse_args()
    api = HfApi()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": []}
    done = {(f["repo"], f["path"]) for f in manifest["files"] if f["ok"]}
    revisions = manifest.setdefault("revisions", {})
    plan = planned_files(api, args.domains.split(","), args.code_files)
    for repo in sorted({r for _, r, _ in plan}):
        revisions.setdefault(repo, api.dataset_info(repo, token=False).sha)
    for domain, repo, path in plan:
        if (repo, path) in done:
            print(f"skip (verified) {repo}/{path}", flush=True)
            continue
        rev = revisions[repo]
        info = api.get_paths_info(repo, [path], repo_type="dataset", revision=rev, token=False)[0]
        expected = info.lfs.sha256 if info.lfs is not None else None
        start = time.time()
        local = Path(hf_hub_download(repo, path, repo_type="dataset", revision=rev,
                                     local_dir=RAW / domain, token=False))
        got = sha256_of(local)
        size = local.stat().st_size
        ok = expected is not None and got == expected and size == info.size
        entry = {"domain": domain, "repo": repo, "revision": rev, "path": path,
                 "local": str(local.relative_to(ROOT)).replace("\\", "/"), "size": size,
                 "sha256": got, "expected_sha256": expected, "ok": ok}
        manifest["files"] = [f for f in manifest["files"] if (f["repo"], f["path"]) != (repo, path)] + [entry]
        MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"{'OK ' if ok else 'BAD'} {size / 1e6:9.1f} MB {time.time() - start:6.1f}s {repo}/{path}",
              flush=True)
        if not ok:
            raise SystemExit(f"sha256/size mismatch for {repo}/{path}: got {got}, expected {expected}")
    total = sum(f["size"] for f in manifest["files"])
    print(f"verified files: {sum(f['ok'] for f in manifest['files'])}, total {total / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
