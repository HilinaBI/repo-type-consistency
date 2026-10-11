import json
import os
from collections import Counter

CLEAN_REPOS = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\cleaned_repos"

FILES = [
    ("Training alpha=1.0",            "CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS.jsonl"),
    ("Training alpha=1.0 (filtered)", "CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS_filtered.jsonl"),
    ("Validation alpha=0.5",          "CODET5_VALIDATION_NOTYPE_NORMALIZED_α0.5.jsonl"),
    ("Test inference",                "TEST_SET_INFERENCE_CODET5_NORMALIZED_clean.jsonl"),
    ("Test correction",               "TEST_SET_CORRUPTED_CODET5_NORMALIZED_clean.jsonl"),
]


def strip_prefix(example_id):
    for prefix in ("val_", "infer_", "test_"):
        if example_id.startswith(prefix):
            return example_id[len(prefix):]
    return example_id


def repo_of(example_id):
    return strip_prefix(example_id).split("_", 1)[0]


def kind_of(example_id):
    parts = strip_prefix(example_id).split("_")
    for p in parts:
        if p in ("arg", "return", "kwarg", "vararg"):
            return p
    return "unknown"


def measure(path):
    total = 0
    per_repo = Counter()
    per_type = Counter()
    per_kind = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            eid = obj.get("id", "")

            repo = repo_of(eid)
            if repo:
                per_repo[repo] += 1

            kind = kind_of(eid)
            per_kind[kind] += 1

            target = obj.get("target", "").strip()
            if target:
                per_type[target] += 1

    return total, per_repo, per_type, per_kind


def write_report(out):
    for label, filename in FILES:
        path = os.path.join(CLEAN_REPOS, filename)
        if not os.path.exists(path):
            out.write(f"\nSKIP (not found): {filename}\n")
            continue

        total, per_repo, per_type, per_kind = measure(path)
        if total == 0:
            out.write(f"\n{label}: empty\n")
            continue

        out.write("=" * 70 + "\n")
        out.write(f"{label}\n")
        out.write(f"file: {filename}\n")
        out.write(f"total examples: {total}\n")
        out.write("=" * 70 + "\n")

        out.write("\n  REPOSITORY DIVERSITY\n")
        out.write(f"  distinct repositories : {len(per_repo)}\n")
        top3 = sum(c for _, c in per_repo.most_common(3))
        top5 = sum(c for _, c in per_repo.most_common(5))
        out.write(f"  top 3 share           : {top3 / total * 100:.1f}%\n")
        out.write(f"  top 5 share           : {top5 / total * 100:.1f}%\n")
        out.write("  top 5 repositories:\n")
        for repo, count in per_repo.most_common(5):
            pct = count / total * 100
            out.write(f"    {repo:25s} {count:>8}   {pct:>5.1f}%\n")

        out.write("\n  TARGET TYPE DIVERSITY\n")
        out.write(f"  distinct target types : {len(per_type)}\n")
        top3t = sum(c for _, c in per_type.most_common(3))
        top10t = sum(c for _, c in per_type.most_common(10))
        top50t = sum(c for _, c in per_type.most_common(50))
        out.write(f"  top 3 types share     : {top3t / total * 100:.1f}%\n")
        out.write(f"  top 10 types share    : {top10t / total * 100:.1f}%\n")
        out.write(f"  top 50 types share    : {top50t / total * 100:.1f}%\n")
        out.write("  top 10 types:\n")
        for t, count in per_type.most_common(10):
            pct = count / total * 100
            short = t if len(t) <= 35 else t[:32] + "..."
            out.write(f"    {short:35s} {count:>8}   {pct:>5.1f}%\n")

        out.write("\n  ANNOTATION-KIND MIX\n")
        for k in ("arg", "return", "kwarg", "vararg", "unknown"):
            if per_kind.get(k):
                pct = per_kind[k] / total * 100
                out.write(f"    {k:10s} {per_kind[k]:>8}   {pct:>5.1f}%\n")

        out.write("\n")


def main():
    out_path = os.path.join(CLEAN_REPOS, "diversity_full_report.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        write_report(out)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()