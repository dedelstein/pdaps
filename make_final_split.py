import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the fixed patient-disjoint final comparison split."
    )
    parser.add_argument(
        "--maps-dir",
        default="/dtu/blackhole/1d/214141/Thesis/data/knee/multicoil_val_sens_maps_espirit",
        help="Directory containing one ESPIRiT map HDF5 per fastMRI validation patient.",
    )
    parser.add_argument("--out", default="final_split.json")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--expected-count", type=int, default=199)
    return parser.parse_args()


def main():
    args = parse_args()
    maps_dir = Path(args.maps_dir)
    filenames = sorted(path.name for path in maps_dir.glob("*.h5"))
    if args.expected_count is not None and len(filenames) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} HDF5 files in {maps_dir}, found {len(filenames)}"
        )
    needed = args.val_count + args.test_count
    if len(filenames) < needed:
        raise SystemExit(f"Need at least {needed} files, found {len(filenames)}")

    shuffled = list(filenames)
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)
    val = shuffled[:args.val_count]
    test = shuffled[args.val_count:needed]
    reserved = shuffled[needed:]
    split = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_maps_dir": str(maps_dir),
        "shuffle_seed": args.seed,
        "source_count": len(filenames),
        "val_count": len(val),
        "test_count": len(test),
        "reserved_count": len(reserved),
        "val": val,
        "test": test,
        "reserved": reserved,
        "all_sorted": filenames,
    }
    overlap = set(val) & set(test)
    if overlap:
        raise SystemExit(f"Internal error: val/test overlap: {sorted(overlap)}")

    out = Path(args.out)
    with open(out, "w") as f:
        json.dump(split, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}: {len(val)} val, {len(test)} test, {len(reserved)} reserved")


if __name__ == "__main__":
    main()
