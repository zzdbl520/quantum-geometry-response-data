#!/usr/bin/env python3
"""Read-only packager for the JCP B2 v1p8/v1p9 SA3 checkpoint release.

Run from /scr/zdb/B2_quantum_geometry/B2_polar_qgeom_package.
Requires Python standard library only. Preserves original relative paths.
"""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import argparse, csv, hashlib, json, tarfile

SOURCES = [
    ("B2_polar_qgeom_package_v1p8", "B2_stage2c_runs"),
    ("B2_polar_qgeom_package_v1p9", "B2_stage2c_addon_runs")
]

def read_csv(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--out",type=Path,default=Path("B2_release"))
    args=ap.parse_args()
    root=args.root.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if out==root: raise SystemExit("Output must be separate from source")
    paths=set(); pairs=defaultdict(dict); count=0; zeros=0
    def add(path):
        if not path.is_file() or path.is_symlink(): raise RuntimeError(f"Missing or symlink: {path}")
        paths.add(path.relative_to(root).as_posix())
    for folder,run_name in SOURCES:
        src=root/folder; run=src/run_name
        for n in ["config_stage1b.json","config_stage2c.json","config.json",
                  "b2_stage2c_transport.py","b2_polar_qgeom.py",
                  "b2_stage1b_symmetry_audit.py","b2_stage2b_sa3_manifold.py"]:
            add(src/n)
        for n in ["field_task_manifest.csv","zero_task_manifest.csv"]: add(run/n)
        tasks=read_csv(run/"field_task_manifest.csv")
        for task in tasks:
            i=int(task["task_index"]); sign=task["sign"]
            h5=run/"field"/f"task_{i:03d}"/"sa3_field.h5"
            result=run/"field"/f"task_{i:03d}"/"result.csv"
            add(h5); add(result)
            rows=read_csv(result)
            if len(rows)!=1 or rows[0]["status"]!="PASS": raise RuntimeError(f"Unexpected result: {result}")
            for k in ["task_index","R_Angstrom","field_au","field_magnitude_au"]:
                if Decimal(task[k])!=Decimal(rows[0][k]): raise RuntimeError(f"{k} mismatch: {result}")
            for k in ["axis","sign"]:
                if task[k]!=rows[0][k]: raise RuntimeError(f"{k} mismatch: {result}")
            k=(str(Decimal(task["R_Angstrom"]).normalize()),task["axis"],
               str(Decimal(task["field_magnitude_au"]).normalize()))
            if sign in pairs[k]: raise RuntimeError(f"Duplicate field key: {k} {sign}")
            pairs[k][sign]=h5.relative_to(root).as_posix(); count+=1
        zz=sorted((run/"zero").rglob("sa3_zero.h5"))
        if len(zz)!=len(read_csv(run/"zero_task_manifest.csv")): raise RuntimeError("Zero count mismatch")
        for z in zz: add(z)
        zeros+=len(zz)
    if count!=234 or zeros!=13 or len(pairs)!=117 or any(set(v)!={"plus","minus"} for v in pairs.values()):
        raise RuntimeError(f"Incomplete archive: fields={count} zeros={zeros} pairs={len(pairs)}")
    for n in ["audit_report.json","pair_mapping.csv","wavefunction_manifest.csv"]:
        p=root/"B2_pair_audit"/n
        if p.is_file(): add(p)
    for n in ["round4_report.json","zero_field_overlap_reconstruction.csv","plus_minus_overlap_summary.csv"]:
        p=root/"B2_pair_audit"/"round4"/n
        if p.is_file(): add(p)
    manifest=[(rel,sha256(root/rel)) for rel in sorted(paths)]
    (out/"SHA256SUMS").write_text("".join(f"{h}  {rel}\n" for rel,h in manifest))
    with (out/"file_inventory.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["path","bytes","sha256"])
        w.writerows((rel,(root/rel).stat().st_size,h) for rel,h in manifest)
    with (out/"pair_mapping.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["R_Angstrom","axis","field_magnitude_au","plus","minus"])
        for key,v in sorted(pairs.items()): w.writerow([*key,v["plus"],v["minus"]])
    (out/"README_ARCHIVE.md").write_text(
       "# B2 SA3-CASSCF wavefunction archive\n\n"
       "234 signed-field and 13 zero-field checkpoints, in 117 +/- pairs at 13 geometries.\n"
       "Source code, original configurations, manifests and per-task results are included.\n"
       "Files are archived without modifying their contents.\n"
       "Check integrity after extraction with: sha256sum -c SHA256SUMS\n"
       "Reconstruction using the original production frame_overlap is not an independent-algorithm verification.\n"
       "This archive is not a complete clean-environment SCF/CASSCF reproduction.\n")
    dest=out/"B2_SA3_wavefunctions_v1.tar.gz"
    with tarfile.open(dest,"w:gz") as tar:
        for rel in sorted(paths): tar.add(root/rel,arcname=rel,recursive=False)
        for n in ["SHA256SUMS","file_inventory.csv","pair_mapping.csv","README_ARCHIVE.md"]:
            tar.add(out/n,arcname=n,recursive=False)
    report={"field_checkpoints":count,"zero_checkpoints":zeros,"field_pairs":len(pairs),
      "archived_files":len(paths),"archive":str(dest),"archive_sha256":sha256(dest),
      "archive_bytes":dest.stat().st_size}
    (out/"release_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
