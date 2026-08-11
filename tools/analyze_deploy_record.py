"""Compare commanded vs measured joint angles from a deploy_linker --record CSV.

Hardware bring-up diagnostic: the live node logs, per tick, the measured joint
angles (``q*``), the integrated targets it commanded (``tgt*``), the raw 0..255
command (``cmd*``) and the 0..255 state read back (``state*``).  A joint whose
measured angle never approaches its commanded angle is either mis-mapped
(wrong slot/sign), saturated against a hardware limit the sim does not model, or
mechanically blocked by the object.

Usage:
    python tools/analyze_deploy_record.py live.csv
    python tools/analyze_deploy_record.py startup.csv --phase hold
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from screwdriver_rl.deploy import linker_sdk_map as sdkmap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--calib", default="linker_calib_deploy.json",
                   help="Calibration overlay the capture ran with (for the joint table).")
    p.add_argument("--phase", default=None,
                   help="Only rows whose 'phase' column matches (e.g. hold, approach).")
    p.add_argument("--last", type=int, default=20,
                   help="Average over the last N matching rows (settled pose). 0 = all.")
    args = p.parse_args()

    try:
        sdkmap.apply_calibration(args.calib)
    except Exception as exc:  # keep going with defaults; the table only labels joints
        print(f"[warn] could not load {args.calib}: {exc}")
    joints = sdkmap.active_joints()

    rows = [r for r in csv.DictReader(args.csv.open())]
    if args.phase:
        rows = [r for r in rows if r.get("phase") == args.phase]
    rows = [r for r in rows if (r.get("q0") or "").strip() != ""]
    if not rows:
        print("no rows with measured joint angles (q*) — was this a --dry-run capture?")
        return 1
    if args.last > 0:
        rows = rows[-args.last:]

    def col(name: str) -> list[float]:
        out = []
        for r in rows:
            v = (r.get(name) or "").strip()
            if v:
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        return out

    print(f"rows analysed: {len(rows)}"
          f"{f' (phase={args.phase})' if args.phase else ''}\n")
    print(f"{'joint':16s} {'slot':>4s} {'cmd_rad':>8s} {'meas_rad':>9s} {'err_rad':>8s} "
          f"{'err_deg':>8s} {'hw_lo':>7s} {'hw_hi':>7s}  note")
    for i, js in enumerate(joints):
        tgt, q = col(f"tgt{i}"), col(f"q{i}")
        if not tgt or not q:
            continue
        t, m = statistics.fmean(tgt), statistics.fmean(q)
        err = m - t
        lo, hi = js.lo, js.hi
        note = ""
        if abs(err) > 0.15:
            note = "LARGE DEVIATION"
        # Targets and readback are in the active semantic calibration domain.
        # Comparing them to the SDK's internal arc table is wrong for a measured
        # physical LUT, whose semantic joint coordinate is intentionally distinct.
        if t > max(lo, hi) or t < min(lo, hi):
            note = (note + " " if note else "") + "CMD OUTSIDE CALIBRATED RANGE"
        print(f"{js.name:16s} {js.slot:4d} {t:8.4f} {m:9.4f} {err:8.4f} "
              f"{err * 57.2958:8.1f} {lo:7.3f} {hi:7.3f}  {note}")

    print("\nerr = measured - commanded (both in training-URDF radians).")
    print("LARGE DEVIATION (>0.15 rad ~ 8.6deg) = mapping error, hardware limit, or blocked by object.")

    # Raw 0..255 view.  This is the only table-independent check: cmd/state are
    # the wire protocol itself, so a mismatch here is real actuator behaviour and
    # does not rely on the vendored radian calibration being correct.
    print(f"\n--- raw 0..255 (protocol units, no radian conversion) ---")
    print(f"{'slot':>4s} {'joint':16s} {'cmd':>6s} {'state':>6s} {'diff':>6s}  note")
    slot_names = {js.slot: js.name for js in joints}
    for slot in range(20):
        if slot in sdkmap._RESERVED_SLOTS:  # no actuator; always 0
            continue
        cmd, st = col(f"cmd{slot}"), col(f"state{slot}")
        if not cmd or not st:
            continue
        c, s = statistics.fmean(cmd), statistics.fmean(st)
        note = ""
        if abs(s - c) > 12:
            note = "NOT TRACKING"
        if c <= 1 or c >= 254:
            note = (note + " " if note else "") + "COMMAND AT RAIL"
        print(f"{slot:4d} {slot_names.get(slot, '-'):16s} {c:6.1f} {s:6.1f} {s - c:6.1f}  {note}")
    print("\nCOMMAND AT RAIL = we are asking for the extreme of the actuator's travel;")
    print("NOT TRACKING    = the actuator did not reach the commanded position (blocked or limited).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
