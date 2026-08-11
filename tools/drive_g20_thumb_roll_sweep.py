#!/usr/bin/env python3
"""Drive one gated G20 thumb-CMC-roll sweep, alternating CAN and D435."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
CAMERA_PYTHON="/home/user/miniconda3/bin/python3"
SDK_PYTHON="/home/user/miniconda3/envs/env_isaaclab/bin/python"
SLOT=5
MAX_BLOCK_RANGE_DEG=.60
MAX_PALM_DRIFT_DEG=.80
MAX_STEP_ANGLE_DEG=8.0

def load(p): return json.loads(Path(p).read_text())
def median(p,key): return load(p)["angle_summary_deg"][key]["median"]
def run(cmd,label):
    print(f"\n=== {label}\n$ {' '.join(map(str,cmd))}",flush=True)
    completed=subprocess.run([str(v) for v in cmd])
    if completed.returncode: raise SystemExit(f"STOP: {label} exited {completed.returncode}")

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--session",type=Path,required=True); p.add_argument("--reference",type=Path,nargs=2,required=True); p.add_argument("--start-raw",type=int,required=True); p.add_argument("--direction",choices=("up","down"),required=True); p.add_argument("--stop-at",type=int,required=True); p.add_argument("--previous-deg",type=float,required=True); p.add_argument("--tag",required=True); p.add_argument("--calib",type=Path,required=True); p.add_argument("--step-raw",type=int,default=14); p.add_argument("--min-step-raw",type=int,default=8); a=p.parse_args()
    key="thumb_cmc_roll_deg_3d"; palm="palm_heading_deg_2d"
    zero=sum(median(x,key) for x in a.reference)/2; palm_zero=sum(median(x,palm) for x in a.reference)/2
    current=a.start_raw; previous=a.previous_deg; results=[]
    while True:
        remaining=(a.stop_at-current) if a.direction=="up" else (current-a.stop_at)
        if remaining<=0: break
        step=min(a.step_raw,remaining)
        if step<a.min_step_raw:
            print(f"[driver] remaining {remaining} raw below {a.min_step_raw} deadband floor; clean stop",flush=True); break
        target=current+step if a.direction=="up" else current-step
        point=a.session/f"{a.tag}_cmd{target:03d}_from{current:03d}"; point.mkdir(parents=True,exist_ok=True)
        motion=point/f"motion_raw{current:03d}_to_raw{target:03d}.json"
        run([SDK_PYTHON,"-m","tools.run_g20_thumb_cmc_roll_candidate_step","--sdk-root","/home/user/linkerhand-ros-sdk","--calib",a.calib,"--can","can0","--expected-start-raw",current,"--target-raw",target,"--yaw-hold-raw",125,"--pitch-hold-raw",247,"--mcp-hold-raw",254,"--speed",5,"--samples",20,"--hz",5,"--settle-tolerance-raw",4,"--out",motion,"--execute"],f"motion {current}->{target}")
        prefix=point/"camera_fixed_exp_180f"
        run([CAMERA_PYTHON,"-m","tools.measure_g20_thumb_roll","--frames",180,"--out-prefix",prefix],f"camera at command {target}")
        summary=load(f"{prefix}_summary.json"); rb=load(motion)["result"]["settled_state20_median"][SLOT]
        blocks=summary["block_medians_deg"][key]; block_range=max(blocks)-min(blocks) if len(blocks)>1 else float("inf")
        absolute=summary["angle_summary_deg"][key]["median"]; physical=absolute-zero
        palm_drift=summary["angle_summary_deg"][palm]["median"]-palm_zero
        delta=physical-previous
        failures=summary["capture"]["failure_fraction"]
        print(f"[gate] rb={rb} roll={physical:+.4f} delta={delta:+.4f} block={block_range:.4f} palm={palm_drift:+.4f} fail={failures:.3f}",flush=True)
        if block_range>MAX_BLOCK_RANGE_DEG: raise SystemExit(f"STOP: block range {block_range:.4f} deg")
        if abs(palm_drift)>MAX_PALM_DRIFT_DEG: raise SystemExit(f"STOP: palm drift {palm_drift:+.4f} deg")
        if failures>.10: raise SystemExit(f"STOP: failure fraction {failures:.3f}")
        if abs(delta)>MAX_STEP_ANGLE_DEG: raise SystemExit(f"STOP: implausible step angle {delta:+.4f} deg")
        if a.direction=="up" and delta<=0: raise SystemExit(f"STOP: roll reversed {delta:+.4f} deg on up sweep")
        if a.direction=="down" and delta>=0: raise SystemExit(f"STOP: roll reversed {delta:+.4f} deg on down sweep")
        if rb==current: raise SystemExit("STOP: no readback movement; endpoint requires review")
        results.append({"command_raw":target,"stable_readback_raw":rb,"roll_deg_from_zero":physical,"delta_deg":delta,"block_range_deg":block_range,"palm_drift_deg":palm_drift,"failure_fraction":failures,"motion":str(motion),"camera_summary":f"{prefix}_summary.json"})
        (a.session/f"{a.tag}_progress.json").write_text(json.dumps(results,indent=2)+"\n")
        current=rb; previous=physical
    print(json.dumps({"final_readback_raw":current,"final_roll_deg":previous,"points":len(results)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
