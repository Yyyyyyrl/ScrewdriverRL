#!/usr/bin/env python3
"""Measure thumb CMC roll as a palm-relative 3D orientation from the D435.

One broad fixed palm marker supplies a 3D basis. Two depth-separated thumb-link
markers supply the moving vector. Physical roll is the signed angle change from
a separately captured reference in the same camera pose.
"""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
from typing import Any,Sequence
import cv2
import numpy as np
from tools.measure_g20_tape_angles import _parse_roi,_summary

ANGLE_KEYS=("thumb_cmc_roll_deg_3d","palm_relative_roll_deg_3d","thumb_vector_elevation_deg_3d","thumb_marker_separation_m","palm_plane_rms_m","palm_heading_deg_2d","thumb_pair_heading_deg_2d")

def _blue(color,args):
    hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,np.asarray(args.hsv_lower,np.uint8),np.asarray(args.hsv_upper,np.uint8))>0

def _white(color,args):
    hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    mask=(hsv[:,:,1]<=80)&(hsv[:,:,2]>=100)
    return cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))>0

def _components(mask,depth_m,band,roi,min_area,expected,label):
    region=np.zeros(mask.shape,bool); x0,y0,x1,y1=roi; region[y0:y1,x0:x1]=True
    selected=mask&region&(depth_m>=band[0])&(depth_m<=band[1])
    count,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8),8)
    keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=min_area]
    if len(keep)!=expected:
        raise ValueError(f"expected exactly {expected} {label} components in {band[0]:.3f}..{band[1]:.3f} m, found {len(keep)}")
    out=[]
    for i in keep:
        ys,xs=np.where(labels==i); out.append(np.column_stack([xs.astype(float),ys.astype(float)]))
    return out

def _points3(pixels,depth_m,intr):
    cols=pixels[:,0].astype(int); rows=pixels[:,1].astype(int); z=depth_m[rows,cols]
    valid=np.isfinite(z)&(z>.05)&(z<3); u=pixels[valid,0]; v=pixels[valid,1]; z=z[valid]
    return np.column_stack([(u-intr["ppx"])*z/intr["fx"],(v-intr["ppy"])*z/intr["fy"],z])

def _heading(pixels):
    q=pixels-pixels.mean(0); values,vectors=np.linalg.eigh(q.T@q); axis=vectors[:,int(np.argmax(values))]
    return (math.degrees(math.atan2(axis[1],axis[0]))+90)%180-90

def _detail(pixels,points3,depth_m):
    rows=pixels[:,1].astype(int); cols=pixels[:,0].astype(int)
    return {"area_px":int(len(pixels)),"centroid_uv":[float(v) for v in pixels.mean(0)],"centroid_xyz_m":[float(v) for v in np.median(points3,0)],"depth_median_m":float(np.median(depth_m[rows,cols])),"heading_deg_2d":_heading(pixels)}

def measure_roll_frame(color,depth,depth_scale,intr,args):
    depth_m=depth.astype(float)*depth_scale; blue=_blue(color,args); white=_white(color,args)
    palm_pixels=_components(white,depth_m,args.palm_depth_band,args.palm_roi,args.palm_min_area,1,"palm")[0]
    thumb_pixels=_components(blue,depth_m,args.thumb_depth_band,args.thumb_roi,args.thumb_min_area,2,"thumb")
    thumb_pixels.sort(key=lambda p:float(p[:,1].mean()))
    palm3=_points3(palm_pixels,depth_m,intr); thumb3=[_points3(p,depth_m,intr) for p in thumb_pixels]
    if len(palm3)<1000 or min(map(len,thumb3))<300: raise ValueError("too few valid 3D marker pixels")
    palm_center=np.median(palm3,0); centred=palm3-palm_center; _,_,vt=np.linalg.svd(centred,full_matrices=False)
    long_axis=vt[0]; normal=vt[2]
    if long_axis[0]<0: long_axis=-long_axis
    if normal[2]>0: normal=-normal
    short_axis=np.cross(long_axis,normal); short_axis/=np.linalg.norm(short_axis)
    if short_axis[1]<0: short_axis=-short_axis; normal=-normal
    centers=[np.median(p,0) for p in thumb3]; thumb_vector=centers[1]-centers[0]; separation=float(np.linalg.norm(thumb_vector))
    if not args.min_thumb_separation_m<=separation<=args.max_thumb_separation_m:
        raise ValueError(f"thumb marker separation {separation:.4f} m outside gate")
    thumb_vector/=separation
    palm_relative=math.degrees(math.atan2(float(np.dot(thumb_vector,short_axis)),float(np.dot(thumb_vector,long_axis))))
    roll=math.degrees(math.atan2(float(thumb_vector[1]),float(thumb_vector[0])))
    elevation=math.degrees(math.atan2(float(thumb_vector[2]),float(np.hypot(thumb_vector[0],thumb_vector[1]))))
    plane_rms=float(np.sqrt(np.mean((centred@normal)**2)))
    if plane_rms>args.max_palm_plane_rms_m: raise ValueError(f"palm plane RMS {plane_rms:.5f} m above gate")
    details={"palm":_detail(palm_pixels,palm3,depth_m),"thumb_upper":_detail(thumb_pixels[0],thumb3[0],depth_m),"thumb_lower":_detail(thumb_pixels[1],thumb3[1],depth_m),"palm_long_axis_xyz":[float(v) for v in long_axis],"palm_short_axis_xyz":[float(v) for v in short_axis],"palm_normal_xyz":[float(v) for v in normal],"thumb_pair_vector_xyz":[float(v) for v in thumb_vector]}
    values={"thumb_cmc_roll_deg_3d":roll,"palm_relative_roll_deg_3d":palm_relative,"thumb_vector_elevation_deg_3d":elevation,"thumb_marker_separation_m":separation,"palm_plane_rms_m":plane_rms,"palm_heading_deg_2d":details["palm"]["heading_deg_2d"],"thumb_pair_heading_deg_2d":_heading(np.asarray([details["thumb_upper"]["centroid_uv"],details["thumb_lower"]["centroid_uv"]]))}
    return values,details

def _blocks(rows,key,size):
    values=[r[key] for r in rows]; return [float(np.median(values[i:i+size])) for i in range(0,len(values)-size+1,size)]

def capture(args):
    import pyrealsense2 as rs
    pipeline=rs.pipeline(); config=rs.config(); config.enable_device(args.serial)
    config.enable_stream(rs.stream.color,args.width,args.height,rs.format.bgr8,args.fps); config.enable_stream(rs.stream.depth,args.width,args.height,rs.format.z16,args.fps)
    profile=pipeline.start(config)
    try:
        sensor=next(s for s in profile.get_device().query_sensors() if "RGB" in s.get_info(rs.camera_info.name))
        sensor.set_option(rs.option.enable_auto_exposure,0); sensor.set_option(rs.option.exposure,args.rgb_exposure); sensor.set_option(rs.option.gain,args.rgb_gain)
        sensor.set_option(rs.option.enable_auto_white_balance,0); sensor.set_option(rs.option.white_balance,args.rgb_white_balance)
        controls={"exposure":sensor.get_option(rs.option.exposure),"gain":sensor.get_option(rs.option.gain),"white_balance":sensor.get_option(rs.option.white_balance),"enable_auto_exposure":sensor.get_option(rs.option.enable_auto_exposure),"enable_auto_white_balance":sensor.get_option(rs.option.enable_auto_white_balance)}
        ci=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(); intr={k:float(getattr(ci,k)) for k in ("fx","fy","ppx","ppy")}
        align=rs.align(rs.stream.color); depth_scale=profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(args.warmup_frames): pipeline.wait_for_frames(5000)
        started=time.time(); rows=[]; failures=[]; last={}
        for index in range(args.frames):
            frames=align.process(pipeline.wait_for_frames(5000)); color=np.asanyarray(frames.get_color_frame().get_data()); depth=np.asanyarray(frames.get_depth_frame().get_data())
            try: values,details=measure_roll_frame(color,depth,depth_scale,intr,args)
            except ValueError as exc: failures.append({"frame":index,"reason":str(exc)}); continue
            rows.append({"frame":index,**values}); last={"color":color,"depth":depth,"values":values,"details":details}
    finally: pipeline.stop()
    if not rows: raise RuntimeError(f"no usable frames; first failures={failures[:5]}")
    prefix=args.out_prefix; prefix.parent.mkdir(parents=True,exist_ok=True)
    with Path(f"{prefix}_samples.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["frame",*ANGLE_KEYS]); writer.writeheader(); writer.writerows(rows)
    cv2.imwrite(f"{prefix}_color.png",last["color"]); np.save(f"{prefix}_depth_raw.npy",last["depth"])
    canvas=last["color"].copy()
    for roi,shade in ((args.thumb_roi,(0,255,255)),(args.palm_roi,(255,200,0))): cv2.rectangle(canvas,roi[:2],roi[2:],shade,2)
    cv2.putText(canvas,f"roll3d {last['values']['thumb_cmc_roll_deg_3d']:+.3f} deg",(args.thumb_roi[0],args.thumb_roi[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2,cv2.LINE_AA); cv2.imwrite(f"{prefix}_annotated.png",canvas)
    summary={"schema_version":1,"camera_only":True,"hand_motion_sent":False,"camera":{"serial":args.serial,"intrinsics":intr,"depth_scale_m_per_unit":depth_scale,"rgb_controls":controls},"detector":{"mode":"thumb_3d_vector_azimuth_with_white_palm_drift_reference","primary_value":"signed thumb 3D vector azimuth; white palm marker is the rigid drift witness","thumb_roi_xyxy":list(args.thumb_roi),"palm_roi_xyxy":list(args.palm_roi),"thumb_depth_band_m":list(args.thumb_depth_band),"palm_depth_band_m":list(args.palm_depth_band)},"capture":{"requested_frames":args.frames,"usable_frames":len(rows),"failure_count":len(failures),"failure_fraction":len(failures)/args.frames,"started_wall_time_s":started,"ended_wall_time_s":time.time()},"angle_summary_deg":{k:_summary([float(r[k]) for r in rows]) for k in ANGLE_KEYS},"block_medians_deg":{k:_blocks(rows,k,args.stability_block_frames) for k in ANGLE_KEYS},"last_markers":last["details"],"failures":failures}
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); return summary

def parse_args(argv:Sequence[str]|None=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--serial",default="143322073091"); p.add_argument("--width",type=int,default=1280); p.add_argument("--height",type=int,default=720); p.add_argument("--fps",type=int,default=30); p.add_argument("--warmup-frames",type=int,default=60); p.add_argument("--frames",type=int,default=180); p.add_argument("--rgb-exposure",type=float,default=166); p.add_argument("--rgb-gain",type=float,default=32); p.add_argument("--rgb-white-balance",type=float,default=4600)
    p.add_argument("--thumb-roi",type=_parse_roi,default=(450,380,750,630)); p.add_argument("--palm-roi",type=_parse_roi,default=(670,230,850,315)); p.add_argument("--thumb-depth-band",type=float,nargs=2,default=(.43,.50)); p.add_argument("--palm-depth-band",type=float,nargs=2,default=(.275,.310)); p.add_argument("--thumb-min-area",type=int,default=600); p.add_argument("--palm-min-area",type=int,default=5000); p.add_argument("--hsv-lower",type=int,nargs=3,default=(75,40,20)); p.add_argument("--hsv-upper",type=int,nargs=3,default=(165,255,255)); p.add_argument("--min-thumb-separation-m",type=float,default=.025); p.add_argument("--max-thumb-separation-m",type=float,default=.080); p.add_argument("--max-palm-plane-rms-m",type=float,default=.008); p.add_argument("--stability-block-frames",type=int,default=60); p.add_argument("--max-failure-fraction",type=float,default=.1); p.add_argument("--out-prefix",type=Path,required=True); return p.parse_args(argv)

def main(argv=None):
    args=parse_args(argv); summary=capture(args); print(json.dumps({"usable_frames":summary["capture"]["usable_frames"],"failure_fraction":summary["capture"]["failure_fraction"],"angle_summary_deg":summary["angle_summary_deg"],"block_medians_deg":summary["block_medians_deg"]},indent=2)); return int(summary["capture"]["failure_fraction"]>args.max_failure_fraction)
if __name__=="__main__": raise SystemExit(main())
