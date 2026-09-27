"""ROS-independent RGB/depth validation helpers."""
import numpy as np

def depth_meters(data,height,width,step,encoding,bigendian=False):
    if encoding not in ("16UC1","32FC1"):
        raise ValueError("Depth encoding must be 16UC1 or 32FC1")
    dtype=np.dtype((">" if bigendian else "<")+("u2" if encoding=="16UC1" else "f4"))
    if height<=0 or width<=0 or step<width*dtype.itemsize or step%dtype.itemsize or len(data)!=height*step:
        raise ValueError("Invalid image dimensions/stride/buffer size")
    pixels=np.frombuffer(data,dtype=dtype).reshape(height,step//dtype.itemsize)[:,:width]
    return pixels.astype(float)*(0.001 if encoding=="16UC1" else 1.0)

def stamp_seconds(stamp):
    return stamp.sec+stamp.nanosec*1e-9

def validate_snapshot(rgb,depth,info,rgb_header,now,max_skew=0.06,max_age=0.5):
    if rgb.shape[:2]!=(depth.height,depth.width) or (info.height,info.width)!=(depth.height,depth.width):
        raise ValueError("Aligned RGB/depth/CameraInfo dimensions must match")
    frames={rgb_header.frame_id,depth.header.frame_id,info.header.frame_id}
    if len(frames)!=1 or not next(iter(frames)):
        raise ValueError("Aligned image and calibration frames must match")
    times=[stamp_seconds(rgb_header.stamp),stamp_seconds(depth.header.stamp)]
    if abs(times[0]-times[1])>max_skew or any(t>now+max_skew or now-t>max_age for t in times):
        raise ValueError("Stale or unsynchronized image pair")
    k=np.asarray(info.k,dtype=float)
    if k.size!=9 or not np.all(np.isfinite(k)) or k[0]<=0 or k[4]<=0:
        raise ValueError("Invalid camera intrinsics")
    if any(abs(float(x))>1e-8 for x in info.d):
        raise ValueError("Use rectified images and matching rectified calibration")
    return k[0],k[4],k[2],k[5]
