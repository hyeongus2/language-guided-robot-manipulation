"""Recorded arm geometry and pulse conversions; angles are radians."""
import json
from pathlib import Path
import numpy as np

def get_urdf():
    data=json.loads((Path(__file__).parents[1]/"geometry.json").read_text())
    return {k:np.asarray(v,dtype=float) for k,v in data.items()}

def pulse2angle(pulse):
    return np.deg2rad((np.asarray(pulse,dtype=float)-500)*(-240/1000))

def angle2pulse(angle):
    return np.rint(500-np.rad2deg(np.asarray(angle,dtype=float))*1000/240).astype(int)
