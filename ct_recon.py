import argparse, glob, os, re, sys, time
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

def parse_xtekct(path):
    cfg = {}
    with open(path, "r", errors="ignore") as f: # opens file in read mode stores it in varible f
        for line in f:
            line = line.strip() # removes spaces in the line
            if "=" in line and not line.startwith("["): 
                k, v = line.split("=",1) # splits at the equal to sign and store in k and v e.g srcobject = 500  now k = srcobject and v = 500
                try:
                    cfg[k.strip()] = float(v) 
                expect ValueError:
                    cfg[k.strip()] = v.strip()
    return cfg

def read_angles(path):
    ang = []
    with open(path, "r",errors="ignore") as f:
        next(f) # next(f) assumes there is header in the angles table/file
        for line in f:
            m = re.match(r"\s*\d+\s*:\s*([-+]?\d+\.?\d*)", line)
            #\s -  matches zero or more spaces(handles any indentation at the start of the line)
            #\d+ Matches one or more digits (this is the frame or projection number, like 1)
            #\s* matches zero or more spaces befroe the colon
            #: matches the actual colon
            #\s* ,amtches zero or more spaces after the colon
            # ([-+]?\d+.?\d*) Cature group defined by the parratheses
            # [-+] optionally matches a single postiiiive or negative sign
            # \d+ matches one Optionanlly matches a literal decimal point
            # \d* matches zero or more digts after the decimal point
            if m:
                ang.append(float(m.group(1)))
                # extracts the text inside e.g ("-000.010")
    return np.array(ang)

def find_projection_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")) +
                    glob.glob(os.path.join(folder, "*.tiff")))
    #globbing finding a list of files or folders that match a specifc pattern using wilcard characters
    files = [f for f in files 
             if not re.search(r"(flat|dark|white|black|ref))", os.path.basename(f), re.I)]
    return files




def bin2d(a, b): 
    if b == 1:
        return a
    h, w = (a.shape[0]// b) * b, (a.shape[1]//b) * b 
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))

def load_projections(
        files, # projection image
        binning=1, # how much to reduce each images dimensions
        step=1, # how many positions to advance through the file list each time 
        flat=None,# flat-field calibration image
        dark=None,#
        white_level=60000.0 # reference intensity used when flat-field calibration is not supplied
):
    idx = np.arange(0, len(files), step)
    flat_bined = (bin2d(tifffile.imread(flat).astype(np.float32), binning) if flat else None ) 
    dark_bined = (bin2d(tifffile.imread(flat).astype(np.float32), binning) if flat else None ) 
    out = None # to store output array but first the function needs tp read projection image to determine its dimensions, especailly after binning
    for n, i in enumerate(idx):
        im = bin2d(tifffile.imread(files[i].astype(np.float32)), binning)  # converts pixel value to floating-point numbers
        if flat_bined is not None:
            d = dark_bined if dark_bined is not None else 0.0
            im = (im -d)/ np.maximum(flat_bined - d, 1.0)
        else:
            im = im / white_level
        im = np.clip(im, 1e-4, None)
        if out is None:
            out = np.empty( (len(idx),) + im.shape, np.float32)# the comma in len(idx), makes this one-element tuple in python
        out[n] = -np.log(im)
        if n % 100 == 0:
            print(f"loaded {n + 1}/{len(idx)}", end="\r") # end="\r"  returns the cursor to the beginning of the same terminal line
    print()
    return out, idx


class Geometry:
    def __init__(self, sod, sdd, du, dv, nu, nv, cor_px=0.0):
        self.sod, self.sdd = sod, sdd # source to object & soucre to detector
        self.du, self.dv = du, dv # detector pixel spacing horizontally(du) and vertically(dv)
        self.nu, self.nv = nu, nv # Number of detector pixels
        self.cor_px = cor_px # centre of rotation
        self.mag = sdd / sod # geometric magnification

        # Virtual detector 
        self.du_v, self.dv_v = du / self.mag, self.mag  
        self.voxel = self.du_v  #natural voxel size(mm)

    @property
    def uc(self): #  A horizontal detector-coordinate 
        return (self.nu - 1) / 2.0 + self.cor_px

    def __subpixel_peak():
        
