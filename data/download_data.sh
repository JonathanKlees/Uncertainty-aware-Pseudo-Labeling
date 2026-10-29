#!/bin/bash

DOI="doi:10.7910/DVN/L7QCUG"

OUTDIR="soft_datasets"
# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"


curl -s \
"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=${DOI}" |
OUTDIR="$OUTDIR" python3 -c '
import json, subprocess, sys, os

outdir = os.environ["OUTDIR"]
data = json.load(sys.stdin)
for f in data["data"]["latestVersion"]["files"]:
    fid = f["dataFile"]["id"]
    name = f["dataFile"]["filename"]
    print(f"Downloading {name}...")
    subprocess.run([
        "curl", "-L",
        f"https://dataverse.harvard.edu/api/access/datafile/{fid}",
        "-o", f"{outdir}/{name}"
    ], check=True)
'