"""Example script for checking the data and which scanner data was recorded with.

This script has to be run on a computer with a MaxFilter license.
"""

from subprocess import PIPE, run
import numpy as np


raw_dir =  "/home/mtrubshaw/Documents/ALS_dyn/data/raw/"

files = ["sXX_resting.fif"]

ids = []
dates = []
scanners = []

print()
for file in files:
    path = raw_dir + file
    cmd = f"/neuro/bin/util/show_fiff -v -t 100:206 {path}"
    result = run(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout = result.stdout.splitlines()
    date = str(stdout[0]).split("    ")[-1][2:-1]
    scanner = str(stdout[1]).split("    ")[-1][2:-1]

    print(file)
    print(f"date:    {date}")
    print(f"scanner: {scanner}")
    print()
    ids.append(file)
    dates.append(date)
    scanners.append(scanner)
    
    output = [ids,dates,scanners]
    
print("See: https://github.com/OHBA-analysis/osl/tree/examples/examples/oxford/maxfilter for what to use for the --scanner argument.")
