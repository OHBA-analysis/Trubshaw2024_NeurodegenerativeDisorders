"""Fit a GLM and perform statistical significance testing.

"""

import os
import numpy as np
import pandas as pd
from scipy import stats

import glmtools as glm

from osl_dynamics.analysis import power

os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('plots/compiled_plots', exist_ok=True)

# Load target data
f = np.load("../../data/static/f.npy")
psd = np.load("../../data/static/p.npy")

power_ = []
for band in [[1, 4], [4, 8], [8, 13], [13, 30], [30, 48], [52, 80]]:
    power_.append(power.variance_from_spectra(f, psd, frequency_range=band))
power_ = np.swapaxes(power_, 0, 1)

# Load regressor data
demographics = pd.read_csv("../../demographics/demographics_als_dyn.csv")

category_list = demographics["Group"].values
category_list[category_list == "HC"] = 1
category_list[category_list == "ALS"] = 2
category_list[category_list == "rALS"] = 2
category_list[category_list == "AC9"] = 2
category_list[category_list == "ADCT"] = 2
category_list[category_list == "AFIG"] = 2
category_list[category_list == "rAFIG"] = 2
category_list[category_list == "ASOD"] = 2
category_list[category_list == "PC9"] = 3
category_list[category_list == "PSOD"] = 4

uniques = np.unique(category_list)

age = demographics["Age"].values

gender = []
for g in demographics["Gender"].values:
    if g == "Male":
        gender.append(0)
    else:
        gender.append(1)
gender = np.array(gender)

missing_struc = demographics["Missing_struc"].values

# Create GLM dataset
data = glm.data.TrialGLMData(
    data=power_,
    category_list=category_list,
    age=age,
    gender=gender,
    dim_labels=["Subjects", "Frequencies", "Parcels"],
    missing_struc=missing_struc,
)

# Design matrix
DC = glm.design.DesignConfig()
DC.add_regressor(name="HC", rtype="Categorical", codes=1)
DC.add_regressor(name="ALS", rtype="Categorical", codes=2)
DC.add_regressor(name="PC9", rtype="Categorical", codes=3)
DC.add_regressor(name="PSOD", rtype="Categorical", codes=4)
DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
DC.add_regressor(name="Gender", rtype="Parametric", datainfo="gender", preproc="z")
DC.add_regressor(name="Missing Structural", rtype="Parametric", datainfo="missing_struc", preproc="z")


DC.add_contrast(name="ALS-HC", values=[-1, 1, 0, 0, 0, 0, 0])
DC.add_contrast(name="PC9-HC", values=[-1, 0, 1, 0, 0, 0, 0])
DC.add_contrast(name="PSOD-HC", values=[-1, 0, 0, 1, 0, 0, 0])
DC.add_contrast(name="PC9-PSOD", values=[0, 0, 1, -1, 0, 0, 0])

design = DC.design_from_datainfo(data.info)
design.plot_summary(savepath="plots/glm_design.png", show=False)
design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

# Fit the GLM
model = glm.fit.OLSModel(design, data)

def do_stats(contrast_idx, metric="tstats"):
    # Max-stat permutations
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=1000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1,2),  # pool over frequencies and channels
        nprocesses=16,
    )
    null_dist = perm.nulls

    # Calculate p-values
    if metric == "tstats":
        tstats = abs(model.tstats[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        copes = abs(model.copes[contrast_idx])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    return pvalues

pvalues_bin=[]
tstats_bin = []
contrast_bin = []
regions_bin = []
for i in range(model.copes.shape[0]):
    cope = model.copes[i]
    pvalues = do_stats(contrast_idx=i)
    tstats = model.tstats[i]
    np.save(f"data/contrast_{i}.npy", cope)
    np.save(f"data/contrast_{i}_pvalues.npy", pvalues)
    pvalues_bin.append(pvalues)
    tstats_bin.append(tstats)
    contrast_bin.append(np.zeros(model.copes.shape[1:])+i)
    regions_bin.append(np.full((model.copes.shape[1:]),list(range(52))))
pvalues_bin = np.array(pvalues_bin)
tstats_bin = np.array(tstats_bin) 
contrast_bin = np.array(contrast_bin)
regions_bin = np.array(regions_bin)
dofs = np.full((model.copes.shape),model.dof_model)
    
contrast_names = model.contrast_names

sig_ps_m = pvalues_bin<0.05
sig_ps = pvalues_bin[sig_ps_m]
sig_ts = tstats_bin[sig_ps_m]
sig_cont = contrast_bin[sig_ps_m]
sig_regs = regions_bin[sig_ps_m]
sig_dofs = dofs[sig_ps_m]

results = np.vstack((sig_cont,sig_regs,sig_dofs,sig_ts,sig_ps)).T
np.savetxt('data/results.csv',results,delimiter=',',fmt='%.3f')
with open('data/results_contrast_names.txt', 'w') as file:
    for item in contrast_names:
        file.write(f"{item}\n")
        
for unique in uniques:
    count = np.count_nonzero(category_list==unique)
    print('Group',unique,' - ',count)
