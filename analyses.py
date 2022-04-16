#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
EXPOsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Yalin Li <zoe.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/EXPOsan/blob/main/LICENSE.txt
for license details.
'''


# %%

import systems, models
from qsdsan.utils import copy_samples

sysAB = sysA, sysB = systems.sysA, systems.sysB
# get_cost = systems.get_daily_cap_cost
# get_GHG = systems.get_daily_cap_GHG
# run_MCDA = systems.run_MCDA
# for sys in (sysA, sysB):
#     get_cost(sys)
#     get_GHG(sys)
# run_MCDA()

modelAB = modelA, modelB = models.modelA, models.modelB
run_uncertainty = models.run_uncertainty

def run_uncertainties(N=100, seed=None, rule='L',
                      percentiles=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1),):
    for model in modelAB:
        run_uncertainty(model, N, seed, rule, percentiles,
                        only_load_samples=True, only_organize_results=False)
    copy_samples(modelA, modelB)

    for model in modelAB:
        run_uncertainty(model, N, seed, rule, percentiles,
                        only_load_samples=False, only_organize_results=False)

    return modelA, modelB

run_uncertainties()
