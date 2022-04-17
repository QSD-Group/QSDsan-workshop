#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <zoe.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/EXPOsan/blob/main/LICENSE.txt
for license details.
'''


# %%

# =============================================================================
# Systems
# =============================================================================

from systems import (
    sysA, sysB,
    get_daily_cap_cost as get_cost,
    get_daily_cap_ghg as get_ghg,
    run_mcda,
    )

for sys in (sysA, sysB):
    get_cost(sys)
    get_ghg(sys)
run_mcda()


# %%

# =============================================================================
# Models
# =============================================================================

from models import create_model, run_uncertainty
from qsdsan import stats as s
from qsdsan.utils import copy_samples

modelA = create_model('A')
modelB = create_model('B')
modelAB = modelA, modelB

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

def get_param_metric(name, model, kind='parameter'):
    kind = 'parameters' if kind.lower() in ('p', 'param', 'parameter', 'parameters') \
        else 'metrics'
    for obj in getattr(model, kind):
        if obj.name == name: break
    return obj

def plot_spearman(model, metric='Net cost', top=None):
    metric = get_param_metric(metric, model, 'metric')
    df = s.get_correlations(modelA, input_y=metric, kind='Spearman')[0]
    fig, ax = s.plot_correlations(df, top=10)


plot_spearman()


# %%

# =============================================================================
# Country-specific
# =============================================================================

from country_specific import get_val_df, get_results, plot

vals = []
results = []
figs = []
for country in ('Uganda', 'USA', 'India', 'China', 'South Africa', 'Germany'):
    vals.append(get_val_df(country))
    results = get_results(country)
    figs.append(plot(results))
