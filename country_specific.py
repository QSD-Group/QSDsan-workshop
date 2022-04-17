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

import os, numpy as np, pandas as pd, country_converter as coco
from matplotlib import pyplot as plt
from qsdsan.utils.colors import Guest

import systems, models
sysAB = sysA, sysB = systems.sysA, systems.sysB
get_cost = systems.get_daily_cap_cost
get_ghg = systems.get_daily_cap_ghg
run_mcda = systems.run_mcda
get_ppl = systems.get_ppl

uganda_dct = {} # cache Uganda results for comparison
for sys in sysAB:
    sys.simulate()
    AB = sys.ID[-1]
    uganda_dct[f'cost{AB}'] = get_cost(sys, print_msg=False)
    uganda_dct[f'ghg{AB}'] = get_ghg(sys, print_msg=False)

create_model = models.create_model
modelA = create_model('A', country_specific=True)
modelB = create_model('B', country_specific=True)
modelAB = modelA, modelB


# %%

# =============================================================================
# Update values and calculate results
# =============================================================================

# Import data
dir_path = os.path.dirname(__file__)
path = os.path.join(dir_path, 'data/contextual_parameters.xlsx')
file = pd.ExcelFile(path)

read_excel = lambda name: pd.read_excel(file, name) # name is sheet name

countries = read_excel('Countries')

def get_country_val(sheet, country, index_col='Code', val_col='Value'):
    df = read_excel(sheet) if isinstance(sheet, str) else sheet
    idx = df[df.loc[:, index_col]==country].index
    val = df.loc[idx, val_col]

    # When no country-specific data or no this country, use continental data
    if (val.isna().any() or val.size==0):
        region = countries[countries.Code==country].Region.item()
        idx = df[df.loc[:, index_col]==region].index
        val = df.loc[idx, val_col]

        # If not even continental data or no this country, use world data
        if (val.isna().any() or val.size==0):
            idx = df[df.loc[:, index_col]=='World'].index
            val = df.loc[idx, val_col]
    return val.values.item()

# Country-specific input values
country_val_dcts = {} # for cached results
def lookup_val(country):
    val_dct = country_val_dcts.get(country)
    if val_dct: return val_dct

    country = coco.convert(country)
    if country == 'not found': return

    val_dct = country_val_dcts[country] = {
        'Caloric intake': get_country_val('Caloric Intake', country),
        'Vegetable protein intake': get_country_val('Vegetal Protein', country),
        'Animal protein intake': get_country_val('Animal Protein', country),
        'N fertilizer price': get_country_val('N Fertilizer Price', country),
        'P fertilizer price': get_country_val('P Fertilizer Price', country),
        'K fertilizer price': get_country_val('K Fertilizer Price', country),
        'Food waste ratio': get_country_val('Food Waste', country),
        'Price level ratio': get_country_val('Price Level Ratio', country),
        'Income tax': get_country_val('Tax Rate', country)/100,
        }
    return val_dct

# Update the baseline values of the models based on the country
paramA_dct = {param.name: param for param in modelA.parameters}
paramB_dct = {param.name: param for param in modelB.parameters}
country_params = models.country_params
def get_results(country):
    val_dct = lookup_val(country)

    global result_dfs
    result_dfs = {}
    for model in modelAB:
        param_dct = paramA_dct if model.system.ID[-1]=='A' else paramB_dct
        for name, param in param_dct:
            param.baseline = val_dct[name]
        result_dfs[model.system.ID] = model.metrics_at_baseline()
    return result_dfs


# %%

# =============================================================================
# Prettify things for displaying
# =============================================================================

country_val_dfs = {} # for cached results
def get_val_df(country):
    val_df = country_val_dfs.get(country)
    if val_df: return val_df

    val_dct = lookup_val(country)
    if not val_dct:
        return '', f'No available information for country {country}'

    val_df = country_val_dfs[country] = pd.DataFrame({
        'Parameter': val_dct.keys(),
        'Value': val_dct.values(),
        'Unit': [p.units for p in paramA_dct.values() # A or B are the same
                 if p.name in country_params.values()]
        })
    return val_df

def plot(results_dct):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    midx = np.arange(len(uganda_dct))
    bar_width = 0.3
    ax.bar(midx-bar_width/1.8, uganda_dct.values(), label='Uganda',
           width=bar_width, color=Guest.green.RGBn)
    ax.bar(midx+bar_width/1.8, results_dct.values(), label='Simulated',
           width=bar_width, color=Guest.blue.RGBn)

    ax.set_xticklabels(('', 'A-cost', 'A-GHG', 'B-cost', 'B-GHG'))
    ax.set_ylabel('Cost [¢/cap/d] or GHG [g CO2-e/cap/d]', weight='bold')
    ax.legend(loc='best')

    return ax