#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
QSDsan: Quantitative Sustainable Design for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <zoe.yalin.li@gmail.com>

Part of this module is based on the EXPOsan repository:
https://github.com/QSD-Group/EXPOsan

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt
for license details.
'''

# %%

# Filter out warnings related to solid content
import warnings
warnings.filterwarnings('ignore', message='Solid content')

import numpy as np, pandas as pd, qsdsan as qs
from sklearn.linear_model import LinearRegression as LR
from matplotlib import pyplot as plt
from qsdsan import (
    Flowsheet, main_flowsheet,
    WasteStream,
    sanunits as su,
    ImpactIndicator, ImpactItem, StreamImpactItem,
    System, SimpleTEA, LCA,
    )
from _cmps import create_bwaise_components
from _mcda import MCDA

# =============================================================================
# Unit parameters
# =============================================================================

_ImpactItem_LOADED = False
currency = qs.currency = 'USD'
qs.CEPCI = qs.CEPCI_by_year[2018]
cmps = create_bwaise_components(True)

household_size = 4
household_per_toilet = 4
get_toilet_user = lambda: household_size * household_per_toilet

# Number of people served by the existing plant (sysA and sysC)
# ppl_exist_sewer = 4e4
# ppl_exist_sludge = 416667
# get_ppl = lambda: ppl_exist_sewer+ppl_exist_sludge
ppl = 20000
get_ppl = lambda: ppl

exchange_rate = 3700 # UGX per USD
get_exchange_rate = lambda: exchange_rate

discount_rate = 0.05

# Time take for full degradation, [yr]
tau_deg = 2
# Log reduction at full degradation
log_deg = 3
# Get reduction rate constant k for COD and N, use a function so that k can be
# changed during uncertainty analysis
def get_decay_k(tau_deg=2, log_deg=3):
    k = (-1/tau_deg)*np.log(10**-log_deg)
    return k

max_CH4_emission = 0.25

# Model for tanker truck cost based on capacity (m3)
# price = a*capacity**b -> ln(price) = ln(a) + bln(capacity)
UGX_price_dct = np.array((8e4, 12e4, 20e4, 25e4))
capacities = np.array((3, 4.5, 8, 15))
emptying_fee = 0.15 # additional emptying fee, fraction of base cost
def get_tanker_truck_fee(capacity):
    price_dct = UGX_price_dct*(1+emptying_fee)/get_exchange_rate()
    ln_p = np.log(price_dct)
    ln_cap = np.log(capacities)
    model = LR().fit(ln_cap.reshape(-1,1), ln_p.reshape(-1,1))
    predicted = model.predict(np.array((np.log(capacity))).reshape(1, -1)).item()
    cost = np.exp(predicted)
    return cost

# Flow rates for treatment plants
sewer_flow = 2750 # m3/d
sludge_flow_exist = 500 # m3/d
sludge_flow_alt = 60 # m3/d
get_sludge_flow = lambda kind: \
    sludge_flow_exist if kind.lower() in ('exist', 'sysa', 'sysc', 'a', 'c') else sludge_flow_alt

# Nutrient loss during application
app_loss = dict.fromkeys(('NH3', 'NonNH3', 'P', 'K', 'Mg', 'Ca'), 0.02)
app_loss['NH3'] = 0.05


# =============================================================================
# Prices and GWP CFs
# =============================================================================

# To account for different in deployment locations
price_ratio = 1
get_price_ratio = lambda: price_ratio

# Recycled nutrients are sold at a lower price than commercial fertilizers
price_factor = 0.25
get_price_factor = lambda: price_factor

lifetime = 8
get_lifetime = lambda: lifetime

price_dct = {
    'Concrete': 194,
    'Steel': 2.665,
    'N': 1.507*get_price_factor(),
    'P': 3.983*get_price_factor(),
    'K': 1.333*get_price_factor(),
    }

GWP_dct = {
    'CH4': 28,
    'N2O': 265,
    'N': -5.4,
    'P': -4.9,
    'K': -1.5,
    }

indicator_path = 'data/impact_indicators.tsv'
qs.ImpactIndicator.load_from_file(indicator_path)
indicators = qs.ImpactIndicator.get_all_indicators()
GWP = ImpactIndicator.get_indicator('GWP')

item_path = 'data/impact_items.xlsx'
qs.ImpactItem.load_from_file(item_path)
items = qs.ImpactItem.get_all_items()

ImpactItem.get_item('Concrete').price = price_dct['Concrete']
ImpactItem.get_item('Steel').price = price_dct['Steel']

# =============================================================================
# Universal units and functions
# =============================================================================

def batch_create_stream_items():
    for k, v in GWP_dct.items(): StreamImpactItem(ID=f'{k}_item', GWP=v)
    global _ImpactItem_LOADED
    _ImpactItem_LOADED = True


def batch_create_streams(prefix):
    if not _ImpactItem_LOADED: batch_create_stream_items()

    stream_dct = {}
    item = ImpactItem.get_item('CH4_item').copy(f'{prefix}_CH4_item', set_as_source=True)
    stream_dct['CH4'] = WasteStream(f'{prefix}_CH4', phase='g', stream_impact_item=item)

    item = ImpactItem.get_item('N2O_item').copy(f'{prefix}_N2O_item', set_as_source=True)
    stream_dct['N2O'] = WasteStream(f'{prefix}_N2O', phase='g', stream_impact_item=item)

    if prefix == 'A':
        mixed_or_liq = 'mixed'
    else:
        mixed_or_liq = 'liq'
        item = ImpactItem.get_item('N_item').copy(f'{prefix}_sol_N_item', set_as_source=True)
        stream_dct['sol_N'] = WasteStream(f'{prefix}_sol_N', phase='l', price=price_dct['N'],
                                                  stream_impact_item=item)
        item = ImpactItem.get_item('P_item').copy(f'{prefix}_sol_P_item', set_as_source=True)
        stream_dct['sol_P'] = WasteStream(f'{prefix}_sol_P', phase='l', price=price_dct['P'],
                                          stream_impact_item=item)
        item = ImpactItem.get_item('K_item').copy(f'{prefix}_sol_K_item', set_as_source=True)
        stream_dct['sol_K'] = WasteStream(f'{prefix}_sol_K', phase='l', price=price_dct['K'],
                                          stream_impact_item=item)
    get_ID = lambda element: f'{prefix}_{mixed_or_liq}_{element}_item'
    item = ImpactItem.get_item('N_item').copy(get_ID('N'), set_as_source=True)
    stream_dct[f'{mixed_or_liq}_N'] = WasteStream(
        f'{prefix}_liq_N', phase='l', price=price_dct['N'], stream_impact_item=item)
    item = ImpactItem.get_item('P_item').copy(get_ID('P'), set_as_source=True)
    stream_dct[f'{mixed_or_liq}_P'] = WasteStream(
        f'{prefix}_liq_P', phase='l', price=price_dct['P'], stream_impact_item=item)
    item = ImpactItem.get_item('K_item').copy(get_ID('K'), set_as_source=True)
    stream_dct[f'{mixed_or_liq}_K'] = WasteStream(
        f'{prefix}_liq_K', phase='l', price=price_dct['K'], stream_impact_item=item)

    return stream_dct

def update_toilet_param(unit):
    # Use the private attribute so that the number of users/toilets will be exactly as assigned
    # (i.e., can be fractions)
    unit._N_user = get_toilet_user()
    unit._N_toilet = get_ppl()/get_toilet_user()
    unit._run()


def adjust_NH3_loss(unit):
    unit._run()
    # Assume the slight higher loss of NH3 does not affect COD,
    # does not matter much since COD not considered in crop application
    unit.outs[0]._COD = unit.outs[1]._COD = unit.ins[0]._COD


# %%

# =============================================================================
# Pit latrine system
# =============================================================================

# Set flowsheet to avoid stream replacement warnings
flowsheetA = Flowsheet('pit')
main_flowsheet.set_flowsheet(flowsheetA)
streamsA = batch_create_streams(prefix='A')

#################### Human Inputs ####################
A1 = su.Excretion('A1', outs=('urine', 'feces'), waste_ratio=0.02) # Uganda

################### User Interface ###################
CH4_item = ImpactItem.get_item('CH4_item')
N2O_item = ImpactItem.get_item('N2O_item')
pit_CH4 = WasteStream('pit_CH4', stream_impact_item=CH4_item.copy(set_as_source=True))
pit_N2O = WasteStream('pit_N2O', stream_impact_item=N2O_item.copy(set_as_source=True))
A2 = su.PitLatrine('A2', ins=(A1-0, A1-1,
                              'toilet_paper', 'flushing_water',
                              'cleansing_water', 'desiccant'),
                   outs=('mixed_waste', 'leachate', 'A2_CH4', 'A2_N2O'),
                   N_user=get_toilet_user(), N_toilet=get_ppl()/get_toilet_user(),
                   OPEX_over_CAPEX=0.05,
                   decay_k_COD=get_decay_k(tau_deg, log_deg),
                   decay_k_N=get_decay_k(tau_deg, log_deg),
                   max_CH4_emission=max_CH4_emission,
                   price_ratio=get_price_ratio())
A2.specification = lambda: update_toilet_param(A2)

##################### Conveyance #####################
A3 = su.Trucking('A3', ins=A2-0, outs=('transported', 'conveyance_loss'),
                 load_type='mass', distance=5, distance_unit='km',
                 interval=A2.emptying_period, interval_unit='yr',
                 loss_ratio=0.02, price_ratio=get_price_ratio())
def update_A3_param():
    A3._run()
    truck = A3.single_truck
    truck.interval = A2.emptying_period*365*24
    truck.load = A3.F_mass_in*truck.interval/A2.N_toilet
    rho = A3.F_mass_in/A3.F_vol_in
    vol = truck.load/rho
    A3.fee = get_tanker_truck_fee(vol)
    A3.price_ratio = get_price_ratio()
    A3._design()
A3.specification = update_A3_param

################## Reuse or Disposal ##################
A4 = su.CropApplication('A4', ins=A3-0, outs=('liquid_fertilizer', 'reuse_loss'),
                        loss_ratio=app_loss)
A4.specification = lambda: adjust_NH3_loss(A4)

A5 = su.Mixer('A5', ins=(A2-2,), outs=streamsA['CH4'])
A5.line = 'fugitive CH4 mixer'

A6 = su.Mixer('A6', ins=(A2-3,), outs=streamsA['N2O'])
A6.line = 'fugitive N2O mixer'

A7 = su.ComponentSplitter('A7', ins=A4-0,
                           outs=(streamsA['mixed_N'], streamsA['mixed_P'], streamsA['mixed_K'],
                                 'A_liq_non_fertilizers'),
                           split_keys=(('NH3', 'NonNH3'), 'P', 'K'))

############### Simulation, TEA, and LCA ###############
sysA = System('sysA', path=flowsheetA.unit)

teaA = SimpleTEA(system=sysA, discount_rate=discount_rate, income_tax=0.3, # Uganda
                 start_year=2018, lifetime=get_lifetime(), uptime_ratio=1,
                 lang_factor=None, annual_maintenance=0, annual_labor=0)

lcaA = LCA(system=sysA, lifetime=get_lifetime(), lifetime_unit='yr', uptime_ratio=1,
           annualize_construction=True)

def update_sys_pit_lifetime():
    A7._run()
    A7.outs[0].price = price_dct['N']
    A7.outs[1].price = price_dct['P']
    A7.outs[2].price = price_dct['K']
    teaA.lifetime = lcaA.lifetime = get_lifetime()
A7.specification = update_sys_pit_lifetime


# %%

# =============================================================================
# Urine-diverting dry toilet (UDDT) system
# =============================================================================

flowsheetB = Flowsheet('uddt')
main_flowsheet.set_flowsheet(flowsheetB)
streamsB = batch_create_streams('B')

#################### Human Inputs ####################
B1 = su.Excretion('B1', outs=('urine', 'feces'), waste_ratio=0.02) # Uganda

################### User Interface ###################
B2 = su.UDDT('B2', ins=(B1-0, B1-1,
                        'toilet_paper', 'flushing_water',
                        'cleaning_water', 'desiccant'),
             outs=('liq_waste', 'sol_waste',
                   'struvite', 'HAP', 'B2_CH4', 'B2_N2O'),
             N_user=get_toilet_user(), N_toilet=get_ppl()/get_toilet_user(),
             OPEX_over_CAPEX=0.1,
             decay_k_COD=get_decay_k(tau_deg, log_deg),
             decay_k_N=get_decay_k(tau_deg, log_deg),
             max_CH4_emission=max_CH4_emission,
             price_ratio=get_price_ratio())
B2.specification = lambda: update_toilet_param(B2)

##################### Conveyance #####################
handcart_fee = 0.01 # USD/cap/d
truck_fee = 23e3 # UGX/m3

# Handcart fee is for both liquid/solid
get_handcart_and_truck_fee = \
    lambda vol, ppl, include_fee: truck_fee/get_exchange_rate()*vol \
        + int(include_fee)*handcart_fee*ppl*B2.collection_period

# Liquid waste
B3 = su.Trucking('B3', ins=B2-0, outs=('transported_l', 'loss_l'),
                 load_type='mass', distance=5, distance_unit='km',
                 loss_ratio=0.02)

# Solid waste
B4 = su.Trucking('B4', ins=B2-1, outs=('transported_s', 'loss_s'),
                 load_type='mass', load=1, load_unit='tonne',
                 distance=5, distance_unit='km',
                 loss_ratio=0.02)
def update_B3_B4_param():
    B4._run()
    truck3, truck4 = B3.single_truck, B4.single_truck
    hr = truck3.interval = truck4.interval = B2.collection_period*24
    N_toilet = B2.N_toilet
    ppl = get_ppl() / N_toilet
    truck3.load = B3.F_mass_in * hr / N_toilet
    truck4.load = B4.F_mass_in * hr / N_toilet
    rho3 = B3.F_mass_in/B3.F_vol_in
    rho4 = B4.F_mass_in/B4.F_vol_in
    B3.fee = get_handcart_and_truck_fee(truck3.load/rho3, ppl, True)
    B4.fee = get_handcart_and_truck_fee(truck4.load/rho4, ppl, False)
    B3.price_ratio = B4.price_ratio = get_price_ratio()
    B3._design()
    B4._design()
B4.specification = update_B3_B4_param

################## Reuse or Disposal ##################
B5 = su.CropApplication('B5', ins=B3-0, outs=('liquid_fertilizer', 'liquid_reuse_loss'),
                        loss_ratio=app_loss)
B5.specification = lambda: adjust_NH3_loss(B5)

B6 = su.CropApplication('B6', ins=B4-0, outs=('solid_fertilizer', 'solid_reuse_loss'),
                        loss_ratio=app_loss)
def adjust_B6_NH3_loss():
    B6.loss_ratio.update(B5.loss_ratio)
    adjust_NH3_loss(B6)
B6.specification = adjust_B6_NH3_loss

B7 = su.Mixer('B7', ins=(B2-4,), outs=streamsB['CH4'])
B7.line = 'fugitive CH4 mixer'

B8 = su.Mixer('B8', ins=(B2-5,), outs=streamsB['N2O'])
B8.line = 'fugitive N2O mixer'

B9 = su.ComponentSplitter('B9', ins=B5-0,
                           outs=(streamsB['liq_N'], streamsB['liq_P'], streamsB['liq_K'],
                                 'B_liq_non_fertilizers'),
                           split_keys=(('NH3', 'NonNH3'), 'P', 'K'))

B10 = su.ComponentSplitter('B10', ins=B6-0,
                           outs=(streamsB['sol_N'], streamsB['sol_P'], streamsB['sol_K'],
                                 'B_sol_non_fertilizers'),
                           split_keys=(('NH3', 'NonNH3'), 'P', 'K'))


############### Simulation, TEA, and LCA ###############
sysB = System('sysB', path=flowsheetB.unit)

teaB = SimpleTEA(system=sysB, discount_rate=discount_rate, income_tax=0.3, # Uganda
                 start_year=2018, lifetime=get_lifetime(), uptime_ratio=1,
                 lang_factor=None, annual_maintenance=0, annual_labor=0)

lcaB = LCA(system=sysB, lifetime=get_lifetime(), lifetime_unit='yr', uptime_ratio=1,
           annualize_construction=True)

def update_sysB_params():
    B10._run()
    B9.outs[0].price = B10.outs[0].price = price_dct['N']
    B9.outs[1].price = B10.outs[1].price = price_dct['P']
    B9.outs[2].price = B10.outs[2].price = price_dct['K']
    teaB.lifetime = lcaB.lifetime = get_lifetime()
B10.specification = update_sysB_params


# %%

# =============================================================================
# Analyses
# =============================================================================

def get_recovery(system, nutrient='N', print_msg=True):
    if nutrient not in ('N', 'P', 'K'):
        raise ValueError('`nutrient` can only be "N", "P", or "K", '
                         f'not "{nutrient}".')
    sum_up = lambda streams: sum(getattr(s, f'T{nutrient}')*s.F_vol for s in streams) # g/hr
    tot_in = sum_up(system.path[0].outs)
    tot_out = sum_up(A7.ins) if system is sysA else sum_up((B9.ins[0], B10.ins[0]))
    recovery = tot_out / get_ppl() / tot_in
    if print_msg: print(f'{nutrient} recovery for {system.ID} is {recovery:.1%}.')
    return recovery


def get_daily_cap_cost(system, kind='net', print_msg=True):
    tea = system.TEA
    ppl = get_ppl()
    if kind == 'net':
        cost = (tea.annualized_equipment_cost-tea.net_earnings)
    elif kind in ('CAPEX', 'capital', 'construction'):
        cost = tea.annualized_equipment_cost
    elif kind in ('OPEX', 'operating', 'operation'):
        cost = tea.AOC
    elif kind in ('sales', 'revenue'):
        cost = tea.sales
    else:
        raise ValueError(f'Invalid `kind` input "{kind}", '
                         'try "net", "CAPEX", "OPEX", or "sales".')
    cost = cost / ppl / 365 * 100 # from $/cap/yr to ¢/cap/d
    if print_msg: print(f'Daily {kind} cost for {system.ID} is ¢{cost:.2f}/cap/d.')
    return cost


def get_daily_cap_ghg(system, kind='net', print_msg=True):
    lca = system.LCA
    ppl = get_ppl()
    ind_ID = 'GlobalWarming'
    if kind == 'net':
        ghg = lca.total_impacts[ind_ID]
    elif kind in ('CAPEX', 'capital', 'construction'):
        ghg = lca.total_construction_impacts[ind_ID]
    elif kind in ('transportation', 'transporting'):
        ghg = lca.total_transportation_impacts[ind_ID]
    elif kind == 'direct':
        ghg = lca.get_stream_impacts(kind='direct_emission')[ind_ID]
    elif kind == 'offset':
        ghg = -lca.get_stream_impacts(kind='offset')[ind_ID] # make it positive for credits
    elif kind in ('OPEX', 'operating', 'operation'):
        ghg = lca.total_transportation_impacts[ind_ID] + lca.get_stream_impacts()[ind_ID]
    else:
        raise ValueError(f'Invalid `kind` input "{kind}", '
                         'try "net", "construction", "transportation", '
                         '"operating", "direct", or "offset".')
    ghg = ghg / lca.lifetime / ppl / 365 * 1000 # from kg CO2-e/lifetime to g CO2-e/cap/d
    if print_msg: print(f'Daily {kind} emission for {system.ID} is {ghg:.1f} g CO2-e/cap/d.')
    return ghg

def plot_tea_lca(tea_metrics=('net',), lca_metrics=('net',)):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    ax1, ax2 = axs
    ylabel_size = 12
    xticklabel_size = 10

    # Cost
    bar_width = 0.3
    x = np.array(range(len(tea_metrics)))

    ax1.bar(x-bar_width,
            [get_daily_cap_cost(sysA, m, False) for m in tea_metrics],
            label='sysA', width=bar_width)
    ax1.bar(x+bar_width,
            [get_daily_cap_cost(sysB, m, False) for m in tea_metrics],
            label='sysB', width=bar_width)
    ax1.set_ylabel('Cost [¢/cap/yr]', fontsize=ylabel_size)
    ax1.set_xticks(x, tea_metrics, fontsize=xticklabel_size)

    # Emission
    x = np.array(range(len(lca_metrics)))
    ax2.bar(x-bar_width,
            [get_daily_cap_ghg(sysA, m, False) for m in lca_metrics],
            label='sysA', width=bar_width)
    ax2.bar(x+bar_width,
            [get_daily_cap_ghg(sysB, m, False) for m in lca_metrics],
            label='sysB', width=bar_width)
    ax2.set_ylabel('Emission [g CO2-e/cap/d]', fontsize=ylabel_size)
    ax2.set_xticks(x, lca_metrics, fontsize=xticklabel_size)

    for ax in axs: ax.legend()
    fig.tight_layout()

    return fig


score_df = pd.DataFrame({
    'Econ': (0, 0),
    'Env': (0, 0),
    })
def get_indicator_scores(systems=(sysA, sysB), tea_metric='net', lca_metric='net'):
    if not systems: systems = (sysA, sysB)
    for num, sys in enumerate(systems):
        score_df.loc[num, 'Econ'] = get_daily_cap_cost(sys, tea_metric, print_msg=False)
        score_df.loc[num, 'Env'] = get_daily_cap_ghg(sys, lca_metric, print_msg=False)
    return score_df

alt_names = (sysA.ID, sysB.ID)
mcda = MCDA(alt_names=alt_names, indicator_scores=get_indicator_scores((sysA, sysB)))

cr_wt = mcda.criterion_weights.copy()
def update_criterion_weights(econ_weight):
    cr_wt.Econ = econ_weight
    cr_wt.Env = env_weight = 1 - econ_weight
    if econ_weight == 0: cr_wt.Ratio = '0:1'
    elif econ_weight == 1: cr_wt.Ratio = '1:0'
    else: cr_wt.Ratio = f'{round(econ_weight,2)}:{round(env_weight,2)}'
    return cr_wt

def run_mcda(systems=(sysA, sysB), tea_metric='net', lca_metric='net',
             econ_weight=0.5, print_msg=True):
    indicator_scores = get_indicator_scores(systems, tea_metric, lca_metric) \
        if systems else mcda.indicator_scores
    mcda.run_MCDA(criterion_weights=update_criterion_weights(econ_weight),
                  indicator_scores=indicator_scores)
    if print_msg:
        scoreA = mcda.performance_scores[alt_names[0]].item()
        scoreB = mcda.performance_scores[alt_names[1]].item()
        winner = mcda.winners.Winner.item()
        print(f'The score for {sysA.ID} is {scoreA:.3f}, for {sysB.ID} is {scoreB:.3f}, '
              f'{winner} is selected.')
    return mcda.performance_scores


def plot_mcda(systems=(sysA, sysB), tea_metric='net', lca_metric='net',
             econ_weights=np.arange(0, 1.1, 0.1)):
    dfs = [run_mcda(systems, tea_metric, lca_metric, wt, False) for wt in econ_weights]
    scoresA = [df.sysA.item() for df in dfs]
    scoresB = [df.sysB.item() for df in dfs]
    fig, ax = plt.subplots()
    ax.plot(econ_weights, scoresA, '-', label='sysA')
    ax.plot(econ_weights, scoresB, '--', label='sysB')
    ax.set_xlabel('Economic Weight', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.legend()
    return fig

if __name__ == '__main__':
    for sys in (sysA, sysB):
        get_daily_cap_cost(sys)
        get_daily_cap_ghg(sys)
    run_mcda()
