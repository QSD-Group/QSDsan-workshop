# Session plan (organizer notes)

Draft for review. Tutorial numbers refer to the QSDsan documentation tutorials (`docs/source/tutorials`, QSDsan 1.5.3). Runtimes below are not yet measured; they are checked in the testing step (D).

## Frame

Each session is 2 hours. Zoom opens 11:45 ET for setup help.

| Time (ET) | Block |
|---|---|
| 12:00 to 12:10 | Welcome and setup block (session 2: 5 minutes) |
| 12:10 to 1:00 | Part 1 (50 minutes) |
| 1:00 to 1:05 | Break |
| 1:05 to 1:55 | Part 2 (50 minutes) |
| 1:55 to 2:00 | Wrap-up and what to read next |

Questions are taken in the chat during the parts, with 2 short pauses per part for Q&A.

Running example: `qs.utils.create_example_treatment_systems()` (the aerobic and anaerobic plant used in tutorials 7, 8, and 9), so participants meet one system in session 1 and reuse it in uncertainty analysis. The dynamic part uses the BSM1 system from EXPOsan (tutorial 11). Every exercise has a level tag: **B** (basic, edit values), **I** (intermediate, short code), **A** (advanced, optional extension). Every exercise has a hint and a solution file.

Framing for people who know other tools: component = state variable list, `WasteStream` = stream with composition, `SanUnit` = unit block, `System` = flowsheet, TEA and LCA as add-ons that read the simulated system. Mention BioWin, SUMO, GPS-X, and SuperPro Designer comparisons where they help, in the first 5 minutes of each session.

## Session 1: basics

### Part 1: components, streams, units, systems

Objectives: after this part, participants can (1) define components and make a wastewater stream, (2) build and simulate a small system from existing units, (3) read results and diagrams, (4) write a minimal custom unit, and (5) find their way around an EXPOsan project.

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 5 | What QSDsan is, what it can do, how a project is laid out (EXPOsan package: `system.py`, `model.py`, `data/`, `load()`) | Tutorial 0, EXPOsan `bsm1` | Demo |
| 5 to 15 | Components and `WasteStream`: load default components, `codbased_inf_model` | Tutorials 2, 3 | Demo, exercise 1 (B) |
| 15 to 30 | `SanUnit` and `System`: build a chain of existing units, simulate, diagram, results | Tutorials 4, 6 | Demo, exercise 2 (B, I) |
| 30 to 42 | Custom unit: `_run`, `_design`, `_cost` for a simple reactor | Tutorial 5 section 2 | Live coding, exercise 3 (I, A) |
| 42 to 50 | Explore a ready-made system from EXPOsan; where to read about pitfalls | Tutorial 14, EXPOsan | Demo, pointers |

Exercises: (1) change the influent COD and TKN and read the composition (B). (2) change a tank volume or add a unit and re-simulate, read the recycle convergence report (B/I). (3) extend the custom unit with an extra output or a cost term (I); add a `Reaction` or `Equipment` (A).

### Part 2: cost and environmental impacts (TEA and LCA)

Objectives: after this part, participants can (1) explain what costs `TEA` includes and how the financial metrics are computed, (2) set up impact items and read an LCA table, (3) change assumptions and see the effect, and (4) name the common pitfalls (cost year, purchase vs installed cost).

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 5 | How QSDsan joins simulation, TEA, and LCA; comparison with SuperPro and LCA software | Tutorials 7, 8 intros | Talk |
| 5 to 22 | TEA: cost components, capital cost (unit `_cost`, the `cost` decorator, cost indices), parameters, NPV and EAC, depreciation | Tutorial 7 sections 1 and 3, tutorial 5 section 3.2 | Demo, exercise 4 (B, I) |
| 22 to 42 | LCA: impact indicators and items, construction, electricity, stream impacts; impact tables and allocation | Tutorial 8 | Demo, exercise 5 (B, I) |
| 42 to 50 | Cost vs environment trade-off; pitfalls (CEPCI year, purchase vs installed, static inventory) | Tutorial 8 section 5.6, tutorial 14 section 3 | Demo |

Exercises: (4) change the discount rate, lifetime, and electricity price and read how NPV changes (B); override one unit's cost (I). (5) read the impact breakdown and change a characterization factor (B); add a stream impact item (I).

## Session 2: advanced

### Part 1: uncertainty and sensitivity analysis

Objectives: after this part, participants can (1) set up a `Model` with parameters and metrics, (2) run an uncertainty analysis and read the results, (3) choose between rank correlation, Morris, and variance-based sensitivity methods, and (4) know when a run is too heavy for live use.

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 5 | Recap of the example system; why uncertainty analysis; distributions (chaospy) | Tutorial 9 | Talk |
| 5 to 20 | `Model`, `Parameter`, `Metric`; coupled vs isolated parameters | Tutorial 9 section 1 | Demo, exercise 6 (B, I) |
| 20 to 35 | Uncertainty analysis: live with a small N; precomputed with a large N | Tutorial 9 section 2.1 | Demo |
| 35 to 48 | Sensitivity: Spearman live; Morris and Sobol or FAST precomputed | Tutorial 9 section 2.2 | Demo, exercise 7 (B, I) |
| 48 to 50 | Pointers: scaling to bigger systems | Docs | Talk |

Exercises: (6) add one parameter with a distribution and one metric (I; B version: change a distribution's bounds). (7) read a sensitivity plot and rank the inputs (B); run Morris with a small N (I).

### Part 2: dynamic simulation

Objectives: after this part, participants can (1) run and inspect a dynamic simulation, (2) choose basic solver settings, (3) set up time-series inputs, and (4) see how a simple control rule can be written as a controller unit.

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 5 | Static vs dynamic; comparison with WEST, GPS-X, BioWin | Tutorial 11 section 1 | Talk |
| 5 to 20 | Load BSM1, run a short simulation, track units and streams, plot time series | Tutorial 11 section 1.1, EXPOsan `bsm1` | Demo, exercise 8 (B) |
| 20 to 32 | Solver settings: method (BDF, RK23), time span, output times, state reset; effect on runtime and accuracy | Tutorial 11, tutorial 14 sections 2.3 and 2.6 | Demo, exercise 9 (I) |
| 32 to 44 | Time-series inputs: `ExogenousDynamicVariable`, `DynamicInfluent` from a file | Tutorial 11 section 3 | Demo, exercise 10 (I, A) |
| 44 to 50 | Control with a controller unit: a custom dynamic unit that reads a state and adjusts a setpoint or input during simulation | Tutorial 11 section 2 | Optional demo, low priority (see below) |

Exercises: (8) track a different unit or state and plot it (B). (9) compare two solver methods on a short horizon by runtime and result (I). (10) supply a custom influent time series (I); write a small dynamic unit (A, tutorial 11 section 2).

## Coverage of the requested themes

| Theme | Status | Where | How it is handled |
|---|---|---|---|
| End-to-end workflow, project structure, interpreting results | Partial | Tutorial 0, 6; EXPOsan packages | 5-minute project walk-through in session 1 part 1, plus a short project-layout note in the notebook. No existing tutorial covers project structure. |
| Build and customize units and systems (including for teaching) | Covered | Tutorials 4, 5, 6, 14 | Live-coded simple custom unit (10 minutes). Teaching is a pointer to QSDedu and the qsdsan-teaching apps. |
| TEA and LCA integration and assumptions | Covered | Tutorials 7, 8, 14 | Part 2 of session 1. |
| Capital cost modeling in QSDsan and EXPOsan | Partial | Tutorial 5 section 3.2, tutorial 7 | Tutorial text covers the mechanism. Add one 3-minute walk through how a real EXPOsan unit computes its cost. |
| Dynamic simulation: ODE solver settings | Partial | Tutorial 11 | Tutorial shows `method` and `t_span` only. Exercise 9 covers the main choices; confirm which `solve_ivp` options `System.simulate` passes through before teaching them. |
| Dynamic simulation: time series setup | Covered | Tutorial 11 section 3 | Part 2 of session 2. |
| Control applications | Partial, unverified, low priority | No formal controller module. Control can be built as a controller unit (a custom dynamic `SanUnit`, tutorial 11 section 2) | 6-minute demo of a simple controller unit, once a working example is verified. Otherwise a short discussion. |
| Hybrid modeling | Not supported | No formal mechanism in QSDsan | Not taught. State this plainly in the Q&A and point to the docs and Slack for discussion. |
| Running QSDsan in Jupyter | Covered | Jupyter tips, prework | Prerequisites sheet and tips page. |
| Overview of capabilities, adding features, community | Partial | Docs, GitHub, Slack | 5 minutes at the start of session 1 and the wrap-up. No new material. |

## Open verification: control in dynamic simulation (low priority)

Priority is low. Work on this only after the notebooks, precomputed results, and testing are done. Until then the default for the last 6 minutes of session 2 is a short discussion, not a demo.

Approach to try first: a controller unit, written as a custom dynamic `SanUnit` that reads a state (for example dissolved oxygen in a reactor) and adjusts a setpoint or input (for example an aeration rate). It is not yet known whether the dynamic framework lets one unit change another unit's parameters during integration, so the first step is a small test on BSM1.

Fallback options, in order: a state-dependent rule applied between short segmented `simulate` calls, then a discussion only.

Specifications (`add_specification`) are documented for static simulation and are not the first choice. In a quick test (BSM1, EXPOsan 1.5.3, `t_span=(0, 0.2)`, `BDF`), the unmodified system ran, but adding a no-op specification to the first reactor made `sys.simulate` fail with `AttributeError: <CSTR: O3> undefined composition; cannot set flow rate`. That was not investigated further.

## Cuts made to fit the time

- Process models, `Process`, and ADM1 (tutorials 10, 12, 13) are pointers only.
- Component creation in depth (tutorial 2) and `SanUnit` basics (tutorial 4) are shortened to demos on the default components.
- Allocation and transportation detail in LCA is a pointer.
- Optimization is not covered.

## Proposed workshop files

```
editions/2026-virtual/
  notebooks/   s1p1_systems.ipynb  s1p2_tea_lca.ipynb  s2p1_uncertainty.ipynb  s2p2_dynamic.ipynb
  solutions/   one solutions notebook per part
  data/        precomputed uncertainty and dynamic results, example influent time series
```

Each notebook starts with a setup cell, runs top to bottom, has its exercises as marked cells, and has a reduced-size switch (`LIVE = True`) for the heavy runs. Notebooks are adapted from the tutorials listed above, with the tutorial text shortened.

## Decisions made

1. Setup block of 10 minutes and 50-minute parts: accepted.
2. Layout `notebooks/`, `solutions/`, `data/`: accepted. Keep file names short so that paths stay short on Windows.
3. BSM1 for session 2 part 2: accepted, provided it starts from a correct (steady-state) initial state. Check this when building the notebook. Full runs are precomputed and only a short horizon is run live.
4. Control: try a controller unit first (low priority). No formal mechanism for hybrid modeling exists in QSDsan, so it is not taught.
