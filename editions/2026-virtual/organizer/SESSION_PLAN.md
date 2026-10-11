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

Running example: `qs.utils.create_example_treatment_systems()` (the aerobic and anaerobic plant used in tutorials 7, 8, and 9), so participants meet one system in session 1 and reuse it in uncertainty analysis. The dynamic part uses the BSM1 system from EXPOsan (tutorial 11). Every exercise has a level: **1** (change a value), **2** (short code), **3** (advanced, optional extension). Every exercise has a hint and a hidden solution (a "Show solution" fold that attendees open if they want).

Framing for people who know other tools: component = state variable list, `WasteStream` = stream with composition, `SanUnit` = unit block, `System` = flowsheet, TEA and LCA as add-ons that read the simulated system. Mention BioWin, SUMO, GPS-X, and SuperPro Designer comparisons where they help, in the first 5 minutes of each session.

## Session 1: basics

### Part 1: components, streams, units, systems

Objectives: after this part, participants can (1) look up how any class works from their editor, and say what is needed to create each key class, (2) define a component and make a wastewater stream, (3) build and simulate a small system from existing units and read its results and diagram, (4) write a minimal custom unit, and (5) find their way around an EXPOsan project.

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 3 | What QSDsan is and what it can do | Tutorial 0 | Talk |
| 3 to 11 | Reading the documentation (Jupyter, VS Code, PyCharm, Spyder); the key classes and what each needs; make a component | Notebook section 0, tutorial 2 | Demo, try it (level 1) |
| 11 to 19 | Components and `WasteStream`: default components, `codbased_inf_model`, copy and scale | Tutorials 2, 3 | Demo, exercise 1 (level 1) |
| 19 to 31 | `SanUnit` and `System`: build a chain of existing units, simulate, diagram, results | Tutorials 4, 6 | Demo, exercises 2, 3 (levels 1, 2) |
| 31 to 42 | Custom unit: `_run`, `_design`, `_cost` for a simple reactor | Tutorial 5 section 2 | Live coding, exercises 4, 5 (levels 2, 3) |
| 42 to 50 | An EXPOsan project: files, `load()`, diagram (`bsm1`); where to read about pitfalls | Tutorial 14, EXPOsan | Demo, exercise 6 (level 1) |

Exercises: (try it) make a component from its documentation (level 1). (1) copy the influent and scale a component, read the composition (level 1). (2) change the clarifier split and compare the flows (level 1). (3) add a second aerated tank in a new system and compare the cost (level 2). (4) change the conversion of the custom reactor and read the results (level 2). (5) extend the reactor with an electricity term (level 3). (6) load another EXPOsan system and compare layouts (level 1).

Timing note: this part is full. In a dry run, if it overruns, make exercises 3 and 5 optional homework, since they are the longest.

### Part 2: cost and environmental impacts (TEA and LCA)

Objectives: after this part, participants can (1) explain what costs `TEA` includes and how the financial metrics are computed, (2) set up impact items and read an LCA table, (3) change assumptions and see the effect, and (4) name the common pitfalls (cost year, purchase vs installed cost).

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 5 | How QSDsan joins simulation, TEA, and LCA; comparison with SuperPro and LCA software | Tutorials 7, 8 intros | Talk |
| 5 to 22 | TEA: cost categories, capital cost (unit `_cost`, `F_BM`, the `cost` decorator with size scaling and CEPCI), NPV, EAC, break-even fee | Tutorial 7 sections 1 and 3, tutorial 5 section 3.2 | Demo, exercises 1, 2 (levels 1, 2) |
| 22 to 42 | LCA: impact indicators and items, construction, transportation, stream impacts, electricity as a function; totals and per m3 | Tutorial 8 | Demo, exercise 3 (level 1) |
| 42 to 50 | Cost vs environment trade-off; pitfalls (CEPCI year, purchase vs installed, static inventory) | Tutorial 8 section 5.6, tutorial 14 section 3 | Demo, exercise 4 (level 2) |

Exercises: (1) change the discount rate and compare both plants (level 1). (2) change the anaerobic plant's capital cost and compare break-even fees (level 2). (3) change the grid carbon intensity and compare footprints (level 1). (4) add an environmental impact to the bicarbonate dose and recompute with a new `LCA` (level 2).

## Session 2: advanced

### Part 1: uncertainty and sensitivity analysis

Objectives: after this part, participants can (1) set up a `Model` with parameters (distribution, coupled or isolated) and metrics, (2) run an uncertainty analysis and read the results, (3) choose between rank correlation, Morris, and variance-based sensitivity methods, and (4) estimate the cost of an analysis and work with precomputed results when a model is expensive.

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 4 | Why uncertainty and sensitivity analysis; the question: aerobic or anaerobic under uncertainty | Tutorial 9 | Talk |
| 4 to 14 | Recap of both plants; `Model`, `Parameter` (distribution, kind), `Metric`; one model for both plants | Tutorial 9 section 1 | Demo |
| 14 to 24 | Uncertainty analysis: sample, evaluate, probability that the anaerobic plant wins, joint plot; how many samples | Tutorial 9 section 2.1 | Demo, exercises 1, 2 (levels 1, 2) |
| 24 to 38 | Sensitivity: Spearman and Morris; comparing the two methods. Sobol and FAST are described in a table and left to tutorial 9 | Tutorial 9 section 2.2 | Demo, exercise 3 (level 1) |
| 38 to 50 | Expensive models: time a small run, run counts per method (including Sobol and FAST), precomputed results from EXPOsan `bwaise` (137 parameters) | EXPOsan `bwaise` | Demo, exercise 4 (level 2) |

Exercises: (1) repeat the uncertainty analysis with N = 50, 500, 5000 and see when the results settle (level 1). (2) widen the influent COD range and find the share of samples in which the anaerobic plant becomes cheaper (level 2). (3) compare the two sensitivity methods for the carbon advantage (level 1). (4) rank the influential inputs of another `bwaise` result from the precomputed table (level 2).

Live and precomputed: the model of both plants evaluates 1,000 samples in about 1 second, so everything for that part runs live. Only the `bwaise` model is heavy (about 0.6 to 0.8 seconds per sample). Its N = 1000 table was computed ahead of time (765 seconds on one laptop) and is in `data/bwaise_modelA_N1000.csv.gz`. A live run with N = 20 (about 15 seconds) shows the workflow, controlled by `LIVE` and `N_LIVE` in the notebook.

### Part 2: dynamic simulation

Objectives: after this part, participants can (1) run and inspect a dynamic simulation of BSM1, (2) choose a solver and tolerances and know what they cost, (3) start an experiment from a steady state, (4) set up time-series inputs, and (5) say where control and hybrid modeling stand in QSDsan.

| Min | Content | Source | Type |
|---|---|---|---|
| 0 to 5 | Static vs dynamic; comparison with WEST, GPS-X, BioWin, SUMO | Tutorial 11 section 1 | Talk |
| 5 to 20 | Load BSM1, why a dynamic system needs `t_span`, a 100-day spin-up, scopes and trackers, plot time series | Tutorial 11 section 1.1, EXPOsan `bsm1` | Demo, exercise 1 (level 1) |
| 20 to 34 | Solver settings: `method`, `rtol` and `atol`, `t_eval`, `state_reset_hook`; start from a steady state | Tutorial 11, tutorial 14 sections 2.3 and 2.6 | Demo, exercise 2 (level 2) |
| 34 to 44 | Time-series inputs: `ExogenousDynamicVariable`, `DynamicInfluent` with the default and with your own file | Tutorial 11 section 3 | Demo, exercise 3 (level 2) |
| 44 to 50 | Control and hybrid modeling: what exists and what does not; practical advice | Discussion | Talk |

Exercises: (1) track another reactor and plot its oxygen and ammonium (level 1). (2) compare three solvers by run time and result (level 2). (3) make your own influent time series and simulate it (level 2).

Measured on one Windows laptop (QSDsan 1.5.3): a 100-day BSM1 run takes about 2 seconds with `BDF`; a 5-day run takes about 2 seconds with `BDF` or `LSODA`, about 10 seconds with `RK23`, about 17 seconds with `RK45`, and about 18 seconds with `Radau`. Tolerances of `rtol=1e-6`, `atol=1e-8` took 80 seconds for the same 5 days. The notebook runs in under a minute.

Decisions made along the way: the plan said to precompute the BSM1 runs, but they are fast enough to run live. The influent of BSM1 is not replaced by a dynamic influent in the notebook: building a second `System` around the BSM1 units re-initialized the states and failed to converge on later passes, so time-series inputs are taught with the standalone `DynamicInfluent` and with `ExogenousDynamicVariable`. Control is a discussion, not a demo: there is no formal controller module.

## Coverage of the requested themes

| Theme | Status | Where | How it is handled |
|---|---|---|---|
| End-to-end workflow, project structure, interpreting results | Partial | Tutorial 0, 6; EXPOsan packages | 5-minute project walk-through in session 1 part 1, plus a short project-layout note in the notebook. No existing tutorial covers project structure. |
| Build and customize units and systems (including for teaching) | Covered | Tutorials 4, 5, 6, 14 | Live-coded simple custom unit (10 minutes). Teaching: say that the notebooks can be reused in a course. Add a pointer to the qsdsan-teaching apps only if they are public by then (to confirm). |
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
  data/        precomputed uncertainty and dynamic results, example influent time series
```

Each notebook starts with a setup cell, runs top to bottom, has its exercises as marked cells, and has a reduced-size switch (`LIVE = True`) for the heavy runs. Notebooks are adapted from the tutorials listed above, with the tutorial text shortened.

## Decisions made

1. Setup block of 10 minutes and 50-minute parts: accepted.
2. Layout `notebooks/` and `data/`: accepted. Solutions are not separate files: each exercise has a hidden "Show solution" fold in the notebook. Keep file names short so that paths stay short on Windows.
3. BSM1 for session 2 part 2: accepted, provided it starts from a correct (steady-state) initial state. Check this when building the notebook. Full runs are precomputed and only a short horizon is run live.
4. Control: try a controller unit first (low priority). No formal mechanism for hybrid modeling exists in QSDsan, so it is not taught.
