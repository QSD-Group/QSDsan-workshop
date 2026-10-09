"""
QSDsan workshop environment check.

Run it once before the workshop, in the environment you plan to use:

    python check_environment.py

or, inside Jupyter or Spyder:

    %run check_environment.py

It prints PASS, WARN, or FAIL for each check, with a hint for anything that is
not PASS, and ends with a summary. The whole run takes about 1 to 2 minutes
(the first dynamic simulation is slow because it compiles code).
Copy the full output into your message if you need help.
"""

import importlib
import platform
import sys
import time
import traceback
import warnings

MIN_PYTHON = (3, 12)
MIN_QSDSAN = (1, 5, 3)  # version the workshop materials were tested with

results = []  # (status, name, detail, hint)


def record(status, name, detail='', hint=''):
    results.append((status, name, detail, hint))
    line = f'[{status}] {name}'
    if detail:
        line += f': {detail}'
    print(line)
    if hint and status != 'PASS':
        print(f'       Fix: {hint}')


def parse_version(text):
    parts = []
    for p in str(text).split('.'):
        digits = ''.join(ch for ch in p if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_python():
    v = sys.version_info
    detail = f'{v.major}.{v.minor}.{v.micro} on {platform.system()} ({sys.executable})'
    if (v.major, v.minor) >= MIN_PYTHON:
        record('PASS', 'Python version', detail)
    else:
        record('FAIL', 'Python version', detail,
               'QSDsan needs Python 3.12 or newer. Create a new environment '
               'with Python 3.12 (see INSTALL.md) and run this check there.')


def check_package(import_name, label=None, required=True, hint=''):
    label = label or import_name
    try:
        mod = importlib.import_module(import_name)
    except Exception as e:
        status = 'FAIL' if required else 'WARN'
        record(status, label, f'cannot import ({type(e).__name__}: {e})',
               hint or f'Run: pip install {label}')
        return None
    version = getattr(mod, '__version__', None)
    if version is None:
        try:
            from importlib.metadata import version as pkg_version
            version = pkg_version(label)
        except Exception:
            version = 'version unknown'
    record('PASS', label, version)
    return mod


def check_qsdsan_version(qs):
    if qs is None:
        return
    have = parse_version(qs.__version__)
    if have >= MIN_QSDSAN:
        return
    want = '.'.join(map(str, MIN_QSDSAN))
    record('FAIL', 'QSDsan version', f'{qs.__version__} is older than {want}',
           f'Run: pip install --upgrade "qsdsan>={want}" "exposan>={want}"')


def check_jupyter():
    # Jupyter is optional for the check itself but needed for the workshop.
    found = []
    for name in ('jupyterlab', 'notebook', 'ipykernel'):
        try:
            importlib.import_module(name)
            found.append(name)
        except Exception:
            pass
    if 'ipykernel' in found and ('jupyterlab' in found or 'notebook' in found):
        record('PASS', 'Jupyter', ', '.join(found))
    else:
        record('WARN', 'Jupyter', 'not found in this environment' if not found
               else 'incomplete: found ' + ', '.join(found),
               'Run: pip install jupyterlab ipykernel. Spyder or VS Code users '
               'can ignore this if they run scripts or notebooks from their editor.')


def check_static_system(qs):
    """Components, streams, a unit, a system, TEA-free simulation."""
    try:
        cmps = qs.Components.load_default()
        qs.set_thermo(cmps)
        ws1 = qs.WasteStream('ws1', H2O=1000, S_F=5, units='kg/hr')
        ws2 = qs.WasteStream('ws2', H2O=500, S_F=10, units='kg/hr')
        M = qs.sanunits.Mixer('M1', ins=(ws1, ws2), outs='mixed')
        sys_ = qs.System('check_sys', path=(M,))
        sys_.simulate()
        total = M.outs[0].F_mass
        expected = ws1.F_mass + ws2.F_mass
        if abs(total - expected) < 1e-6 * expected:
            record('PASS', 'Static system (components, streams, unit, system)',
                   f'mixed flow {total:.1f} kg/hr')
        else:
            record('FAIL', 'Static system', f'mass balance off: {total} vs {expected}',
                   'Reinstall QSDsan in a clean environment (see INSTALL.md).')
    except Exception as e:
        record('FAIL', 'Static system', f'{type(e).__name__}: {e}',
               'Reinstall QSDsan in a clean environment (see INSTALL.md).')
        traceback.print_exc()
    finally:
        # Do not leave the check system in the global flowsheet.
        try:
            qs.main_flowsheet.clear()
        except Exception:
            pass


def check_uncertainty(qs):
    """A tiny Model with 4 Monte Carlo samples (needs SALib and chaospy)."""
    try:
        import numpy as np
        from chaospy import distributions as shape

        cmps = qs.Components.load_default()
        qs.set_thermo(cmps)
        ws = qs.WasteStream('ws_uncert', H2O=1000, S_F=5, units='kg/hr')
        P = qs.sanunits.Pump('P1', ins=ws, outs='pumped')
        sys_ = qs.System('uncert_sys', path=(P,))
        model = qs.Model(sys_)

        @model.parameter(name='S_F flow', element=ws, kind='coupled', units='kg/hr',
                         baseline=5, distribution=shape.Uniform(lower=1, upper=10))
        def set_flow(x):
            ws.imass['S_F'] = x

        @model.metric(name='Outlet flow', units='kg/hr')
        def get_flow():
            return P.outs[0].F_mass

        samples = model.sample(N=4, rule='L', seed=3118)
        model.load_samples(samples)
        model.evaluate()
        table = model.table
        if table.shape[0] == 4 and not np.isnan(table.values.astype(float)).any():
            record('PASS', 'Uncertainty analysis (Model, 4 samples)')
        else:
            record('FAIL', 'Uncertainty analysis', 'unexpected result table',
                   'Run: pip install --upgrade qsdsan SALib chaospy')
    except Exception as e:
        record('FAIL', 'Uncertainty analysis', f'{type(e).__name__}: {e}',
               'Run: pip install --upgrade qsdsan SALib chaospy')
        traceback.print_exc()
    finally:
        try:
            qs.main_flowsheet.clear()
        except Exception:
            pass


def check_dynamic():
    """A very short dynamic simulation of the BSM1 system from EXPOsan."""
    start = time.time()
    try:
        from exposan import bsm1
        bsm1.load()
        sys_ = bsm1.sys
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # numba performance notes are harmless
            sys_.simulate(t_span=(0, 0.05), method='BDF', state_reset_hook='reset_cache')
        eff = sys_.flowsheet.stream.effluent
        if eff.F_vol > 0:
            record('PASS', 'Dynamic simulation (BSM1, 0.05 day)',
                   f'{time.time() - start:.0f} s')
        else:
            record('FAIL', 'Dynamic simulation', 'effluent flow is zero',
                   'Reinstall in a clean environment (see INSTALL.md).')
    except Exception as e:
        record('FAIL', 'Dynamic simulation', f'{type(e).__name__}: {e}',
               'Often a numba cache problem. Set the environment variable '
               'NUMBA_CACHE_DIR to a short, writable folder and run again '
               '(see the common errors in INSTALL.md).')
        traceback.print_exc()


def check_diagram():
    """Optional: system diagrams need the Graphviz program, not only the Python package."""
    import shutil
    if shutil.which('dot'):
        record('PASS', 'Graphviz program (optional)', shutil.which('dot'))
    else:
        record('WARN', 'Graphviz program (optional)', 'dot not found on PATH',
               'Only needed for sys.diagram(). Install Graphviz from graphviz.org, '
               'or with conda: conda install graphviz. You can skip this.')


def main():
    print('QSDsan workshop environment check')
    print('=' * 40)
    check_python()
    qs = check_package('qsdsan', required=True,
                       hint='Run: pip install qsdsan exposan')
    check_qsdsan_version(qs)
    check_package('biosteam')
    check_package('thermosteam')
    check_package('exposan', hint='Run: pip install exposan')
    for pkg in ('numpy', 'scipy', 'pandas', 'matplotlib', 'SALib', 'chaospy', 'seaborn', 'numba'):
        check_package(pkg)
    check_jupyter()
    check_diagram()
    if qs is not None:
        check_static_system(qs)
        check_uncertainty(qs)
        check_dynamic()

    print('=' * 40)
    n_fail = sum(r[0] == 'FAIL' for r in results)
    n_warn = sum(r[0] == 'WARN' for r in results)
    n_pass = sum(r[0] == 'PASS' for r in results)
    print(f'{n_pass} passed, {n_warn} warnings, {n_fail} failed')
    if n_fail:
        print('NOT READY. Fix the FAIL items above, run this check again, '
              'and report any problem that remains (include this full output).')
    elif n_warn:
        print('READY, with warnings. Warnings do not block the workshop.')
    else:
        print('READY. You are all set for the workshop.')

if __name__ == '__main__':
    # No sys.exit here, so that %run in Jupyter or Spyder does not show a traceback.
    main()
