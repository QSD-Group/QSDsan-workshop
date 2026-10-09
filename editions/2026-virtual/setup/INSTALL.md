<!-- MAINTAINER NOTE, delete before sending: tested on Windows 11 with Python 3.12.10 and venv only.
     macOS, Linux, and conda commands are standard but not yet tested. -->

# Installing QSDsan for the workshop

Time needed: 15 to 30 minutes, mostly downloads. Please finish by the deadline in the setup email, then run the [environment check](#4-run-the-environment-check).

No local install possible? Use the browser option in section 6.

## 1. What you need

| Item | Requirement |
|---|---|
| Python | **3.12 or newer** (3.13 is fine). Older versions will not work. |
| QSDsan and EXPOsan | **1.5.3** (the version the workshop notebooks were tested with) |
| Disk space | About 2 GB |
| Internet | Needed for installation and the Zoom session |
| Editor | JupyterLab (used in the workshop). VS Code or Spyder also work, see section 5. |

**Already have QSDsan?** Versions older than 1.5.3 are likely to break the notebooks. Do not upgrade your existing environment. Create a new one as below. It keeps your current work safe and avoids version conflicts.

## 2. Create a clean environment and install

Pick one option. Use a **terminal**: Anaconda Prompt or PowerShell on Windows, Terminal on macOS and Linux.

### Option A: venv (comes with Python)

Check your Python first:

```
python --version
```

On macOS and Linux the command may be `python3`. If the version is below 3.12, install a newer Python from [python.org](https://www.python.org/downloads/) or use Option B.

Windows (PowerShell):

```
python -m venv C:\qsdsan-env
C:\qsdsan-env\Scripts\Activate.ps1
```

macOS and Linux:

```
python3 -m venv ~/qsdsan-env
source ~/qsdsan-env/bin/activate
```

Keep the Windows path short. Long paths can make the install fail (see common errors).

Then, in the activated environment (the prompt starts with `(qsdsan-env)`):

```
python -m pip install --upgrade pip
python -m pip install qsdsan==1.5.3 exposan==1.5.3 jupyterlab
```

### Option B: conda (Anaconda, Miniconda, or Miniforge)

```
conda create -n qsdsan python=3.12
conda activate qsdsan
python -m pip install qsdsan==1.5.3 exposan==1.5.3 jupyterlab
```

Install QSDsan with `pip` inside the conda environment, as shown.

### What gets installed

`pip` installs everything the workshop needs, so there is nothing extra to add for the uncertainty and sensitivity analysis (SALib, chaospy) or the dynamic simulation (numba, SciPy). One optional extra is the Graphviz program, which draws system diagrams (`system.diagram()`). Install it from [graphviz.org](https://graphviz.org/download/) or with `conda install graphviz`. The workshop does not depend on it.

## 3. Get the workshop files

Download the workshop folder from the link in the setup email (or clone the repository), then start JupyterLab from inside it:

```
cd path/to/QSDsan-workshop
jupyter lab
```

Always activate your environment first (`C:\qsdsan-env\Scripts\Activate.ps1`, `source ~/qsdsan-env/bin/activate`, or `conda activate qsdsan`).

## 4. Run the environment check

With the environment activated, from the folder that contains `check_environment.py`:

```
python check_environment.py
```

It takes 1 to 2 minutes (the first dynamic simulation compiles code). Each line shows PASS, WARN, or FAIL. A FAIL line includes a suggested fix. The last line should say **READY**. WARN lines do not block the workshop.

You can also run it from a notebook or Spyder with `%run check_environment.py`.

If something still fails, reply to the setup email with the full output.

## 5. Other editors

- **VS Code:** install the Python and Jupyter extensions, open the workshop folder, then choose your environment as the interpreter (Ctrl+Shift+P, "Python: Select Interpreter") and as the notebook kernel (top right of a notebook).
- **Spyder:** Spyder uses its own Python by default. In the environment where you installed QSDsan, run `python -m pip install spyder-kernels`, then in Spyder go to Preferences, Python interpreter, "Use the following interpreter" and select the `python` of that environment. Restart the console.
- **Plain JupyterLab:** nothing more to do if you started it from the activated environment.

Whatever you use, the environment check must be run with the same Python that you will use in the workshop.

## 6. No local install: run in the browser

Use the Google Colab or Binder links in the setup email. They need no installation. Colab requires a Google account. Binder needs none but can take a few minutes to start, and sessions end after a period of inactivity, so download anything you want to keep. If one does not work, try the other.

## Common errors

| Symptom | Cause and fix |
|---|---|
| `python` is not recognized (Windows) | Python is not on your PATH. Reinstall Python and tick "Add python.exe to PATH", or use Anaconda Prompt. |
| `ERROR: No matching distribution found for qsdsan==1.5.3` or "requires a different Python" | Python is older than 3.12. Check with `python --version` and create the environment with a newer Python. |
| `OSError: [Errno 2] No such file or directory: '...\site-packages\...'` during install (Windows) | The path is too long. Create the environment in a short path such as `C:\qsdsan-env`, or enable long paths: run PowerShell as administrator and enter `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, then restart. |
| Activation is blocked: "running scripts is disabled" (Windows PowerShell) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Anaconda Prompt or Command Prompt (`C:\qsdsan-env\Scripts\activate.bat`). |
| `ModuleNotFoundError: No module named 'qsdsan'` in a notebook | The notebook uses a different Python than the one you installed into. Start Jupyter from the activated environment, or pick the right kernel. In a notebook cell, `import sys; print(sys.executable)` shows which Python is running. |
| Older QSDsan version shown by the check | You have a pre-existing install. Create a new environment (section 2) instead of upgrading. |
| Dynamic simulation check fails with a numba or cache error | Set `NUMBA_CACHE_DIR` to a short, writable folder and run again. Windows PowerShell: `$env:NUMBA_CACHE_DIR="C:\nbc"`. macOS and Linux: `export NUMBA_CACHE_DIR=~/nbc`. |
| Plot window does not appear, or a Tk or backend error appears when running scripts | Run in Jupyter, or set `MPLBACKEND=Agg` to skip showing windows (plots are then saved, not shown). |
| `dot` not found or `ExecutableNotFound` when drawing a diagram | The Graphviz program is not installed. It is optional. Install it (section 2) or skip `diagram()` calls. |
| Corporate or university network blocks `pip` | Try a different network (home or hotspot), or use the browser option in section 6. |
| `pip` takes very long or times out | Add `--default-timeout 100` to the pip command, and retry. |
| Apple silicon Mac: build errors during install | Use a native (arm64) Python, not an Intel build under Rosetta. Check with `python -c "import platform; print(platform.machine())"`, which should print `arm64`. |

## Reporting a problem

Reply to the setup email before the workshop and include:

1. Your operating system.
2. The output of `python --version` and `python -m pip list`.
3. The full output of `python check_environment.py`.
4. The full error message (copy the text, not a photo, if possible).
