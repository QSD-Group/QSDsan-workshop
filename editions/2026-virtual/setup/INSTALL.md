<!-- MAINTAINER NOTE, delete before sending: tested on Windows 11 with Python 3.12.10 and venv only.
     macOS, Linux, and conda commands are standard but not yet tested. -->

# Installing QSDsan for the workshop

Time needed: 15 to 30 minutes, mostly downloads. Please finish by the deadline in the setup email, then run the [environment check](#4-run-the-environment-check).

No local install possible? Use the browser option in section 6.

## 1. What you need

| Item | Requirement |
|---|---|
| Python | **3.12 or newer** QSDsan 1.5.3 declares 3.12 as its minimum, and `pip` will refuse to install it on older Pythons. The workshop was tested with 3.12. |
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
python -m venv $HOME\qsdsan-env
& $HOME\qsdsan-env\Scripts\Activate.ps1
```

macOS and Linux:

```
python3 -m venv ~/qsdsan-env
source ~/qsdsan-env/bin/activate
```

On Windows the environment goes in your home folder (for example `C:\Users\yourname\qsdsan-env`). Do not put it inside deeply nested folders, because long paths can make the install fail (see common errors).

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

`pip` installs everything the workshop needs for the uncertainty and sensitivity analysis (SALib, chaospy) and the dynamic simulation (numba, SciPy), so there is nothing extra to add for those.

### Graphviz (required, install separately)

We draw system diagrams (`system.diagram()`) throughout the workshop. This needs the **Graphviz program**, which `pip` does not install (the Python `graphviz` package that `pip` installs is only a wrapper around it). Install it once, then **close and reopen your terminal or editor** so it is found:

| System | Command |
|---|---|
| Windows | `winget install Graphviz.Graphviz`, or the installer from [graphviz.org](https://graphviz.org/download/) (choose "Add Graphviz to the system PATH for current user") |
| macOS | `brew install graphviz` (needs [Homebrew](https://brew.sh)) |
| Ubuntu or Debian | `sudo apt install graphviz` |
| Fedora | `sudo dnf install graphviz` |
| conda (any system) | `conda install -c conda-forge graphviz` (installs into the active conda environment) |

Check it with `dot -V`, which should print a version.

## 3. Get the workshop files

Download the workshop folder from the link in the setup email (or clone the repository), then start JupyterLab from inside it:

```
cd path/to/QSDsan-workshop
jupyter lab
```

Always activate your environment first (`& $HOME\qsdsan-env\Scripts\Activate.ps1` on Windows, `source ~/qsdsan-env/bin/activate`, or `conda activate qsdsan`).

## 4. Run the environment check

Run `check_environment.py` (in the same folder as this guide) with the **same Python environment you will use in the workshop**. Choose one way:

- **Terminal** (Anaconda Prompt, PowerShell, or Terminal): activate your environment, go to the folder containing the file, and run:

  ```
  python check_environment.py
  ```

- **Editor or IDE** (VS Code, Spyder, PyCharm, others): open the file, select your QSDsan environment as the interpreter (section 5), and use the editor's Run button (Spyder: F5; VS Code: "Run Python File"; PyCharm: right-click, Run).
- **Jupyter** (notebook, JupyterLab, IPython console): with the QSDsan kernel selected, run `%run check_environment.py` in a cell. Use the full path if the file is not in the notebook's folder.

It takes 1 to 2 minutes (the first dynamic simulation compiles code). Each line shows PASS, WARN, or FAIL. A FAIL line includes a suggested fix. The last line should say **READY**. WARN lines do not block the workshop. The first line of output shows which Python ran the check, so you can confirm it is the right one.

If something still fails, reply to the setup email with the full output.

## 5. Other editors

- **VS Code:** install the Python and Jupyter extensions, open the workshop folder, then choose your environment as the interpreter (Ctrl+Shift+P, "Python: Select Interpreter") and as the notebook kernel (top right of a notebook).
- **PyCharm:** open the workshop folder, then Settings, Project, Python Interpreter, Add Interpreter, "Existing" (or "Conda"/"Virtualenv"), and pick the `python` of your environment (`%USERPROFILE%\qsdsan-env\Scripts\python.exe` on Windows, `~/qsdsan-env/bin/python`, or the conda environment). Notebooks in PyCharm Professional use the same interpreter.
- **Spyder:** Spyder uses its own Python by default. In the environment where you installed QSDsan, run `python -m pip install spyder-kernels`, then in Spyder go to Preferences, Python interpreter, "Use the following interpreter" and select the `python` of that environment. Restart the console.
- **Plain JupyterLab:** nothing more to do if you started it from the activated environment.

Whatever you use, the environment check must be run with the same Python that you will use in the workshop.

## 6. No local install: run in the browser

Use the Google Colab or Binder links in the setup email. They need no installation. Colab requires a Google account. Binder needs none but can take a few minutes to start, and sessions end after a period of inactivity, so download anything you want to keep. If one does not work, try the other.

## Common errors

| Symptom | Cause and fix |
|---|---|
| `python` is not recognized (Windows) | Python is not on your PATH. Reinstall Python and tick "Add python.exe to PATH", or use Anaconda Prompt. |
| `ERROR: No matching distribution found for qsdsan==1.5.3` or "requires a different Python" | QSDsan 1.5.3 requires Python 3.12 or newer. Check with `python --version` and create the environment with a newer Python. |
| `OSError: [Errno 2] No such file or directory: '...\site-packages\...'` during install (Windows) | The path is too long. Create the environment in your home folder or another short path (for example `python -m venv C:\qsdsan-env`), or enable long paths: run PowerShell as administrator and enter `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, then restart. |
| Activation is blocked: "running scripts is disabled" (Windows PowerShell) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Anaconda Prompt or Command Prompt (`%USERPROFILE%\qsdsan-env\Scripts\activate.bat`). |
| `ModuleNotFoundError: No module named 'qsdsan'` in a notebook | The notebook uses a different Python than the one you installed into. Start Jupyter from the activated environment, or pick the right kernel. In a notebook cell, `import sys; print(sys.executable)` shows which Python is running. |
| Older QSDsan version shown by the check | You have a pre-existing install. Create a new environment (section 2) instead of upgrading. |
| Dynamic simulation check fails with a numba or cache error | Set `NUMBA_CACHE_DIR` to a short, writable folder and run again. Windows PowerShell: `$env:NUMBA_CACHE_DIR="C:\nbc"`. macOS and Linux: `export NUMBA_CACHE_DIR=~/nbc`. |
| Plot window does not appear, or a Tk or backend error appears when running scripts | Run in Jupyter, or set `MPLBACKEND=Agg` to skip showing windows (plots are then saved, not shown). |
| `dot` not found or `ExecutableNotFound` when drawing a diagram | The Graphviz program is not installed, or was installed after your terminal or editor started. Install it (section 2), then fully close and reopen the terminal or editor. If `dot -V` still fails, add Graphviz's `bin` folder to your PATH. |
| Corporate or university network blocks `pip` | Try a different network (home or hotspot), or use the browser option in section 6. |
| `pip` takes very long or times out | Add `--default-timeout 100` to the pip command, and retry. |
| Apple silicon Mac: build errors during install | Use a native (arm64) Python, not an Intel build under Rosetta. Check with `python -c "import platform; print(platform.machine())"`, which should print `arm64`. |

## Reporting a problem

Reply to the setup email before the workshop and include:

1. Your operating system.
2. The output of `python --version` and `python -m pip list`.
3. The full output of `python check_environment.py`.
4. The full error message (copy the text, not a photo, if possible).
