# QSDsan virtual workshop, October 2026

Two 2-hour Zoom sessions on Tuesdays, October 20 and October 27, 2026, 12:00 to 2:00 pm US Eastern. The sessions are hands-on: you run and edit short Jupyter notebooks. The notebooks are written so that you can also follow them on your own later.

## Notebooks

| Session | Part | Topic | Notebook | Open in browser |
|---|---|---|---|---|
| 1 (basics) | 1 | Components, streams, units, and systems | [s1p1_systems.ipynb](notebooks/s1p1_systems.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2026-virtual/editions/2026-virtual/notebooks/s1p1_systems.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QSD-Group/QSDsan-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FQSD-Group%252FQSDsan-workshop%26urlpath%3Dlab%252Ftree%252FQSDsan-workshop%252Feditions%252F2026-virtual%252Fnotebooks%252Fs1p1_systems.ipynb%26branch%3D2026-virtual) |
| 1 (basics) | 2 | Cost and environmental impacts (TEA and LCA) | [s1p2_tea_lca.ipynb](notebooks/s1p2_tea_lca.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2026-virtual/editions/2026-virtual/notebooks/s1p2_tea_lca.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QSD-Group/QSDsan-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FQSD-Group%252FQSDsan-workshop%26urlpath%3Dlab%252Ftree%252FQSDsan-workshop%252Feditions%252F2026-virtual%252Fnotebooks%252Fs1p2_tea_lca.ipynb%26branch%3D2026-virtual) |
| 2 (advanced) | 1 | Uncertainty and sensitivity analysis | [s2p1_uncertainty.ipynb](notebooks/s2p1_uncertainty.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2026-virtual/editions/2026-virtual/notebooks/s2p1_uncertainty.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QSD-Group/QSDsan-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FQSD-Group%252FQSDsan-workshop%26urlpath%3Dlab%252Ftree%252FQSDsan-workshop%252Feditions%252F2026-virtual%252Fnotebooks%252Fs2p1_uncertainty.ipynb%26branch%3D2026-virtual) |
| 2 (advanced) | 2 | Dynamic simulation | [s2p2_dynamic.ipynb](notebooks/s2p2_dynamic.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2026-virtual/editions/2026-virtual/notebooks/s2p2_dynamic.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QSD-Group/QSDsan-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FQSD-Group%252FQSDsan-workshop%26urlpath%3Dlab%252Ftree%252FQSDsan-workshop%252Feditions%252F2026-virtual%252Fnotebooks%252Fs2p2_dynamic.ipynb%26branch%3D2026-virtual) |

How the exercises work: each notebook has exercises at level 1 (change a value), level 2 (a few lines of code), and level 3 (advanced, optional). Every exercise has a hint and a solution that stays hidden until you click **Show solution**.

The folder `data/` holds precomputed results used by the uncertainty notebook (an EXPOsan model that is too slow to run live). If you run that notebook locally, keep the folder structure so that the notebook finds `../data`.

## Before the workshop

1. Install QSDsan: follow [setup/INSTALL.md](setup/INSTALL.md).
2. Run the environment check: [setup/check_environment.py](setup/check_environment.py). The last line should say READY.
3. Check what you need to know: [prework/PREREQUISITES.md](prework/PREREQUISITES.md).
4. Optional reading: [prework/PREREADING.md](prework/PREREADING.md).

## No local install? Run in the browser

Use one of these. If one does not work, try the other.

### Binder (no account needed)

Opens JupyterLab with QSDsan, EXPOsan, and Graphviz already installed. Use the Binder badge of a notebook in the table above, or open the whole folder with this badge. Sessions close after a period of inactivity, so download your work.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QSD-Group/QSDsan-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FQSD-Group%252FQSDsan-workshop%26urlpath%3Dlab%252Ftree%252FQSDsan-workshop%252Feditions%252F2026-virtual%26branch%3D2026-virtual)

Run `check_environment.py` there to confirm it works (open it from the file browser, or in a notebook cell run `%run setup/check_environment.py`).

### Google Colab (Google account needed)

Colab does not include QSDsan, so each notebook starts with an install cell that you run once in every new session:

1. Open the notebook with its Colab badge in the table above.
2. Run the first code cell. On Colab it installs QSDsan 1.5.3 and EXPOsan and tells you when it is done. Red text about "dependency conflicts" is a harmless warning.
3. Restart the session (menu **Runtime**, **Restart session**), even if Colab does not ask.
4. Run the notebook from the top. The install cell now does nothing. On your own computer that cell never does anything.

To check your Colab setup first, use the environment check notebook. It installs QSDsan 1.5.3, then runs the same environment check.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2026-virtual/editions/2026-virtual/setup/colab_check.ipynb)

## Folder contents

- `setup/`: installation guide, environment check script, and the Colab check notebook
- `prework/`: prerequisites and optional reading
- `notebooks/`: the four workshop notebooks
- `data/`: precomputed results
