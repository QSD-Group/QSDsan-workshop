====================================================
27th Environmental Engineering and Science Symposium
====================================================

This folder contains materials initially developed for a QSDsan workshop on April 22, 2022 during the `27th Environmental Engineering and Science (EES) Symposium <https://publish.illinois.edu/2022-environmentalsymposium>`_ and updated for later workshops. You can find the `recording <https://youtu.be/C4Wk2bhsvnk>`_ of the EES Symposium workshop and a `demo video <https://youtu.be/cO3LZpwOit8>`_ at our YouTube channel. Slides used for this workshop can be viewed and downloaded through `this link <https://uofi.box.com/s/ysjoo1dfmddrhkdp8xttmlggaa9k9ubl>`_.

Launching in your browser
-------------------------
Both options need no installation, and both use the ``2022-EES`` git tag, so they build the environment exactly as it was when this workshop was taught.

Binder
******
Launches the whole folder in JupyterLab.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/QSD-Group/QSDsan-workshop/2022-EES?urlpath=lab/tree/editions/2022-EES

Google Colab
************
Opens a single notebook (requires a Google account). The first code cell installs the pinned packages and fetches the supporting files; it takes a few minutes the first time. If Colab asks you to restart the runtime after the installation, restart and run the notebook again from the top.

* ``Example_complete.ipynb``:

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2022-EES/editions/2022-EES/Example_complete.ipynb

* ``Example_interactive.ipynb``:

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/QSD-Group/QSDsan-workshop/blob/2022-EES/editions/2022-EES/Example_interactive.ipynb

Materials
---------
* Jupyter Notebook examples

    - Example_complete.ipynb (fully populated with additional notes)
    - Example_interactive.ipynb (interactive module that does not require any coding skills)

* Python modules to construct the systems and analyses

    - country_specific.py (country-specific analysis)
    - models.py (uncertainty and sensitivity analyses)
    - systems.py (systems)

* data folder with data used in the analysis (e.g., location-specific parameters)
* results folder with results generated from the analysis
* ``dmsan`` folder with the multi-criteria decision analysis (MCDA) module used by the examples
* ``files`` folder with images used in the notebooks

The pinned environment (``requirements.txt``, ``runtime.txt``) is at the repository root.
