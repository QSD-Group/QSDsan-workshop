# What you need to know before the workshop

One page. The workshop is hands-on: you will run and edit short Jupyter notebooks. You do not need TEA, LCA, or QSDsan experience. We explain those as we go.

## Everyone

You should be able to:

- Open a Jupyter notebook and run a cell (Shift+Enter).
- Change a number or a name in a code cell and run it again.
- Read an error message and say which line it points to.
- Restart the kernel (menu: Kernel, Restart) when asked.

If any of these is new to you, start with the [Jupyter tips page](https://qsdsan.readthedocs.io/en/latest/tutorials/jupyter_tips.html) and the [Jupyter documentation](https://docs.jupyter.org/en/latest/), then run `setup/check_environment.py` (or the Colab notebook) to practice.

## Pick your level

| Level | You can | What to expect | What to do beforehand |
|---|---|---|---|
| **New to Python** | Not write code yet | Notebooks are pre-filled, so you can run cells and follow along. You will not be able to do every exercise on your own, and that is fine. | Use Colab or Binder (no install). Work through the primer below, parts 1 to 3. Do not worry about finishing it. |
| **Basic** | Read code, change values | You can do the core exercises by editing values and running cells. | Skim the primer parts you are unsure about. Know what a variable, list, dictionary, function call, and `import` are. |
| **Intermediate** | Write scripts and small functions | Core exercises plus tasks that need a short function or loop. | Nothing required. Know basic pandas (`DataFrame`, selecting columns) and matplotlib plots. |
| **Advanced** | Build packages, write classes | Optional extension tasks, for example writing a custom unit as a Python class. | Nothing required. Look at tutorials 4 and 5 (SanUnit) if you want to start early. |

You do not need to be at the same level for the whole workshop. Each exercise states the minimum level it needs, and each has a short hint and a full solution file.

## Python primer (optional)

Pick one. Both use Jupyter-style examples and need no installation if you use Colab.

- [Software Carpentry, Plotting and Programming in Python](https://swcarpentry.github.io/python-novice-gapminder/): the gentlest start. Parts 1 to 3 (Python basics, data types, lists) cover what you need.
- [The Python Tutorial](https://docs.python.org/3/tutorial/) (official), sections 3 to 5: faster, for people who have programmed before.
- For tables of results: [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html).
