# QSDsan-workshop

Materials for workshops on [QSDsan](https://github.com/QSD-Group/QSDsan), organized by edition. Each edition is a self-contained folder with what was used for that event (agenda, slides link, setup, exercises, pinned environment).

## Editions

| Edition | Date | Venue / audience | Length | Folder | Recording |
|---|---|---|---|---|---|
| 2022 EES Symposium | 2022-04-22 | 27th Environmental Engineering and Science Symposium (in person) | One Hour | [2022-EES](2022-EES) | [Video](https://youtu.be/C4Wk2bhsvnk) |
| 2026 Virtual Workshop | 2026-10-20 and 2026-10-27 | Open registration, on Zoom | Two 2-hour sessions | [2026-virtual](2026-virtual) | [Session 1](https://youtu.be/YrBc9CSg9kU), [Session 2](https://youtu.be/7zbhwiWNN-A) |

## Running an edition in your browser
Each edition's README gives its launch links (e.g., Binder, Google Colab), so nothing needs to be installed. Launchers differ in where they read the environment from (each edition's pins are in its own `requirements.txt`, which CI installs into a fresh environment; the fast Binder links use a prebuilt environment from [QSDsan-env](https://github.com/QSD-Group/QSDsan-env), selected by an environment tag), but all of them should point to the git tag named after the edition (e.g., `2022-EES`), so that an earlier edition is rebuilt exactly as it was taught.

## Adding a new edition
1. Copy an existing edition folder to `<year>-<venue>/` at the repository root.
2. Fill in its README (audience, length, learning goals, agenda, slides/recording links).
3. Set the pins in the new edition's `requirements.txt` (and `runtime.txt`) to the versions it was tested against, and re-run its notebooks.
4. Once delivered, tag the repo `<year>-<venue>` and treat the folder as frozen; fix only broken links.

## Related
- [QSDsan tutorials](https://qsdsan.readthedocs.io/en/latest/tutorials/index.html): source of truth for how QSDsan works
- [QSDedu](https://github.com/QSD-Group/QSDedu): course modules for quantitative sustainable design
