# QSDsan-workshop

Materials for workshops on [QSDsan](https://github.com/QSD-Group/QSDsan), organized by edition. Each edition is a self-contained folder with what was used for that event (agenda, slides link, setup, exercises, pinned environment). Where possible, editions link to the [QSDsan tutorials](https://qsdsan.readthedocs.io/en/latest/tutorials/index.html) instead of copying notebooks.

## Editions

| Edition | Date | Venue / audience | Length | Folder |
|---|---|---|---|---|
| 2022 EES Symposium | 2022-04-22 | 27th Environmental Engineering and Science Symposium (in person) | TODO | [editions/2022-EES](editions/2022-EES) |

## Running an edition in your browser
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QSD-Group/QSDsan-workshop/main?urlpath=lab/tree/editions/2022-EES)

Binder reads its configuration from the repo root, so `requirements.txt` and `runtime.txt` there are pinned to the **latest** edition. To rebuild an earlier edition exactly as it was taught, use the git tag named after it (e.g., `2022-EES`) in place of `main`, both in the Binder link and when cloning.

## Adding a new edition
1. Copy an existing folder in `editions/` to `editions/<year>-<venue>/`.
2. Fill in its README (audience, length, learning goals, agenda, slides/recording links).
3. Update the root `requirements.txt` and `runtime.txt` to the versions the new edition was tested against, and re-run its notebooks.
4. Once delivered, tag the repo `<year>-<venue>` and treat the folder as frozen; fix only broken links.

## Related
- [QSDsan tutorials](https://qsdsan.readthedocs.io/en/latest/tutorials/index.html): source of truth for how QSDsan works
- [QSDedu](https://github.com/QSD-Group/QSDedu): course modules for quantitative sustainable design
