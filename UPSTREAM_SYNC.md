# Upstream sync

This fork tracks [mitdbg/KramaBench](https://github.com/mitdbg/KramaBench)
(remote `upstream`). It is synced through upstream `main` at `b2e0d77`
(2026-07-24). The sync was done on 2026-10-05 on branch `single-task-final`.

Rule: upstream's task fixes win (answers, questions, data sources). Where
this fork already fixed the same problem, the fork's fix stays. Upstream's
runner and baseline systems are not taken.

## Taken from upstream

| Upstream | Change | Here |
|---|---|---|
| `c832df3` | Delete `data/astronomy/input/download_tle.py` | `4dd5c84` |
| `8a127e2` | wildfire-hard-20 gold 0.0465 → 4.65 (the question asks for a percentage) | `ad1e84f` |
| `03bd9bb` | astronomy-hard-7 data sources | `e2d0ce6` |
| `228fc8d` | legal-easy-21 retargeted: question, source file, gold 13596, subtasks; legal-hard-22-7 subtask id | `7d481c2`; conflicted with this fork's `627fec5` and was resolved to upstream |
| `f753b91` | legal-hard-15 wording (Metropolitan or Micropolitan) | `811c14d` |
| `932e79b`, `b2e0d77` | legal-hard-29 restricted to single-state areas, gold Atlanta-Sandy Springs-Roswell; legal-hard-14 reference script | `483427f`, `24de6ea` |
| `b9ef450` (task file only) | archeology-hard-12-6 answer type `list_exact` | `9a51e34` |
| `4e81c65` (script only) | environment-hard-10 reference script | `95aace2` |
| `fad06c0`, `b9ef450`, `2de935e`, `f52be4e` (parts) | `success`: numbers written as text, trailing `%`, missing answer; `f1`: comma-separated lists; aggregation casts `success` to 0/1 | `2920c81` |
| `fad06c0` (part) | LLM judge model `gpt-5-mini` | `959756c` |

## Already equivalent here

- `f08fa38` (legal-easy-21 CSV reader): synced by this fork's `627fec5`, then
  superseded by `228fc8d`.
- `8fdf7f7` (environment-hard-20 answer type `list_exact`): same change as this
  fork's `c0e7d1d`.
- The headline score counts `llm_paraphrase` instead of `string_bootstrap`:
  fixed on both sides.

## Kept from this fork

- **legal-hard-24 gold:** keeps the comma, "Los Angeles-Long Beach-Anaheim, CA
  Metropolitan Statistical Area". That is the value in the data; upstream's
  comma-less gold matches no correct answer (`627fec5`).
- **astronomy-hard-9, -11 and -12 data sources** (`6b33f01`): upstream still
  declares `omni2_low_res/omni2.txt` (the file is `omni2.text`) and a
  `swarm/POD` glob with four `?` per six-digit time.
- **`success` comparison:** rounds the answer to the target's displayed
  precision. Upstream's `fad06c0` accepts a relative error of 1e-4 or an
  absolute error of 1e-6. The absolute term passes any answer for targets near
  1e-13 (astronomy-easy-3, -hard-7, -hard-11).
- **Judge reliability:** a verdict cache keyed by model, a shortcut for text
  that differs only in spacing or case, and whole-word yes/no parsing.
- **Scoring:** failed judge calls are left out of the headline denominator.
- **Data:**
  - links that flatten `data/legal/input/csn-data-book-2024-csv/CSVs`;
  - a 0-byte `data/wildfire/input/ZHVI.csv` placeholder. The real file is a
    Kaggle download (see `CLAUDE.md`), and wildfire-hard-21 lists it.
- **`format_hint/`** (this fork only): updated for wildfire-hard-20 and the
  three rewritten legal tasks (`0f9a5c9`).

## Not taken

- Upstream's runner refactor: the Benchmark/Executor/Evaluator split,
  per-task caching and parallel workers (`056b0b2`, `6835169`, `f942c37`,
  `2f5a974`, the rest of `b9ef450`).
- smolagents and Claude 4.5 baselines, `tests/`, `pyproject.toml` packaging,
  dependency bumps, README install text, annotator-agreement scripts.
- `workload/environment-tiny.json` (`2e74e9a`), a fixture for upstream's tests.
- Pipeline-judge prompt changes (`fad06c0`). This fork's runs pass
  `--no_pipeline_eval`.
- Unmerged upstream branches such as `dsguru-gt`.

## Notes

- Upstream's `solutions/wildfire/wildfire-hard-20.py` divides a DataFrame
  index label by the fire count and prints 1221.99% on this data. The gold 4.65
  is still right: 22 of the 473 fires of 2008 hold 90% of the damage (4.6512%).
- Checked against this data:
  - legal-easy-21: Alabama's row in
    `2024_CSN_State_Rankings_Identity_Theft_Reports.csv` holds 13,596;
  - legal-hard-29: upstream's script reproduces Atlanta-Sandy Springs-Roswell.

## Next sync

```bash
git fetch upstream main
git log --oneline b2e0d77..upstream/main -- workload data solutions benchmark evaluate.py
```
