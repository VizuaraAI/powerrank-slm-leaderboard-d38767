# Experiment log

One entry per run. The rows here map directly onto the results tables in the
paper draft, so a filled-in log is most of a results section.

Rules that will save you a week later:

- Write the entry **before** you start the run, filling in everything except
  the outcome. If you cannot state what you expect to happen, the run is not
  ready.
- Record the measured **discordance rate** every time. It is the input that
  decides every sample size in this project, and it is the one number you
  cannot get back without re-running the models.
- Record what you expected. A run that matches expectation teaches you almost
  nothing; the gap between expected and observed is where the paper is.
- Never overwrite an entry. Add a new one.

---

## Template

### Run ID: `YYYY-MM-DD-short-name`

| Field | Value |
| --- | --- |
| Date | |
| Goal (one sentence) | |
| Paper table/figure this feeds | e.g. Table 1, Figure 2b |
| Command | `powerrank evaluate --bank ... --out ...` |
| Commit | `git rev-parse --short HEAD` |
| Models | |
| Items (n) | |
| Prompt templates | |
| Seeds | |
| Scored responses | models x items x prompts x seeds |
| Wall clock / GPU | |

**Expected before running:**

**Observed:**

| Metric | Value |
| --- | --- |
| Accuracy range across models | |
| Median pairwise discordance | |
| MDE at this n (points) | |
| Pairs significant after BH (of N) | |
| Number of rank bands | |
| Largest band size | |

**Did the result change what I believe?**

**Next run this implies:**

---

## Worked example (replace with your own first entry)

### Run ID: `2026-10-20-smoke-two-models`

| Field | Value |
| --- | --- |
| Date | 2026-10-20 |
| Goal (one sentence) | Prove the pipeline runs end to end before spending GPU time. |
| Paper table/figure this feeds | none, infrastructure only |
| Command | `bash scripts/01_smoke_run.sh` |
| Commit | `a1b2c3d` |
| Models | SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct |
| Items (n) | 60 |
| Prompt templates | plain |
| Seeds | 0 |
| Scored responses | 120 |
| Wall clock / GPU | 4 min, CPU |

**Expected before running:** Both models near chance on a 60 item mixed slice,
no significant difference, one single rank band. At n=60 the MDE is around 15
points, so anything that does look significant is a bug in my scoring code,
not a finding.

**Observed:** Fill this in.

| Metric | Value |
| --- | --- |
| Accuracy range across models | |
| Median pairwise discordance | |
| MDE at this n (points) | |
| Pairs significant after BH (of 1) | |
| Number of rank bands | |
| Largest band size | |

**Did the result change what I believe?** Fill this in.

**Next run this implies:** Scale items to 1,000 with three prompt templates and
check that the measured discordance rate is near the 0.20 assumed in
`configs/default.yaml`. If it is much lower, every sample size in the paper
gets smaller and the headline number needs updating.

---

## Running summary: the number that matters

Keep this table current. It is Table 1 of the paper in miniature.

| Date | n items | models | measured discordance | MDE (pt) | bands | note |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |
