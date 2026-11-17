# Live test of `tfimpact recommend` — how to prepare, run and record it

The live test is the real-world part of the validation: the trained model
(`models/ga_weights.json`) is used on Terraform repositories it has never seen, on edits made in
front of an audience, and 2–3 people rate the suggestions. It complements the replay-based
numbers (cross-validation, optional external set), which measure the same thing on past commits
but without users.

## 1. Pick the repositories

Use repositories that are **not** in the training data (`data/raw/repos.csv`) and not in the
optional external set (TerraDS star ranks 151–200). Candidates checked on 2026-11-17 (TerraDS,
≥ 20 AWS resources, star ranks 301+):

| Repo | Kind | Commits (`.tf`) | AWS resources / dirs | Notes |
|---|---|---|---|---|
| **`tedilabs/terraform-aws-network`** | module library | 241 (121) | 44 / 12 | Recommended. Network modules (VPC, subnets, NAT, …) wired by variables, outputs and module calls; linear history; a call takes seconds. |
| `stroeer/terraform-aws-ecs-fargate` | module library | 309 (145) | 47 / 4 | Alternative: one dense ECS module. |
| **`dfds/infrastructure-modules`** | deployment (a company's infrastructure) | 2,713 (1,597) | 352 / 119 | Recommended. Real infrastructure, long history (strong history rules); reaches the 2,000-commit limit, ~30–40 s per call. |
| `schubergphilis/terraform-aws-mcaf-landing-zone` | platform / landing zone | 363 (188), merge PRs | 68 / 3 | Middle ground: interlinked account-baseline resources. |
| `nanlabs/terraform-aws-starter` | educational / starter | 269 (89) | 37 / 13 | Only if an educational repo is wanted. |

Good set: one module library + one deployment repo + your own repository.

## 2. Preparation (the day before)

1. **Clone** each repo normally, outside this project:
   ```bash
   mkdir -p ~/demo && git clone https://github.com/tedilabs/terraform-aws-network ~/demo/network
   git -C ~/demo/network status          # must be clean
   ```
2. **Dry run** to check mining, parse warnings and the time per call:
   ```bash
   uv run tfimpact recommend --repo ~/demo/network --block <some block, e.g. module.vpc>
   ```
3. **Write 5–6 scenarios** (section 3). For each one: make the edit, save it as a patch, reset.
   ```bash
   mkdir -p ~/demo/scenarios/network
   # ... edit files ...
   git -C ~/demo/network diff > ~/demo/scenarios/network/01-rename-variable.patch
   git -C ~/demo/network checkout . && git -C ~/demo/network clean -fd
   ```
4. **Write the expected answer before running the tool** (`expected.md` next to the patches): the
   blocks a careful developer would also change. Find them by reading the code and with
   `grep -rn "var.old_name" ~/demo/network`. Where possible, get an objective list from
   Terraform itself:
   ```bash
   git -C ~/demo/network apply ~/demo/scenarios/network/01-rename-variable.patch
   cd ~/demo/network/<module dir> && terraform init -backend=false && terraform validate
   ```
   `validate` lists every reference broken by a rename or removal. Never adjust the expected
   list after seeing the tool's output (that would bias the hit rate); extra useful suggestions
   are recorded separately (section 6).
5. **Backup output** for every scenario, in case something fails live:
   ```bash
   git -C ~/demo/network apply ~/demo/scenarios/network/01-rename-variable.patch
   uv run tfimpact recommend --repo ~/demo/network --diff --top 5 --json \
       > results/live/network/01-rename-variable.json
   git -C ~/demo/network checkout . && git -C ~/demo/network clean -fd
   ```
6. **Rating sheet** (`results/live/<repo>/ratings.csv`, template in section 6) and a short
   instruction for the raters (section 5).
7. Terminal: large font, wide window (the `--explain` column is wide), `--top 5`.

## 3. Scenarios

Cover the tool's main signals, plus one case that shows its limits honestly. Choose concrete
blocks from the repo; the table gives the pattern.

| # | Edit | A good answer contains | Main signal (`--explain`) |
|---|---|---|---|
| 1 | Rename a variable used in several places | every block / module call still using the old name | `graph_propagation_conf` (strongest learned weight) |
| 2 | Delete a resource referenced by outputs or other resources | those outputs / resources | propagation on removal (58 % of direct neighbours of a removed block change) |
| 3 | Add a new resource (e.g. a NAT gateway, an S3 bucket) | "types to add": the companion resource type, `output`, `variable` | type rule (`hist_type_cochange_conf`) + add-type offset |
| 4 | Change an attribute in one of several near-identical blocks (per-AZ / per-environment copies) | the sibling copies, even without references between them | `text_content_similarity` |
| 5 | Add an input variable to a module | the `module` calls in `examples/` or in other modules | graph + module plumbing |
| 6 | (limits) Edit an isolated block with no history or references | few / weak suggestions | shows the tool does not invent links |

Your own repository: reuse the same patterns on code you know well; your judgement of the
expected answer is then the most reliable.

## 4. Running the session

**Opening (2 minutes).** What the tool does (given an edited Terraform block, it suggests the
other blocks to change and the block types to add); the two modes (`--block` = "what is this
block related to", `--diff` = "given my actual edits, what else must change"); that the model
learned its 13 weights from 53 other repositories and never saw this one; what it reads (only
the repo's own git history and code, nothing is sent anywhere).

**Per scenario (3–4 minutes each).**
1. Apply the edit and show it:
   ```bash
   git -C ~/demo/network apply ~/demo/scenarios/network/01-rename-variable.patch
   git -C ~/demo/network diff
   ```
2. Ask the audience: "what else would you have to change?" (makes the comparison concrete).
3. Run the tool:
   ```bash
   uv run tfimpact recommend --repo ~/demo/network --diff --explain --top 5
   ```
4. Walk through both lists and use the contributions to say *why* each block is there, e.g.
   "`module.vpc` is first because the removed variable feeds it (`propagation_conf`)".
5. Compare with the expected answer written beforehand: hits in the top 5, and any useful
   suggestion that was not on the list.
6. Raters fill in the sheet for this scenario.
7. Reset: `git -C ~/demo/network checkout . && git -C ~/demo/network clean -fd`.

**Closing (2 minutes).** The offline numbers (cross-validated MRR 0.746, Hit@5 0.846 on 53
repos) next to the session's hit rate and mean rating; the limits seen in scenario 6.

## 5. Instructions for raters

For each scenario, look at the five "Blocks to modify" and the "Block types to add":
- per suggestion: **relevant** (1) if you would want to check or change it for this edit,
  otherwise 0;
- one **overall usefulness** score for the scenario:
  1 = useless, 2 = mostly noise, 3 = some useful reminders, 4 = useful, 5 = would have prevented
  a mistake;
- optional free comment.

Raters judge independently (no discussion before scoring).

## 6. What to record

```
results/live/<repo>/
  scenarios/NN-<name>.patch     the edit
  expected.md                   expected blocks per scenario, written before running
  NN-<name>.json                tool output (--json, --top 5)
  ratings.csv                   one row per (rater, scenario, suggestion)
  notes.md                      what happened live, extra useful suggestions, audience remarks
```

`ratings.csv` template:

```csv
repo,scenario,rater,list,rank,suggestion,in_expected,relevant,overall,comment
network,01-rename-variable,R1,modify,1,module.vpc,1,1,5,"caught the module call"
network,01-rename-variable,R1,modify,2,output.vpc_id,0,0,,
network,01-rename-variable,R1,add_type,1,variable,1,1,,
```

`in_expected` is filled from `expected.md` (not by the rater); `overall` is given once per
(rater, scenario), on the rank-1 row.

## 7. What goes into the report

- Hit rate against the expected lists: per scenario, the share of expected blocks found in the
  top 5 (Recall@5) and whether the first suggestion was expected (Hit@1).
- Rater precision: share of top-5 suggestions marked relevant, per scenario type; mean overall
  score; agreement between raters (share of suggestions all raters marked the same).
- Suggestions marked relevant but not in the expected list (the tool caught something the
  author missed), and irrelevant ones (false alarms) with the likely cause from `--explain`.
- Two or three quoted comments; one worked example with the `--explain` output.
- Threats: few scenarios and raters, scenarios chosen by the author, raters may know the
  author.
