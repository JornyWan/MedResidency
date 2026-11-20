<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="MedResidency" width="440">
  </picture>
</p>

<p align="center">
  <b>Open post-training for medical vision-language models</b><br>
  Supervised fine-tuning, distillation and reinforcement learning on one protocol
</p>

<p align="center">
  <a href="https://github.com/JornyWan/MedResidency/actions/workflows/tests.yml"><img src="https://github.com/JornyWan/MedResidency/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10 or newer"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Code style: Ruff"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="Pull requests welcome"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#methods">Methods</a> •
  <a href="#tasks">Tasks</a> •
  <a href="#models">Models</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#documentation">Documentation</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

Pre-training is medical school. Residency is where a graduate becomes a specialist: by studying reference answers,
by working next to attending physicians, and by seeing how cases turn out. MedResidency is that stage for a medical
VLM. It trains a student with supervised fine-tuning on labels, with distillation from expert teachers, and with
reinforcement learning on task rewards, all in one code base and under one protocol.

Medical VLM papers usually post-train in their own code, on their own split, with their own backbone and
hyper-parameters. A new method is then compared with numbers copied from another setup, and it is hard to tell how
much of a difference comes from the method. In MedResidency a run is a **task**, a **method** and a **seed**:

```bash
medres train --task cxr_report --method gkd --seed 42
```

The task fixes the student, the teachers, the data, the training budget and the evaluation. The method only changes
the loss. Every hyper-parameter lives in a YAML file in `configs/`, and every run records the configuration and the
exact commands it executed.

## Highlights

- **Eight methods on one protocol.** Label SFT, offline distillation, on-policy distillation with one or several
  teachers, GKD, two-stage training, CHORD, CHORD-φ and GRPO share the student, the data, the budget and the
  evaluation of a task.
- **Four medical tasks.** Chest X-ray report generation, reasoning VQA, both at once with two teachers, and eight
  modality experts distilled into one student.
- **Layered YAML configuration.** `defaults ← task ← method ← recipe ← key=value`. Unknown keys are errors, so a typo
  cannot silently train with a default.
- **One launcher.** It starts the teacher servers, checks that each one serves the model it should, trains, and stops
  everything it started, also when the run fails or is cancelled. An interrupted run continues from its last complete
  checkpoint.
- **Reproducibility records.** Each run writes its resolved configuration, its commands, the package versions and the
  commit of the code to `<run>/medres/`.
- **Evaluation included.** The report metrics (RadCliQ-v1, RadGraph-F1, RaTEScore, CheXbert, GREEN), VQA accuracy,
  and paired bootstrap comparisons of two runs.
- **A benchmark with its commands.** The result of every method on every task, and the command behind each row.
- **No trainer of its own.** Training and the teacher servers are [ms-swift](https://github.com/modelscope/ms-swift)'s
  and sampling is [vLLM](https://github.com/vllm-project/vllm)'s. A method is one YAML file and one short function.

## Quick start

The core package needs no GPU:

```bash
git clone https://github.com/JornyWan/MedResidency.git
cd MedResidency
pip install -e .
medres list
medres train --task med_vqa --method opd --dry-run
```

The dry run prints everything a run would execute: the environment, the teacher servers and the training command
with all its arguments.

Training needs the training environment (ms-swift and vLLM, pinned) and, for the report metrics, a second small
environment. Two scripts build them; [docs/installation.md](docs/installation.md) explains both.

```bash
bash scripts/install_train_env.sh ~/envs/medres
conda activate ~/envs/medres
```

A complete run on public data, from download to scores:

```bash
bash scripts/download_models.sh med_vqa
medres data med-vqa --pmc-vqa raw/pmc_vqa --medxpertqa raw/medxpertqa --mmmu raw/mmmu --out data/med_vqa

medres train --task med_vqa --method opd --seed 42
medres eval run --task med_vqa --model runs/med_vqa_opd_s42/checkpoint-300
```

By default one run uses four GPUs of one node: two for the student and two for the teacher server. The benchmark was
run on 48 GB cards. [docs/concepts.md](docs/concepts.md) describes the layout and how to change it, and
`scripts/slurm/` has job templates.

## Methods

`medres list methods` prints them, and [docs/methods.md](docs/methods.md) says what each one optimizes.

| Method | `--method` | Learns from |
|:--|:--|:--|
| Label SFT | `sft` | reference answers |
| Offline distillation | `offline_kd` | answers the teacher wrote beforehand |
| On-policy distillation (OPD), one or several teachers | `opd` | the teacher's score of every token the student samples |
| GKD | `gkd` | the teacher's top-k distribution on the student's samples |
| Two-stage | `two_stage` | offline distillation, then OPD |
| CHORD | `chord` | SFT on teacher answers that fades into OPD |
| CHORD-φ | `chord_phi` | OPD with a small token-weighted SFT term |
| GRPO | `grpo` | a task reward, without a teacher |

## Tasks

`medres list tasks` prints them, and [docs/tasks.md](docs/tasks.md) describes each one.

| Task | Student | Teachers | Data |
|:--|:--|:--|:--|
| `cxr_report` | Qwen3-VL-2B | UniRG-CXR (8B) | MIMIC-CXR; tested on ReXrank MIMIC-CXR and IU X-ray |
| `med_vqa` | Qwen3-VL-2B | MediX-R1 (8B) | PMC-VQA; tested on PMC-VQA, MedXpertQA-MM and MMMU health |
| `cxr_report_vqa` | Qwen3-VL-2B | both of the above | both of the above, mixed |
| `modality_experts` | Qwen2-VL-2B | the eight Med-R1 experts (2B each) | OmniMedVQA |

## Models

`medres list models` prints the model zoo with the hub name of every entry.

| Role | Models |
|:--|:--|
| Students | Qwen3-VL-2B, Qwen3-VL-4B, Qwen2-VL-2B |
| Teachers of Qwen3-VL students | UniRG-CXR, MediX-R1, QoQ-Med3-VL and SonoReasoner (8B each) |
| Teachers of Qwen2-VL students | the eight modality experts of Med-R1 (2B each) |

The teachers are specialists: UniRG-CXR writes chest X-ray reports, MediX-R1 reasons through visual questions,
SonoReasoner reads ultrasound, and each Med-R1 expert covers one imaging modality. Distillation on single tokens needs
a teacher that shares the student's tokenizer, so a teacher comes from the student's model family. A model is one line
in `configs/models.yaml`; [docs/extending.md](docs/extending.md) shows how to add a model, a task, a method, a reward
or a test set.

## Benchmark

<p align="center">
  <a href="benchmarks/README.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/benchmark-dark.svg">
      <img src="assets/benchmark-light.svg" alt="Report generation: 1/RadCliQ and truncated reports of every method" width="880">
    </picture>
  </a>
</p>

Chest X-ray report generation with a 2B student and an 8B teacher; a bar is the mean of three seeds. 1/RadCliQ is the
headline report metric. A report is truncated when it runs into the generation limit, which is what a student that
has collapsed into repetition does: plain OPD collapses on some seeds of this task, and GKD, CHORD and the two-stage
recipe do not.

The tables behind the figure, the other five settings and the command for every row are in
[benchmarks/README.md](benchmarks/README.md). That page also says where the numbers come from and what has been
checked.

## How a run is put together

```mermaid
flowchart LR
    C["configuration<br/>defaults, task, method,<br/>recipe, key=value"] --> B["method builder"]
    B -- stages --> L["launcher"]
    L --> T["teacher servers<br/>(vLLM)"]
    L --> W["reward server<br/>(some rewards)"]
    L --> S["ms-swift trainer<br/>with the plugin"]
    T -. token scores .-> S
    W -. rewards .-> S
    S --> R["run directory<br/>checkpoints and records"]
```

The configuration is plain YAML merged in a fixed order. The builder of the method turns it into one or more stages,
each an ms-swift command with its arguments. The launcher starts what a stage needs around the trainer, checks it,
runs the trainer and cleans up. Evaluation is a separate command that only needs a model directory.
[docs/concepts.md](docs/concepts.md) explains each step, and [docs/design.md](docs/design.md) says which conventions
come from TRL, LLaMA-Factory, verl, lmms-eval and rl-baselines3-zoo, and what was left out.

## Documentation

| | |
|:--|:--|
| [Installation](docs/installation.md) | the two environments, downloads, SLURM templates |
| [Concepts](docs/concepts.md) | runs, configuration layers, recipes, the launcher, resuming |
| [Methods](docs/methods.md) | what each method optimizes and which keys it has |
| [Tasks](docs/tasks.md) | the four built-in tasks |
| [Data](docs/data.md) | sources, licences, builders, file formats |
| [Evaluation](docs/evaluation.md) | scoring, validation, comparing runs |
| [Extending](docs/extending.md) | adding a model, a task, a method, a reward or a test set |
| [Troubleshooting](docs/troubleshooting.md) | failures we ran into, and their fixes |
| [Design](docs/design.md) | what the framework borrows from general-domain ones, and what it leaves out |

## Data and privacy

The repository contains code, configuration and documentation only: no images, no reports, no model weights, no
per-sample outputs. MIMIC-CXR is credentialed data. You obtain it from PhysioNet under your own data-use agreement,
and what you build or generate from it stays under that agreement. The tools that touch such data print counts and
aggregate statistics, never text. Details are in [docs/data.md](docs/data.md).

MedResidency is research software. Models trained with it are not medical devices and must not be used for clinical
decisions.

## Contributing

New tasks, methods, models and benchmark rows are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Results are
contributed as aggregate numbers together with the configuration that produced them.

## Citation

If you use MedResidency, please cite the software. GitHub's "Cite this repository" button gives the same entry
from [CITATION.cff](CITATION.cff).

```bibtex
@software{medresidency,
  title   = {MedResidency: open post-training of medical vision-language models},
  author  = {{The MedResidency authors}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/JornyWan/MedResidency}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Every dataset and model keeps its own licence and terms
of use.

## Acknowledgements

MedResidency stands on [ms-swift](https://github.com/modelscope/ms-swift),
[vLLM](https://github.com/vllm-project/vllm) and [RadEval](https://github.com/jbdel/RadEval), and on the teams that
released the models and datasets of its tasks: Qwen-VL, UniRG-CXR, MediX-R1, Med-R1, MIMIC-CXR, ReXrank, IU X-ray,
PMC-VQA, MedXpertQA, MMMU and OmniMedVQA.
