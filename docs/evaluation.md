# Evaluation

Evaluation is a step of its own, after training. A checkpoint answers the test prompts of its task with greedy
decoding, the answers are scored, and the corpus-level numbers go into one summary file. The benchmark evaluates the
final checkpoint of every run.

## One command

```bash
medres eval run --task cxr_report --model runs/cxr_report_gkd_s42/checkpoint-300
```

`medres eval run` does three things:

1. It generates a response for every prompt of every test set of the task, with the response limit and context
   length of the task's `generation.eval` block. Generation runs in its own process on one GPU.
2. It scores the responses: report test sets with the report metrics, in the metrics environment
   ([installation.md](installation.md#metrics-environment)), and VQA test sets by accuracy.
3. It writes `summary.json` and prints the headline numbers.

`--model` is any model directory: a checkpoint, an untrained student or a teacher. Results go to
`evals/<run>/<checkpoint>`, the last two parts of the model path, unless `--out` names another directory.

| Option | Effect |
|:--|:--|
| `--sets mimic_cxr` | only some of the task's test sets |
| `--split validation` | the task's validation set instead of its test sets |
| `--green` | also GREEN on report test sets (a 7B judge; about 20 GPU-minutes per 2,000 reports) |
| `--tp 2` | spread a model that does not fit on one GPU over two |
| `--out DIR` | another output directory |

A generation file that already exists is kept, so an interrupted evaluation continues where it stopped, and a
second call with `--green` repeats the scoring but not the generation. A later call with other `--sets`, or with
`--split validation`, adds its sets to the `summary.json` of the checkpoint.

## Output

```
evals/cxr_report_gkd_s42/checkpoint-300/
    summary.json                 corpus-level numbers of every test set
    mimic_cxr.jsonl              the responses: {id, text, n_tokens, truncated}
    mimic_cxr.scores.jsonl       per study: id and metric values, no text
    mimic_cxr.summary.json       corpus-level numbers of this test set
    iu_xray.jsonl  ...
```

For VQA test sets the per-item file is `<set>.items.jsonl` (id, domain, correct, answered, tokens, truncated).

The per-item files hold numbers only. They are what `medres stats compare` reads. The response files hold text;
for MIMIC-CXR that text is credentialed data ([data.md](data.md#credentialed-data)).

## Report metrics

Reports are scored with [RadEval](https://github.com/jbdel/RadEval), in two views:

| View | Compared | On |
|:--|:--|:--|
| `fi` | the whole response against `Findings: ... Impression: ...` | studies whose reference has both sections |
| `f` | the findings section of the response against the reference findings | studies with findings |

The benchmark reports the `fi` view. The keys of `summary.json` carry the view as a prefix (`fi_inv_radcliq`,
`f_radgraph`, ...).

| Metric | Summary key | Notes |
|:--|:--|:--|
| 1/RadCliQ-v1 | `inv_radcliq` | the headline: reciprocal of the mean RadCliQ-v1 over the studies; higher is better |
| RadGraph-F1 | `radgraph` | part of RadCliQ-v1 |
| BERTScore, SembScore, BLEU-2 | `bertscore`, `semb`, `bleu2` | the other parts of RadCliQ-v1 |
| RaTEScore | `ratescore` | |
| CheXbert F1 | `f1chexbert_5_micro_f1`, `f1chexbert_all_micro_f1`, and the macro variants | corpus level, over 5 and over 14 findings |
| GREEN | `green` | with `--green` |
| Truncation | `truncated` (no prefix) | share of responses that reached the response limit |

Three details differ from calling RadEval directly, and they matter when you compare with numbers from elsewhere:

- **BERTScore inside RadCliQ-v1** follows the reference code of the metric (distilroberta-base, rescaled, without
  IDF weights), because the weights of RadCliQ-v1 were fitted on that variant.
- **GREEN has a length cap.** GREEN gives the prompt and the judge's answer 2,048 tokens together and pads each
  batch, so one report that loops for a thousand tokens leaves its batch-mates no room for the judge's answer; they
  score zero, or the batch fails. MedResidency cuts each response to 300 words and 512 judge tokens before GREEN
  and sorts the batches by length. Only degenerate responses are cut; `green_cut` in the summary counts them.
- **Truncation is reported next to every score.** A student that has collapsed into repetition often writes a
  plausible first sentence, and some metrics reward that. The truncation rate shows the collapse directly.

## VQA accuracy

Each test row names how an answer is read from a response (`parser`):

- `letter`: the option letter in the final answer, i.e. in the last `\boxed{}` or `<answer></answer>`, or else in the
  text after the reasoning;
- `answer_tag`: the word inside `<answer></answer>`, compared exactly. This is Med-R1's rule, used by
  `modality_experts`.

A response without a recognizable answer is wrong; it is never dropped. The summary has the accuracy per `domain`
(the test set, or the imaging modality), the micro average over all items and the macro average over domains,
together with the share of responses without an answer and of truncated responses.

The rewards of `grpo` call the same parsing code, so a model is trained and tested on the same definition of a
correct answer.

## Validation

`cxr_report` and `med_vqa` have a validation file, built from training-side data that no training prompt uses:

```bash
medres eval run --task cxr_report --model runs/cxr_report_gkd_s42/checkpoint-300 --split validation
```

Use it for every choice you make after looking at a score: a hyper-parameter, a learning rate, a checkpoint. The
benchmark reports the test sets once, at the final checkpoint.

## The steps as separate tools

`medres eval run` chains three tools that can also be called one by one, for instance to spread a large test set
over several GPUs or to score many checkpoints in one pass of the metric models.

```bash
OUT=evals/med_vqa_opd_s42/checkpoint-300
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i medres eval generate --task med_vqa --model runs/med_vqa_opd_s42/checkpoint-300 \
        --data data/med_vqa/test_medxpertqa_mm.jsonl --out $OUT/shards/medxpertqa_mm.$i.jsonl --shard $i/4 &
done
wait
cat $OUT/shards/medxpertqa_mm.?.jsonl > $OUT/medxpertqa_mm.jsonl
medres eval vqa --data data/med_vqa/test_medxpertqa_mm.jsonl --generations $OUT/medxpertqa_mm.jsonl
```

`medres eval report-score` and `medres eval vqa` take several generation files of the same test file, which loads
the metric models once for all of them:

```bash
medres eval report-score --data data/cxr_report/test_mimic.jsonl \
    --generations evals/cxr_report_gkd_s42/checkpoint-*/mimic_cxr.jsonl
```

`--add-green` adds GREEN to files that were scored without it.

## Comparing two runs

```bash
medres stats compare evals/cxr_report_gkd_s42/checkpoint-300/mimic_cxr.scores.jsonl \
    evals/cxr_report_chord_s42/checkpoint-300/mimic_cxr.scores.jsonl \
    --metric inv_radcliq --cluster-data data/cxr_report/test_mimic.jsonl
```

The tool pairs the two runs item by item and prints the difference with a 95% bootstrap interval (1,000 resamples).
For accuracy it adds an exact McNemar test per domain. With `--cluster-data` it resamples patients rather than
studies, since the studies of one patient are not independent.

The interval describes the test set. It says nothing about what a second training run with another seed would give,
and that spread is not small: in the benchmark, the same recipe trained twice on two GPU types differed by 0.048 in
1/RadCliQ. So:

- train several seeds, compare seed by seed, and report the mean over seeds;
- evaluate the checkpoints you compare on the same GPU type;
- do not read a single-seed difference of a few hundredths as a result.

The rules by which the benchmark calls a comparison a win, a tie or undetermined are on the
[benchmark page](../benchmarks/README.md#comparing-two-runs).

## References

- RadCliQ: Yu et al., "Evaluating progress in automatic chest X-ray radiology report generation", Patterns, 2023.
- RadGraph: Jain et al., "RadGraph: Extracting clinical entities and relations from radiology reports",
  arXiv:2106.14463, 2021.
- CheXbert: Smit et al., "Combining automatic labelers and expert annotations for accurate radiology report labeling
  using BERT", EMNLP 2020.
- RaTEScore: Zhao et al., "RaTEScore: A metric for radiology report generation", EMNLP 2024.
- GREEN: Ostmeier et al., "GREEN: Generative radiology report evaluation and error notation", Findings of EMNLP 2024.
