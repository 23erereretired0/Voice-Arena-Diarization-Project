# Results: measured, provisional (2026-10-18)

These files come from the real run: 2 systems × the same 5 AMI excerpts, scored with dscore commit `e02f949…` (0 ms collar, overlap included, no speaker count supplied). Nothing here is estimated or invented. Values that were not measured, such as NeMo or cost, read `not_run` or `n/a`.

| System | Pooled DER | Macro DER | Median DER | Mean JER | Missed | False alarm | Confusion | Spk-count MAE | RTF | Failures |
|---|---|---|---|---|---|---|---|---|---|---|
| pyannote-community-1 | 22.18 | 23.56 | 27.63 | 46.56 | 12.29 | 3.71 | 6.17 | 0.80 | 0.777 (CPU) | 0/5 |
| api-system (Deepgram nova-3, diarizer v2) | 46.97 | 48.50 | 40.94 | 66.41 | 38.11 | 1.86 | 7.00 | 1.00 | 0.024 (end to end) | 0/5 |
| nemo-msdd | not_run | | | | | | | | | not attempted |

| File | Contents |
|---|---|
| `summary.csv` | Per system: pooled and macro DER, mean JER, pooled components, speaker-count MAE, runtime and RTF, failures, CI statement, ranking statement |
| `per_recording.csv` | The required per-recording table (System, Exact model/version, Recording, DER, JER, components, runtime, RTF, predicted speaker count, collar, overlap policy, speaker-count condition) plus diagnostics |
| `error_breakdown.csv` | Component seconds and percentages, dominant component, overlap-region and non-overlap DER, 250 ms collar diagnostic |
| `scores_raw.json` | Raw scorer output per recording, including md-eval seconds and boundary distances |
| `pairwise_bootstrap.csv` | One comparison (pyannote vs Deepgram): `insufficient_sample`, because n = 5 < 20 |
| `recording_summaries.json` | Generated per-recording statements (labelled measured, hypothesis, inspection or status) and cross-recording findings |
| `system_configs.json` | Declarations and run statuses for every system, scorer info, environment |
| `environment.json` | Captured environments (sandbox and the Windows evaluation machine) |
| `validation_report.json` | RTTM validation of every prediction (0 hard errors) |

**Units:** DER, JER and components are % of scored reference speaker time; runtime is in seconds; RTF is processing seconds ÷ audio seconds.

**Interpretation rule:** these numbers describe five 3-minute AMI meeting excerpts. They are not a general ranking: there is no confidence interval below 20 recordings.
