# Voice Arena Speaker Diarization Evaluation Methodology

**Document version:** 0.1 (proposal for review) · **Prepared:** 15 October 2026 · **Status of numbers:** this document contains **no measured leaderboard results**. The only measurements are the provisional 5-recording mini benchmark in §15; companion dashboard values come from `mini_eval/results/`.

### How to read this document

Two labels are used throughout so that readers can separate what the field already does from what Voice Arena is choosing:

- **[Established]**: practice taken from a cited benchmark, scorer, or dataset documentation.
- **[Proposed]**: a Voice Arena design decision. It is a choice and can be changed in a later methodology version, but only through a versioned change log.

Citations appear as bracketed numbers, e.g. [5], and resolve to the reference list in §17. Product facts about APIs and models were checked against vendor documentation in October 2026 and **must be re-verified on the execution date**, because hosted services change without notice.

---

## 1. Executive summary

**What Voice Arena evaluates.** Speaker diarization: given an audio recording, a system outputs time-stamped segments with anonymous speaker labels ("who spoke when"). Voice Arena compares open-source models run locally, hosted versions of open models, commercial diarization APIs, and speech-to-text APIs that attach speaker labels to words. It evaluates batch and streaming systems in separate tracks.

**Who it is for.** Product and engineering teams choosing a diarization system for meeting notes, contact-centre analytics, medical scribing, captioning, voice products and live agent-assist. It is also for researchers who need condition-specific comparisons and buyers who need to know whether a system fits *their* audio rather than whether it ranks first overall.

**Why one overall score is insufficient.** Diarization difficulty varies enormously with the recording. DIHARD III evaluated 11 domains and found that performance had improved markedly for two-party conversations, while domains such as web video remained far from solved [6]. A single average mixes clean two-person calls with noisy far-field meetings and overlapping speech, so a system can lead the average while failing exactly the slice a buyer cares about. Voice Arena therefore always shows results overall *and* by scenario.

**Main recommendation.**

1. **Primary accuracy metric: strict DER.** That means 0 ms collar, overlapping speech scored, no reference speaker count, and the same pinned scorer for every system. This follows the DIHARD III evaluation plan, which applied no forgiveness collar and scored overlap [5]. **[Established]**
2. **Official ranking statistic: pooled (duration-weighted) DER** over the hidden test set within each track. Macro-averaged recording DER is shown beside it as a secondary measure. **[Proposed]**
3. **Supporting evidence, always visible, never folded into one weighted score:** JER, DER components (missed speech, false alarm, speaker confusion), scenario-slice DER, speaker-count error, boundary timing, per-recording distribution, real-time factor, streaming latency, cost per audio hour, failure rate, deployment and privacy information. **[Proposed]**
4. **Rankings are statistical, not ordinal.** Systems are grouped into tiers using paired recording-level bootstrap confidence intervals. A "winner" is declared only when the difference is both statistically reliable and at least 1.0 absolute DER percentage point. **[Proposed]**
5. **Recommendations are conditional.** The dashboard recommends a system for a use case only when the relevant slice has enough recordings and the leader is separated from the runner-up. Otherwise it explicitly says the evidence is insufficient. **[Proposed]**

---

## 2. What speaker diarization is

**"Who spoke when."** Diarization partitions a recording into regions of speech and assigns each region to a speaker. It does not say *who* the speakers are by name; it says that the same unknown voice spoke here and here. Speaker identification, which attaches real names, is a different task.

**Expected output.** A list of segments, each with a start time, an end time, and a label. The standard exchange format is RTTM, a ten-field text line per segment; the dscore documentation and the DIHARD III plan both specify it [11][5]:

```
SPEAKER  meeting_017  1  12.340  3.120  <NA> <NA>  spk_2  <NA> <NA>
         file id     ch  start   dur                 label
```

**Why labels are anonymous within each recording.** A system has no way to know that its "speaker 2" is the reference's "Alice". The scorer therefore searches for the best one-to-one mapping between system labels and reference labels (the Hungarian algorithm in JER, an optimal mapping in DER) before counting errors [5][11]. Labels carry no meaning across recordings: dscore assumes speaker sets of different recordings are disjoint [11].

**Common applications.** Meeting transcripts and summaries ("what did each participant commit to"); call analytics (agent versus customer talk ratio, interruptions); medical scribing (separating clinician from patient speech); subtitles and captions (speaker changes and colours); conversational AI and live agent-assist (knowing who is currently talking).

**Main challenges** (the review by Park et al. surveys these at length [10]):

- **Overlapping speech.** Two or more people talking at once. Many systems output only one speaker per moment, which guarantees missed speech in overlaps.
- **Short turns and backchannels.** "Yeah", "mm-hm" and quick interjections are short, carry little voice information and are easy to miss or mis-assign.
- **Similar voices.** Same-gender or related speakers produce close voice embeddings, causing confusion errors.
- **Many speakers.** Estimating how many people are present gets harder as the number grows; DIHARD's meeting and restaurant domains contained up to 7 speakers [5].
- **Background noise and reverberation.** Noise creates false alarms; reverberation smears speech and blurs speaker characteristics.
- **Far-field microphones.** A single distant microphone records every speaker at a different level and distance. The AMI corpus records the same meetings with close-talking and far-field devices, which makes this effect directly measurable [1].
- **Telephone audio.** Narrowband (8 kHz) channels remove voice information. CALLHOME audio is 8 kHz telephone speech [3].
- **Code-switching and accent variation.** Models trained mostly on one language or accent may confuse speakers whose speech differs from training data.

---

## 3. Purpose and user decisions

| Audience | Decision they make | What Voice Arena must show them |
|---|---|---|
| Meeting-product teams | Which model works on far-field, multi-speaker meetings? | 3–5 and 6–10 speaker meeting strata, far-field tag, overlap-region DER, speaker-count error |
| Contact-centre teams | Which model works on two-party telephone calls? | Telephone stratum DER, confusion component, boundary error, cost per hour at volume, data retention |
| Voice-product teams | What is the best trade-off between accuracy, latency, cost, privacy and deployment? | Streaming track, latency percentiles, label revision rate, RTF, self-hosted versus API, price assumptions |
| Research teams | Which approach is strongest on a specific acoustic condition? | Per-tag slices, DER components, per-recording distribution, exact configurations |
| Buyers | Which system is appropriate for *my* use case, not merely ranked first? | Use-case views, tiers with confidence intervals, explicit "insufficient evidence" states, evaluation cards |

The leaderboard succeeds if a buyer can filter to their condition and get either a defensible shortlist with trade-offs or an honest statement that the data cannot separate the candidates.

---

## 4. Models and services evaluated

The launch roster below has 13 entries covering open/self-hosted, hosted-open, commercial diarization APIs, STT APIs with speaker labels, batch and streaming. **Version rule:** no row states a model version as fact beyond what vendor documentation showed in October 2026. Every run must freeze and record the exact model identifier, API version or diarizer build, and the execution date. Where documentation exposes a pinnable version, the pinned value is required and "latest" aliases are forbidden.

| # | System | Company | Exact model / product | Type | Mode | How Voice Arena runs it | Speaker-count input | Overlap output | Label granularity | Version & date requirement |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Community-1 (self-hosted) | pyannoteAI (open source) | `pyannote/speaker-diarization-community-1` with pyannote.audio 4.x; CC-BY-4.0 [14] | Open, self-hosted | Batch | Pinned container; model cloned at a fixed commit for offline use, as the model card describes [14]; GPU reference hardware | Optional `num_speakers`, `min_speakers`, `max_speakers` [14]. **None** in primary track | Yes. The regular output can overlap; an *exclusive* single-speaker variant also exists [14]. Score the regular output | Segment | Record HF commit SHA, pyannote.audio and torch versions |
| 2 | Precision-3 | pyannoteAI | API model string `precision-3`; opt-in, while `precision-2` is the default when no model is sent [15] | Commercial API | Batch | REST API; model string sent explicitly on every request | `numSpeakers`, `minSpeakers`, `maxSpeakers` available [15]; none in primary | Overlap-aware; exposes crosstalk probability [15]; exclusive mode exists [15] | Segment | Record model string and date. Omitting `model` silently selects Precision-2 [15], so the request body is archived |
| 3 | Community-1 (hosted) | pyannoteAI | API model string `community-1` [15] | Hosted open model | Batch | Same API harness as #2 | As #2 | As #1 | Segment | As #2. Compare against #1 to isolate hosting differences |
| 4 | Live-1 | pyannoteAI | Streaming over WebSocket; vendor states up to 8 speakers and 100 ms chunks [15] | Commercial API | Streaming | Real-time streaming harness (§9.4) | Not applicable | Vendor documentation describes training on overlapping speech [15]; verify | Start/end events [15] | Record endpoint and date. The vendor's sub-300 ms latency is a *vendor claim* until measured |
| 5 | NeMo MSDD | NVIDIA | NeMo `NeuralDiarizer` (multi-scale diarization decoder) with TitaNet-L embeddings and MarbleNet VAD; config family documented in NeMo [16] | Open, self-hosted | Batch | Pinned NGC container; resolved YAML archived | Config switches for oracle speaker count and maximum speakers [16]; primary uses estimated count | MSDD estimates speaker activity per scale; overlap handling is configuration-dependent and must be verified | Segment | Record nemo_toolkit version, container digest, resolved YAML |
| 6 | Streaming Sortformer | NVIDIA | `nvidia/diar_streaming_sortformer_4spk-v2.1` [17] | Open, self-hosted | Streaming | Real-time harness; chunk/right-context presets declared (latency presets on the card run from about 0.3 s to 30.4 s) [17] | None. The model handles **at most 4 speakers** by design [17][18] | End-to-end multi-speaker activity (overlap-capable by design; verify) | Frame → segment | Record model revision and streaming preset. 6–10 speaker strata are scored but flagged "outside declared capability" |
| 7 | SpeechBrain diarization recipe | SpeechBrain (open source) | AMI diarization recipe (speaker embeddings + clustering) [27] | Open, self-hosted | Batch | Recipe at pinned commit; hyper-parameter YAML archived | Recipe supports an oracle-count setting; primary requires estimation | Typically one speaker per segment; verify recipe | Segment | Record git commit, embedding model and YAML |
| 8 | AssemblyAI speaker diarization | AssemblyAI | `speaker_labels=true`; diarization page lists `universal-3-5-pro` and `universal-2` [19] | STT API | Batch | REST API; `speech_model` from response archived | `speakers_expected` (exact) or `speaker_options` min/max as **hard** limits; default maximum depends on file length [19]. None sent in primary | No. Word/utterance labels assign one speaker per word | Word + utterance | Freeze and record `speech_model` at execution time. Note that default caps are part of "audio-only" behaviour |
| 9 | Deepgram diarization | Deepgram | `diarize_model=v2` (batch); `latest` resolves to v2 for batch and v1 for streaming [20] | STT API | Batch & streaming (separate tracks) | REST (batch) and WebSocket (streaming) | No speaker-count parameter on the diarization page [20] | No. One speaker per word [20] | Word | Pin `v2`/`v1`, never `latest`. Archive `metadata.diarize_info` (diarizer UUID and arch) [20] |
| 10 | Google Cloud STT diarization | Google Cloud | Speech-to-Text v2 `diarization_config` [21]; model (e.g. a Chirp variant) must be declared | STT API | Batch & streaming | REST/gRPC client in a declared region | **Required** `min_speaker_count` and `max_speaker_count` [21]. Primary uses one fixed declared range for every file | No. Word-level labels [21] | Word | Record model, region, API version and date |
| 11 | Amazon Transcribe speaker partitioning | AWS | `ShowSpeakerLabels=true` with `MaxSpeakerLabels` (both required together) [22] | STT API | Batch (streaming support to be verified) | Batch job via SDK in a declared region | **Required** maximum. Primary uses one fixed declared maximum | No. Speaker label per item/segment | Word/segment | Record region, SDK version and date |
| 12 | Azure real-time diarization | Microsoft Azure | Speech SDK `ConversationTranscriber` real-time diarization [23] | STT API | Streaming | SDK harness in a declared region | Verify | Verify | Phrase/word | Record SDK version, region and date. The multichannel conversation-transcription preview was retired on 28 March 2025 [25], so the evaluated product is the current single-channel diarization |
| 13 | Azure fast transcription with diarization | Microsoft Azure | Fast transcription REST API with diarization enabled [24] | STT API | Batch | REST in a declared region | Verify | Verify | Phrase/word | As #12. By default, fast transcription merges input channels into one [24] |

**Product-name mapping.** The brief asked for "Azure Conversation Transcription". Microsoft's current documentation routes real-time diarization through the conversation-transcriber SDK and file diarization through the fast transcription API [23][24]. Voice Arena evaluates both under their current names (#12, #13).

**Capability honesty.** STT-based systems that label words (#8–#13) cannot represent overlapping speech in their output. They will lose overlap-region speaker time by construction. This is a real property of what a buyer receives, so it is scored, not excused, and the "overlap output" column makes it visible.

---

## 5. Fair-comparison rules

### 5.1 Primary track **[Proposed, modelled on DIHARD III Track 2 [5]]**

| Rule | Requirement |
|---|---|
| Input | Audio only: the recording file and nothing else. |
| Speaker count | Not given. Systems must estimate it. Mandatory API range parameters (Google, AWS) use **one** fixed declared range for all files (e.g. min 1, max 10), published on the evaluation card. |
| VAD | No reference voice-activity segmentation. DIHARD III separated "diarization from reference SAD" from "diarization from scratch" for this reason [5]. |
| Tuning | No manual tuning per test recording. One declared configuration per system per test-set version. DIHARD III prohibited manual listening to or annotation of evaluation data before results [5]. |
| Data | Identical audio files for every system. |
| Scoring | Identical reference annotations, UEM scoring regions and pinned scorer version. |
| Version | One declared model version or configuration, frozen before the run. |
| Language hint | Allowed only if the same hint is given to every system that accepts one, and disclosed. It is not a speaker-count hint. |

### 5.2 Separate tracks and conditions

| Track | Who belongs | Ranked? |
|---|---|---|
| **A. Audio-only batch** (primary) | Systems that process a complete file | Yes (headline) |
| **B. Audio-only streaming** | Systems that emit labels while audio arrives in real time (§9.4) | Yes, separately |
| **C. Known speaker count (assisted)** | Any system given the reference count | Separate table; never merged with A or B |
| **D. Oracle-VAD diagnostic** | Systems given reference speech regions | Diagnostic only; not ranked |

Batch and streaming systems are never mixed in one ranking, because streaming systems must decide without future context. A system given the reference speaker count is never compared with one that had to discover it.

### 5.3 Operational rules **[Proposed]**

- **API rate limits.** Runs are paced below each provider's published limits, so throttling does not change results. HTTP 429 responses are retried and logged.
- **Retries.** At most 3 attempts per request, retrying only network errors, HTTP 429 and HTTP 5xx, with exponential backoff from 2 s. Other 4xx errors are not retried. The same policy applies to every API (implemented in `mini_eval/scripts/run_api.py`).
- **Timeouts.** End-to-end limit per file for batch APIs = max(900 s, 3 × audio duration). The mini benchmark's 180 s excerpts therefore use 900 s (`run_api.py --timeout 900`, the default). A timeout is a failure.
- **Failed files.** A missing, invalid or empty output is scored as an **empty hypothesis**: all reference speech counts as missed, so DER is 100% for that file. The failure is also counted in the failure rate. Failed files are never removed from the denominator, so a system cannot improve its DER by failing on hard files.
- **Model warm-up.** Self-hosted systems get one untimed warm-up call. Model loading time is reported separately and excluded from RTF.
- **Hardware.** Self-hosted systems run on one declared reference GPU instance and one declared CPU instance. RTF is reported per hardware class and never compared across classes without a label.
- **Audio preprocessing.** Original audio is preserved. Systems that need 16 kHz mono receive the same deterministic conversion (ffmpeg downmix, resample, 16-bit PCM). No denoising, normalization or VAD is applied by the harness.
- **Channel handling.** Single-channel recordings in the primary track. Multichannel recordings are downmixed by averaging (the pyannote card documents the same behaviour [14]). Telephone recordings with separate caller and agent channels are summed to mono for the primary track, because channel separation would leak speaker identity. A channel-aware diagnostic may be added later.
- **API region.** One declared region per provider, ideally the same geography for all. Region is published because latency and data residency depend on it.
- **Privacy and data retention.** API runs use zero-retention or no-training settings where offered, under a data processing agreement. Hidden-test participants must have consented to third-party processing (§6.5). Vendor retention terms are published on the evaluation card; pyannoteAI, for example, documents a data-retention policy [15].
- **Price measurement.** API cost uses the provider's public list price for the evaluated configuration on the run date, archived as a dated snapshot, with no negotiated discounts. Self-hosted cost is the reference instance's on-demand hourly price × wall-clock processing hours ÷ audio hours. Both are labelled as list-price estimates.

---

## 6. Test-set design

### 6.1 Composition **[Proposed]**

- **Hidden test set:** about **120 hours** (acceptable range 100–150 h), never released.
- **Public development set:** about **25 hours** (range 20–30 h), released with references, mirroring the hidden set's strata.

### 6.2 Primary strata and diagnostic tags

The brief listed seven target slices whose hours overlap (a far-field meeting is also a 3–5 speaker meeting). To prevent double counting, Voice Arena separates two concepts. **[Proposed]**

**Primary strata** are mutually exclusive and exhaustive. Every recording belongs to exactly one, hours sum to the total, and they are the basis of stratified sampling and the default scenario view.

| Stratum | Definition | Hidden test | Dev |
|---|---|---|---|
| S1 Two-party calls | 2 speakers, telephone or VoIP, mono or summed channels | 24 h | 5 h |
| S2 Small meetings | 3–5 speakers, meeting setting (any microphone) | 30 h | 6 h |
| S3 Large meetings | 6–10 speakers, meeting setting | 24 h | 5 h |
| S4 Informal conversations | 2–6 speakers, home or public spaces, non-meeting | 18 h | 4 h |
| S5 Interviews, consultations and media | 2–4 speakers: role-played clinical consultations, interviews, podcasts, online video | 24 h | 5 h |
| **Total** | | **120 h** | **25 h** |

**Diagnostic tags** overlap freely, may be attached to recordings in any stratum, and are used only for slice reporting, never for computing the overall score a second time.

| Tag | Definition (applied by annotators or from collection metadata) | Target hours (hidden) |
|---|---|---|
| far-field | All speakers ≥ 1 m from the recording device, or room/array microphone | 15–20 h |
| noisy | Annotator noise rating ≥ 3 on a 5-point scale, or collected in a declared noisy environment | 10–15 h |
| multilingual / accent / code-switching | Includes consented self-reported Indian-accented English and code-switching (e.g. Hindi–English) | 15–20 h |
| high-overlap | Reference overlap ratio ≥ 15% of speech time; the threshold is calibrated on the dev set (approximately its top quartile) | 10–15 h |
| low-overlap | Reference overlap ratio < 5% of speech time (diagnostic contrast to high-overlap) | Recorded, no target |
| close-talk | Headset, lapel or handset microphone | Recorded, no target |
| speaker-count bins | 2 / 3–5 / 6+, plus a 4+ bin used by the dashboard's "4+ speaker DER" column; all derived from the reference | Derived |

Tag targets are met by *selecting* recordings within strata, not by adding hours. Collection plans are checked against both tables before recording begins.

### 6.3 Per-recording metadata

Each recording stores: recording ID; dataset or source; duration; language(s); consented accent and language metadata (self-reported only); number of speakers (from the reference); device type; channel count; room condition; noise category; overlap ratio (computed from the reference); stratum; tags; annotation quality status (single-pass, double-annotated, adjudicated, audited, or unscorable regions present); and licence/consent record ID.

**Sensitive attributes are never inferred from audio.** Voice Arena does not estimate gender, age, ethnicity, health or accent with classifiers. Only attributes participants declared on a consent form are stored, and they are reported only in aggregates large enough to prevent re-identification.

### 6.4 Feasibility, licensing and annotation compatibility **[verified against sources; targets are proposals]**

The proposed sizes are **targets, not executable today**. The checks below explain why the hidden set must be newly collected and what that costs.

| Source | Size and conditions | Licence / access | Usable for |
|---|---|---|---|
| AMI | About 100 h of meetings with close-talking and far-field devices [1] | Signals and transcripts under CC BY 4.0 [1] | Public dev set and mini benchmark; **not hidden** (public, widely used in training) |
| ICSI | About 70 h of meetings on close-talking microphones [2] | CC BY 4.0 [2] | Public dev set; not hidden |
| VoxConverse | Multi-speaker YouTube clips [7] | CC BY 4.0 for research purposes; video copyright stays with original owners [7] | Dev set and research use. **Commercial redistribution needs legal review** |
| AliMeeting | 118.75 h of Mandarin meetings, 2–4 participants, far-field array and headset [8] | CC BY-SA 4.0 [8]. Share-alike may bind derived annotations | Dev diagnostic (multilingual far-field); not hidden |
| CALLHOME (LDC97S42) | About 56 h from 120 American English telephone conversations, 8 kHz [3] | LDC user agreement, fee-based [3] | **Not redistributable.** Licensed-diagnostic track only. Note that common diarization benchmarks use a different LDC release, LDC2001S97 (cited by the pyannote card [14]), so references are not interchangeable |
| DIHARD III | 11 domains; dev 34.15 h and eval 33.01 h ("full" sets) [5] | Distributed via LDC licence [5] | Licensed-diagnostic track only |
| CHiME-8 DASR | CHiME-6, DiPCo, Mixer 6 and NOTSOFAR-1 scenarios, 2–8 speakers [9] | Per-dataset licences | Far-field diagnostic; licence check per subset |

**Conclusions.**

1. **Public corpora cannot form the hidden test set.** They are public, and open models are commonly trained on them. For example, the pyannote 3.0 card (mirrored copy) lists training on AMI, AliMeeting, DIHARD and VoxConverse among others [28]. Using them for headline ranking would reward memorization.
2. **The hidden set must be newly commissioned** under explicit consent, with ownership that permits sending audio to third-party APIs.
3. **Annotation cost dominates.** The DIHARD III organizers reported that fully manual segmentation to their specification ran at typically more than 15× real time and sometimes more than 30× [5], which led them to adopt careful transcription plus forced alignment for some domains [5]. At 15–30× per pass, two independent passes over 120 h is roughly 3,600–7,200 annotator-hours *before* adjudication and audit. This is Voice Arena's arithmetic on the cited rates. Voice Arena therefore proposes transcription plus forced alignment plus human boundary correction, with a pilot of about 5 h to measure the actual rate before committing to the full size.
4. **Annotation conventions differ between sources.** AMI's public diarization references are derived from word-level annotations, merging adjacent words of a speaker but never across pauses, and exclude vocal sounds by default [26]. DIHARD III splits turns at pauses longer than 200 ms and counts vocal noises such as laughter and breaths as speech [5]. The same system output would receive different DER under the two conventions. Voice Arena therefore (a) annotates its hidden set with one written guideline, and (b) reports any public-corpus results per source, never pooled across conventions.

### 6.5 Consent and privacy for new collection **[Proposed]**

Participants give informed consent covering: evaluation use; processing by named categories of third-party APIs and their regions; retention period; withdrawal; and optional self-reported language and accent. Medical-scribe audio uses **role-played consultations** with trained actors rather than real patient encounters, avoiding protected health information. This is a known limitation: role-play under-represents clinical noise and emotional speech. Personal identifying information spoken in recordings is excluded from scoring via UEM, the same mechanism DIHARD III used for its PII regions [5].

---

## 7. Reference annotation

Reference quality is the foundation of the leaderboard. If the reference marks a backchannel that a strong system detected as missing, or misplaces a boundary by 300 ms, the strict 0 ms-collar DER charges the system for the annotator's error. Systematic annotation errors can re-order rankings, especially between systems with different speech-detection behaviour.

**Procedure [Proposed; guideline elements adapted from DIHARD III [5]]**

1. **Written guideline.** It defines speech (including backchannels and filled pauses), vocal noises (laughter, coughs, breaths, following DIHARD's inclusive definition [5]), pause bridging (pauses ≤ 200 ms do not split a turn [5]), boundary placement precision, and handling of crosstalk.
2. **Two trained annotators work independently** on each file. They mark speech regions, speaker activity, speaker changes, overlap, and *uncertain* regions such as inaudible or ambiguous speech.
3. **Speaker labels remain anonymous** (spk_1, spk_2 …). Names heard in the audio are never used as labels.
4. **Inter-annotator agreement is measured before adjudication.** DER and JER are computed between the two annotators' RTTMs with the same scorer and 0 ms collar, plus boundary-agreement statistics. Files with annotator-versus-annotator DER above a threshold (proposed: 8%) go to priority review.
5. **A senior adjudicator resolves disagreements** and produces the final reference.
6. **At least 10% of files receive a senior quality audit** after adjudication, sampled across strata. Audit failure triggers re-annotation of that annotator's batch.
7. **Unscorable regions** (corrupted audio, PII, untranscribable noise bursts) are excluded via UEM files, as in DIHARD III [5], and are recorded with a reason code.
8. **Hidden references stay private.** They are stored in an access-controlled scoring enclave, never sent to vendors, and never published per file.
9. **Every reference file carries an annotation quality status**, and summary agreement statistics are published per test-set version.

---

## 8. Metrics

### 8.1 Primary metric: strict DER **[Established definition [5][11]; Voice Arena settings proposed]**

$$\text{DER} = \frac{\text{Missed speech} + \text{False alarm speech} + \text{Speaker confusion}}{\text{Total reference speaker time}}$$

DER was introduced for the NIST RT-03S evaluation and is defined in the DIHARD III plan in these terms [5]:

- **Missed speech:** reference speaker time that no system speaker covers. The system did not notice someone talking.
- **False alarm:** system speaker time not attributable to any reference speaker. The system labelled silence, noise or music as speech, or added an extra speaker during a single-speaker region.
- **Speaker confusion:** reference speaker time attributed to the wrong speaker after optimal mapping. The system heard the speech but gave it to the wrong person.
- **Lower is better.** 0% is perfect, and DER can exceed 100% because false alarms are unbounded [11].

The denominator counts *speaker time*. Two people talking for one second contribute two seconds, so overlapping speech is part of the total.

**Primary settings.** Collar 0 ms. Overlap included. No reference speaker count (primary track). RTTM input. UEM scoring regions. Same implementation for every system: dscore at a pinned commit, whose DER uses NIST md-eval with a default 0 ms collar and overlap included [11]. DIHARD III used dscore v1.0.1 with md-eval v22 [5].

**Why strict.** A collar removes a band (commonly ±250 ms) around every reference boundary from scoring. For captioning and transcript alignment, those boundary errors are exactly what users see. Ignoring overlap removes the hardest and most conversation-like regions; on meetings, that flatters systems that never output overlap. DIHARD III adopted both strict choices, departing from earlier NIST RT practice [5]. A 250 ms-collar DER is computed only as a labelled diagnostic.

### 8.2 Metric specification table

| Metric | Formula | Denominator | Units | Overlap treatment | Aggregation | Direction | Status |
|---|---|---|---|---|---|---|---|
| **DER (headline)** | (miss + FA + conf) / ref speaker time | Total reference speaker time in UEM | % | Included; overlap seconds count once per active speaker | **Pooled**: Σ errors ÷ Σ ref time over recordings | Lower better | Established metric; pooled-as-rank proposed |
| Macro DER | mean over recordings of per-recording DER | Number of recordings | % | Included | Unweighted mean | Lower | Proposed secondary |
| JER | 1 − mean over reference speakers of Jaccard(ref spk, mapped sys spk) [5][11] | Number of reference speakers | % | Included, frame-based, no collar [11] | Per-recording JER, then macro mean (dscore also reports an overall value; its pooling is recorded) | Lower | Established |
| Missed speech rate | miss / ref speaker time | Ref speaker time | % | Included | Pooled | Lower | Established component |
| False alarm rate | FA / ref speaker time | Ref speaker time | % | Included | Pooled | Lower | Established component |
| Confusion rate | conf / ref speaker time | Ref speaker time | % | Included | Pooled | Lower | Established component |
| Overlap-region DER | DER on UEM restricted to reference regions with ≥ 2 active speakers | Ref speaker time inside overlap regions | % | Only overlap scored | Pooled | Lower | Proposed slice |
| Non-overlap DER | DER on single-speaker reference regions (md-eval `-1`, the dscore `--ignore_overlaps` behaviour [11]) | Ref speaker time in single-speaker regions | % | Overlap excluded | Pooled | Lower | Established option, used as slice |
| Stratum / tag DER | DER on recordings in the slice | Ref speaker time in the slice | % | Included | Pooled within slice | Lower | Proposed |
| Speaker-count MAE | mean over recordings of \|ŷ − y\| | Number of recordings | speakers | n/a | Mean; a failed file counts as ŷ = 0 | Lower | Proposed diagnostic |
| Boundary error | For each reference turn boundary, distance to nearest system boundary (speaker-agnostic, capped at 5 s) | Number of reference boundaries | seconds | Overlap boundaries included | Median and p95 over pooled boundaries | Lower | **Proposed diagnostic, not a standard** |
| Per-recording distribution | Mean, median, p75, p90 and worst-decile mean of per-recording DER | n/a | % | Included | Distribution | Lower | Proposed reporting |
| RTF | processing seconds / audio seconds | Audio seconds | ratio | n/a | Σ processing ÷ Σ audio | Lower | Established concept |
| Streaming latency | See §9.4 | Events | seconds | n/a | p50, p95 | Lower | Proposed definitions |
| Cost per audio hour | List price or instance cost ÷ audio hours | Audio hours | USD/h | n/a | Total ÷ total | Lower | Proposed |
| Failure rate | failed recordings / attempted recordings (also by hours) | Attempted recordings | % | n/a | Ratio | Lower | Proposed |

### 8.3 Supporting metrics in plain language

**A. JER (Jaccard Error Rate).** For each reference speaker, JER finds the best-matching system speaker (Hungarian mapping) and measures how much their talk regions fail to coincide, then averages *across speakers* [5][11]. Because every speaker counts equally, JER prevents a dominant talker from hiding poor performance on quieter participants. A system that nails the meeting chair but loses two shy participants will look fine on DER and worse on JER. JER is typically higher than DER when one speaker dominates, and unlike DER it never exceeds 100% [11]. Lower is better.

**B. DER components.** Missed, false-alarm and confusion rates diagnose *why* a system fails. High miss suggests weak speech detection or no overlap output. High false alarm suggests noise or music sensitivity. High confusion suggests speaker clustering or counting problems. Two systems with equal DER can need opposite fixes.

**C. Scenario slices.** Overlap and non-overlap DER; telephone (S1); far-field and noisy tags; 3–5 speakers (S2); 4+ speakers (derived bin); 6+ speakers (S3); language and accent tags *when sample sizes are adequate* (§10.4). Slices are what make the leaderboard useful to a specific buyer.

**D. Speaker-count MAE.** The average absolute difference between the number of speakers a system outputs and the true number. Over-counting fragments one person into several; under-counting merges people. Both corrupt per-speaker analytics.

**E. Boundary timing.** Median and 95th-percentile distance between reference turn boundaries and the nearest system boundary. Subtitles need speaker changes to land on the right line, and transcript alignment needs the right words attributed at turn edges. A median of 0.1 s with a p95 of 2 s means most changes are crisp but some are badly late.

**F. Per-recording distribution.** Averages hide catastrophes. A system with 12% mean DER might have 5% DER on 90% of files and 70% on the rest. For a meeting product, that worst decile is the angry-customer decile, so mean, median, p75, p90 and the worst-decile mean are all published.

**G. Operational metrics.** RTF (processing seconds ÷ audio seconds; 0.05 means 20× faster than real time); streaming median and p95 latencies; cost per processed audio hour; failure and timeout rate; output capabilities (word versus segment labels, overlap); batch or streaming; self-hosted or API; hardware; privacy and data handling.

**No hidden weighted score.** DER determines the accuracy rank within a track. Operational metrics are separate columns and filters. A blended "Voice Arena score" would encode Voice Arena's priorities instead of the buyer's.

### 8.4 Experimental metrics, including TTRS and CEWDS **[Proposed policy]**

The brief mentions newly proposed metrics such as **TTRS** and **CEWDS**. This methodology does not define them, and no definitions are assumed here. Any such metric, from Voice Arena or a submitter, is listed in an **Experimental** registry and is **never used for ranking or recommendations** until it passes all of the following:

1. **Published definition and reference implementation:** formula, denominator, units, overlap treatment and aggregation (the same fields as §8.2), with versioned code.
2. **Relationship to established metrics:** rank correlation with DER and JER across systems and recordings on the dev set, plus an analysis of where it disagrees.
3. **Relationship to downstream outcomes:** comparison with speaker-attributed transcription metrics such as cpWER or tcpWER (implemented in MeetEval [13]; CHiME-8 DASR ranks by scenario-averaged tcpWER [9]) and with task metrics such as attribution accuracy of meeting action items.
4. **Human judgment study:** at least 3 raters giving pairwise preferences on output pairs, with inter-rater agreement reported. The metric must predict human preference at least as well as DER.
5. **Statistical stability:** bootstrap CI width comparable to DER at the same sample size.
6. **Pre-registered acceptance thresholds**, decided before the validation study runs.

Until then the dashboard shows experimental metrics only in a clearly labelled tab, if at all.

---

## 9. Scoring protocol

### 9.1 Batch protocol **[Proposed unless cited]**

1. **Audio input preserved.** Originals are archived with SHA-256 hashes. If resampling is required, the same documented ffmpeg conversion (mono, 16 kHz, 16-bit PCM) is used for all systems.
2. **Predictions normalized to RTTM.** Word-labelled STT output is converted by bridging same-speaker gaps of ≤ 0.2 s, mirroring the DIHARD 200 ms pause convention [5]. Same-speaker overlapping segments are merged. No other post-processing is applied.
3. **Validation before scoring.** File IDs must match recording IDs exactly. Segments must start at or after 0 and end no later than the audio duration plus 0.5 s. Durations must be positive. A prediction for an unknown recording ID is an error. Invalid outputs are counted as failures and scored as empty hypotheses, never silently removed.
4. **Speaker mapping is done by the scorer** (optimal mapping), never by hand.
5. **Primary collar 0 ms; overlap included; UEM supplied** (whole recording minus excluded regions, as in DIHARD III [5]).
6. **Primary track provides no speaker count.**
7. **Diagnostics:** 250 ms collar DER and non-overlap DER are reported separately and never replace the headline.
8. **Same scorer for all systems.** dscore at a pinned commit, with Python and package versions recorded. pyannote.metrics [12] is a capable alternative but is not mixed with dscore: one scorer per test-set version. Components and slice DERs come from the md-eval script bundled in the same checkout; its recomputed DER must match dscore's within 0.05 pp or the file is flagged.
9. **API retries and timeouts** follow §5.3, fixed before execution.

### 9.2 Micro versus macro averaging

- **Micro (pooled) DER:** Σ over recordings of (miss + FA + conf) ÷ Σ over recordings of reference speaker time. Long, talkative recordings weigh more. This is what "DER on the test set" means in the dscore OVERALL row and in md-eval's ALL block [11].
- **Macro DER:** the mean of per-recording DERs. Every recording counts equally, so short files with high DER pull it up.

They diverge most when failures or difficult files are short. Worked example (arithmetic only, encoded as a unit test in `mini_eval/tests/`): recording 1 has 280 s of reference speaker time and 55 s of error (DER 19.6%); recording 2 is a failed file with 20 s of reference speaker time, scored as empty (DER 100%). Pooled DER = 75 ÷ 300 = 25.0%, while macro DER = (19.6 + 100) ÷ 2 = 59.8%. One short failure moves the macro average far more than the pooled one, which is why both are shown.

### 9.3 Official ranking aggregation rule **[Proposed]**

> **Within each track and speaker-count condition, systems are ordered by pooled, duration-weighted DER over every recording of the frozen hidden test-set version, with failed files scored as empty hypotheses. Macro-average recording DER is shown alongside as the secondary measure. Ordering is then grouped into statistical tiers (§10.3).**

Pooled DER depends on the test-set mix: more meeting hours means meetings weigh more. That is why stratum hours are fixed per test-set version (§6.2) and every stratum's pooled DER is shown side by side. DIHARD III handled the same tension by publishing results on a balanced "core" set and an unbalanced "full" set [5]. CHiME-8 DASR instead macro-averages across scenarios [9]. Voice Arena keeps pooled DER as the headline because it answers the question "what fraction of the speaker time I process will be mis-attributed", and shows strata separately for buyers who care about one scenario.

### 9.4 Streaming evaluation **[Proposed definitions]**

Streaming systems are evaluated by a harness that **replays audio in real time**.

- **Input packetization:** 16 kHz mono 16-bit PCM sent in fixed **100 ms packets** at 1× real-time pace. If a system's API requires a different packet size, it uses its required size and the size is published. A system's own internal buffering (e.g. Streaming Sortformer's chunk plus right context, configured in 80 ms frames [17]) is part of the system, and its declared preset is published.
- **Event logging:** every emitted label or segment event is logged with (a) the *media time* it refers to and (b) the *wall-clock arrival time* at the client, along with whether the event is final or provisional.
- **Final DER:** computed with the batch scorer on the final committed output at stream end. Ranks are by final DER within the streaming track.
- **First-emission DER:** DER of labels as first emitted, before any revision. This shows what a live caption viewer actually saw.
- **Label revision rate:** the fraction of speaker time whose label changed after first emission. Revisions improve final accuracy but cause visible flicker.
- **Speaker-switch delay:** for each reference speaker change at media time *t*, the wall-clock time from the moment the packet containing *t* was sent to the first emitted event that assigns the region after *t* to the correct mapped speaker. Mapping comes from the optimal mapping on final output. Reported as p50 and p95. Changes never correctly reflected within 10 s are counted as misses and reported separately.
- **End-of-turn latency:** for each reference turn end, wall-clock time from sending the packet containing the end to receiving the system's end-of-segment event. Reported as p50 and p95.
- **Network overhead:** for APIs, headline latency is **end-to-end and includes network**, measured from a declared client region with round-trip time to the endpoint recorded. Self-hosted streaming runs on reference hardware with the client on the same host (loopback), reported as compute latency. The two are labelled and never presented as the same thing.
- **Vendor latency claims** (for example, pyannoteAI's sub-300 ms claim for Live-1 [15], or Sortformer's preset input-buffer latencies [17]) are shown only as vendor-reported, never as Voice Arena measurements.

---

## 10. Statistical confidence and ranking

### 10.1 Paired recording-level bootstrap **[Established technique; constants proposed]**

The unit of resampling is the **recording**, because errors within a recording are correlated. Comparisons are **paired**: both systems are scored on the same resampled recordings.

| Constant | Value |
|---|---|
| Replicates (B) | 10,000 |
| Random seed | 20261015 (numpy `default_rng`) |
| Interval | 95% percentile interval |
| Statistic | Difference in pooled DER (ratio of sums recomputed on each resample, not a mean of ratios) |
| p-value | Two-sided bootstrap: 2 × min(P*(Δ ≤ 0), P*(Δ ≥ 0)), with +1 smoothing |

Procedure:

1. Keep per-recording error seconds for systems A and B and the shared reference speaker time, aligned by recording ID.
2. Resample recording indices with replacement, using the same draw for A and B.
3. Recompute pooled DER for each system and store Δ = DER_A − DER_B.
4. Report the 95% interval of Δ and the p-value.

Implemented in `mini_eval/scripts/bootstrap.py` and unit-tested.

### 10.2 Multiple comparisons, and statistical versus practical significance

- **Comparison family:** all pairwise comparisons within one track and speaker-count condition for one test-set version.
- **Correction:** Holm–Bonferroni on the bootstrap p-values across the family.
- **Statistical significance** means Holm-adjusted p < 0.05: the difference is unlikely to be resampling noise.
- **Practical significance** means |Δ| ≥ **1.0 absolute DER percentage point**, a proposed threshold reflecting a difference users could notice. It is a policy choice, not a statistical fact.
- **Verdicts:** *practically different* (both conditions met); *statistically different only* (significant but below 1 pp, shown as "differences too small to matter"); *not distinguishable*; *insufficient sample*.

### 10.3 Tiers

1. Sort systems by pooled DER.
2. Tier 1 contains the best system plus every system whose comparison with it is not "practically different".
3. Remove tier 1 and repeat with the best remaining system.
4. Systems in one tier share a rank (displayed as "1=").

The dashboard shows each system's DER with its own 95% bootstrap interval, the tier, the number of recordings and hours, and the number of failed files.

### 10.4 Minimum sample sizes **[Proposed]**

- No confidence interval or verdict is published for fewer than **20 recordings**.
- A slice is shown with a "low confidence" label at 20–29 recordings and normally at 30 or more. With fewer than 20 recordings it is shown only as "insufficient sample".
- **No definitive claim is ever made from a handful of recordings.** The three-recording mini benchmark (§15) cannot support any ranking, and its CI fields are deliberately set to "not reported".

---

## 11. Evaluation process

1. **Define evaluation tracks:** batch, streaming, assisted and diagnostic, plus the speaker-count conditions.
2. **Prepare development and hidden test data:** collection, consent, annotation, adjudication and audit (§6–7).
3. **Freeze test-set metadata and annotations:** version tag, file hashes, strata and tags.
4. **Release the scoring kit:** pinned scorer commit, validation scripts, dev references, RTTM and UEM specifications.
5. **Collect model declarations:** system card covering identity, version, configuration, track, hardware and data-handling terms.
6. **Freeze model versions and configurations:** pinned commits, container digests and API version strings, with no `latest` aliases.
7. **Run all systems under controlled conditions:** self-hosted in network-isolated containers on reference hardware; APIs from a declared region with the fixed retry and timeout policy.
8. **Normalize outputs into RTTM** using only the documented normalization.
9. **Validate timestamps and file IDs;** failures are recorded, not fixed by hand.
10. **Score with the pinned scorer:** collar 0, overlap included, UEM.
11. **Compute overall and slice metrics:** strata, tags, components, distributions.
12. **Measure runtime, latency, cost and failures.**
13. **Run bootstrap analysis:** pairwise comparisons, Holm correction, tiers.
14. **Human-review anomalous results,** for example DER above 100%, sudden version-to-version jumps, or a component profile that contradicts the system's capabilities. Review checks the *pipeline* (data, format, scorer), never re-tunes the system.
15. **Approve results:** a second evaluator signs off on the evaluation cards.
16. **Publish the leaderboard and evaluation cards,** with a 14-day vendor fact-check window for errors in system metadata, not in scores.
17. **Archive previous versions and schedule refreshes:** quarterly for API systems (drift check), and an annual hidden-set partial refresh (§13).

---

## 12. What is published next to each score (evaluation card)

Model/company · exact model or API version (commit, build or UUID) · run date · evaluation track · test-set version · audio preprocessing · speaker-count condition (and declared range, if the API required one) · collar · overlap policy · pooled DER and macro DER · JER · DER components · slice scores (strata and tags) with recording counts · 95% CI and tier · number of recordings and hours · failure rate and failure reasons · runtime and RTF (with hardware class) · streaming latencies (if streaming) · cost assumptions and price snapshot date · hardware and API region · privacy and deployment notes (retention, training opt-out, self-hosting) · link to the system card and submitted configuration · scorer commit.

---

## 13. Integrity safeguards **[Proposed]**

| Risk | Safeguard |
|---|---|
| **Test-set leakage via APIs.** Hidden audio must be sent to vendors to evaluate them | Zero-retention / no-training settings and data processing agreements; consent covers third-party processing; references never leave the scoring enclave; per-file scores on the hidden set are never released; 20–25% of the hidden set is refreshed annually, with retired files moving to the public dev set |
| **Leakage via public corpora.** Models trained on AMI, VoxConverse and similar | Hidden set newly commissioned; public corpora used only for dev and diagnostics, labelled as possibly seen in training [28] |
| **Repeated-submission overfitting** | At most one scored hidden-set run per system per 30 days; only aggregate and slice scores returned, never per-file; scores published to 1 decimal place; the dev set is the place for iteration. DIHARD III allowed up to 50 submissions per track in a research setting [5]; a buyer-facing board is stricter |
| **Unauthorized network access** by self-hosted systems | Containers run with network egress disabled after model files are staged; image digest recorded; egress attempts logged and flagged |
| **Model-version drift** (hosted APIs change silently) | Pinned version parameters where offered (e.g. Deepgram `diarize_model=v2` rather than `latest` [20]; explicit `model: "precision-3"` [15]); response metadata archived; a **sentinel subset** (about 5% of hidden recordings) is re-run monthly. Output-hash changes or pooled-DER shifts above 1 pp mark the entry "drift detected – re-evaluation pending" |
| **Per-file tuning or human-in-the-loop** | Declared config only; no manual inspection of hidden data (as DIHARD III required [5]); Voice Arena runs systems itself wherever possible |
| **Selective reporting** | Every attempted run, including failures and withdrawn submissions, is logged in the archive |

---

## 14. Use-case recommendations and limitations

### 14.1 When the dashboard recommends **[Proposed]**

A use-case view (for example "contact centre: two-party calls, API, data retention ≤ 30 days") shows a recommendation only if all of these hold:

1. The data is measured on the hidden set, not illustrative and not mini benchmark data.
2. The relevant slice has ≥ 30 recordings for every candidate.
3. The leader is "practically different" from the runner-up on the slice's pooled DER (Holm-adjusted, ≥ 1 pp).
4. No candidate's failure rate exceeds 2% on the slice.

Otherwise the view states **"Evidence is insufficient to identify a winner"**, lists the tier-1 candidates, and discloses the accuracy, cost, latency, privacy and deployment trade-offs among them. In every case the view shows trade-offs rather than a single answer, because a buyer bound by on-premises rules should never be pointed to an API.

### 14.2 Limitations

DER treats every second equally, but users do not: a confusion during a key decision matters more than one during small talk. The hidden set reflects Voice Arena's chosen mix, which will not match any single buyer's traffic; strata views mitigate this but do not eliminate it. Role-played clinical audio under-represents real clinical acoustics. Self-reported accent and language metadata will be sparse for some groups, so many language slices will remain "insufficient sample" in early versions. Commercial APIs can change between runs despite pinning. List prices ignore volume discounts and self-hosting labour. Word-level STT outputs are disadvantaged on overlap by construction; that reflects the product a buyer receives, but it is not a statement about the vendor's underlying diarizer. Finally, a strict 0 ms collar makes DER sensitive to annotation boundary precision, so annotation quality bounds measurement quality.

---

## 15. Optional mini benchmark: status and design

**Status: run on 2026-10-18 (provisional).** Two systems were run on 5 AMI meeting excerpts on a Windows 11 laptop (CPU inference) and scored with dscore commit `e02f949…`, at 0 ms collar with overlap included. Pooled DER was **22.2% for pyannote Community-1 and 47.0% for Deepgram nova-3** (diarizer v2). pyannote had the lower DER on all 5 excerpts.

The 5 VoxConverse excerpts were selected by the rules below but could not be downloaded (HTTP 500 from the dataset server), and NeMo was not attempted. Full results are in `docs/benchmark_report.md` and `mini_eval/results/`. The earlier build environment had no network, GPU or credentials, so all code was first tested on synthetic data. The real run then exposed eight bugs, all fixed (README, "Bugs found only by the real run").

**Design** (details in `mini_eval/README.md`):

- **Recordings** (10 fixed 180 s windows chosen before any run; manifest in `mini_eval/configs/recordings.json`):
  - Five AMI excerpts (300–480 s): ES2004a close-talk and far-field (*same meeting and window*, a controlled contrast), IS1009a far-field, TS3003a close-talk, EN2002a far-field.
  - Five VoxConverse v0.3 test files (0–180 s), chosen by rules applied to the reference annotations inside the window, each file used once: exactly 2 speakers; 3–5 speakers; 6 or more; highest overlap ratio; most speaker changes per minute. Chosen IDs, selection statistics and repository commits are recorded at preparation time.
  - Condition tags come from dataset documentation (recording type, microphone) or the cropped reference (speaker count, overlap level). Audio quality stays "unknown" without a documented listening check; telephone audio is not represented.
  - References: AMI "only_words" RTTMs [26] and VoxConverse RTTMs [7], cropped to the window. Edge-truncated turns are a documented artefact.
- **Systems:**
  - pyannote Community-1, local, speaker count unknown; an optional assisted run goes to a separate directory and track.
  - NeMo MSDD with TitaNet-L, with the resolved YAML archived.
  - One commercial API: Deepgram (`nova-3`, `diarize_model=v2`), AssemblyAI or pyannoteAI Precision-3, chosen at run time.
- **Scoring:** dscore (DER, JER) plus its bundled md-eval (components, overlap and non-overlap DER), collar 0, overlap included, UEM. Failed files are scored as empty hypotheses.
- **Outputs:** `summary.csv`, `per_recording.csv`, `error_breakdown.csv`, `system_configs.json`, `pairwise_bootstrap.csv` (which will read "insufficient_sample" with n = 10), plus generated per-recording summaries (`recording_summaries.json`) and `docs/benchmark_report.md`.

**Planned analysis** (to be written only after a real run):

- Which system had the lowest pooled DER on these excerpts.
- Whether JER ordered the systems the same way.
- Which component dominated per recording.
- How much DER changed between the AMI close-talk and far-field versions of the same audio.
- Whether RTF or cost changes the practical choice.

**What it cannot show.** The 5 excerpts actually run total 15 minutes from one corpus (AMI meetings with 3–4 speakers), below the 20-recording minimum for any confidence interval (§10.4). They include no telephone, media or 6+ speaker audio. **The mini benchmark cannot support a statistically reliable global ranking**, and its results must not be read as one. Once it has actually been run, its value is to demonstrate the pipeline on real audio and to illustrate failure modes on recognizable recordings. Until then it is an implemented pipeline whose components are unit-tested only on synthetic data; no model inference or dscore scoring has been executed.

---

## 16. Final reflection

**1. Which design choice am I least sure about, and why?**
The exact composition of the hidden set: the split across strata and the target hours for languages, accents, devices and overlap. Representativeness pulls toward many languages, devices and acoustic conditions, each with enough recordings for a confidence interval. Annotation cost pulls the other way: at the 15–30× real-time rates DIHARD III reported for careful manual segmentation [5], every extra slice is expensive. I chose 120 h with fixed strata and tagged diagnostics, but the right balance should be revisited after the annotation pilot and after seeing which slices buyers actually filter on.

**2. Which benchmark influenced me most, and how?**
DIHARD III [4]. It shaped the core stance:

- Difficult multi-domain audio, with 11 domains including clinical, restaurant, web video and telephone [5][6].
- **No collar and overlap scored** [5].
- A clean split between diarization from scratch and from reference speech activity [5], which became Voice Arena's track structure.
- Explicit handling of PII regions through UEM [5].
- Honesty about annotation cost [5].
- Its balanced-core versus full-set reporting [5] informed the decision to fix stratum hours and show strata beside the pooled headline.

**3. What could make the leaderboard misleading, and how did I guard against it?**

| Risk | Guardrail |
|---|---|
| One average hiding slice failures | Strata and tag slices; per-recording distributions |
| Small samples | Recording-level paired bootstrap; minimum-n rules; tiers instead of strict ranks; no claims from the mini benchmark |
| Changing APIs | Version pinning; archived metadata; sentinel re-runs with drift flags |
| Different speaker-count conditions | Separate assisted track; declared fixed ranges for APIs that require one |
| Mixing batch and streaming | Separate tracks with their own latency definitions |
| Public test-set overfitting | Newly collected hidden set; submission limits; aggregate-only feedback; annual partial refresh |
| Gaps in context around each number | Evaluation cards with every setting; explicit limitations; an "insufficient evidence" state in place of forced winners |

**4. What would I improve with more time?**

- More languages and recording devices, especially consented Indian-language code-switching and low-cost phone microphones.
- A larger private test set, to support language and accent slices with real confidence intervals.
- More streaming evaluations, with a validated latency harness across vendors.
- Downstream metrics: speaker-attributed WER (cpWER/tcpWER via MeetEval [13]), action-item attribution for meeting notes, and talk-ratio error for call analytics.
- Better privacy measurement (verified retention behaviour, not just policy text) and better cost measurement (volume tiers, self-hosting total cost of ownership).
- Continuous benchmark refresh.
- A human-impact analysis of who is harmed when diarization fails, for example quieter or accented speakers losing attribution in meeting minutes, measured with consented demographic metadata.

---

## 17. References

All URLs accessed October 2026. Vendor documentation changes frequently, so re-verify on the execution date.

1. AMI Meeting Corpus, University of Edinburgh. https://groups.inf.ed.ac.uk/ami/corpus/
2. ICSI Meeting Corpus (Edinburgh distribution). https://groups.inf.ed.ac.uk/ami/icsi/
3. CALLHOME American English Speech, LDC97S42, Linguistic Data Consortium. https://catalog.ldc.upenn.edu/LDC97S42
4. The Third DIHARD Speech Diarization Challenge (website). https://dihardchallenge.github.io/dihard3/
5. N. Ryant et al., *Third DIHARD Challenge Evaluation Plan*, v1.2, 30 Oct 2020. https://dihardchallenge.github.io/dihard3/docs/third_dihard_eval_plan_v1.2.pdf
6. N. Ryant et al., "The Third DIHARD Diarization Challenge," arXiv:2012.01477. https://arxiv.org/abs/2012.01477
7. VoxConverse dataset repository (v0.3). https://github.com/joonson/voxconverse
8. AliMeeting, OpenSLR SLR119. https://www.openslr.org/119
9. CHiME-8 Task 1 (DASR). https://www.chimechallenge.org/challenges/chime8/task1/index
10. T. J. Park et al., "A Review of Speaker Diarization: Recent Advances with Deep Learning," arXiv:2101.09624. https://arxiv.org/abs/2101.09624
11. N. Ryant, dscore: diarization scoring tools. https://github.com/nryant/dscore
12. pyannote.metrics documentation (alternative implementation; not used for Voice Arena scoring). https://pyannote.github.io/pyannote-metrics/
13. MeetEval: meeting transcription evaluation toolkit. https://github.com/fgnt/meeteval
14. pyannote/speaker-diarization-community-1 model card. https://huggingface.co/pyannote/speaker-diarization-community-1
15. pyannoteAI documentation, "Models". https://docs.pyannote.ai/models.md (index: https://docs.pyannote.ai/llms.txt)
16. NVIDIA NeMo, speaker diarization configuration documentation. https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html
17. nvidia/diar_streaming_sortformer_4spk-v2.1 model card. https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
18. "Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering," arXiv:2507.18446. https://arxiv.org/abs/2507.18446 (4-speaker limit also stated in NVIDIA NeMo Curator docs: https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-filtering/speaker-separation)
19. AssemblyAI, "Speaker Diarization." https://www.assemblyai.com/docs/pre-recorded-audio/label-speakers
20. Deepgram, "Speaker Diarization." https://developers.deepgram.com/docs/diarization
21. Google Cloud, "Detect different speakers in an audio recording." https://docs.cloud.google.com/speech-to-text/docs/multiple-voices
22. AWS SDK for Kotlin reference, Amazon Transcribe `Settings.showSpeakerLabels`. https://docs.aws.amazon.com/sdk-for-kotlin/api/latest/transcribe/aws.sdk.kotlin.services.transcribe.model/-settings/show-speaker-labels.html
23. Microsoft Learn, "Quickstart: Create real-time diarization." https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization
24. Azure Speech, "Use the fast transcription API" (Azure China mirror of the Microsoft Learn page). https://docs.azure.cn/en-us/ai-services/speech-service/fast-transcription-create
25. Microsoft Q&A, answer noting retirement of conversation-transcription multichannel diarization (preview) on 28 March 2025. https://learn.microsoft.com/en-us/answers/questions/2085176/has-diarization-in-speech-sdk-been-implemented-for
26. AMI diarization setup (BUT Speech@FIT; pyannote fork). https://github.com/pyannote/AMI-diarization-setup
27. SpeechBrain AMI diarization recipe. https://github.com/speechbrain/speechbrain/tree/develop/recipes/AMI/Diarization (verify the path at execution time)
28. pyannote speaker-diarization-3.0 model card, mirrored copy listing training corpora. https://huggingface.co/statsmaths/diarize

---

## Appendix A. Established standards versus Voice Arena proposals

| Element | Established (source) | Voice Arena proposal |
|---|---|---|
| DER definition, RTTM, UEM | DIHARD III plan [5]; dscore [11] | — |
| 0 ms collar, overlap scored | DIHARD III [5]; dscore defaults [11] | Adopted as headline |
| JER | DIHARD II/III [5]; dscore [11] | Adopted as supporting |
| Scratch versus reference-SAD tracks | DIHARD III [5] | Extended to batch, streaming, assisted and oracle-VAD tracks |
| Pooled DER as official rank; macro secondary | — | **Proposed** |
| Strata versus tags; 120 h / 25 h sizes | — | **Proposed** (targets; feasibility checked in §6.4) |
| Failure = empty hypothesis | — | **Proposed** |
| Bootstrap constants, Holm correction, 1 pp threshold, tiers, min-n rules | Bootstrap and Holm are standard statistics | **Proposed** constants and policy |
| Streaming latency definitions | — | **Proposed** |
| Boundary error, speaker-count MAE | — | **Proposed diagnostics** |
| TTRS, CEWDS, other new metrics | — | **Experimental only** (§8.4) |
