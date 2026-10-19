# Voice Arena: Speaker Diarization Evaluation (one-page report)

**Problem.** Buyers of diarization ("who spoke when") face vendor claims measured on different data and settings. Difficulty varies sharply by audio: DIHARD III found two-party conversation much improved but web video far from solved ([paper](https://arxiv.org/abs/2012.01477)). One average hides the slice a buyer needs.

**Users.** Teams building meeting notes, contact-centre analytics, medical scribes, captions and voice agents, plus researchers.

**Test set (proposed).** A newly commissioned, consented **hidden set of ~120 h** and a **public dev set of ~25 h**. The hidden set has five mutually exclusive strata: two-party calls, 3–5 and 6–10 speaker meetings, informal conversations, and interviews and media. Overlapping tags (far-field, noisy, multilingual and accent, high-overlap) are reported as slices without double-counting hours. Public corpora (AMI, VoxConverse, …) are used only for dev and diagnostics, because models may have trained on them. References are double-annotated, adjudicated, and at least 10% audited.

**Main metric.** **Strict DER**: (missed + false alarm + confusion) ÷ reference speaker time, with a 0 ms collar, overlap scored, no reference speaker count, and one pinned scorer (dscore). The official rank uses **pooled, duration-weighted DER**; macro DER is shown alongside. Failed files count as 100% missed.

**Supporting metrics.** JER; DER components; overlap and stratum slices; per-recording spread; speaker-count error and boundary timing (proposed diagnostics); RTF; streaming latency; cost per hour; failure rate. There is no blended score.

**Fairness.** Batch, streaming and known-speaker-count tracks are never ranked together. Every system gets the same audio, references and scorer, runs a frozen version with no per-file tuning, and follows a fixed retry and timeout policy.

**Statistics.** A paired recording-level bootstrap (10,000 resamples, fixed seed) with Holm correction. A difference counts only if it is significant *and* at least 1.0 DER point; inseparable systems share a tier ("1="). No interval is shown below 20 recordings.

**Dashboard.** A simplified leaderboard with every metric and its formula in a tooltip, a "Customize Ranking" panel that reranks systems from matching recordings only (or says the evidence is insufficient), per-recording summaries, and an upload flow backed by a small local server. It shows only real benchmark output.

**Mini benchmark: run (provisional).** On 5 AMI meeting excerpts, pooled DER was **22.2% for pyannote Community-1** (local, CPU) and **47.0% for Deepgram nova-3**, with missed speech dominating both. On the far-field copy of one meeting, Deepgram recognised 85% fewer words in audio about 12 dB quieter and missed 91% of the speech. VoxConverse (server error) and NeMo were not run. Five recordings support no confidence interval.

**Key design choices.** (1) Strict, pooled DER with failures counted. (2) Strata and slices instead of one average. (3) Statistical tiers with a practical threshold and an explicit "insufficient evidence" state.

**Takeaway.** No diarizer is best everywhere. Voice Arena should show which systems are indistinguishable *on a buyer's kind of audio*, and their cost, latency and privacy trade-offs.

**Limitations.** DER weighs every second equally. The test mix will not match any one buyer's traffic. Role-played clinical audio is unrealistic. Accent slices start small. APIs drift despite pinning, and list prices ignore discounts. Word-level STT cannot represent overlap.

**Next steps.** Add VoxConverse and NeMo. Pilot ~5 h of annotation (DIHARD III reported 15–30× real time). Then commission the hidden set.
