# Voice Arena: two-minute walkthrough script

Target length about 2:00. On-screen actions are in the right-hand column; every one exists in `dashboard/index.html`.

| Time | Voice-over | On screen |
|---|---|---|
| **0:00–0:20** | "Voice Arena is a leaderboard for speaker diarization: working out who spoke when in a recording. It's for teams building meeting notes, call analytics, captions and voice agents, who need to know which system fits *their* audio, not just which one tops a list." | Dashboard header: name, status line ("5 recordings · Measured · provisional"), Customize Ranking and Upload Audio buttons. |
| **0:20–0:50** | "The full benchmark is designed around a hidden, newly recorded test set covering calls, meetings of different sizes, informal conversation and media. This mini benchmark ran on five public AMI meeting excerpts, including the same meeting recorded on a close and a distant microphone. Five VoxConverse videos were also selected, but their server was down, so they are not included." | Scroll to Recordings; click a recording to open "What happened in this recording?" and point at the documented conditions. |
| **0:50–1:20** | "Systems are ranked by strict DER: missed speech, false alarms and wrong-speaker time over total speaker time, with no collar and overlap included. JER, the three error components, overlap DER, speaker-count error, boundary timing, runtime and failures sit beside it. Hover any heading for its formula." | Hover the DER heading, then tab to the JER heading to show the tooltip by keyboard. Open one system's Details. |
| **1:20–1:45** | "Customize Ranking reranks systems using only the recordings that match your conditions. Unknown conditions never match, and with fewer than three scored recordings the page says the evidence is insufficient instead of naming a winner. Every system gets the same audio, no speaker count, a frozen version and the same scorer." | Click Customize Ranking, choose Microphone: Far-field, Apply to show the reranked three-recording table, then choose 6+ speakers to show the insufficient-evidence message, then Back to overall ranking. |
| **1:45–2:00** | "Results: pyannote scored 22 percent DER, Deepgram 47, mostly missed speech. On the distant microphone Deepgram recognised 85 percent fewer words. Five recordings is a provisional result, not a winner." | Leaderboard with the two measured rows; open the ES2004a far-field recording and point at the documented inspection. |

**Spoken length:** 231 words, roughly 1:32–1:46 at a natural pace (150–130 words per minute).

## Shot list

1. **Dashboard overview:** header, status line, the two measured rows, and NeMo marked "Not run".
2. **Filter interaction:** Customize Ranking → Microphone distance: Far-field → Apply (3 recordings; pyannote 25.0%, Deepgram 55.2%) → add 6+ speakers (insufficient evidence) → Back to overall ranking.
3. **Metric tooltip:** hover "DER ↓"; then keyboard-focus "JER ↓" to show the tooltip without a mouse.
4. **System details:** click Details on pyannote to show model revision, packages, CPU hardware and scoring settings.
5. **Recording view:** click ami_ES2004a_sdm; show the per-system metrics, the measured statements and the documented inspection.
6. **Methodology:** open `docs/methodology.md` from the footer link (strata table in §6.2).
7. **Benchmark report:** open `docs/benchmark_report.md` from the footer link and show the cross-recording findings.

**Recording notes.** Say explicitly that the result is provisional: 5 recordings, one corpus, no confidence interval. Do not call pyannote the better system in general.
