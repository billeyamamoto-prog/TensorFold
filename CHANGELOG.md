# What's new in TensorFold

Each release's page on GitHub has its notes and measurements. See [the 1.0.0 release notes](RELEASE-NOTES-1.0.0.md) for the native binary's supported models and migration details.

## Unreleased

- Flash Next on CUDA runs the routed NVFP4 experts of prompt chunks of 1,024 rows or more (the NVIDIA FP4 checkpoint) on shared-memory row tiles of 64 pairs, `nvfp4_experts_tile.cu`, with the decode kernel's bytes: a pair's sums run in the same order, so output does not depend on the kernel or the plan's tile. `TF_FLASHNEXT_PROMPT_EXPERT_TILE=2` (the default at one rank) takes gate/up and down, `1` gate/up only, `0` keeps the decode kernel on 16-pair items; `_CFG=pairs,gate_up,down` and `_ROWS` tune it. `tf-cuda-test nvfp4-experts-tile` compares the bytes and times a layer on a GPU. `tf-cuda-test` and `install` build again: `w4a8_tests.zig` passes the quantizer flag `Kernels.prompt` takes, and `contract_controls.cu` includes `act_round.cuh` before it uses it.

## 1.2.8

- GLM-5.3-Flash on CUDA: a tensor-parallel abort reaches the ranks' communicator from the engine's start, once, and a start-up failure after an abort no longer destroys the communicator again; serving and replies are unchanged (#2918, part of #2560). Thanks to @laudney.

- Metal: MiniMax-M3, Spark2.5 and Llama take their NaN-aware bf16 argmax from one shared source; the compiled code is unchanged (#2949, part of #2251). Thanks to @jschmied.

- Metal: the image towers of GLM, Step-5, Gemma 4 and Qwen share one source for bf16 rounding and threadgroup sum and max; GLM's, Step-5's and Gemma 4's row softmax share one body, and Step-5's, Qwen's and Gemma 4's LayerNorm statistics and output share one (GLM's, which rounds twice, keeps its own); the compiled code is unchanged (#2948, part of #2251). Thanks to @jschmied.

- Tests: the dashboard page's test runs on every OS, so Linux CI covers it, and no longer expects a paged prefix cache tile the page never had; `zig build test` passes on Macs again (#2946).

- Video input: libtfvideo decodes MPEG-4 Part 2 and H.264 streams that declare no B-frames on up to four frame threads, 2.1x to 3.3x faster on an M5 Max and 2.5x to 3.4x on an x86_64 Threadripper 3960X (the decoder alone), with every intact frame equal to before; on damaged clips the return codes, frame counts and frames before the damage equal before, and frames after it stay within main's own run-to-run spread. H.264 with B-frames keeps main's slice-threaded path, because frame threads conceal damage there differently, and a stream that contradicts its own headers falls back to it (#2863). Thanks to @swbiggart.
- Video on Linux: `libtfvideo.so` links with a version script that exports only its four `tf_video_*` functions, so on x86_64 `zig build video` links `tools/zig/build_ffmpeg.sh`'s FFmpeg with its assembly, where before the link stopped on thousands of `R_X86_64_PC32` relocations against FFmpeg's own tables, and a CI check (`check-video-exports-linux`) refuses a build that exports more. Its frames equal PyAV 19.0.1's byte for byte on 23 clips. On aarch64 the library stops exporting FFmpeg's API; real video there is unchecked. macOS is unchanged (#2861). Thanks to @swbiggart.
- Qwen 3.5, 3.6 and 3.8 on CUDA: decode windows of 2 to 8 rows run at 4 or 8 rows instead of being padded to 16, about 1.5% faster at four streams on a GB10 with Qwen3.8-27B (MLX 4-bit), with the same tokens (#2934, fixes #2909).

- Server: `POST /v1/requests/{id}/cancel` cancels a running or queued request by its `chatcmpl-`, `cmpl-` or `resp_` id through the same engine cancel a disconnect sends; the reply ends `cancelled` with the tokens it had, and other streams are unchanged. With API keys only a full or `[admin]` key may cancel; a server without keys answers only direct loopback callers and refuses proxied requests (403 `loopback_only`) (#3193, fixes #3172).

- Tests: the Responses cancel test waits for the turn's first token before it cancels, as the chat cancel test already does, so a slow runner no longer fails it on an empty output; the server is unchanged (#3255).

- Qwen 3.5, 3.6 and 3.8 on CUDA: a served one-row step no longer copies the recurrent state for a rollback that can't happen, so decode is 1-2% faster at one stream and about 4% at four on a GB10 with Qwen3.8-27B (MLX 4-bit), with the same tokens (#2933, fixes #2908).

- Metal: Recursive's Flint (40B-A2.5B MoE) is served natively from `RecursiveSI/flint-int4` on M5-class Macs, with its kernels and 2048-wide router under the kernel contract, a 262,144-token window, copy drafts and the prompt cache; `FLINT_LOAD_LIMIT_GB` caps its memory, and Step-5 now shares the same load budget, so a `STEP5_LOAD_LIMIT_GB` above the Mac's RAM is refused by name (#1177, fixes #1175). Thanks to @BobClawblaw.

- CI: a newer push to main cancels main's older No GPU run (a landing train pushes one commit a PR, and the train's own run has tested the tree), the nightly keeps its own run, and the app checks run on pushes to main only (#3230).

- Intel XPU: Qwen3.8-27B's prompt `gdn_out` kernel keeps its scale bookkeeping in cheaper exact forms, for 22.6% less kernel time on an Arc Pro B70 (736 to 570 ms over a 32K prompt) with byte-identical outputs, its fallback included (#2921, refs #2859). Thanks to @dinadur.

- OCR: the native HIP engine reads page images through `POST /v1/ocr` in Mistral's OCR contract (its Python and TypeScript SDKs), with dots.mocr's reader (a box for each layout block). Mistral crops share one image budget per request (256 images, 4 times the page pixels, 16 MiB plus 2 MiB a page). An option that changes content and that TensorFold cannot apply, such as annotations, is refused, and so are PDF documents and `pages`. A page cut at the context limit or out of its model's format fails the request with HTTP 422; `docs/api.md` lists the contract's fields, statuses and limits (#2570, #2440). Thanks to @laudney.

- Docs: the Intel Arc (XPU) page notes that its B70 prefill row predates 1.1.0's more precise prompt DeltaNet, which costs about 5% of a 32,768-token prompt with GPTQ g128 in the raw runner, with tokens unchanged (#2906, measured in #2859). Thanks to @dinadur.

- Internal: the CUDA kept prompt-state identity is written once through any hasher, so the execution witness (#2741) can hash the same parts with SHA-256; every kept key stays byte for byte what it was, pinned by tests against 1.2.7's sequence and a fixed key, so saved prompt states still reopen. Nothing a user sees changes (#2903, part of #2741). Thanks to @kotevcode.

- Tests: the CUDA source check (`tools/zig/check_cuda_sources.py`) declares the SDK names main's kernels use, so it passes all 918 checks again (#2902). Thanks to @Bizuayeu.

- Qwen3.8-27B on Metal: slot 0's rewind point is now made even beside a prompt cache that can hold a whole-context state, and the warm pass keeps it, so a client that sends the last reply back without its reasoning resumes in place at its history's end instead of a turn further back (on an M5 Max at 32k with a 5 GiB cache, turns 1-6 of a 7-turn chat took 16.5 s and 18.9 s of summed time to first token against 217 s, with identical replies); a client that sends the reply back as generated makes the same fill calls as before; a background pass at depth runs in narrower chunks, so a request arriving mid-pass at 128k waited 0.5 s against 4.1 s; slot 0's 146 MiB rewind buffer counts against `--prompt-cache-gib` room (#2723). Thanks to @ThinkOffApp.

- Prompt cache on the core pool (the `TF_KV_SHARE` families, Qwen 3.5/3.6 on CUDA, and Qwen 3.5 and Nemotron on HIP): the radix store holds each pool page once, however many entries name it, and pins the node a keep or restore is using. A snapshot save that takes pool pages (Gemma 4, Kolibri-1, Spark2.5, Dory) can no longer make reclaim prune the leaf it is writing to. Replies and cached counts are unchanged. The store also learns checkpoint parts and prefix groups, which no family uses yet (#2875, part of #2444). Thanks to @laudney.

- GLM-5.3-Flash on CUDA at four ranks or more: when the default split prefill's start-up exchange with every rank fails, as on a switchless ring of four, the failing rank now prints a refusal naming `TF_GLM_PREFILL_REDUCE=gather`, and rank 0 names it too when another rank fails first. Before, every rank stopped on a bare `ncclGroupEnd` error. Start-ups that succeed are unchanged (#2868). Thanks to @Bizuayeu.

- HIP on gfx1151: the routed prompt tile runs an expert of at most 8 rows in K-parallel passes inside the same launch, with the matrix path's bytes; with Qwen3.6-35B-A3B (MLX 4-bit) on a Ryzen AI Max+ 395 (gfx1151) at one stream, served cold prompts of 64 to 256 tokens are 11% to 22% faster (1.5% at 1,024), and logits and served tokens equal main's. gfx1150, gfx1152 and gfx1153 share gfx1151's table and take the same cut, unmeasured; other cards' tables leave it off (#2878, fixes #2795). Thanks to @AbrahamCete.

- Internal: one block-FP8 reader in `core/quant/fp8.zig` loads the FP8 expert weights and their scales of Flash Next's MTP head, Drama 3 on CUDA and Kolibri-1, widening bf16 scales by the same bit shift and copying f32 ones as before; every packed buffer is byte-identical to main's on synthetic checkpoints of each family (no real checkpoint was loaded), and only Flash Next refuses non-finite scales, as before (#2892, #2460 item 7). Thanks to @jschmied.

- CI: the macOS nightly runs `zig build test-native`, the Metal engine host tests, installing the Metal toolchain when the runner lacks it and printing its cold and warm wall time (#2891, fixes #2880).

- Tests: `qwen27_served`'s reply-history test follows #2699's rule (each conversation's newest state is kept first under pressure), and a new case checks that a regenerate restores at the history mark when the cache has room, so `zig build test-native` passes on main again (#2889, fixes #2881).

- Serve: a later `--compact-at` replaces an earlier one, as for every other serve flag; `--compact-at auto --compact-at 0.5` used to keep automatic compaction because the fraction never cleared it (#2883). Thanks to @laudney.

- GLM-5.3-Flash on CUDA: `TF_GLM_CONTEXT_CEILING` (opt-in) lets GLM open windows past 1,048,576 tokens, read once when the model opens; with it unset nothing changes, and cold prompts and decode on two and three GB10s read level with main (#2765, refs #2663). Thanks to @Bizuayeu.

- Metal on M5: prompt lane projections at every width (8-, 6-, 5-, 3- and 2-bit and the oQ mixes) take a chunk's 16-row blocks in one dispatch, as 4-bit already did; Qwen3.8-27B MLX 8-bit cold prompts are about 33% faster on an M5 Ultra (769 to 1,029 tok/s at 2K, 769 to 1,027 at 8K) with identical tokens, and decode is unchanged (#2851, for #2818).

- GLM-5.3-Flash: the DFlash2 drafter's startup line and docs give its licence as Apache-2.0, as its model card now says (#3084, for #1583).

- CUDA kernels: a bf16 format for grouped experts runs a checkpoint's own bf16 expert weights in mma fragment blocks, with relu² up and fp32 or bf16 down, in decode and prompt forms; no served checkpoint selects it (it was written for Nemotron 3.5 Transcribe, #1631), and every other expert format's dispatch is unchanged (#1630). Thanks to @BobClawblaw.

- DeepSeek-V4.1 on CUDA: `zig build dsv41-assets` now builds `tf-dsv41-assets` at ReleaseSafe whatever `-Doptimize` says, so `aot-needs` plans the same Triton variants about 15 times faster (713 s against 45 s at a 65,536-token context; a GB10 plan at 524,288 tokens had taken about 45 minutes) (#3007, found on #2270). Thanks to @Defilan.

- Docs: the CUDA memory plan says what each setting decides and which file reads it, including that an unset `--context` takes the family's default capped at the model's own window and `--context 0` takes the model's window (#3028). Thanks to @Bizuayeu.

- Qwen3.8-27B with DFlash2 on M5 Macs: one greedy request at the default 16-row window uses the long-copy driver again, so copy-heavy replies such as a whole-file rewrite run at about 792 tok/s instead of 241 on an M5 Ultra (#3013). Every copied token is still verified by the target, so replies are the same tokens. Sampled requests, concurrent streams, wider windows and M1-M4 Macs keep the lane loop; `TF_QWEN27_LANES=1` keeps every request on lanes (#3069).

- Cortheon: a page link survives whitespace around `href` or inside its URL: `href ="…"` and a quote after a newline are kept, an unquoted link ends at a tab, CR or LF, and a tab or newline wrapped inside a link is not part of the address (#3006). Thanks to @ppardus.

- Qwen3.8-Flash-Next on HIP: verify windows run their DeltaNet from the round's plan, so a window writes no state snapshot a row, and its keep replays the kept rows through the same scan. Drafted and undrafted replies are unchanged: every drafted reply equals the same request with drafts off, and concurrent streams equal solo ones. Decode is 2–5% faster on Strix Halo (gfx1151) at one to four streams; shapes the plan kernels cannot take keep the per-row path, and no variable selects it (#2728). Thanks to @gilby.

- Slide rollback: a fact graph is removed only when its own folders file names the folder being rolled back, and its folders file goes with it; a graph whose folders file can't be read is left in place and printed (#3073).

- Chat templates: a template is held to a nesting depth at load, and its render to a step count, stack depth and scratch memory, so a crafted template can't hang or exhaust the server; real templates stay well under each limit (at most about 30% of the memory and 15% of the steps), render byte-identically, and render as fast as before (#3072).

- DeepSeek-V4.1 on Metal (experimental, opt in with `TF_EXPERIMENTAL=deepseek`): Macs now serve the release checkpoint natively with `tensorfold-dsv41` (`zig build dsv41-metal-serve`) for text prompts only. Prompts and replies share rounds through the lane core with DSpark drafts, and the prompt cache reports `cached_tokens`. Rounds with three or more decoding streams decode plain (no draft pass, drafter ingest or copy windows), and when fewer streams decode again the drafter catches up from the target, so replies are unchanged. On an M3 Ultra and an M5 Ultra with 512 GB, drafts make one stream 1.21x to 1.50x faster than `--no-drafts` and two streams up to 1.13x faster, three and four streams are level, and `token_sha256` is equal in every cell (#1720, part of #1676). Thanks to @gilby, and to @3utsu for an independent M5 Ultra receipt.

- Internal: `Admit` and `Ledger` move unchanged from `core/engine_api.zig` to `core/engine_admission.zig` and keep their names under `engine_api`, making room for the execution witness's hook; nothing a user sees changes (#2896, refs #2741). Thanks to @kotevcode.

- Qwen3.8-27B on Metal: prompt passes no longer write the GDN tails buffer, which nothing read. Replies are unchanged; on an M5 Ultra with the MLX 4-bit checkpoint, time to first token is 0.45% to 1.16% shorter at 2K, 8K and 32K prompts, one and four streams (#2915, for #2905).

- Flash Next on CUDA at `--tp 2`: a prompt pass wider than 64 rows reduce-scatters its routed experts' FP32 slots by row halves, sums each rank's rows and gathers the sums, where it all-reduced every slot; decode rounds and verify windows keep the all-reduce. Every reduced value is the same single add and the slots are summed in the writeback's order, so replies keep their tokens. Over two GB10s' RoCE link rank 0 sends 3.43 MiB a prompt token instead of 5.81, and cold prefill of 2k to 32k tokens is 26.5% to 26.8% faster there and 30.9% to 33.9% faster on two RTX PRO 6000s (an 8,192-token prompt: 7.12 to 5.62 s and 3.61 to 2.70 s) (#2977). Thanks to @swbiggart.

- Tests: `tf-cuda-test flashnext-rank RANK PORT timeout` passes again on two GPUs. NCCL start-up keeps its 30 s bound and only the stalled collective gets 200 ms, and both ranks meet once their communicators are up, so `timeout` and `eof` fail the collective under test, not the start-up (#2976, fixes #2975). Thanks to @swbiggart.

- Tokenizer: a split pattern is held to a work allowance per piece (2,048 instructions a byte, counted, not timed) and refused at load when too deep or too large, so a crafted tokenizer or prompt can't hang the pre-tokenizer; every split of the patterns we serve is unchanged (#3071).

- Cluster pull: a pull that would leave a file held only in part is refused unless the caller allows unverified bytes (off by default); a pull that ends whole still resumes and is checked against the manifest's hash. No shipped command calls the cluster loader yet, so nothing changes for users (#3074).

- Server: under connection pressure (96 of 128 slots open) one client address holds at most a quarter of the slots, and an idle keep-alive connection gives its slot up only when every slot is taken. New flags: `--max-connections` (1–1024), `--max-connections-per-address` (0 turns the share off) and `--trusted-proxy ADDRESS` (repeatable; never counted, like loopback). IPv6 clients are counted by /64. Below pressure nothing is refused (#3048).

- Grammar budget: exploring a grammar counts its parser work and stops at a fixed limit (2^38 steps); a grammar past it is refused with HTTP 400 and never served a partial mask, and every grammar under it explores and masks exactly as before. There is no time limit, so the answer does not depend on the machine or its load (#3037).

- XPU (Nemotron, Gemma 4, Qwen3.8-27B and LFM3): a token id at or past the embedding table is refused before the lookup, and the loaders that took the embedding table as stored check its shape when it loads; an 8-bit Qwen3.8-27B MLX checkpoint on XPU is now refused at load by name (#3047).

- HIP (Nemotron, Flash Next, Qwen3.5 and Nemotron Parse's draft path): a token id at or past the embedding table is refused before the lookup, and the loaders that took the embedding table as stored check its shape when it loads (#3046).

- Tokenizer: a token id past the cap (2^24) is refused when the tokenizer loads, instead of growing the id table without bound (#3033).

- Cluster: a node on a real host takes only the nodes its cluster file names and believes a node only about itself; relays and news about third nodes are dropped (#3039).

- HIP ranks: a follower checks each step's counts against its message before using them, so a malformed step is refused instead of read past; a zero-row prompt chunk that is not the last is still taken (#3042).

- Cluster pull: a file pulled whole is checked against the manifest's hash; a mismatch deletes the file and fails the load (#3043).

- Pull: a repository file under the engine's reserved names (`.tensorfold*` paths and `*.tensorfold-rollback`) is refused, so a model repository can't overwrite the engine's own state (#3032).

- Server media: a media URL whose host resolves to more addresses than the lookup's queue holds no longer hangs the fetch; every address is still checked against private ranges (#3035).

- Server: compaction notes and the fact graph are written owner-only through a unique temporary file renamed into place, so a planted link can't redirect the write (#3031).

- Sessions: a saved session's chunk can be no larger than the reader's buffer, so a crafted file is refused instead of read past it, and the disk tier's folders and files are owner-only (#3030).

- Server: a request body that no route reads is capped at 64 KiB, and a larger one gets HTTP 400 (#3029).

- DeepSeek-V4.1: both ranks print the startup settings line, and its transport field now reads `comm=roce` or `comm=nccl` with its source from `TF_COMM_BACKEND`; DeepSeek says once at startup that it does not read `TF_ROCE` when that is set, where the line used to report `roce=1(set)` for `TF_ROCE=0` (#2877, for #2715).

- GLM-5.3-Flash EXL3 on CUDA: a whole prompt pass routes its experts in one call instead of 1,024-row calls, and writes straight into its output slots, with bit-identical rows (#2852, for #2819).

- GLM-5.3-Flash on CUDA: with its MTP head and `--mtp-confidence`, a temperature-0 stream now drafts the depth its acceptance rates support instead of the full 15-step chain every round, while the confidence floor still cuts what the window verifies; replies are unchanged, and at `--mtp-confidence 0.5` one greedy stream decodes 16% (code) to 28% (chat) faster at TP=2 on two GB10s and 22-46% faster at TP=3 on three (#2864). Thanks to @Bizuayeu.

- XPU (Intel Arc Pro B70) with Qwen3.8-27B: a prompt chunk whose output nothing reads stops its last layer at the keys and values, and the chunk that is read runs the rest of that layer on its last 64-row tile only, as CUDA and Metal already do; logits, keys and values are byte-identical at ten prompt lengths, and the cold first token comes 1.3-2.5% sooner at 8K to 66K prompts with identical replies (#2854, fixes #2849). Thanks to @SergiioB and @Mirrdhyn for the B70 runs.

- Qwen 3.5 on CUDA: a stream's whole prompt fill runs exactly once, so with `--prompt-cache-disk` or `--learn-conversations` the startup probe no longer refuses with `ProbeUnmarked`, and a readout no longer runs a second, unread text pass; served replies are unchanged (#2838, fixes #2804).

- HIP: the qwen3_5 engine has a DFlash2 draft engine for chains: it walks the core `draft_tree`, shares one `prior` table with CUDA, and refuses an absorb whose start differs from its features; in the drafter test with Qwen3.8-27B on gfx1151, 0.86 of first drafts are right and a round makes about 5 tokens (greedy), and without a drafter the 27B's and 35B-A3B's logits on HIP are byte-identical to main's; the server doesn't use it yet (#2726, part 4 of #1563). Thanks to @AbrahamCete.

- Tests: the cancelled-benchmark test checks that a cancelled run saves one cancelled sample and never publishes, instead of counting requests that race the stop, so it no longer fails on a busy runner (#2865, fixes #2853).

- HIP Flash Next: a decode round's n-gram table gather spreads over the parked host workers and asks for its pages first (`MADV_POPULATE_READ`), so first-time page faults overlap; on one Radeon 8060S (gfx1151) with Qwen3.8-Flash-Next oQ4e at 4 streams, cold decode is about 45% faster sampled and 18% greedy and the cold 8K first token about 10% sooner, with warm decode level and tokens unchanged (#2695). Thanks to @gilby.

- CI: the serving pins (`test-gradient-serving` and `test-qwen-serving-host`) run on every pull request, not only in the nightly, so a change to a pinned serving file carries its re-pin. `experts.zig` is re-pinned for #2843 (#3011).

- Tests: the images route test fixture can no longer hang a test run on macOS. Its route closes its end of the socket on every exit and the client's writes are bounded, so a route that answers before reading a large body ends the write at once (#3010).

- CUDA EXL3 grouped experts launch and size their storage at the caller's tile and pair count, so decode no longer launches three quarters of its grid empty, and Qwen3.6-35B-A3B EXL3 sizes its grouped storage for 36,864 pair rows instead of 163,904 (about 3 GB less); outputs are bit-equal (#2843, for #2823).

- CUDA: dots.mocr's text decoder serves text prompts on a GB10 from the publisher's BF16 weights, on a new shared engine for dense GQA decoders (`zig/src/cuda_dense/`); `--vision`, prompt caching and fp8 caches are refused by name, and served CUDA checkpoints are unchanged (#2673, part of #1927). Thanks to @laudney.

- The prompt cache can keep a bounded ladder of intermediate checkpoints under a long prompt's history mark, so a later, compacted prompt resumes inside its retained prefix instead of the shared cut. On one GB10 with `nvidia/Qwen3.8-Flash-Next-NVFP4` and drafts off, a 104K-token request that kept its first 100K resumed 65,536 tokens instead of 2, cutting TTFT from 50.7 s to 19.9 s with the same tokens; a cold 165K prompt paid 0.9 s more, and the 8 GiB cache held two more states. A follow-up after that turn resumed 65,536 tokens too, because the compacted turn evicted the larger states. `TF_PROMPT_INTERMEDIATE=1` enables it for Flash Next on CUDA; it stays off by default. Its checkpoints go after a conversation's other redundant copies but before any conversation's newest state, and take only the room left after the endpoint, the rewind state and other requests' promised keeps (#1953, fixes #1909). Thanks to @javierbits.

- Metal: kernels nothing launches are removed: Qwen3.8-27B's `prompt_glue.metal` and `prompt_state.metal` (six `q27_prompt_*` kernels) and core's `tf_lane_rope` with `Attention.rotate`; saved Qwen3.8-27B prompt states on Metal are recomputed once after upgrading, since their identity covers the kernel sources (#3065, part of #2251). Thanks to @jschmied.
- Tests: main's nightly is green again. The Flash Next learn, prompt and compare tools build again on Linux (#2585 had put one file in two modules), and the dense serving and serving-identity pins take the changes #2579, #2585, #2285, #2205 and #2789 made, each named beside its pin. Serving is unchanged (#3002).
- HIP: Qwen3.8-Flash-Next's MTP head takes a prompt chunk on the prompt tiles, not a decode window at a time, so cold prompts finish about 5% sooner at 8K and 32K on a Radeon 8060S (gfx1151). Replies are unchanged; the head's caches and drafts are not bit-identical to before (#1878). Thanks to @gilby.

- Cortheon (`--cortheon`): DuckDuckGo results are kept, and its ads still skipped, when only the spelling differs (either quote style, a `class` attribute in any letter case, an unquoted `href`, spaces around its `=`, or a line break before it); a duplicate whose scheme or host differs only in letter case is dropped while paths and queries keep their case, and the PyPI, npm and crates.io allow-list takes the host in any case with the path exact (#2750). Thanks to @ppardus.

- GLM-5.3-Flash on CUDA: with `--vision` and no video library, rank 0 says at startup how much each stream sets aside for `--vision-video-tokens` rows that no request can fill (128 MiB a stream at the default 16,384), and that `--vision-video-tokens 64` on every rank cuts it to 512 KiB (#2813). Thanks to @Bizuayeu.

- CUDA: `Qwen/Qwen3-Embedding-0.6B`, `-4B` and `-8B` load on the native engine from the published BF16 weights and serve `/v1/embeddings` (last-token pooling, unit-normed, `dimensions` crops). On one GB10 the worst cosine distance to float32 is 1.55e-4, 1.34e-4 and 1.08e-4 (transformers' own BF16: 1.88e-4, 1.79e-4 and 1.77e-4), a request's vectors equal each input sent alone, and a reranker folder or a flag the family does not serve is refused by name (#2285). Thanks to @Arminova.
- CUDA: `Qwen/Qwen3-Reranker-0.6B`, `-4B` and `-8B` load on the native engine from the published BF16 weights and serve `/v1/rerank`, scoring a request's documents as one decision whose shared start runs once a pass. On one GB10 the worst score from float32 is 5.5e-5, 6.2e-4 and 2.9e-4 on 16 documents of up to 2,579 tokens (transformers' own BF16: 3.4e-5, 2.8e-4 and 2.2e-4), a document scored alone equals its score in a request of 16, and a prompt past the window is refused (#2995, part of #2261). Thanks to @Arminova.
- Qwen3.8-27B on Metal: the startup line says how many streams draft ("for both streams", "for all 4 streams", "for 8 of 16 streams"), the smaller of the stream count and `TF_Q27_DRAFT_SLOTS`, instead of "for one of them"; log text only (#2806).

- Server: `--profile fast` refuses as not planned, since a speed mode that changes output tokens is not shipped; `--profile fast-exact` now names the token gates it still needs, and an unknown spelling is told that `stable` is the only served profile (#2794). Thanks to @devopsfortunato.
- Qwen3.8-27B on Metal: three glue kernels nothing launched (`q27_norm`, `q27_mlp`, `q27_gdn_post`) and their host mirrors are removed; saved prompt states on Metal are recomputed once after upgrading, since their identity covers the kernel sources (#2816, part of #2251). Thanks to @jschmied.

- Pipeline stages: the shared stage driver pipelines rounds, so rank 0 sends round r + 1 while the later stages still run round r, and a stage that echoes a round short or empty is refused by name; tokens are checked equal to one engine on the same binary (#2579, part of #2005). Thanks to @jschmied.

- GLM-5.3-Flash on CUDA: tensor-parallel shares and layer sums go through one mesh layer (declared shares, one reduction boundary over a `Collective`), with the same gathers, bytes and order and tokens equal to main at TP=2. In any family on CUDA, an aborted `tp_group` call now ends as `Aborted` instead of stopping the process (#2789, part of #2560). Thanks to @laudney.

- Tensor parallel on CUDA: with `TF_ROCE=1` on every rank, three ranks or more take the one-shot RoCE all-gather for decode-sized gathers, byte-equal to NCCL's, falling back to NCCL with one warning when a peer can't be reached, and GLM-5.3-Flash's DFlash2 drafter MLP splits across every rank; without it they keep NCCL, two ranks run as before, and MiniMax-M3 keeps NCCL at three ranks or more. With `TF_ROCE=1`, GLM-5.3-Flash decode is about 6% faster at one and two streams on three GB10s (four streams level) and 12.9%, 10.4% and 6.6% faster at one, two and four streams on four GB10s, with the target's tokens unchanged; drafter acceptance can move slightly, since the split MLP sums in another order (#2205). Thanks to @shanedertrain.

- Flash Next: twenty-nine `TF_FLASHNEXT_*` settings refuse a malformed value by name at startup, before any device work on CUDA, and a Metal load refuses them too; blank `TF_FLASHNEXT_KV_PAGE_TEST` and `TF_FLASHNEXT_KEEP_LATE` read as unset, and a non-finite confidence is refused (#2585, part of #2370). Thanks to @devopsfortunato.
  device or checkpoint work, through the shared environment table: unset and blank keep the default, a valid value keeps
  its meaning, and the switches are exactly `0` or `1` (#2585, for #2370). Thanks to @devopsfortunato.
- MOSS-Image on Metal: `tf-moss-image-generate` makes a picture from options alone (`--prompt`, `--negative`, `--width`, `--height`, `--steps`, `--seed`, `--guidance`) at the card's defaults, and refuses skip-layer guidance by name; `tf-moss-image` still runs a reference case (#2719, for #1531). Thanks to @devopsfortunato.
- Tensor parallel: `fabric/tp` gains a TCP transport for hosts without RDMA (`Kind.tcp`), the same collective contract over one stream socket a pair of ranks, failing by name on a desynchronised call, a late frame or a lost peer; `tf-tp-test tcp-host` checks two hosts bit for bit over it. No serving backend can select it in this release, so every engine is unchanged (#2718, part of #2065). Thanks to @Shoestring-dev.

- Tests: `zig build tf-tp-check` builds a checker for the per-rank collective and logits dumps of a mixed Metal and CUDA tensor-parallel run (dump versions 1 to 3), and `zig build test-tp-check` runs its tests on synthetic dumps. Nothing in this release writes those dumps, so serving is unchanged (#2321, part of #1236). Thanks to @ThinkOffApp.
  writes, and `zig build test-tp-check` runs its tests (#2321, part of #1236). Thanks to @ThinkOffApp.
- Tests: `core/lanes` gains a shared depth sweep any core drafter can run, a fresh run at each depth from 1 to the block and one zig-zag run, each checked against a reference pass, so an adapter that ignores its hold depth or drafts past its block fails (#2729, part of #2460). Thanks to @jschmied and @EugeneClaw.

- CUDA: `TF_DSPINE_WINDOW=N` (16 to 8,192, opt-in) narrows a Qwen3.8-27B DSpine drafter's sliding window to N rows without editing the checkpoint; a value outside the block size and the checkpoint's own window is refused when the drafter opens, and unset keeps the checkpoint's window. Drafts are verified, so replies are unchanged (#2705, after #2633). Thanks to @jschmied.

- Metal: Spark2.5's and Llama's bf16 decode matvec kernels come from one source, with the same compiled code; Qwen3.8-27B saved prompt states on Metal are recomputed once after upgrading (#2792, part of #2251). Thanks to @jschmied.

- Metal: the image towers' and Spark2.5's and Llama's prompt GEMMs share one `gemm_nax` kernel macro, with the same preprocessed source; Qwen3.8-27B saved prompt states on Metal are recomputed once after upgrading (#2787, part of #2251). Thanks to @jschmied.

- GLM-5.3-Flash on Metal: the TP2 sparse-attention and KDA-rows kernels are the full kernels' templates at 32 heads, with the same compiled code; saved GLM prompt states on Metal are recomputed once after upgrading (#2786, part of #2251). Thanks to @jschmied.

- Metal: Qwen 3.5, LFM2-MoE, MiniMax-M3 (with its DFlash2 and DSpark drafters), Step-5 and Spark-X read MLX affine modules through one core reader, `core/quant/mlx_affine.zig`, and LFM3 takes its modules' stored shapes from it while keeping its own checks; loads are unchanged, and a module the reader refuses now reports `UnexpectedAffineTensor` and is named in the log (#2583, part of #2460). Thanks to @jschmied.

## 1.2.7

- GLM-5.3-Flash on CUDA: with `TF_GLM_LAYERS=N`, serve's start-up memory check counts the weights by tensor from the shard headers, only those the load reads (the first N layers, the MTP layer when drafts are on, the embedding, norm and head), where it counted every checkpoint file. A 20-layer thin load of the 163.6 GiB checkpoint now counts 66.6 GiB without drafts. Unset, the count is the files' as before (#2788, fixes #2745). Thanks to @wojo.

- Build: `zig build -Dgpu=<gpu>` compiles again; the `-Dgpu` products did not set `is_release`, which the server has read since 1.2.5, so every such build failed (#2879). Thanks to @tacosta for the fix and @Defilan for finding it.

- Lanes: where a family learns its shared rows' cost on the machine (`learn_row_costs`, GLM on Macs without tensor units), a fit that shows rows cost less than the table says now lowers their price instead of counting as nothing, so a table timed on a slower build no longer keeps drafts out that would pay. The fall clears the same noise test as a rise, and a round never prices below its streams' plain width. While a stream count has no fit, its rounds now reach the width they measure (the likeliest drafts first), not only stop there, so a table that prices every draft out still gets its wider widths measured. Only how many drafts a round verifies changes; replies are unchanged, and families that do not learn row costs are unchanged (#2777, fixes #2776).

- Tests: the serving pins name the changes #2721, #2753 and #2802 made to six pinned sources, so `test-gradient-serving` passes on main again (#2872).

- Flash Next on Metal: the host's status reads the FSR cache counters through a pointer the host keeps from open, not through the engine, so `zig build test-native` no longer crashes in two host tests that build a host without one; served behaviour is unchanged (#2876).

- Docs: `docs/optimizing.md` says how to approach speed work in TensorFold: measure first, keep every token exact, and write what lands from our own code, with a worked example (#2740, #2739). Thanks to @jschmied.

- Responses API: WebSocket mode. `GET /v1/responses` with a WebSocket upgrade takes `response.create` events and streams the same events as SSE, token-equal to it; `previous_response_id` continues `store: false` chains from a per-socket cache, with `stream_id` lanes, `generate: false` warmups and OpenAI's 60-minute limit, so `client.responses.connect()` works unchanged (#2749, fixes #2746). Thanks to @laudney.

- Tests: host tests that rely on the operating system refusing access skip by name when run as root, so those tests no longer fail in containers that run as root; as an ordinary user nothing changes (#2757, part of #2748).

- Flash Next on CUDA: a draft-head row that comes back non-finite now fails as `ProposalNonfinite` instead of `ProposalPaddingPick`, sampled draft candidates are checked for count, finite values and column range before sampling, and verify rows are checked the same way after the one completion wait their path already makes, before any output token is written (grammar-masked columns excepted); a refused row logs one line naming the stream and the draft length, with the row's top and log-sum-exp where it has them. Finite rows pick exactly as before, and prompt passes and decode (chat and code, 1 to 8 streams) read level on a GB10 (#2753, refs #2734).

- GLM-5.3-Flash on CUDA with a DFlash2 drafter: shared rounds size each stream's drafts by its last grant plus the slack again, trimmed at verify, instead of pricing them first from chances the drafter's chain cut zeroed, so a stream no longer stays at no drafts while its peers decode. On two GB10s at TP=2 with distinct prompts, four-stream decode is 3.7% faster greedy and 3.5% faster at temperature 0.7; one stream, the MTP head and every reply's tokens are unchanged (#2793, fixes #2720). Thanks to @kelchm.

- Tests: three serving pins take their files' new hashes after #2609, #2680 and #2709 changed them, so `zig build test-gradient-serving` passes on main again; test files only (#2798).

- Qwen3.8-27B on Metal: a client that sends earlier replies back without their reasoning now resumes a follow-up turn at the end of its last message instead of far short of it. On an M3 Ultra with `--prompt-cache-gib 5` and a 34K history, such turns resume about 34,100 tokens instead of 7,424, with the first token in about 0.3 s instead of 85 s, while clients that keep the reasoning resume as before; replies are unchanged (#2811, fixes #2791).

- Living Weights: a lesson's subject is never a number group ("12 500", "7,400", including thin-space and no-break-space forms) or a sentence-opening determiner such as "Any" on its own, so "Any invoice above 7,400 euros requires a second signature by Brightwell Logistics." is about Brightwell Logistics (#2807).

- Qwen 3.5 on CUDA: `--decode-share`'s 512-row cap now reaches the prompt pass, cut inside the cold chunks on a 128-row grid so every cold cut still lands and kept states stay at cold cuts; decoders beside a long prompt wait one short step instead of a whole chunk, the long prompt's own first token comes later, and replies are unchanged. `TENSORFOLD_PREFILL_POLICY=fcfs` now cuts Qwen's prompt steps beside decoders too, at its row count floored to 128 (#2802, refs #840). Thanks to @jschmied.

- Lane host: when the backend fails a round, only the streams it held at the failure (paused ones included) and the prompts whose last piece the round planned fail with it; a prompt still filling keeps filling, and a stream that had already finished delivers its reply instead of an error (#2727). Thanks to @ciprianveg.

- Dashboard (`--dashboard`) and `/stats`: the prompt cache shows its states, hits, held bytes against budget, cached tokens, prefill reuse and its 64 most recently used states; its disk tier, when present, shows reads and writes, latency, queue depth and read-ahead; on Metal, Flash Next's SSD-expert cache shows hits, evictions and occupancy; state ids are hashed only with `--dashboard` on (#2682, fixes #2655). Thanks to @volfco.

- Server: a GLM tool call whose argument value runs into the next key/value pair, or never closes before the call ends, now stays in `content` as text instead of becoming a call with corrupt arguments, whole and streamed (a stream may already have sent the call's first pieces); other GLM calls parse as before, except that a value containing the literal text `<arg_key>` now stays as text (#2831, fixes #2801). Thanks to @devopsfortunato.

- Internal: one core reader for ModelOpt projection parts (weight, NVFP4 or FP8 scales, input scale), with shape checks and refusals that name the module and tensor, now serves Dory's NVFP4 and bf16 projections, Qwen 3.5/3.6's NVFP4 experts and `check-precision`; Dory's MXFP8, Nemotron, Qwen 3.5/3.6's other ModelOpt projections and GLM keep their own readers, and the loaded weights are unchanged (#2691, part of #2460). Thanks to @jschmied.

- Tests: `zig build test-qwen35-copies-host` compiles again, passing the drafter kind `copyMatch` now takes (#2836).

- Internal: core gains one walk for copying kept prompt states, and GLM's copies on CUDA and Metal go through it, with the same copies in the same order; other families keep their own loops. Nothing a user sees changes (#2215, part of #1293). Thanks to @jschmied.

- CUDA, opt-in build (`-Dnative-fp-decode`): qmmf's FP4 decode rows use the shared native e2m1 pair decode, `tf_common::e2m1_pairs`, which equals the existing decode on all 2^33 inputs; the default build's 113 fatbins are byte-identical and GB10 decode is level (#2721). Thanks to @jschmied.

- Qwen3.8-27B on Metal: while a request runs, the keep-warm ticker now commits the residency sets' probe each interval, so the first command buffer after a wide prompt pass no longer waits in the kernel (likely for macOS to wire the model again; not traced). On an M5 Ultra with a 14 GiB prompt cache filling memory, five cold 24K- to 64K-token prompts in a row reach their first tokens 3.5% to 5.9% sooner in total; with the cache off, cold prompts are level. Replies are unchanged (#2725, fixes #2724). Thanks to @mrv777.

- CUDA: beside a DSpine drafter, the dense Qwen3.8-27B copies from its context after 12 matching tokens instead of 20 (DFlash2 keeps 20), so agent turns decode about 6-7% faster at one stream on a GB10 with chat and code level; greedy replies are unchanged (#2703, after #2633). Thanks to @jschmied.

- Tests: a HIP GPU test runs one synthetic DeltaNet layer of Flash Next through `linear()` with the fused prompt prep and with the chain, compares the two bit for bit, and checks from its scratch use that the fused branch ran (#2706, after #2582). Thanks to @AbrahamCete.

- HIP on gfx1151: a routed prompt product takes the 64-row tile only while its experts average at most 64 rows each. #1658's 1,024-row chunk cap never fired, so Qwen3.8-Flash-Next's default 4,096-row chunks (80 rows an expert) had been running on it; they now take the skewed 128-row tile. Flash Next 8K-token prompts reach their first token about 5% sooner. Served Qwen3.5/3.6 MoE prompts (1,024-row chunks, at most 32 rows an expert) are unchanged; Nemotron's 2,048-row prompt chunks (96 rows an expert on Nemotron 3.5 Lightning 30B-A3B) now take the skewed 128-row tile, and its 8K-token prompts reach their first token about 6% sooner. Replies are unchanged (#2576). Thanks to @gilby.

- `TF_GLM_REQUEST_STARTS=1` (opt-in, off by default) lets GLM on CUDA cut prompt passes at a request's message starts, so a chat's next turn resumes at its previous turn's history instead of the last 2,048-row point. GLM refuses it at startup with `TF_KV_SHARE` above 0 or with `--learn-conversations`. On two GB10s at TP=2 (EXL3 TR3, DFlash2), a chat of about 58.7K tokens resumed its later turns at 58,735 instead of 57,344, with replies identical to `--prompt-cache-gib 0` and to the switch off (#2404, part of #2396). Thanks to @laidick.

- Flash Next on Metal (M1 through M4): a decode window of five rows or more now runs its 4-bit experts grouped by expert, so rows that pick the same expert share its launch and replies keep their tokens. A copy stretch decodes 7.7% faster on an M3 Ultra (256.6 to 276.3 tok/s); prose rounds, four rows or fewer, are unchanged. `TF_FLASHNEXT_GROUPED_ROWS` sets the first grouped width and `0` turns it off (#2784, part of #2764).

- Docs: the optimization ledger marks the rows that have open PRs and adds DSpine as row 29 (#2681). Thanks to @jschmied.

- GLM-5.3-Flash on Metal with `--vision`: a prompt chunk sends the vision tower only the images that end after its start, so two interleaved image prompts no longer re-encode images their earlier chunks already injected (12 images through the tower instead of 18 in a host test); each image is still encoded on its own and read through its own span, so the rows a chunk reads are unchanged (#2744).

- Flash Next on Metal: `TF_FLASHNEXT_SCALE_INDEX=1` (off by default; M1 through M4) has the 4-bit expert kernels read each group's scale and bias through a 16-bit index into its tensor's table of pairs: the same values from 10% fewer weight bytes, so replies keep their tokens. On an M3 Ultra with the MLX 4-bit checkpoint, drafted decode on one stream is 1.3% to 3.9% faster (prose 137.3 to 139.1 tok/s, a copy-heavy edit 262.8 to 273.2) and `"draft": false` 0.8% to 2.2% faster, with every token unchanged. The index sits beside the stored scales and biases, which prompt passes still read: 7.3 GiB more in use and about 4 s more load (#2760, part of #2751).

- Qwen 3.5 and 3.8 configs: one tied-head rule reads `tie_word_embeddings` from `text_config` then the root, and a checkpoint that sets neither loads untied, as transformers does (HIP had loaded it tied; Metal's 2B still requires the key); a root and `text_config` that disagree, or a value that isn't a flag, are refused with `BadTiedHead`. Flash Next, LFM2, Kolibri-1 and Qwen 3.5 read `rope_parameters`, legacy `rope_scaling` and their top-level keys through one core reader, and every config they accepted parses as before (#2714, fixes #2561, part of #2460). Thanks to @jschmied.

- CUDA: a family can declare that it keeps no prompt states, so the pool plans no prompt cache for it and `--prompt-cache-gib` is refused by name before the weights load, and that its replies carry no reasoning block, as HIP already allows; every family on main is unchanged (#2710). Thanks to @laudney.

- Tests: `zig build test-fsr-cli-host` compiles again (its module now gets the server's shared json imports), so `zig build test` no longer stops there on a Mac (#2782). Thanks to @BobClawblaw for the report.

- Images: a damaged JPEG whose decode stops before every pixel is written is refused as undecodable, where it could be accepted with rows the decoder never wrote; clean JPEGs decode to the same bytes as before (#2759). Thanks to a community tester who reported it privately.

- CUDA: LightOnOCR-3 0.8B and 4B serve image chat on a GB10 from the publisher's BF16 weights: a plain BF16 Qwen3.5 checkpoint with an image tower the CUDA tower reads and no Clef joint head now opens as chat, its tied head packed from the embedding, instead of the decision-only refusal. The flags it is not qualified on (`--drafter`, `--slide`, `--kv-cache fp8`, `--prompt-precision int8`), towers the CUDA tower cannot read and a joint head beside a tied backbone are refused by name before the weights load. MLX, EXL3, ModelOpt and Clef checkpoints load and answer as before (#2597, part of #1927). Thanks to @laudney.

- GLM-5.3-Flash on CUDA: the router's eight K slices run on the shared mma tiles (`common/router_mma.cuh`), so Flash Next and GLM share one router template; every partial keeps its bytes, and replies, prompt and decode speed at TP=2 on two GB10s are unchanged. The unaligned copy (`tf_glm_router_part`) is gone, and `glue.router` refuses unaligned rows, which no caller passes (#2557, fixes #2535). Thanks to @laudney.

- CUDA: Flash Next's router logits template (`route_logits`) moves into the shared header `common/router_mma.cuh`, with a K-sliced form for GLM's router; its kernels compile to the same machine code on sm_80, sm_86, sm_89, sm_120 and sm_121 (#2556, part of #2535). Thanks to @laudney.
- Qwen3.8-Flash-Next on HIP: a decode round's n-gram table rows are gathered over parked workers, and the pages they need are asked for first, so the page faults of first-time text overlap instead of running one after another on one thread; on a Strix Halo iGPU four cold distinct streams decode about 38% faster, and a cold 8K prompt's first token comes about 10% sooner with its reply decoding about 25% faster. Replies are unchanged.

- Prompt cache: long conversations survive 8 to 16 agents sending follow-ups at once. Redundant states are evicted before each conversation's newest one, admission never evicts a state a queued request resumes, concurrent keeps reserve their room, and the default budget scales with the room left after streams on Metal and unified-memory CUDA (discrete CUDA hosts keep 8 GiB, Flash Next's batch host 16 GiB). On Qwen3.8-27B with DFlash2 at 16 agents, a follow-up's median first token drops from 546 s to 16.5 s on an M3 Ultra and from 90-114 s to 28 s on a GB10, with every reply unchanged (#2699).

- Tests: the thinking benchmark check accepts an unmeasured delivery, so it no longer flakes on busy Linux runners, and the image routes' fixture starts its reader before writing, so its two large-body tests no longer hang where socket buffers are small, as on macOS (#2743, closes #2735).

- Conformance: a Qwen3.8-27B XPU profile (`tools/conformance_profiles/qwen27-xpu.json`, GPTQ 4-bit group 128 with its DFlash2 drafter, four streams on an Arc Pro B70); its first turn carries a 600-item document because the XPU prompt cache keeps states only at multiples of 64 tokens (#2677, landed as #2690). Thanks to @Mirrdhyn.

- Docs: the GLM CUDA recipe covers a four-GB10 switchless ring (gather starts there, the NCCL settings it needs, and the opt-in relay with its measured effect), and `docs/api.md` gives `presence_penalty`'s measured cost to drafting, about 4-10% of tokens a round (#2742, closes #1971 and #2645). Thanks to @Bizuayeu, @BobClawblaw and @jschmied for the measurements.

- Tests: when spawning a rank fails, `host.runWith`, the in-process tensor-parallel transport only tests use, aborts and joins the ranks it already started instead of leaving them waiting on a dead frame, with a test that fails without it (#2700, follow-up to #2491). Thanks to @Defilan.

- HIP: Flash Next sizes its scratch arena from the dry run's peak, counting the MTP head and a verify round's logits. On two Radeon AI PRO R9700s at tensor parallel 2 with a 32k window, each card holds 0.74 GiB less and `--policy prefill_step=2048` loads where main refused it, with replies equal to main's (#2670). Thanks to @millaguie.

- Qwen3.8-27B on Metal: a sampled row's top-k skips 32 logits at a time that cannot beat the k-th candidate, still refusing any non-finite logit, with the same candidates and replies. On an M5 Ultra with Qwen3.8-27B MLX 4-bit and its DFlash2 drafter, sampled verify rounds take 1.6 ms less and sampled decode is 6% faster; greedy is unchanged. The Qwen 3.5/3.6 CUDA engine shares this host draw for each first token and for grammar-masked windows; its unmasked sampled verify rows draw on the GPU. CUDA is not measured (#2507). Thanks to @mrv777.

- CUDA kernels: `tf_dots_qkv_rope` adds dots.mocr's q, k and v biases and rotates its queries and keys in one pass at Transformers' BF16 rounding points (BF16 cosines and sines of an FP32 angle, each product and their sum rounded), writing the rows the paged key and value writer reads, with a contract against FP64 at 1, 16, 128 and 1,024 rows and positions up to 131,071, a planted control and records for five SMs. No model serves through it in this release (#2709, part of #1927). Thanks to @laudney.

- CUDA kernels: `pattn_kernel` gains instances at two query heads a block for head dims 128 and 64, the prompt attention that dots.mocr (12 heads over 2), GLM-OCR (16 over 8) and Nemotron Parse (16 over 8) need on CUDA, with contract entries, planted controls and records for five SMs; paged decode at head 128 gets contract entries at those layouts and `tf_nemo_attn_merge_segs` its record. No served kernel's code changes (#2609, part of #1927). Thanks to @laudney.

- CUDA: the shared KDA module gains `kda.geometry`, which refuses KDA heads or a conv window other than the kernels' (128 keys, 128 values, 4 taps) with `KdaGeometry`, for a family to call at load before anything is allocated or launched, and a host test ties those numbers to `kda.cuh`. No family calls it in this release, so loading is unchanged (#2338, refs #850). Thanks to @ajensenwaud.

- Tests: `tp.device` runs N tensor-parallel ranks as threads on one GPU with device copies, so DeepSeek-V4.1's exchanges (embedding, Engram's column gather and the mHC folds) are checked bit for bit at worlds 1 to 4 on one GPU, and its uneven sampling, dense split KV and lockstep tests run at three ranks. `zig build dsv41-test` builds `tf-dsv41-test` over the family's own kernel images. Serving still refuses three ranks (#2491, refs #2270). Thanks to @Defilan.

- CUDA: the start-up note about pages parked on the kernel's per-CPU lists logs at debug level, so whether it shows no longer depends on the box's memory state; the budget still counts those pages (#2164, follow-up to #2127). Thanks to @jschmied.

- CUDA: seven families (Spark 2.5, Kolibri-1, Dory, Gemma 4, Qwen4, Nemotron and MiniMax M3) take their sampled rule from one function, `cuda.sample.ruleFor`, in place of their own copies (MiniMax M3's hand-filled `Rule` included); a host test checks each old form byte for byte over 25,480 settings, and the fatbins are unchanged. Qwen 3.5/3.6, GLM and LFM2 MoE keep their own forms (#2680, #2460 item 3). Thanks to @jschmied.

- Tests: the dense serving and gradient-serving pins follow three sources main changed (the DSpine draft kernels, Flash Next's shared RMS helpers with the same SASS, and its disk-state keys), so `test-qwen-serving-host` and `test-gradient-serving` pass on main again (#2738, refs #2031).

- Tests: the sessions-writes mixed load holds its pool to one slot's capacity, so it exercises spills to the tier (#2731). Thanks to @garvrao80.

- Prompt cache: when two disk restores in a row fail (a damaged or missing state file), the request now resumes from the shorter state it already had in RAM instead of prefilling from token 0, so the reply's `cached_tokens` reports that state's position (#2694). Thanks to @ThinkOffApp.

## 1.2.6

- Prompt cache: long conversations survive 8 to 16 agents sending follow-ups at once. Redundant states are evicted before each conversation's newest one, admission never evicts a state a queued request resumes, concurrent keeps reserve their room, and the default budget scales with the room left after streams on Metal and unified-memory CUDA (discrete CUDA hosts keep 8 GiB, Flash Next's batch host 16 GiB). On Qwen3.8-27B with DFlash2 at 16 agents, a follow-up's median first token drops from 546 s to 15.9 s on an M3 Ultra and from 105 s to 25.8 s on a GB10, with every reply unchanged (#2699).

- Evidence index: reads saved native capabilities, including unknown and XPU chip labels; advertised facts stay unqualified, and invalid supplied input stops with a bounded error (#2708).

- `--instruct-sampling card` (or `field=value` pairs) serves the model card's instruct sampling to sampled chat requests with thinking off that leave those fields unset; request fields, then the serve flags, then the pairs decide, and the server's own requests never take it (#2626). Thanks to @BobClawblaw.

- XPU: Qwen3.8-27B's DFlash2 drafts go through the core Drafter and Drafted, and each stream's draft tree is keyed with its own request (a sampled shared round read another's) (#2672, #2460 item 5). Thanks to @jschmied, and to @Mirrdhyn for the Arc Pro B70 runs.

- `POST /v1/images/generations` serves an engine's image capability: one picture a request as a base64 PNG, with an `n` other than 1, sizes past 1536 a side or 1,572,864 pixels, `response_format` `url`, unknown fields and a `negative_prompt` that is null or not valid UTF-8 refused by name; a text model's routes are unchanged (#2253, part of #1573). Thanks to @devopsfortunato.

- Server: the tokenizer's memo keeps segments of 64 bytes or more, not 1,024, so a chat's short earlier messages and tool calls are tokenized once instead of on each of a request's renders, with the same ids. On real agent conversations (Qwen3.8-27B tokenizer, up to 60K tokens) a warm encode of the eight longest takes 146-151 µs instead of 680-691 µs, and a one-off prompt is no slower (#2509). Thanks to @mrv777.

- Persisted prompt states (the prompt cache's disk tier and `--learn-conversations`): every family that saves states to disk now keys them by what its loaded run actually computed with, not only the flags as set: resolved arithmetic and kernel choices, every indexed weight shard (subfolders included), and external loaded inputs such as the CUDA Qwen rotary table, a Flash Next pack outside its folder, Flash Next on CUDA's chunk heights and kernel forms, and Nemotron's precision, expert ranks, ModelOpt lane and captured attention variant. A state computed differently now misses instead of resuming, and a resumed state still equals a fresh one. Upgrading starts every kept state once cold, and so does changing a GLM or Qwen Metal prompt chunk height. `GLM_EHPROJ_NAX` now takes only `0` or `1`; other text, which used to read as on, is refused at start (#2287, closes #2131).

- Docs: a reply that quotes a template marker finishes with a plain request, measured on two Qwen checkpoints on a GB10; a client that keeps `ignore_eos` stops on the model's end-of-text and next-turn-open spellings and strips the trailing turn marker, since behind an agent prompt end-of-text alone let 7 of 9 replies run on, against 0 of 9 with both, on Qwen3.8-Flash-Next on an RTX PRO 6000 (#2437, refs #2399). Thanks to @javierbits and @robertung88.
- Conformance: the cancel and evict gates check that a cut reply and an evicted prompt leave the server as a fresh one, with a planted fault for each; the Qwen3.8-27B Metal profile skips evict with its measured reason, and the CUDA profiles run both (#2611). Thanks to @jschmied.

- HIP: the qwen3_5 engine builds the DFlash2 drafter's kernels (tap gathers, key ring, dynamic conv, block attention), tested and in the kernel contract; nothing launches them yet (#2545, part 3 of #1563). Thanks to @AbrahamCete.

- CUDA: Qwen3.8-27B can draft with Tencent's DSpine-b16 when `--drafter` names its folder (never the default). Its code bridge and exact-sender table run on affine-4 projections, and the draft tree walker moves into `core/lanes`. Replies are unchanged: every drafted reply equals the same request with drafts off. On a GB10, agent-shaped requests decoded 2.4-19.8% faster than with DFlash2, and chat and code 10-19% slower, which is why it is opt-in (#2633, refs #2592). Thanks to @jschmied.

- Qwen3.8-27B on Metal: a prompt pass that keeps no row's logits writes only its last layer's keys and values, and skips that layer's attention, MLP and final norm, with KV pages on or off; the prompt state and tokens are unchanged. Served prefill is 1.6% faster at 4K and 2.3% at 16K on an M3 Pro, and 1.2% at 8K and 1.5-1.6% at 16K on an M5 Ultra (#2436, refs #2229). Thanks to @jschmied.

- Nemotron on Metal: the 8 group norms share one source, with the same kernels and bits (#2540, part of #2251). Thanks to @jschmied. Saved Nemotron prompt states on Metal take their identity from the source text, so after upgrading they are not reused: the first long prompt prefills again, and is kept as before from then on.

- Qwen 3.5/3.8: the four config readers check activation, attention gate, bias and recurrent state type from one table with a column per backend; each admits what it did, and a refused config names the part (#2563, part of #2460). Thanks to @jschmied.

- CUDA: Flash Next's PLE gate RMS scales, and the Qwen4 kernel generated from them, call `tf_common::rms_scale`; every function's SASS is unchanged (#2684, refs #548). Thanks to @jschmied.

- CUDA: the stored partial-sum folds before an RMS scale (Flash Next's hyper-connection normalize, hc_down, rinv and hcf prologue, DeepSeek-V4.1's mHC norm folds) call one helper, `tf_common::sum_rn` in `common/fold.cuh`; every function's SASS is unchanged (#2666, refs #548). Thanks to @jschmied.

- CUDA: Qwen's DFlash2 drafter config refuses a `target_layer_ids` list that names one layer twice, as HIP's does (#2674). Thanks to @AbrahamCete.

- Flash Next on CUDA: an empty or whitespace-only `TF_FLASHNEXT_KV_PAGES` reads as unset, so streams take no pool pages, where it used to stop the start with `BadFlashNextKvPages`; every value accepted before reads the same, and numbers with surrounding spaces are still refused (#2619, for #2591). Thanks to @devopsfortunato.

- With `--cortheon`, a DuckDuckGo result whose `uddg` key, `class=` mark (on the full and Lite pages) or `&AMP;` entity differs only in letter case is kept, an ad whose host or `Result--ad` mark differs in case is still skipped, a tool argument keyed `Query` or `URL` is read with the exact key still first, HTML's uppercase named entities decode, and a `data:` link is dropped from a page's links (#2372). Without `--cortheon` nothing changes. Thanks to @ppardus.

- HIP: prompts of Qwen3.5, 3.6 and 3.8 and Flash Next run their DeltaNet recurrence at half a value head a block, with the same bits, so a prompt takes 5-6% less time on Qwen3.8-27B and 6-10% less on Qwen3.6-35B-A3B on gfx1151; Flash Next's prompt conv, split, norms and gates run in one launch (#2582, piece J of #1667). Thanks to @davidtai and @AbrahamCete.

- Server: a reply that max_tokens cuts inside a JSON tool call (`<tool_call>` then `{"name": ..., "arguments": ...`, the form Ripple-65B-Preview writes) returns that call with its arguments as far as written, whole and streamed alike, as XML and GLM cuts already are, instead of leaving the markup in `content`; a call cut before its name is whole or its arguments begin, or naming a tool not offered, stays text; a whole `/v1/messages` reply closes a cut call's input as far as written, as its stream does, where an XML cut was a 500 (#2511). Thanks to @ajensenwaud.

- CUDA: the shared MoE routing glue gains group-limited selection in its sigmoid score form: experts split into equal groups, each scored by its two best values, the best groups kept (the lower group on ties) and the top-k taken from them; a grouped route outside that form is refused before launch. Nemotron, the one family on the score form today, keeps its kernels and bytes (#2334, part of #850). Thanks to @ajensenwaud.

- HIP on gfx1151: a skewed-row matrix tile runs three blocks a WGP for prompts up to 8192 codes deep, with the same results as the padded tile, which stays for every other shape (#2087). Thanks to @AbrahamCete.

- GLM-5.3-Flash on CUDA: decode windows reach 32 rows (from 8), and MTP chains up to 15 drafts (from 3). At TP=2 on two GB10s, prompts of 2k and 8k tokens fill 23% and 13% faster and one-stream code decodes 32% faster, with every reply equal to before (#2257).

- Qwen3.5/3.6 MoE on CUDA: router calls of 257 rows or more (prompt chunks) run on the shared register tiles (`tf_moe_router_tiles`), with every logit's bits unchanged; on a DGX Spark (GB10), Qwen3.6-35B-A3B NVFP4's cold prompts are 13.4% faster at 2K (0.382 to 0.331 s), 14.1% at 8K (1.477 to 1.269 s), 11.8% at 32K (6.772 to 5.970 s) and 9.9% at 64K (15.899 to 14.323 s), with decode and replies unchanged. The 257-row switch is GB10's crossover; `tf-cuda-test moe-router-bench` measures it on other GPUs (#2537, #2499). Thanks to @laudney.
- CUDA: Nemotron's prompt router tiles (`tf_nemo_router_tiles`) move into the shared header `common/router_tiles.cuh`, so other routers can use them; the kernel compiles to the same machine code on sm_80, sm_86, sm_89, sm_120 and sm_121 (#2536, part of #2499). Thanks to @laudney.
- Qwen3.8-27B on Macs: a sampled or grammar-masked verify window draws its rows on up to four host threads, the caller's included, instead of one after another. Every row is still drawn and checked in full, and a failure is still the first failing row's, so tokens and errors are unchanged; on an M5 Ultra with the DFlash2 drafter, sampled decode at one stream is 4 to 8.8% faster (#2644). Thanks to @mrv777.

- Living Weights on Nemotron (`--slide`): a lesson's pocket goes only into the layers whose subject rows carry it and that pass the full verifier, and the neurons it needs are shrunk to what passes; a rejected shrink restores the exact rows and count. A correction overturns a shipped answer with an explicit additive fit and checked deletion of the original, and one that depends on an earlier pocket is refused by fact before fitting. Pretrained weights and kernels are unchanged, and serving without `--slide` is unchanged. `TF_SLIDE_POCKET_DUMP` is read through the environment table: blank means no dump, and a path with a control character is refused at startup (#2025, part of #1968).

- Server: `--unix-socket PATH` serves on a new owner-only Unix socket instead of TCP, so a local client reaches the server without its requests crossing a network interface. It refuses an existing path, a parent directory others can write to (unless sticky) and the TCP-only flags, checks each peer's user, and is refused on Windows (#2389). Thanks to @kotevcode.

- CUDA EXL3 experts: a second finish kernel clamps silu's output, `min(silu(g), L) · clip(u, −L, L)` (Ling-3.1's order), when a layer asks for it. Every other format, and bf16 or SwiGLU-OAI activations, refuse it before launch, and existing EXL3 models are unchanged (#2336). Thanks to @ajensenwaud.

- HIP: the load-time timing of shared rows gives every row a distinct token for any vocabulary; its stride is 7919, or 7927 where 7919 divides the span, so it is always coprime to it. Served vocabularies keep #2652's ids and table, and tokens are unchanged (#2661, refs #2605). Thanks to @AbrahamCete.

- CUDA: GLM-5.3-Flash's KDA chain and saved-row replay move into a shared module (`cuda.kda`) that another family can use. GLM keeps its projections, convolution commit, wide selection and state; its kernels compile to the same SASS on every supported GPU and launch with the same arguments, so replies are unchanged (#2142, part of #850).

- Tests: host pins for the request and flag limits `docs/api.md` and the README state, for the documented `TF_*` defaults and for the server's flag dependencies; a rerunnable replay of the prompt cache's eviction on agent-shaped traces; and a timing of the prompt cache's snapshot index at 16 to 1,024 kept states (#2305).

- GLM-5.3-Flash on CUDA: a malformed `TF_GLM_*` switch, `TF_GLM_DFLASH2_LOG` or `TF_GLM_DRAFT_CONFIDENCE` is refused by name, with its value and accepted form, when GLM's CUDA engine opens, before ranks, NCCL or weights, and only GLM's start stops. The seven 0/1 switches (`TF_GLM_HOST_CACHE`, `TF_GLM_VISION_HOST`, `TF_GLM_PROF`, `TF_GLM_LINK_WAIT`, `TF_GLM_PDL`, `TF_GLM_PROMPT_STEPS`, `TF_GLM_SHARED`) take exactly `0` or `1`, where text such as `off`, `yes` or `2` used to keep the default; the log takes a whole count and the draft cut a finite value from 0 to 1. Five of the seven are registered in the profile registry as identical, so the startup line and `capabilities` report them; the timing `TF_GLM_PROF` and the waited `TF_GLM_LINK_WAIT` move no memory and change no arithmetic, so they stay outside it. Unset and blank keep every default (#2578, part of #2370). Thanks to @Bizuayeu.

- Diagnostics: opt-in startup receipts show Qwen3.8-27B Metal load phases and the first completed non-streaming reply; a bounded standard-library harness records resource counters and the reply token hash (#2640).

- GLM-5.3-Flash on CUDA: with `TF_GLM_PREFILL_REDUCE=gather`, the start-up warm-up posts one all-gather of the same size in place of a send and receive to every peer, so four GB10s cabled as a switchless ring start and serve at TP=4, where on main every rank stopped within 122 s after `ibv_modify_qp` timeouts; split and ring keep their warm-ups, and a gather warm-up that doesn't finish is named in the timeout line as the all-gather (#2569, refs #1971). Thanks to @Bizuayeu.

- Tests: `zig build sessions-writes` drives the real DeepSeek-V4.1 session store and NVMe index with synthetic agent, chat and mixed loads, no model or GPU, and reports the bytes an enabled tier (`TF_DSV41_SESSION_DISK`) writes and reads back; a nightly Linux test checks its accounting against real files (#2660).

- Server: Responses grows each output item in one buffer, and Kimi chat advances its native stream parser through new tokens, so a long reply retains memory in proportion to its length while every reply and streamed event stays the same (#2642).
- Tests: `zig build bench-stream-scan` streams a 32k-token thinking reply (30,000 reasoning tokens, prose, then a Qwen XML, GLM, DSML or Kimi call) through the chat, Messages and Responses routes one or two tokens a chunk, and prints each chunk's request-thread time at 1k to 30k reply tokens, each whole-text step's time and the request arena; under `test-server-cpu` a 300-token reply checks that each route's streamed reasoning, prose and call arguments equal the reply's (#2642).

- Drama 3 on CUDA opens again: since 1.2.5 every open stopped with "unregistered strict environment parser", because `TF_DRAMA_PREFILL_ROWS` and `TF_DRAMA_REFERENCE_CACHE` had no rows in the environment table. Both now have rows, with their old forms and bounds, and the build refuses a CUDA setting read through the table without one (#2658).

- Internal: a CUDA family's disk module may declare its own host staging permit for kept states, so a family that declares a larger permit can write kept states over 1 GiB (none does yet); families that declare none keep the fixed 1 GiB permit, and every family on main today is unchanged (#2337, refs #850). Thanks to @ajensenwaud.

- HIP: the load-time lane cost table times a shared bucket of two or more windows with a distinct token a row, as shared rounds draft, instead of one repeated token; lone windows, one-row buckets, the head step and one-stream servers are timed as before, and only the drafted width can change, never a token (#2652, part of #2605). Thanks to @AbrahamCete.

- Tool calls: a reply's call ids are an HMAC-SHA-256 of the request (its body without `stream` and `stream_options`, the choice and the call's index) under the server's key instead of chance, so a request replayed to the same server gets the same ids, a stream names its calls as its whole reply and the next turn's background prefill do, and a Responses `fc_` item id follows its call; without the key an id reveals nothing about the request, and byte-identical requests under one key share ids. The key is random per server process unless `TENSORFOLD_TOOL_ID_KEY_FILE` names an owner-only file of at least 32 bytes, which keeps ids across restarts and machines sharing it, and a missing, unreadable, linked, shared, short or oversized file stops the start by name; the `call_` and 24 hex digit form and the ids a client sends back are unchanged (#2653).

- Qwen3.8-27B on Metal: after a sampled or grammar-masked verify, only the drafter's taps and head row are waited for; the target's keep follows on the same queue and is settled before the next verify, so the drafter starts sooner, with every token unchanged. One stream sampled with DFlash2 on an M3 Ultra: chat 75.3 to 77.0 tok/s, code 106.8 to 109.8 (#2646). Thanks to @mrv777.

- DeepSeek-V4.1: attention heads now split by whole output groups through the cluster layer's shared rule, so three ranks plan and write their rank files (24/24/16 heads). Serving still refuses three ranks. Worlds 1, 2 and 4 plan, write and emit the same bytes as before; their prepared folders rebuild once after upgrading (at TP=2 on two GB10s, about 10.5 minutes to weights loaded on that first boot against about 5 later) (#2325, part of #2270). Thanks to @Defilan.

- Flash Next on Metal: 6- and 8-bit row products at 2 to 4 rows take two outputs a simdgroup instead of four, so a 2,560 x 6,144 product at 2 rows takes 23.5 to 28.7% less time across the three widths measured on an M3 Ultra; served decode is 2.4-6.0% faster with every token hash equal to main's (#2593, fixes #2553). Thanks to @abm9111.

- Flash Next on Metal: a prompt chunk's staggered segments wait until the first queue reaches the chunk, so after idle, when macOS has unwired the model, a later segment no longer sits in a GPU wait past macOS's limit and fails the request with HTTP 500; warm, nothing waits (#2594, fixes #2552). Thanks to @abm9111.

- Conformance: a Qwen3.8-27B Metal profile (`tools/conformance_profiles/qwen27-metal.json`, MLX 4-bit with its DFlash2 drafter), and a profile key `fresh_env` the kit sets only on the cache-off servers it starts, so the resumed gate gets a fresh side on Metal, where a stream resumes its own last chat (#2612, refs #2460). Thanks to @jschmied.

- Sampling: Qwen 3.5's host draw over a bf16 logits row (a reply's first token and every grammar-masked row on CUDA, every draw on Metal) moves unchanged into `core/lanes` as `sampling_row`, the first step to one sampler; token ids equal main's on greedy, sampled and JSON-schema requests (#2606, part of #2460). Thanks to @jschmied.

- Internal: GLM's CUDA SwiGLU rows move into their own header (`glm_swiglu.cuh`) with a template for a clip after silu; GLM's machine code is the same on all five SMs (#2627, part of #850). Thanks to @ajensenwaud.

- KV pool: a pool's stores join an ordered chain of reclaim hooks, asked in join order until enough pages are free, and one lane host counts a pool, a second refused; every load keeps its one store and its calls, so nothing changes yet (#2528, part of #2444). Thanks to @jschmied.

- Configuration: an empty or whitespace-only `TF_KV_SHARE` (GLM, Gemma 4, Kolibri-1, Spark2.5 and Dory), `TF_QWEN_KV_SHARE` or `TF_QWEN_KV_PAGES` now means unset instead of stopping the start, and GLM's empty or whitespace-only `TF_KV_PAGES` means `auto` instead of `on`, which paged a single GPU and was refused with `--kv-cache fp8`, as `docs/environment.md` says of every variable; so at TP=1 a blank `TF_KV_PAGES` with `TF_KV_SHARE` or `TF_GLM_KV_HOST=1` is refused by name, as an unset one is (#2629, refs #2591). Thanks to @Bizuayeu.

- Prompt cache: `--learn`'s Imprint finds a record by key through an index kept in step with its record list, instead of scanning every record; at 100,000 learned states `find`, `has` and `touch` take 0.02-0.04 µs (were 24-57 µs) and a reload's finds after another writer appended take 1.1 ms (were 2.8 s) on an M3 Ultra, with the same records, order, clock and eviction victims (#2351).

- GLM on CUDA: an empty or whitespace-only value of a setting its start checks (`TF_GLM_MM`, `TF_GLM_PROMPT_TAIL`, `TF_GLM_L2PF` and every other name `Settings.read` reads) now means unset, as `docs/environment.md` says of every variable, instead of stopping the start; `TF_GLM_ROCE_MULTI=1` with a blank `TF_TP_MASTER` or `TF_TP_KERNEL_IMAGE` is refused by name, at TP=2 too, as an unset one already was (#2636, follows #2487, refs #2591). Thanks to @Bizuayeu.

- Tests: compile the runtime Metal libraries selected by each loader, check every registered launch mapping on the host, and keep limits running across unsupported probes, including LFM2 masked prompts on pre-M5 devices (#2601).

- Flash Next on CUDA: a verify window's and a shared pass's graph keys now hash every buffer of a stream's tables (states, conv windows, caches, the n-gram tail and the page table), not its first state and cache alone, so a layer whose buffers move alone gets a new key; nothing changes on today's placement, where every state moves together (#2577, part of #2444). Thanks to @jschmied.
- GLM on CUDA keeps the live half of the double-buffered KDA recurrent state alone in a kept prompt state with `TF_GLM_KV_LIVE_HALF=1` (opt-in): the other half is the state before the last prompt chunk or decode window and nothing reads it after a restore, so a kept state is smaller on GLM-5.3-Flash at TP=2 with replies unchanged, drafts off and on (#2181, part of #2180). Thanks to @volfco.

- HIP: a host-to-device upload copies in 64 MiB windows, one synchronous copy each, so a multi-GiB tensor no longer has its whole source pinned by one copy; uploads of 64 MiB or less are one call as before (#2584). Thanks to @ElderOrb.

- Benchmark clients: `bench_concurrent` and `bench_openai` time content, reasoning and tool-call output separately (header arrival, first payload line, first output of each kind, last output) and classify how each reply ended; token ITL/TPOT is null where a chunk carries several tokens instead of padded with zeros, with a labelled chunk-level estimate beside it; the prompt/TTFT rate is `prompt_ttft_tps`, with `prefill_tps` still written and marked deprecated because it is not the engine's prefill time; `tools/conformance.py --stream` runs the exactness gates on streamed replies, read by the same reader (#2625).

- CUDA: every IEEE RMS scale (Flash Next's hyper-connection norms, DeepSeek-V4.1's mHC norms, Gemma 4's vision-tower q/k norm, the GDN prep and the shared norm and q/k row kernels) calls one helper, `tf_common::rms_scale`, with the same operations in the same order; every function's SASS is unchanged (#2580, refs #548). Thanks to @jschmied.

- Tools: `tools/evidence_index.py` indexes what each family and backend has qualified, on which commit and at which grade (host test, kernel contract, served exactness, served speed), from the receipts, the optimization ledger, the conformance profiles, the LFM3 ledger and known issues, marking stale and failed evidence and leaving unknown what no source states; `docs/evidence-index.md` explains it (#2621).

- File readers: a session header or Linux restart file, a conversation state file (`--learn-conversations`), an EXL3 or EXLR8 trellis shape, a DeepSeek V4.1 Engram sample row, a tokenizer vocabulary id of 2^32 - 1 or an added token whose new id would reach it, or a Drama codec storage count is refused with that reader's own error when its size, offset or count arithmetic would pass 2^64 (2^32 for u32 fields), where a safe build used to panic and a fast build could carry the wrapped value on; every file accepted before reads the same (#2618).

- Sessions: a restore whose page write fails mid-stream gives back its in-flight reads' staging buffers, and a restore or park whose tier queue can't take it leaves no job behind, so neither the next restore nor the store's shutdown waits forever after such an error (#2615).

- Release checks: the release privacy check walks the release tree in Zig instead of the shell `find`, so `check-release-paths` and packaging's stage check run one implementation on every host and `zig build test` passes them on native Windows; a symlink, an unreadable entry or a file given as the tree is refused, and the build-root argument is validated first, as before (#2614).

- Flash Next on Metal and HIP: a config.json with a legacy `rope_scaling` (YaRN) is now refused by name; a `rope_parameters` type other than default already was. LFM2 and Kolibri-1 now refuse both forms; these readers ran plain RoPE under them. Metal Flash Next still takes its long-context factor from `--context` (#2604, fixes part of #2562). Thanks to @jschmied.

- Linux: Nemotron's io_uring checkpoint reads no longer hang at shutdown after a failed read: a read that failed partway through a batch dropped the batch's later completions, so the loader waited for them forever; every reader that skips the page cache or reads aligned spans (row and state files, Engram rows, session restore, Qwen-VL tower reads, GLM's rewritten tensors) is now tested with a file that ends inside the span it asks for (#2608).
- Flash Next on Macs: a server that closes while replies wait in its queue now ends each of them even when it is short of memory. Before, a failed copy at close left those requests waiting for good (#2616).

- Loading: a direct-read load on Linux (Spark 2.5, Dory, Gemma 4 and Qwen 3.5 on CUDA, and Flash Next's experts) that fails now names its reader's error, such as `EndOfFile` when a shard ends inside a tensor it reads, where every failure was `ReadFailed` (#2610).

- Linux: DeepSeek-V4.1's Engram shards on a disk or file system whose direct reads need more than a 512-byte sector, such as an NVMe with 4 KiB logical blocks, are read through the page cache, where their rows used to fail with `EngramReadFailed`; on a 512-byte disk they are still read direct, extent for extent (#2610).

- Linux: an O_DIRECT checkpoint read of a file that ends inside the span it asked for reports end of file instead of a failed read, so Nemotron's CUDA checkpoint reads stop on `EndOfFile` as they do through the page cache (nothing was copied wrongly before; the error was wrong); Nemotron's CUDA checkpoint reader is now tested on the host through its io_uring ring and its thread fallback (#2602).

- CI: the nightly Linux host job now runs `test-pdf-worker`, which spawns the PDF render worker from a Linux `tensorfold-native` so its `/proc/self/statm` memory cap and `close_range` run on Linux, and the new `test-cache-fit-host`, so the prompt-cache budget's no-footprint branch runs at all; both steps were only built into the macOS Metal graph before (#2599).

- CI: the root `pytest` run has its own job beside `zig` instead of being its last step, so a pull request no longer waits about five minutes for it after its tests (#2586).

- CI: the nightly Linux host job gains a `deepseek` shard that runs `test-dsv41-device-host` and `test-dsv41-forward`, so their Linux-only tests (a forked child's fail-fast exit and the model's fatal callback without an NCCL abort) run on Linux; Linux CI only compiled these steps before, and macOS skips those tests (#2598).

- CI: the macOS nightly also runs the thirteen host checks it held back until their fixes landed (#2566).
- Qwen3.8-Flash-Next on HIP fills prompts faster: its sparse attention runs a new kernel with the same sums as before (a lane holds a key and several heads, keys and values staged in the cache type), so replies are unchanged. On a Radeon 8060S (gfx1151), cold prompts of 2,048 to 63,728 tokens finish 29 to 41% sooner; other HIP chips take the kernel but weren't run (#1877). Thanks to @gilby.
- Tools: `tools/conformance.py` runs a family's served exactness gates from one JSON profile, and its own tests run in CI with no device against a fake server with planted faults (#2467, part of #2460). Thanks to @jschmied.
- Dory on CUDA: `TF_KV_SHARE=n` keeps prompts as the KV pool's pages in the shared prompt store, with at most n kept states, and turns Dory's pool pages on, which the startup line and kept states' identity record; with `TENSORFOLD_DORY_KV_PAGES=0` it refuses to load, naming both. On a GB10 at a 4 GiB cache a 25-turn agent loop to 34K tokens spends 10.7% less time to first tokens, about level up to 12K; 72 round-robin turns keep 29 states against 9, with the same replies. Unset: as before (#2430, part of #1657). Thanks to @jschmied.
- KV pages: a window past the engine's own counts toward the startup plan only where the room past every stream's fixed part holds it whole, so a band of rooms no longer leaves the prompt cache at 0 (at TP=3 with a 1M window and eight streams, 38.84 to 40.21 GiB; at TP=2 with a 630,000-token window and four streams, 15.66 to 16.32 GiB), where since #2358 GLM's `TF_KV_SHARE` kept no state and a long conversation's next turn ran its whole prompt again; outside the band every plan is as before. The startup line and the warning under four states a stream name a `--prompt-cache-gib` only where it keeps all the states, and otherwise how many the room holds (#2568, fixes #2547). Thanks to @Bizuayeu.

- Experimental: `TF_TP_ROCE_RELAY=1` relays gathers through both neighbours on a four-rank RoCE ring with no opposite links, instead of falling back to NCCL. Gathers above `TF_TP_ROCE_RELAY_MAX_KB` (64 KiB by default) stay on NCCL, GLM's `TF_GLM_ROCE_MULTI=1` takes the relay at its own TP world, and `tf-tp-test suite` says which NCCL point-to-point references it leaves out on such a ring. Off by default; two- and three-rank behaviour is unchanged (#2352, part of #1971). Idea from blaesus/dsgx10 and Picom-code/pico-dgx-ds4's ring notes, our implementation. Thanks to @Bizuayeu for the four-GB10 ring run.

- Server: control bytes in a request line (C0, DEL and C1) appear in the access log as `\xNN`, so a request cannot write terminal escapes into the log; ordinary lines are unchanged (#2542). Thanks to @chadhurley25075-png.

- Conversation states on disk (`--learn-conversations`): a row block or state file whose trailer sizes add up to the file only by wrapping past 2^64 is refused at open, where a safe build used to panic on the addition (#2544). Thanks to @chadhurley25075-png.

- Docs: the Zig family recipe lists the core blocks on main with the call a family makes, and a family PR names each block it doesn't use and why (#2468, part of #2460). Thanks to @jschmied.

- Spark 2.5 and Dory on CUDA now load with direct reads by default: cold loads on a GB10 go from 5.03 to 2.53 s (Spark X2.6 4B 8-bit) and from 4.79 to 3.79 s (Dory NVFP4), with the same replies. Host memory while loading goes down, not up: Spark peaks at 6.1 GiB resident instead of 9.4 and fills 0.4 GiB of page cache instead of 4.2; Dory stays at 7.2 GiB resident and fills 1.3 GiB of page cache instead of 3.1. `TF_LOAD_DIRECT=0` keeps the old mapped reads (#2449). Thanks to @jschmied.
- Docs: the LFM3 recipe records the qualification rules agreed on #1491 — native no further from an independent FP32 run than the reference's own BF16, partial not pass, the default cold-prefill gap does not block the opt-in text slice, 32k stays the release tier pending long-context fidelity — with an evidence ledger beside the run instructions and a receipts README link (#2416, part of #1491). Thanks to @akol1.
- Gemma 4 on CUDA: `TF_GEMMA4_CHUNK_SOLO` is a row of the shared environment table and the lane core's `solo_chunk.rows` reads it there instead of calling `getenv`, so a family that follows the new-family recipe registers its solo variable; empty or whitespace-only text now means off, where it was refused, and every other value is accepted or refused, with the same message, as before (#2587).
- Tests: the Qwen4 QSA backward host test prints its coordinate and direction lines only when a comparison fails, naming the first wrong one; a passing run writes nothing to stderr (#2588, refs #1852).
- Docs: the image page says how the native server finds libjpeg-turbo 3.1.4.1 for JPEG input: why the release is pinned, the search order and file names on macOS, Linux and Windows, `TENSORFOLD_LIBTURBOJPEG`, and what each startup warning means (#2590, refs #2581).

- Internal: a shared CUDA MLA image (`mla.cu`, `cuda.mla`) for a 512-wide latent with 64 roped columns and 128-wide heads, on GLM's MLA chunk bodies, which move from `glm_glue_attn.cuh` to `common/mla.cuh` and which GLM instantiates at its old widths (its SASS is identical on all five SMs). Its seven launches are under the kernel contract with a planted control each, and `test-mla-twins-host` keeps its absorb, expand and per-rank merge GLM's kernels token for token. No family calls it yet (#2335, refs #850). Thanks to @ajensenwaud.

## 1.2.5

- Tests: the Metal kernel contract's limits pass checks LFM3's launches again; a rebase had dropped their five entries, so the pass stopped at the first LFM3 record with `UnknownLaunch` and left every later launch unchecked (#2565).

- HIP on gfx12 under `--policy matrix=on`: Qwen3 dense checkpoints load each layer's gate and up weights interleaved as one weight, and one WMMA product writes silu(gate) x up in its epilogue instead of two products and `tf_silu_mul`, with the same output bytes: Qwen3-Reranker-8B ranks 16 documents in 0.657 to 0.661 s instead of 0.687 to 0.697 s and Qwen3-Embedding-8B embeds 100 short inputs in 0.901 to 0.911 s instead of 0.940 to 0.948 s on a Radeon AI PRO R9700. The kernel contract checks both fused blocks byte for byte against the unfused launches (#2344, part of #2245). Thanks to @laudney.

- CI: a nightly job runs 246 host checks that no per-PR job ran, in shards with their wall times printed; any move to per-PR jobs waits for those timings, and checks that fail on main stay off until fixed (#2554).

- Nemotron on Macs: the masked prompt shader calls the shared `tf_paged_row` page helper again, so its prompt sources compile; #2322 had used the name #2283 retired the same day (#2548). Thanks to @jschmied, who found it first (#2538).

- Tests: `zig build test-nemotron-prefill-metal-sources` compiles Nemotron's prompt shaders, including both masked sources, on a Metal device without running a kernel; it fails on the helper name #2548 fixed (#2558).

- Lean checks: each new `getenv` that reads a setting outside `platform/environment.zig` is a finding, whether it
  names a `TF_*` or `TENSORFOLD_*` setting or takes a name that is not a literal, such as a new wrapper's; the reads
  already on main, test files and `test` blocks are not counted (#2521). Thanks to @Bizuayeu.

- GLM-5.3-Flash on CUDA: with `TF_KV_SHARE=n`, the startup plan sets a slot aside in the prompt cache for each kept
  state, so the cache keeps as many states as its slots hold, up to n, and the rest of it goes to the KV pool's pages;
  the startup line names the slots and, when fewer than n fit, the `--prompt-cache-gib` that holds them all. A value
  over 64 now stops the server, and a cache too small for one slot (for example `--prompt-cache-gib 0`) keeps none,
  where before it kept them outside the plan (#2358). Thanks to @Bizuayeu.

- Metal: Qwen3.8-27B MLX 8-bit (groups of 64) reads its stored code bytes directly and reuses the producer's group sums, so cold prompts are 18% faster on an M5 Ultra (649 to 768 tok/s at 2K) and decode 7.6% faster, with the same tokens and prompt state; the lane kernel contract now pins 8-bit group-64 projections at the 27B's K splits (#2518, fixes #2502).

- Qwen3.8-27B on Metal: `TF_PROMPT_ROWS` below 128 is refused with the prompt floor's own message again ("needs at least 128 prompt rows … or unset it to use the device default"), which #2405's startup check had replaced with the generic one; other malformed values keep the generic refusal (#2527).

- HIP on gfx12 under `--policy matrix=on`: the WMMA dense tile's big block takes 64-deep K steps over swizzled LDS rows, its barrier orders LDS only, and the block thresholds are tuned again, with the same output bytes: Qwen3-Reranker-8B ranks 16 documents in 0.694 s instead of 0.720 s and Qwen3-Embedding-8B embeds 100 short inputs in 0.946 s instead of 0.974 s on a Radeon AI PRO R9700. The kernel contract checks the new block byte for byte against a frozen copy of the old one (#2280, part of #2245). Thanks to @laudney.

- Fabric: a pipeline stage's boot and round loop is one driver, generic over an engine binding, so a CUDA, Metal or XPU stage writes only its binding; nothing uses it yet. A stop that a middle stage receives now passes on instead of closing the ring (#2241, part of #2005). Thanks to @jschmied.

- KV pool: a family's pool group is sized by its bytes a row over its layers, and a paged CUDA load prints a line of each group's rows a page and padding (the window rows and state outside or beside the pool too), apart from the unchanged startup line; every family stays one group, so nothing else changes (#2532, part of #2444). Thanks to @jschmied.

- LFM2 on Macs: a non-finite value in a masked key no longer reaches prompt attention's output; finite prompts run the same kernels as before, and prefill is level (#2328, part of #1009).

- Release checks: on a Mac release binary, the private-path check no longer reads the code signature's hash bytes as text, which could spell a path by chance and refuse a clean build; the exact build root is still refused there, and the rest of the binary is checked as before (#2524).

- Server: an explicit thinking off that the chat template can't honour (a `</think>` marker but no switch or closable generation prefix) is refused with HTTP 400 before submission instead of being accepted and ignored (Messages' `count_tokens` refuses it too; a request that leaves thinking out, and a server or template default, never counts as one); model discovery's capabilities end with `thinking_switch` and `thinking_applicable`, and the benchmark clients read them (#2170, part of #2139).

- Prompt cache on CUDA: `TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS=1` (opt-in) stops a new prompt state from evicting another conversation's kept states; one that fits only that way isn't kept and is logged once with its size and the free budget, while a conversation's own older states still make room; tokens are unchanged, and unset or `0` keeps today's eviction order (#2533, part of #2346).

- Docs: the Qwen3.8-27B recipe says native serving reads MLX 8-bit with the same flags, above the Python 0.6.6 part where it was easy to miss, with 8-bit measured against 4-bit on an M5 Ultra: decode at its bytes' speed, prompts slower (#2506, refs #2502).

- Qwen3.8-27B on Metal: the startup line reports how many admitted streams DFlash2 can draft for, using `TF_Q27_DRAFT_SLOTS` and the loaded drafter (#2526). Thanks to @mrv777 for the report.

- Tests: the Qwen4 and Dory RoPE word-table host tests print their tables only when a comparison fails, naming the first wrong word; a passing run writes nothing to stderr (#2519, refs #1852).

- Tests: thirteen host checks that no CI job ran build and pass again on main (`test-cluster-loader-host`, `test-exl3-prompt-host`, `test-gemma4-affine-host`, `test-gradient-serving`, `test-metal-residency-owners-host`, `test-qwen-serving-host`, `test-qwen27-prompt-rows-host`, `test-qwen35-draft-frame-host`, `test-qwen35-draft-trees-host`, `test-qwen35-frame-invalidation-host`, `test-qwen35-live-host`, `test-qwen35-shared-host` and `compile-qwen35-rounds-host`). Their fixtures and build wiring follow the engine's current imports and fields. The prompt-rows and shared-cache tests expect blank values as unset, #2527's floor message and the overlaid log and before block. The source pins take main's reviewed bytes, each naming its commit. No engine code changes (#2550).

- Nemotron on Macs: a non-finite value in a masked key no longer reaches prompt attention's output; finite prompts run the same kernels as before, and prefill is level (#2322, part of #1009).

- CUDA: refresh 13 nvcc 13.0 register-count slots from the five-SM 13.0.88 audit; other compiler records and launch geometry stay unchanged (#2515, refs #2395). Thanks to @yguanrong, who found them.

- GLM CUDA: bound the peer warm-up after NCCL init to 300 seconds, aborting its communicator and stopping every rank with the missing peers and observed transport named on timeout (#2512, part of #2347).

- CUDA: a request's finished stats list its prompt pass widths for Qwen3.8-27B, GLM-5.3-Flash and Nemotron, appended
  only where a pass is already known complete. Telemetry only (#2304).

- Prompt cache: `--learn` opens its kept states in linear time, so a server with 100,000 of them starts about 5 s sooner (8.0x faster open); the same states and files are kept (#2316).

- Prompt cache: Qwen3.8-27B on a Mac widens its disk staging permit to one whole state at its `--context`, within `--prompt-cache-gib`, so conversations past about 13K new tokens are saved instead of skipped with `StagingFull` (#2388). Thanks to @mrv777.

- Qwen3.8-27B on Metal, opt-in with `TF_QWEN27_METAL_KV_PAGES=1`: each stream's keys and values live in pages of the core KV pool instead of their own buffers, with greedy replies equal to the default on one and two streams. The paged kernels derive from the default ones, and the page code they share with Nemotron and Qwen3.5 lives in one place. It has run only on Macs without tensor units, and an M5 refuses the flag by name (#2283, part of #1293). Thanks to @jschmied.

- DeepSeek-V4.1 on CUDA: a malformed `TF_DSV41_*` setting is refused by name, with its value and accepted forms, before `Model.open` or the AOT planner touches memory, a device or the checkpoint, and every later reader refuses the same text. All 81 settings are rows of the one startup table, where empty means unset (`TF_DSV41_LOOKUP` keeps blank-means-off). Switches that read any text but `0` as on now take exactly `0` or `1`, so a word such as `false` is refused rather than read the other way; `LOOKUP` and `LOOKUP_PLAN` keep `true`/`on`/`yes`/`false`/`off`/`no`, and `MHC_DEFER` and `COEF_LATE` keep them except `no`, which their old readers read as on and which is now refused. Unset defaults are unchanged, a malformed `TF_DSV41_TOOL_GRAMMAR` is refused at server start unless `TENSORFOLD_TOOL_GRAMMAR` is set, and the startup line shows explicit DeepSeek settings (#2379, #2516, part of #2370).

- HIP: the qwen3_5 engine reads a DFlash2 drafter's `config.json` against its target, refusing one it can't run, and loads its weights as MLX affine-4 (groups of 64), quantized on the host as on CUDA; the server doesn't use the drafter (#2477, part 2 of #1563). Thanks to @AbrahamCete.


- Developers: the Qwen3.8-27B Metal lane check initializes its draft-window costs, and the served draft check compares complete state at the consumed token position, with exact tokens and state bytes (#2493, fixes #2489).

- Developers: `tools/zig/cuda_fragment_check.py` checks that every CUDA file including a function-body fragment (`prefill_attention_rows.cuh`, `qmmf_body.cuh`) declares each name the fragment uses on the branches it takes, and CI runs it beside the shared-copy check. It would have caught the `contract_controls.cu` miss that #2463 fixed (#2495).

- Qwen3.8-27B on Metal: prompt chunks of 64 rows or more run the wide projections with half operands wherever a tile's inputs all fit in half, and keep the fp32 path elsewhere, so logits and tokens are bit-equal to before. Served prefill is 2-3% faster on an M3 Pro (#2435, part of #2089). Thanks to @jschmied.

- TensorFold Web, a local web app laid out like the desktop app (chat, models, learning, API, benchmarks and clusters), ships in `web/`: `tensorfold web` opens it, and on first use offers to install it with uv, pinned to the same release. It starts without network access and reaches a local engine directly, past any system proxy. Adopted from @Stephane-Eden's TensorFold Studio (#1781). Thanks to @Stephane-Eden.

- Qwen3.5 and Qwen3.6 on HIP: sampled streams with an MTP head draft on the graphed head chain, as greedy ones do, with the same tokens: 4-12% more tokens a second at temperature 1.0 on gfx1151 (#2455). Thanks to @AbrahamCete.

- Docs: the README's GLM-5.3-Flash Jammed XSFP4 row and the GLM CUDA recipe say it shipped in 1.2.3, not "on main after 1.2.2" (#2451, for #2041). Thanks to @Bizuayeu.

- Qwen3.8 Flash Next on CUDA: the BF16 prompt tile from #1846 stages less shared memory, gathering K32 fragments from the same K64 pack, with the same output bytes and replies, so cold prompts take another 0.6 to 0.8% less time on a GB10 (#1848). Thanks to @laudney.

- Developers: `python3 -m pytest` from the repository root passes on a machine with no GPU or optional packages, skipping the rest with their reason; the release packaging check is renamed `tools/release/check_package.py`, and CI runs the root suite (#2471).

- Docs and packaging: `docs/evaluating.md` says what to pin when testing TensorFold, how to check a release's signature, which tests need no GPU and which requests to replay. A root `Dockerfile` installs a Linux release archive once its checksum, its signature and every unpacked file's checksum match (#2472).

- Packaging: each stable release publishes its signed Mac arm64 desktop runtime and catalog update from main's release tooling, keeps earlier compatible runtimes and checks each published download (#2465).

- Qwen CUDA with `TF_QWEN_REQUEST_STARTS=1` keeps later chat turns at their own history starts even when turns are under 256 tokens apart. On a GB10 with Qwen3.8-27B 4-bit and its DFlash2 drafter, turns 2 to 6 of a six-turn chat reuse 1,450 to 1,635 prompt tokens and take 0.72 to 1.33 s instead of 1.78 to 2.10 s. Prompt passes are cut at replies, so a reply can differ from the unset path's; resumed and fresh replies match with it set. Unset, nothing changes (#2176, carries #1335). Thanks to @gilby.

- Internal: HIP record counts are pinned in their owning arrays, so additions in different record files no longer edit a shared numeric total; omissions still fail host checks (#2461).

- Qwen3.8 Flash Next on CUDA: BF16 prompt projections take a 128 × 128 tile on sm_12x wherever it fills the GPU, with the same output bytes, so cold prompts take 3.7 to 4.4% less time on a GB10 (NVFP4 checkpoint, 4K to 34K tokens) and replies are unchanged. `TF_FLASHNEXT_PROMPT_BF16=0` turns the tile off, and `TF_FLASHNEXT_PROMPT_ROWS` is gone (#1846, refs #1754). Thanks to @laudney.

- Internal: `compile-cuda-tests` builds again: `contract_controls.cu` declares the prompt-row fragment's page settings (`P256`, `IDS`, `PN`), which its control kernel predated (#2463). Thanks to @jschmied for the report.

- Gemma 4 on CUDA: `TF_GEMMA4_CHUNK_SOLO=2048` lets a prompt pass take 2,048 rows while no other stream decodes, and the usual 512 once one does. On a GB10 with the 31B 4-bit release, a cold prompt's first token comes 4-5% sooner at 4K, 16K and 29K tokens with the same replies, and decode beside a filling prompt keeps its pace; each stream's sliding rings take 1.2 GiB more. Unset, nothing changes. Flash Next's and GLM's solo-chunk switches share the same rule (#2391). Thanks to @jschmied.

- Qwen3.5 and Qwen3.6 on CUDA: a stream's rollback copy shares its verify log's bytes, which are never in use together, so each pooled stream takes 147 MiB less on Qwen3.8-27B and 61 MiB less on the 35B-A3B; tokens and speed unchanged (#2447, part of #1943). Thanks to @jschmied.

- LightOnOCR-3 on HIP: the native server reads `lightonai/LightOnOCR-3-0.8B`, `-1B` and `-4B` (BF16) and serves their OCR prompts, token-exact against Transformers apart from exact BF16 logit ties. With `--vision`, a Qwen 3.5 checkpoint whose vision tower the HIP engine can't read (such as Qwen3.5-9B's 72-wide heads) is now refused at start-up instead of answering each image with a 400 (#1965, for #1927). Thanks to @laudney.

- Vision: PDFs with text no longer crash the render worker on macOS. Each PDF renders in a fresh server process instead of a fork, under the same limits and deadline, and a crash is reported by its signal, for example "the PDF render worker crashed (SIGSEGV)" (#2393, fixes #2385). Thanks to @gordonhatusupy for the report.

- Internal: the kernel contract registry's record joins live in `registry_joins.zig`, one module a line, so PRs that add records for different families no longer edit the same line; every registry list is unchanged (#2422).

- Gemma 4 and Nemotron on CUDA skip the last layers of prompt chunks nothing reads, so cold prompts on a GB10 start up to 3.9% sooner (Gemma 4 31B: 1.3% at 2K, 3.1% at 16K, 3.9% at 32K; Nemotron 3.5 Lightning level at 2K, about 2% sooner at 16K and 32K) with identical replies; a drafter that reads those rows keeps them (#2432, refs #2165). Thanks to @jschmied.

- Docs: the optimization ledger's cells for #2011, #2162 and #2282 say they landed (#2431). Thanks to @jschmied.

- Lane host: a stream paid for at load is a standing lease in the KV ledger, so a family that reserves per request admits as many full-window requests as the window plan, not one fewer, with its prompt cache beside the same streams. No family on main reserves per request, so every current load is unchanged (#2434, part of #1943). Thanks to @jschmied, and to @laudney for finding the gap.

- Nemotron Parse on HIP: NVIDIA's `nvidia/Nemotron-3.5-Parse` is served natively from its FP32 safetensors on a Radeon AI PRO R9700 (gfx1201), with its RADIO image tower, convolution neck and mBART decoder. The token arrays match the published example's on three pages, plain and drafted, solo and concurrent (#1941, part of #1927). Thanks to @laudney.

- Developers: `test-glm-snapshots-host` runs under `test-cuda-host`, as GLM's other host steps do, so CI builds it and runs its 21 tests; its host-cache test sets `TF_GLM_HOST_CACHE` through `sys.env`, so it links on Windows too (#2426, refs #2031). Thanks to @Bizuayeu.

- Conversation states on disk try one shorter compatible state after a refused read before prefilling from the start. Failed reads still name their error in the log (#2442).

- Internal: `families/glm/cuda_engine.zig` goes from 600 to 547 lines; GLM's CUDA chunk gate (`gate`, `shared`, `Wait`) lives in `cuda_engine_gate.zig`, unchanged (#2429).

- Internal: `zig/src/native/metal.zig` goes from 593 to 546 lines; six helpers the Metal open paths share (`chip`, `deviceName`, `prefillStep`, `modelContext`, `gibBytes` and the chip classes) live in `metal_shared.zig`, unchanged (#2425).

- Profiles: Dory's, Gemma 4's and Spark2.5's pool-page switches (`TENSORFOLD_DORY_KV_PAGES`, `TENSORFOLD_GEMMA4_KV_PAGES`, `TENSORFOLD_SPARK25_KV_PAGES`) and Nemotron's paging on Metal (`TF_NEMOTRON_METAL_KV_PAGES`) are registered in the profile registry; the startup line and `capabilities` report them, and the values are read as before with every default unchanged (#2306, part of #1660). Thanks to @devopsfortunato.

- LFM2.5-24B-A2B serves on one NVIDIA GPU from the MLX 4-bit export, on the shared CUDA kernels, with `--parallel` streams and copy drafts; it runs without a drafter or prompt cache (#1492, part of #1488). Thanks to @jschmied.

- HIP: the qwen3_5 engine captures a drafter's requested target-layer taps in each stream's ring, with and without graphs, and planted kernel controls (#2314, part of #1563). Thanks to @AbrahamCete.

- CUDA: the kernel limits audit reads sm_110 resource usage from compiled cubins (#2395, part of #2319). Thanks to @yguanrong.

- Docs: `/health`'s `memory` on Metal is the process footprint from the families with a reader, not `{}`, and `/memory`'s `prompt_cache_plan` row names every Mac family that reports it, with Qwen3.5-2B under `--prompt-cache-gib` (#2439). Thanks to @Whq02.

- Kernel contract: a CPU affine projection (Qwen3.5-2B's 4-bit, group-64 `qmm` on the host's cores) and a CPU RMS norm run through gates 1 to 4 with eight planted controls on any host with no device (`zig build cpu-contract`; CI's `test-kernel-contract`), with output bytes pinned the same at 1 to 16 threads and across column splits on x86_64 and arm64 (#2478, part of #2448). Thanks to @Whq02.

- Docs: `WINDOWS.md` says what the host suite needs on a Windows machine with no usable GPU (Zig alone), how to read the build runner's `failed command:` lines, which tests skip there and why, and how to unzip the Zig download quickly (#2479). Thanks to @Whq02.

- The CUDA Qwen3.5 lanes fill readout prompts in pieces for the joint schema head that follows: from row 0, each piece's final-norm rows go to the readout, 1,024 rows a piece unless the lane host caps it, as on HIP (#2384, part of #2324). Thanks to @laudney.

- Clef-Flash on CUDA: `POST /v1/systemone` serves Cloudflare's BF16 `clef-flash` on one NVIDIA GPU (checked on a
  GB10) with its joint schema head on the GPU (HIP's kernels and products, on cuBLASLt with FP32 compute), and chat
  is refused by name; `TF_LANE_TRACE` gains a `cancel` line for a request cancelled between two prompt calls (#2458,
  part of #2324). Thanks to @laudney.

## 1.2.4

- Docs: four subcommand-flag fixes checked against the code: `slide audit --threshold`'s range, `benchmark --context` and `--kv-cache`, `info` on Flash Next, and `run --device` and `--carveout` on CUDA builds (#2348).

- Configuration: a malformed TF_ or TENSORFOLD_ setting now stops the native server at startup with its name, the text it got and the forms it accepts, before any checkpoint or driver work, instead of being read leniently inside an engine; empty or whitespace-only values still mean unset, and the documented lenient forms stay. `--profile` and the startup line report six more setting groups. A kept disk state's identity now also records explicit prompt widths, Dory's exact-zero setting and the slide group's values, so kept disk states of Qwen 3.5 and Qwen3.8-27B on Metal, and of Nemotron and Qwen 3.5 on CUDA, are prefilled once after upgrading (#2405, part of #2370).

- Tests: `test-quoted-markers-host` checks, with real tokenizers and chat templates named in `TF_TOKENIZER_DIRS` (skipped when unset), that every added-token spelling quoted in a chat request's messages, reasoning, tool-call arguments, tool results or tool schema stays plain text; run on Qwen3.8-Flash-Next and Qwen3.8-27B (#2400, part of #2399).

- Prompt cache on disk: what the store drops as it opens is now said on the log, where it went silently before: a kept state whose file fails its check (a truncated file, a wrong name) or misses one of its blocks, with its cause, one line a cause, and the count of blocks no kept state names any more (a state's file deleted, or a write cut short); what is restored is unchanged (#2397, part of #2349). Thanks to @ThinkOffApp for the run that found it.

- Build: `zig/build/cuda.zig` goes from 600 to 467 lines; the CUDA contracts' CPU references and the server's host tests register from their own build files, with the same steps in the same order (#2420).

- Vision: PDF pages reach the vision tower in their true colours. The render flags set `FPDF_REVERSE_BYTE_ORDER` where printing was meant, which swapped red and blue on every page (#2387, for #2386). Thanks to @gordonhatusupy for the report.

- Tests: `test-gradient-serving` and `test-qwen-serving-host` pass again: their serving source pins take main's bytes after a read of every change since they were set, each a comment, an off-by-default path or a reviewed PR (#2418, part of #2031).

- The engine API carries an optional image capability beside speech: `Info.image` states what an engine's image host serves and a defaulted `VTable.submit_image` hands back one encoded PNG a request through the caller's own list and allocator, so an engine without the entry is unchanged (#2248, part of #1573). Thanks to @devopsfortunato.

- Build: `zig build compile-dsv41-server` and `dsv41-serve` build again: the DeepSeek server module gets the `lanes` module its `/v1/systemone` route reads (#2412). Thanks to @Defilan.

- Tests: four host steps that no CI job runs build and pass again: `test-gemma4-affine-host` pins `affine_mm.metal`'s current bytes, the compile-only Metal contract runners import `masked_sources`, `compile-metal-residency-host` imports `kimi_k3`, and the GLM Metal fact tests give `facts()` a host device object instead of an undefined one, which crashed them at random (#2409, part of #2031).

- Flash Next on CUDA with `TF_FLASHNEXT_KV_PAGES`: a shared round's grouped draft head reads and writes each stream's own pages, so paged streams' replies equal their runs alone again, and paged decode runs as fast as unpaged; default settings are unchanged (#2244). Thanks to @Arminova.

- Tests: `zig build test-glm-snapshots-host` builds again: its copies module imports `core`, which `cuda_snaps.zig` reads for `TF_GLM_PROMPT_TAIL` since #2226 (#2403). Thanks to @Bizuayeu.

- HIP: on RDNA 3.5 (gfx1150 to gfx1153) a prompt row gets the same bytes wherever its chunk starts. The 64-row prompt tile there now takes each key past a row's causal edge as +0, as RDNA 3.5's dot2 rounds a -0 lane apart from +0, so its prompt bits change, and gfx1151's prefill is up to 0.7% slower at 2k and 8k. Other chips keep main's prompt tile and bits, and gfx1201's prompt speed is main's within noise. On every chip, a non-finite value at a key past a row's edge now adds nothing to that row (#2357, closes #2317). Thanks to @laudney.

- GLM CUDA at TP=2 and up: each rank names its startup phases (NCCL init, peer warm-up, weights, calibration, ready), prints NCCL's init device, version and log path, and warns once a minute while NCCL init or the warm-up is unfinished, with the transport seen so far. It only reports; transfers are unchanged (#2354, part of #2347).

- Docs: the serve flags match the CLI: `--metrics-open` opens `/v1/metrics` too, `--learn-conversations` on CUDA serves the Qwen3.5 hybrids and Flash Next as well as Nemotron, Flash Next shares sessions under an explicit `--parallel` above one and refuses `--recall` only on a Mac, and the Metal Qwen3.5-2B keeps prompt states (#2367).

- Internal: the kernel contract's nvcc 13.0 resource records match what nvcc 13.0.88 builds for 57 CUDA kernels (226 per-SM slots), so the five-SM limits gate passes on main; no kernel, flag or launch changes (#2390).

- Flash Next on CUDA: a prompt-cache mark inside a prompt chunk is kept on the way through the pass instead of cutting the pass there, so a new short prompt's first token comes 19% sooner on a GB10, with replies unchanged; `TF_FLASHNEXT_KEEP_INPASS=0` cuts the pass at the mark as before (#1866). Thanks to @Arminova.

- Tests: `tf-moss-image-blocks` gates MOSS-Image's transformer block by block against the publisher's float32 run, within twice the spread of torch bf16's device and CPU runs, on Metal and CUDA (#2294, part of #1529). Thanks to @BobClawblaw.

- Cluster: MCDMA bring-up on real hosts, step 2a of #1594 (#2088). Thanks to @ThinkOffApp.

- Docs: the model recipes match the native server: Gemma 4 serves M1 to M4 through the row path and enforces `"required"` tool calls, Nemotron refuses `--tp` above 1 and reserves 32 rows past a reply, `qwen3_5` serves EXL3 packs on CUDA, the CUDA CLI applies `--alias`, Flash Next on Metal shares sessions through `lane_host` under an explicit `--parallel`, its draft levels are 3, 6 and 8, the grammar library installs into `zig-out/native/lib` on every OS, the Qwen3.5-2B check commands are ones that exist, and three smaller fixes (#2361).

- Tests: `serve_check.py`'s `resumed == fresh` sends its fresh request to an optional cold server (`FRESH_URL`, started with `--prompt-cache-gib 0`) and fails by name if the fresh reply resumed, as the two Zig checks now do (#2381, for #2353).

- Tests: `exl3_load`'s allocation-failure test refuses resizes, as its neighbour in `checkpoint_weights_test.zig` does, so it passes whatever the checkout's path length; it failed at 70 of 102 lengths on main (#2377). Thanks to @jishnuvenugopal.

- Tests: the Qwen 3.5 `resumed == fresh` checks (`check_invariants`' long greedy lanes and `check_radix`'s stress) evict every kept prompt before their fresh run, and fail by name if a fresh reply still resumed, so they compare a resume with a real fresh run (#2374, for #2353). Thanks to @AbrahamCete.

- Internal: 20 more lean findings fixed in core's and the Metal kernels' comments, with no change in behaviour (#2309).

- Internal: the TP host reference and in-process transport add bf16 and f16 sums in fp32 rank order and round once, as documented; sums at one or two ranks are unchanged, and the device checks' expectations don't move (#2366, part of #1971).

- CUDA on integrated GPUs such as the GB10: every other synchronous device-to-host copy now stages through the same pinned buffer as `download` (54 copies in families, tools and tests); discrete cards keep the direct copy, and the bytes copied are unchanged (#2289, for #2102).

- Each native model discovery entry publishes loaded capability booleans, read through the same flag accessor as request admission (#2157, part of #2139).

- Tests: `tf-qwen27-reuse`'s yielded-pass check follows the host's lane loop on M5 chips, comparing state only where the rounds match and tokens always (#2343).

- Tests: the CUDA kernel contract releases each case's buffers when the case ends, so the full run fits in a GB10's memory (#1845). Thanks to @Maxrio9.

- Tools: `tools/zig/bench_sessions.py` replays long agent sessions (a shared harness plus histories of up to 215K tokens) in turn patterns (alternating, bursts, three-way and overlapping), one JSON line per request with `cached_tokens` and timings, reads the prompt-cache series from `/metrics`, can plan without sending (`--plan-only`), and replays one session on a second server to compare token hashes (#2359, for #2346).

- Contributing: `tools/pr_check.py` checks local changes against their merge base before a push (new lean findings, sources over 600 lines, and advisory warnings for private-looking paths and addresses, untracked files included), and the PR template asks for paired main-against-branch cells, the machines a change reaches and switch defaults (#2355).

- Docs: the GLM-5.3-Flash recipe notes that `TF_GLM_CLEAR_THINKING` is read by the native server too, not only the Python engine (#2350).

- LFM3-1.7B on Metal: the text decoder is served from the official MLX 4-, 6- and 8-bit checkpoints, with per-stream convolution and KV state, verification windows, snapshots and the prompt cache. LFM3-VL checkpoints are refused by name until the vision tower lands (#1514, part of #1491). Thanks to @BlivionIaG.

- Docs: ten native `serve` flags that were undocumented now are: `--top-p`, `--top-k` and `--min-p` as sampling defaults, `--compact-at`, `--compact-keep` and `--compact-memory`, Dory's `--loops`, the two silence-onset flags and Drama 3's `--max-frames` (#2345).

- CI: main's `zig` and `zig-windows` jobs reuse main's newest build cache, as a pull request does, so each push compiles and runs only what changed since. `zig_cache_restat --prune` keeps each saved cache to about one build, and a nightly run builds and checks everything from nothing (#2402).

- Qwen3.5 on CUDA loads a plain BF16 checkpoint as a decision backbone (Cloudflare's Clef-Flash), not for chat:
  chat and the flags the joint schema head is not qualified with are refused by name before the weights load,
  and its norms scale by 1 + w in FP32. A quantized Qwen3.5 checkpoint that ships `joint_head_config.json` is now
  refused (`QwenHeadNeedsBf16`), as on HIP: the head runs on the BF16 weights only (#2383, part of #2324). Thanks to
  @laudney.

- Docs: `docs/api.md` lists `DELETE /v1/slide/facts`, and `docs/control.md` no longer says the native server refuses `--spill-gib` and `--vision` (#2341).

- Prompt cache on CUDA: when a kept state is larger than the whole cache budget, the log line now says that `TF_KV_SHARE=n` would keep its prompt rows as the pool's pages, for families on KV pages; nothing else changes (#2333). Thanks to @Bizuayeu.

- Living Weights: `--slide-drift-gate` (opt-in, with `--slide`) measures each lesson's drift on unrelated replies and checks its recall before keeping it; without the flag nothing changes (#2028, part of #1389).

- Internal: 11 more lean findings fixed in core, CUDA, HIP and server comments, with no change in behaviour (#2332).

- Tests: `test-metal-residency-owners-host` compiles and passes again (68 tests), so it can come off CI's exclusion list (#2331, part of #2031). Thanks to @jishnuvenugopal.

- Internal: 9 more lean findings fixed in Metal kernel and server comments, with no change in behaviour (#2329).

- Native DeepSeek V4 and V4.1 prompts follow the recorded encoder outputs, except where those outputs drop a tool's namespace, which we keep. Assistant thinking markers, blank-line text joins and tool schemas change; minimal keeps thinking on at budget 50 and xhigh uses 75. Developer instructions render as system, strict tools keep their existing grammar handling, listings and tool-call history keep qualified function names, and schemas retain namespace descriptions. Invalid user null content, malformed tool-call lists or calls without string id/name/arguments or type function, names outside 1-64 ASCII letters/digits/underscore/hyphen, tool replies without tool_call_id and non-function tool envelopes now return HTTP 400. Ultra keeps its existing refusal because no recorded case establishes a budget (#2193, #2141).

- GLM-5.3-Flash on CUDA: a prompt step takes two chunks while no other stream decodes, each one's exchanges under the
  other's sites, so prompts of 8K to 128K finish about 2% sooner at TP=2, with replies unchanged;
  `TF_GLM_PROMPT_PAIR=0` turns it off (#1423). Thanks to @Bizuayeu.

- TENSORFOLD_PHASE_METRICS=1 adds a bounded request queue-wait histogram and its vLLM alias to native metrics; it is off by default (#2158, part of #2139).

- GLM on CUDA prefetches a decode window's next weights into L2 on a side stream, on by default on GB10 (sm_121) at two ranks or more, inside a 12 MiB budget; replies are identical, and every other GPU is unchanged (#2162, #1580). Thanks to @Bizuayeu.

- Internal: 11 more lean findings fixed in the CUDA kernel headers and the tests' comments, with no change in behaviour (#2326).

- Flash Next on Metal: a Mac with tensor units (M5) now takes the same shared-host defaults as other Macs at an explicit `--parallel 2` or more: replies take 0.8 of a filling prompt's time, and the prompt's steps grow to 512 rows. On an M5 Ultra with the MLX 4-bit checkpoint, two overlapping clients finish in 47 s instead of 127 s, and a 64K prompt beside a 1,024-token reply in 21 s instead of 36-38 s, with the same tokens (#2394, refs #1893).

- Internal: 14 more lean findings fixed in the core library's comments, with no change in behaviour (#2323).

- Pipeline stages across machines: the stage protocol (boot description, round plan, rows layout and keep-alive) with golden bytes for every message, so a CUDA and a Metal stage can be built against the same bytes; nothing serves through it yet (#2060, part of #2005). Thanks to @jschmied.

- Internal: 15 more lean findings fixed in the CLI, cluster and CUDA comments, with no change in behaviour (#2320).

- Internal: 15 more lean findings fixed in the Metal kernels' comments, with no change in behaviour (#2318).

- Kept prompt states: a state saved to disk is reused only by the same build and the same checkpoint files, so a rewritten checkpoint or a new build starts cold instead of reusing an old state; the first long prompt after an upgrade prefills again (#2163, #2131).

- Internal: 42 more lean findings fixed in the tests' comments and the tools' docstrings, with no change in behaviour (#2315).

- HIP builds and loads its runtime on Windows: `hipcc.exe` is preferred and `hipcc.bat` is found through `-Dhipcc`, `PATH` or `HIP_PATH`/`ROCM_PATH` and run through `cmd /c call`, `check-native-hip-host` compiles for the host OS instead of always `linux-gnu`, the runtime is loaded by its Windows names (`amdhip64.dll` and the versioned ones, `TENSORFOLD_HIP` overrides), the admission budget comes from the driver where `/proc/meminfo` and sysfs do not exist, the optional libraries use their Windows names (cuBLAS had been looking for `libcublas.so.13`), and the cooperative grid test runs plain where the driver reports no cooperative launch support (ROCm/HIP#3640) (#2178). Thanks to @ElderOrb.

- HIP: the decode round's residual-add and norm tail issues each row's loads before its branches, and a MoE tail loads its slots' rows together, with output bytes unchanged. On a gfx1151, one-stream decode is 2.7% faster on Qwen3.6-35B-A3B and 1.1% on the 27B. `tf_tail` is now under the kernel contract (#2247). Thanks to @AbrahamCete.

- Qwen3.8-27B on Metal: a round's keep that fails part way (an allocation, a refused keep or path) returns its error instead of aborting the process on an unended command encoder, and after a failed allocation the round can keep again; `tf-qwen27-keep-error` fails each of the keep's allocations in turn and refuses a path after the encoder has ended (#2376). Thanks to @jschmied.

- Docs: `docs/optimization-ledger.md` lists which speed and memory optimizations apply to which native family, and where each stands, with links to the PRs that measured them (#2232). Thanks to @jschmied.


- Docs: the pull request pre-flight asks a PR that adds, finishes or drops a family's optimization to update that family's cell in the optimization ledger (#2243). Thanks to @jschmied.

- Server: a refused `--api-key-file` logs the path it was given, the mode the file system reported and which test failed, once at startup and once when a reload is refused (#2365). Thanks to @Whq02.

- CLI: `tensorfold models --json` prints the cached checkpoints as one document (`schema` 1, each checkpoint's `repo`, `family` and `model_type`); the lines are unchanged (#2364). Thanks to @Whq02.

- Server: tool calls completed just before the thinking-close marker are returned when no answer follows, whole and
  streamed; calls followed by more reasoning or an answer stay in reasoning (#2313, fixes #2310). Thanks to @AdrianBinDC.

- Clusters: a Linux kernel built without transparent hugepages no longer refuses a RoCE start: `prepareRegion` keeps
  `MADV_DONTFORK` strict, asks for `MADV_NOHUGEPAGE` only where THP can apply it, and says at startup when it skips it
  (#2380, for #2378). Thanks to @devopsfortunato.

- CI: the portable-names check runs on every pull request again; since #2274 a reused pass could miss a newly added name
  Windows can't check out (#2311).

- Tests: the Metal kernel contract checks a masked key's non-finite value across every family's prompt attention, with
  the families that still give hidden rows NaN registered as known (#2308, refs #1009).

- Tests: the Qwen3.8-27B decode control in the Metal kernel contract edits the line #2106 changed, so contract runs on
  M5 reach their summary again (#2303, fixes #2301). Thanks to @BobClawblaw.

- Build: `zig build server-cpu` (the CPU-only fake server and engine-less CLI the server tests install) compiles again; its server module gets `core`, as the Metal, CUDA and fuzz graphs give it (#2363). Thanks to @Whq02.

- Serving: `serve` refuses, at start and by name, a `TENSORFOLD_PREFILL_SHARE` outside 0.01 to 1, a `TENSORFOLD_PREFILL_POLICY` other than `fcfs`, and `TENSORFOLD_PREFILL_TOKENS` or `TENSORFOLD_PREFILL_BUDGET` text that is not a whole number. The lane host used to read each one as its default without a word, so a share of 0 meant whole prompt steps (#2369).

- Build: a plain `zig build` on Linux compiles again; the `tf-qwen35-cuda` tool passes the device flag that kept prompt states gained on 4 Nov (#2362). Thanks to @jschmied for the report.

- Developers: `tf-xpu-contract-run` runs the kernel contract's gates 1 to 4 on an Intel GPU through Level Zero, with
  planted controls and gate 5 checked live against the card; its first run covers the Qwen3.8-27B glue on an Arc Pro B70
  (#2269, part of #2080). Thanks to @tolewis.

- Flash Next on CUDA: a shared round's DeltaNet steps run in one launch each across its lanes instead of one launch a lane, so 8 streams decode 3-6% faster on a GB10 with the same tokens; one stream keeps its own launches (#1942). Thanks to @Arminova.

- Qwen4: the rotary table comes from core's exact helper, matching transformers' FP32 values; replies can change where a token was a near-tie (#2223, part of #2000).

- Docs: `docs/api.md` matches the server: the `speculative` draft counters, Responses metadata limits, inline
  `input_file` and Messages `document` PDFs, and the fields it left out (`thinking`, `min_rows`, grammar timings,
  `exact_mode`, `stop_sequence`, health's `memory.limit` and 503 body, `/v1/models`' context and reasoning) (#2299).

- Qwen3.8-27B: a prompt frame below 128 rows (`TF_PROMPT_ROWS=16`) is refused before loading, with the setting, the
  value and the fix named, instead of a bare "TargetBinding" (#2298).

- Docs: the Flash Next CUDA recipe gives `TF_FLASHNEXT_CONFIDENCE`'s real default, -0.4 (the drafts' probabilities
  multiply to at least 0.4), and what its sign means (#2296).

- Internal: 73 more lean findings fixed in the kernels' comments, tests, cluster and fabric, with no change in behaviour
  (#2295).

- GLM-5.3-Flash on CUDA: an EXL3 checkpoint whose `quantization_config` sits inside `text_config` loads as EXL3. Admission already read it there, but the config parser read only the top level, so such a pack was admitted and then asked for NVFP4 dense weights at load (#2327). Thanks to @Bizuayeu.

- Docs: `docs/hermes-agent.md` shows how to use TensorFold from Hermes Agent through the community [hermes-tensorfold](https://github.com/atakann/hermes-tensorfold) plugin (a model provider and a dashboard tab), checked on a GB10 with Qwen3.8-Flash-Next (#2330, part of #2109). Thanks to @atakann.

- Qwen3.5 and Qwen3.6 on CUDA: KV pool pages stay opt-in (`TF_QWEN_KV_PAGES=256`), dense and MoE alike; unset keeps a window each, as before, and `TF_QWEN_KV_PAGES_ALL` unset still leaves the engine's own stream a window. With pages set, `--parallel` is a cap rather than a reservation: a request can wait for pages instead of every stream holding a full-context window. Paged prompt attention stages a chunk's page ids in shared memory, a decode thread looks its page up once a page, and a paged prompt-cache save or restore copies one 2D block a page. Replies are the same tokens on pages and windows (#2011, part of #1943). Thanks to @jschmied.

- Tests: the pull client's security tests build and run on Windows, on a portable fake hub, and CI runs them there
  (#2284). Thanks to @ElderOrb.

- Metal: a request's finished stats list its prompt pass widths (`tensorfold.prefill_widths`, one entry per completed
  target pass) on every Metal text family except Flash Next, whose engine reports none. Telemetry only (#2293).

- Internal: 100 lean findings fixed in core, the server, cluster and the model families (comments reworded to fit one
  line, and Kolibri-1's page geometry in its own file), with no change in behaviour (#2292).

- Build: `zig build -Dgpu=<gpu>` builds the one-GPU server again; its modules now import the profile registry, as the
  full build's do (#2290). Thanks to @atakann.

- Tests: the MOSS-Image per-step teacher gate compares every step with the teacher's fp32 values within bf16's rounding, keeps repeatability as its own check, and calls a missing teacher file inconclusive (#2291, for #1529).

- Docs: `docs/api.md`'s Reasoning section says what a no-thinking GLM-5.3 prompt is on each engine — native appends `</think>` to the template's render, Python 0.6.6 strips the `Reasoning Effort` head — so prompt counts and kept states differ (#2297, refs #1911). Thanks to @devopsfortunato.

- Tests: rollback conformance on the fake lane target (drafts landing 0 to all, cancels between rounds and inside prompt passes, each stream's cache checked after every step), and 12 captured CUDA graph shapes replayed out of order. No defect was found, and no engine code changes (#2161, closes #2138).

- GLM-5.3-Flash on CUDA: `TF_GLM_PROMPT_TAIL=1` (opt-in) keeps a raw `/v1/completions` prompt's state at its last chunk start in the unpaged prompt cache, so a repeat or a longer prompt resumes there, with tokens unchanged. It shares the cache's budget with chat states, so it's off by default; any value other than unset, `0` or `1` is refused by name before the weights load (#2226). Thanks to @Bizuayeu.

- Cortheon: a Windows path whose name holds a change word no longer turns a question into a change request; fallback queries stop at a tabbed sentence end and drop a tabbed list mark; search results with upper-case tags, `HREF=` or single-quoted links are kept; a GitHub file under `Blob` is read as raw text; and an absolute link of another scheme stays as written (#2216). Thanks to @ppardus.

- Developers: `-Dnative-fp-decode=true` (off by default, nvcc 13.2 or later) builds CUDA's prompt16 FP4 tile and FP8 expert prompt tile with the hardware FP4 and FP8 decode on sm_120 and later. Outputs are bit-equal to the default build unless a weight carries an e4m3 NaN code, which only a broken checkpoint has. On a GB10 the FP4 prompt projections took 0.95 to 0.96 of the time. Default builds are unchanged (#2238, refs #1448). Thanks to @jschmied.

- DeepSeek-V4.1 on CUDA with `TF_DSV41_PREFILL=replay`: a conversation saved to the NVMe session tier now restores instead of being refused as `BadBlob` and prefilled again. On two GB10s at TP=2, recalling a 319K-token conversation took 1.2 to 1.4 s instead of 155 to 178 s, with replies byte-identical to a cold prefill (#2182). Thanks to @goshi.

- Docs: `CONTRIBUTING.md` and the pull request template ask authors for their own main-against-branch numbers first; our run follows once the review has no open asks (#2179).

- DeepSeek-V4.x chat renders with its own encoder and schema prompt on the native server, a served image part becomes the image token, and the start-up line names the chat renderer (#1698). Thanks to @jayleaton.

- Tools: compare Living Weights keep rates between two native binaries with fresh checkpoint copies, retained refusal receipts and exact unrelated-token controls (#2027, #1587).

- Tests: the stale serving source pins are re-baselined to current sources; no device re-qualification is claimed (#2175).

- Qwen 3.5 on CUDA: an integrated GPU keeps prompt states in device memory instead of pinning a host copy at each save, still on the host cache budget, and the startup line names device memory; on a GB10, 8k and 32k prompts start 1.6 to 2.1% sooner with the same replies (#2113, #2043). Thanks to @jschmied.

- Kernel contract: an XPU limits backend with gate 5, a `.ze_info` reader and planted controls; no GPU needed (#2155). Thanks to @tolewis.

- Flash Next on Metal enforces `response_format` (`json_schema`, `json_object`) and `tool_choice` `"required"` or a named function at `--parallel 1` too, through the same grammar path `--parallel 2` uses; unconstrained requests take exactly the single-stream path as before (#1969, #1894).

- Prompt cache: a restore the backend refuses now tries the next shorter state kept in memory before prefilling from the start (#2144).

- Native engines that cannot apply presence or frequency penalties refuse nonzero requests with 400; zero, null and omitted penalties keep working (#2166).

- Qwen3.8-27B: a refusal that no `--parallel` or `--context` can fix now names the wired-limit raise, and only suggests one that leaves the system room (#2123). Thanks to @devopsfortunato.

- `api.Structure` keeps the body field that asked for a structure — `response_format`, `guided_json` and the other four, or `text.format` and `output_config.format` on the Responses and Messages routes — so an engine that refuses one can name the field the client sent (#2110, #2084). Thanks to @devopsfortunato.

- Cloudflare's Clef-Flash serves on HIP (Radeon AI PRO R9700): its BF16 Qwen3.5 backbone and joint schema head answer `/v1/systemone` with the release's record encoder, head and answers, the head on the GPU over each prompt piece in the shared rounds; validation errors are 422s with their location, and chat and the other routes refuse the checkpoint naming `/v1/systemone`. `capabilities --json` lists BF16 for HIP Qwen3.5 under a new `decisions` key, apart from the chat formats. Checkpoints whose `tokenizer_config.json` publishes a `pretokenize_regex` split words with it, as transformers does (#2048, #1360, #1952). Thanks to @gilby for the first encoder port (#1361) and to @laudney.

- Tests: the heaviest host test gates build in ReleaseSafe, which keeps every runtime safety check: the Qwen4 host gate's 113 checks take 18 seconds on one 20-core host instead of 4 minutes, and the CUDA host family's gates take a sixth to an eighteenth of their Debug time, for about the same compile cost (#2249, part of #2242). Thanks to @devopsfortunato.

- GLM-5.3-Flash on CUDA: `TF_GLM_KV_SHARE` moves to `TF_KV_SHARE`. With `TF_KV_PAGES=on`, `TF_KV_SHARE=n` keeps GLM's prompts in the shared prompt store that Spark 2.5, Gemma 4 and Kolibri-1 use, with the same tokens, resume marks and memory as GLM's own store, which goes. A server started with `TF_GLM_KV_SHARE` set to anything but 0 or an empty value stops before the weights load and names `TF_KV_SHARE` (#2211). Thanks to @Bizuayeu.

- Qwen 3.5 and 3.6 on CUDA: a prompt chunk's last layer runs q, attention, o and the MLP over the kept row's 128-row tile only, and k and v over every row, so the cache is unchanged and the last row's logits stay bit-equal; on a GB10 first tokens come 1 to 3% sooner. A model served with an MTP head, int8 or images keeps the full last layer (#2229, closes #2165). Thanks to @jschmied and @jungjipdo.

- An engine can refuse a request it cannot serve as asked in words the client can use: the server asks one optional `Engine.VTable.refusal` when the request is whole and answers 400 `invalid_request_error` before any header on chat, completions, Responses and Messages, streamed or not. An engine without the entry behaves exactly as today (#2198, #2082). Thanks to @devopsfortunato, and to @shuuul for the report.

- DeepSeek-V4.1 DSML calls now stream from the invoke name, DSML's own tag, unlike GLM and XML calls, which have none; before, the header waited for the first parameter to close, so a call whose first parameter is the long one now streams while it is written. A DSML call malformed after its name may stream before the whole reply keeps it as text (#2201, #1975). Thanks to @devopsfortunato.

- Vision: image patch preparation is rewritten from published model cards and tested behaviour, with byte-equal grid sizes, normalized pixels and cache identities in 55,503 host cases (#2271).

- GLM on CUDA: the recipe notes that `TF_GLM_CHUNK_SOLO=4096` made prompts 12-14% slower with NVIDIA's NVFP4 checkpoint on two GB10s at `--parallel 1`, so leave it unset there (#2288).

- Flash Next on CUDA at `--tp 2` with drafts keeps prompt states, and no longer needs `--prompt-cache-gib 0` to start. On two GB10s, a chat with about 28,500 tokens of history replies in 1.5 s resumed against 21.2 s fresh, with the same replies (#2214, fixes #1829).

- Flash Next on CUDA: the whole shared expert's down projection writes straight into its output slot, without a scatter, so NVFP4 prompts run about 0.7% faster on a GB10 with the same tokens (#1792). Thanks to @laudney.

- Qwen3.8-27B on Metal: a NaN or infinity in a value a row must not see no longer turns that row's attention output into NaN on the prompt kernels (before M5 and on M5) or the M5 decode prefix. Replies change only where such a value already sits in a masked key, and finite inputs keep their bytes (#2106). Thanks to @cshintov.

- Flash Next on Metal: a lone `"draft": false` reply at `--parallel` above one runs as it does at `--parallel 1`, 17-18% faster on an M3 Ultra with the same tokens (#2272).

- CI: the cache tool's own test runs only when the tool or its test, `no-gpu.yml`, Zig's version or `build.zig.zon` changes, on main too, which saves about 2 minutes of each main `zig` job (#2277).

- Qwen3.8-27B on Metal: a reply that would end well before a long prompt now goes first, and the prompt lends it at most a quarter of its own time. `--decode-share`, `TENSORFOLD_PREFILL_SHARE`, or any `TENSORFOLD_PREFILL_ORDER` other than `srpt` keeps the old order (#2224).

- Flash Next on CUDA: at `--parallel 1` on one GPU, a prompt pass takes up to 8,192 rows, so prompts run 15-23% faster on a GB10 for about 3.2 GiB more memory; `TF_FLASHNEXT_CHUNK_SOLO` still decides (#2246).

- Docs: the slow drafted request after a `draft: false` one no longer reproduces on one Mac, and the expert recipe lists peak memory at each `--ssd-experts` budget (#2237).

- Flash Next with `--ssd-experts`: a request no longer waits for a forecast read it doesn't need, and a forecast that never started keeps the expert it displaced: about 8% faster at 35 GiB and 4% at 17 GiB (#2236).

- CUDA on integrated GPUs such as the GB10: synchronous downloads stage through a per-thread pinned buffer, which guards against a driver fault where a copy into pageable memory could miss pages. Discrete cards keep the direct copy. On an idle GB10, large copies take longer: 212 µs instead of 85 µs at 4 MiB (#2103). Thanks to @shuuul.

- Prompt scheduling: a resumed turn that has not started its prompt pass is ranked by its uncached tail, not its whole prompt, so a long cached history no longer waits behind a cold prefill; a rejected restore ranks the whole prompt again (#2276, #2256). Thanks to @devopsfortunato, and to @laidick for the report.

- CI: a pull request's `zig` and `zig-windows` jobs reuse main's test passes and compiles, C and C++ included, so they run only what its diff changed; main still checks every commit from scratch (#2274).

- Nemotron on Metal: the 16 decode attention partials share one source, with the same kernels and bits (#2263). Thanks to @jschmied. Saved Nemotron prompt states on Metal take their identity from the source text, so after upgrading they are not reused: the first long prompt prefills again, and is kept as before from then on.

- Tests: Metal learning admission checks Kimi's own refusal and runs without shader compilation. Serving is unchanged (#2267).

- Tests: Metal GLM draft-source facts use a defined host capability receiver, removing a random crash from an uninitialized device. Serving is unchanged (#2268).

- CI: main's build cache now fits every runner, because CI compiles host code for one CPU, and `release-paths` and the macOS dev tools use the cache too (#2252).

- Gemma 4 E4B (NVIDIA Nemotron 3.5 Bolt E4B) runs on CUDA from `mlx_lm.convert` 4-bit checkpoints: per-layer inputs in their own launch and layers 24 to 41 on the caches of 22 and 23, with replies equal across streams and resumes (#1222, #2230). Thanks to @jschmied.

- Qwen3-Reranker-8B on HIP computes a rerank request's shared start (the system prompt, the instruction and the query) once a pass instead of once a document, with the same responses byte for byte: 16 documents of 323 tokens, 87 of them shared, take 1.98 s instead of 2.63 s on the dot2 tile and 0.73 s instead of 0.99 s with the gfx12 WMMA tile (opt-in, `--policy matrix=on`), on a Radeon AI PRO R9700. The decide GPU checks now run on the tile and attention path that `TF_QWEN3_MATRIX` and `TF_QWEN3_ATTENTION` choose (#2262, part of #2245). Thanks to @laudney.

- Gemma 4 and Qwen 3.5 on CUDA now load with direct reads by default, 28-31% faster cold on a GB10 with the same weights and replies. A load uses about 0.6 GB more host memory; `TF_LOAD_DIRECT=0` keeps the old read-ahead (#2282). Thanks to @jschmied.

## 1.2.3

- llama on Metal: Nanbeige's own `Nanbeige/Nanbeige4.3-33B-INT4` release (compressed-tensors int4 in groups of 32, 21.0 GiB) is served as published, read in place; any other quantization of the family is refused by name. Against a float64 reference over 1,500 teacher-forced positions on an M5 Ultra, mean KL 0.0044 and 97.1% top-1. Prompts need Metal 4 tensor units, so M1 to M4 Macs are refused at load (#1782, part of #1414). Thanks to @BobClawblaw.
- Nemotron on CUDA: `TF_KV_PAGES=0` and `TF_KV_PAGES=all` are reported as overrides at startup and in `capabilities`, for Nemotron-H and Nemotron Omni, instead of "registered overrides: none"; how the value is read is unchanged (#2061). Thanks to @garvrao80.
- Clusters: the fabric layer gains a pipeline stage link, a TCP ring between ranks on Linux and macOS that carries each round's plan and hidden rows, with a shared key proved both ways before any frame. Nothing serves through it yet (#2012, step 1 of #2005). Thanks to @jschmied.
- Developers: `docs/pr/SKILL.md` is a pre-flight for preparing a pull request, linked from `AGENTS.md` and the README (#1595). Thanks to @jayleaton.
- Developers: `tools/zig/lean_check.py` reads sources as UTF-8 on every platform, so Windows counts the same as Linux and macOS (#2097, fixes #2096). Thanks to @BobClawblaw.
- Flash Next at `--tp 2` on CUDA: each rank computes its own output columns of the dense projections instead of both repeating the whole projection, with the same replies as one GPU. On two GB10s with NVIDIA's NVFP4 release, plain decode is 21-23% faster, drafted decode 9-11% faster at one stream and 76-77% at two, and prompts about 6%. Model memory goes from 45.04 to 42.42 GiB a rank, and a stream's KV at 65,536 tokens from 2.18 to 1.30 GiB (#1784, refs #1754). Thanks to @laudney.
- Flash Next on CUDA, beside another open stream: a greedy chain stops at 0.6 per draft, and a round with more than four streams drafting drafts greedy chains 6 deep. That fixes #1786's loss at eight greedy streams: on a GB10, MLX 4-bit chat goes from 288.8 to 303.8 tok/s and code from 336.9 to 358.5, FP4 stays level and every reply keeps its bits. `TF_FLASHNEXT_GREEDY_WIDE` sets the width and `TF_FLASHNEXT_SHARED_DRAFTS=0` keeps the alone rule (#2160, part of #1762). Thanks to @Arminova.
- With `--ssd-experts`, the expert pool counts uses, so an expert used once leaves before one used again. On an M3 Ultra, Flash Next reads about 11% less from disk at 17 and 35 GiB pools, and a mixed workload took 8% less time at 35 GiB, with the same replies (#2209).

- Flash Next on a Mac without tensor units, with `--parallel` above one: a lone reply runs as fast as at `--parallel 1`. On an M3 Ultra, sampled replies are 12-19% faster and greedy ones about 4%, with the same tokens, and a second request joining a reply leaves it unchanged (#2199).

- Flash Next on Metal reports a filling prompt's progress: the dashboard shows what's left, and `TENSORFOLD_PREFILL_ORDER=srpt` works from a server's first long prompt (#2194).

- CI: `zig` and `zig-windows` keep main's build cache instead of rebuilding from nothing, every check still reruns, and the quick checks run first.

- HIP: the `tf_sum_partials` records follow ROCm 10.1's hipcc, taking 8 limit problems off the contract (#2233). Thanks to @AbrahamCete.

- CUDA docs match the code: README, RUNBOOK and the recipes now say which families refuse an A100 at startup, that no NVFP4 or FP8 model has run on sm_80 to sm_89, and that the largest recorded run is three GB10s (#2227).

- Kimi K3 and K3.1 on Metal through the native server: chat rendering, reasoning and tool calls, with K3.1 capped at 2,047 tokens until its sparse index is qualified. Tested on a synthetic graph; no real weights have run here, and the full 1.45 TB checkpoint doesn't fit one Mac through this path (#2221, part of #2192).

- HIP: a lane round of more than four rows reuses cached weights instead of reading them from DRAM again for each row block, and makes a chunk's code pairs once; gfx1030's spilling tiles keep their layout. On a gfx1151 the 27B at 16 streams goes from about 51 to 75 tok/s on code. The records are @AbrahamCete's own, as we have no AMD GPU (#1694). Thanks to @AbrahamCete.

- CUDA with KV pages: `--prompt-cache-gib N` now keeps N GiB before the pool sets aside windows past the engine's own, so a named cache is no longer cut to almost nothing at some room sizes. A command that already names N splits memory differently after an upgrade; unset, the order is unchanged. The startup line counts the windows the pages hold (#2220). Thanks to @Bizuayeu, and to @MiaAI-Lab for finding it.

- llama `--drafter`: a probe stays armed until a round verifies its drafts, so a request that once stood the drafter aside gets drafting back; replies don't change (#2169, closes #2168). Thanks to @BobClawblaw.
- GLM-5.3-Flash on CUDA: `TF_GLM_CHUNK_SOLO=4096` now serves beside `--drafter` (DFlash2), 80 MiB more on each rank; a cold 64K prompt's first token comes about 7% sooner at TP=2 with the same tokens (#2219). Thanks to @Bizuayeu and @MiaAI-Lab.

- Every retryable 503 on the OpenAI-shaped inference routes names its wait in a fixed `code` (`server_restarting`, `engine_loading`, `engine_reload_failed`, `engine_busy`, `engine_unavailable`, `memory_pressure`); a parked model's reload was reported as `server_restarting` with the drain's message and now answers `engine_loading` or `engine_reload_failed` with the model's own words, while capacity refusals gain the code and keep their status and absent `Retry-After` — an embeddings request that finds the engine busy now names `engine_busy` and says busy instead of unavailable. The drain's body bytes, every status code and every other message are unchanged, and the Messages routes' Anthropic envelope has no code field (#2135). Thanks to @devopsfortunato.

- GLM on CUDA: `TF_GLM_COPIES=1` adds opt-in context copies, verified and priced with the same row costs as drafts. Off by default; tokens are unchanged (#2146).

- Living Weights: a lesson's extra wordings are seeded from the lesson, so the same lesson samples the same wordings on each run (#1587).

- `cached_tokens` and `prompt_tokens_cached_total` can no longer exceed the prompt, whatever a backend reports (#1899).


- GLM on CUDA at TP=2 with `TF_KV_PAGES=on` and `TF_GLM_KV_HOST=1`: a prompt chunk's MTP pass reads the pooled keys at the device stride, so a long first prompt no longer faults (#2173, fixes #2119). Thanks to @Bizuayeu.
- Living Weights: a refused lesson round names the sub-check that failed (#1587).

- Kimi K3.1 Pro's config is recognised: `dsa_config` is checked and refused where unsupported, not silently read as K3, and the family caps it at 2,047 tokens until a sparse index is qualified. Not served yet (#2192).

- DeepSeek-V4.1 logs its resolved serving settings at start-up, so two runs can be compared setting by setting (#1395).

- HIP: the OCR families share one host protocol, with the same replies (#2040). Thanks to @laudney.
- `THIRD_PARTY_NOTICES.md` names the model release and library code that five sets of files follow, with the LFM Open License text (#2197).

- Persisted prompt states include the resolved profile values that can affect a family's arithmetic or experimental/debug path. Changing one moves the disk namespace, so the first long prompt under that setting prefills again; an unset boolean and an explicit 0 share the same key. Native CUDA and the four Metal persistence owners use the same scoped helper. Qwen and Flash Next on CUDA have such rows even with nothing set, so their disk states from before this version aren't reused: the first long prompt after upgrading prefills again (#2131).

- Lane host: the paged KV pool's promises now count in the host's one ledger, with no change in behaviour (#2127). Thanks to @jschmied.

- Qwen3.5 on HIP: the prompt conv writes q, k and v to their own rows, which drops 144 of the 27B's 160 column copies, with the same results (#2117). Thanks to @AbrahamCete.

- README: the GLM-5.3-Flash Jammed XSFP4 row, and shorter FP4 rows that keep their links (#2114). Thanks to @Bizuayeu.

- Tests: the Qwen27 lanes' facts read a dense model's config, so `test-native` no longer crashes (#2122). Thanks to @BobClawblaw.

- `tools/bench_fill.py` measures a long prompt filling beside a running reply, in both arrival orders, against each run alone, with every reply checked against its run alone (#2154).
- DeepSeek-V4 and V4.1 tool calls stream their arguments as they are written, from the first parameter's end, like GLM and XML calls; a call whose first parameter is the long one still arrives when that parameter closes. With `stream: false`, a reply cut by `length` inside a DSML call now returns that call instead of text (#1975).

- The XPU unknown-model error names Gemma 4 E4B when the build includes its DPAS engine (#2189).

- The Qwen3.5-2B recipe says which of the family's listed formats load at its geometry: only `mlx-q4g64` (#2139).
- Decision routes now include `context_length_exceeded` on context-limit errors for text-only prompts, preserving their status and text; a prompt with pictures still gets a plain request error from the engine check (#2183).
- `capabilities --json`: the docs now say what `env` covers, a selected documented subset, and that `profiles.registered_switches` covers the profile registry (#2139).
- Model discovery reads one engine-info snapshot under the idle-unload lock, avoiding reload races (#2186).

- HIP Qwen format discovery names each admitted MLX bit width and group size, such as `mlx-q4g64` (#2188).
- The native `--reasoning-effort` choices include `minimal` and `max`, using the shared request vocabulary. Existing template coercion and `--no-thinking` behavior stay unchanged (#2139).

- Nemotron-H on Metal, when both the checkpoint context and `--context` are absent, now reports its existing 262,193-token prompt-plus-reply limit in `/v1/models`. Requests, default reply limits and the backend refusal remain unchanged.
- Model discovery keeps `reasoning.default_effort` for templates that consume `reasoning_effort` as free text, even when they name no quoted effort levels. It omits the default when the template does not mention that input. Rendering and request behavior are unchanged.

- `--profile stable`'s registry covers GLM's host KV tier: `TF_GLM_KV_HOST` and `TF_GLM_KV_HOST_SHARED` are now rows in the table, with [#1795](https://github.com/ashhart/TensorFold/pull/1795)'s identical tokens at TP=2, and their readers resolve through the registry. A switch looked up by name now goes through `core.profile.lookup`, which serves registered names from the table and every other name from the environment as before. Stable supplies no value, so nothing is enabled or changed by default (#1660).


- LFM3-1.7B runs on Intel GPUs through XPU, Iris Xe first (#1696). Thanks to @jschmied.
- Disk prompt-state startup preserves valid files when allocation fails, and reports the failure instead of treating them as damaged (#2134).

- Fish Audio Drama 3 (`fish_omni`) on Metal lists its native NVFP4 expert weights beside FP8 in `capabilities` (#2139).

- HIP Qwen3.5 and Qwen3.5-MoE no longer list BF16 in `capabilities`, since those servers refuse a BF16 checkpoint for chat (it can only score decisions) (#2139).

- MiniMax-M3 on Metal refuses logprobs with an external drafter before prefill; plain decoding keeps logprobs (#2139).
- CUDA GLM-5.3-Flash capabilities include the EXL3 format its loader already serves (#2139).

- Model discovery omits reasoning for engines that disable it, including HIP OCR and scoring models, even when their template has thinking markers (#2139).

- Qwen3.5 decode on HIP reads the BF16 router rows the model already holds, with the same tokens (#2007). Thanks to @AbrahamCete.

- Qwen3.5 prompts on HIP run the residual add and post-mixer norm in one launch, with the same bytes (#2099). Thanks to @AbrahamCete.

- GLM-5.3-Flash on CUDA reads block 32's scales only for a `csf0_v1` checkpoint, and every rank refuses a scale offset past 0x7E with a named error (#2091). Thanks to @Bizuayeu.
- GLM on CUDA with `TF_GLM_PREFILL_REDUCE=ring` warms up only each rank's ring neighbours at startup; the default split and gather warm-up are unchanged (#1971).

- Qwen3.5-MoE's prompt sorts its experts' rows in one threadgroup, a thread an expert, with the same order as the serial sort (#1799). Thanks to @BobClawblaw.
- A batched `/v1/embeddings` request no longer hangs on a backend that prefills many prompts at once, and the lane core can score decision prompts in 1,024-row pieces (#2047). Thanks to @laudney.
- The native tokenizer reads a Qwen2 checkpoint's slow-tokenizer files when it ships no `tokenizer.json`, converting them to the JSON it would ship; structured output still needs that file (#2225). Thanks to @shuuul.

- XPU's dense 27B takes its kernel sizes from one shared `shape.h`, with 8-head prompt attention, and its SPIR-V images stay byte-identical (#1425, thanks @tolewis).

- PNG pictures with a `tRNS` key past 255, or a 16-bit key, now decode with Pillow's transparency: grayscale and RGB keys compare their low byte with the decoded sample. Pictures without such a key decode as before, and 1-bit grayscale keeps its whole-key rule (#2100).

- GLM on CUDA at TP>1: the DFlash2 drafter reads complete prompt taps on every rank (it read stale residual rows under the split prompt reduce), so acceptance is back to the gathered level with every reply unchanged (#1623, thanks @BadAd84).

- dots.mocr on HIP: the publisher's BF16 `dots-studio/dots.mocr` OCR checkpoint is served natively, with its own template, tokenizer, image processor and EOS ids, token-exact against its model card (#1938, part of #1927, thanks @laudney).
- CUDA on the RTX PRO 4500 (sm_120): the dense 27B's MLP uses a 128-row tile, byte-equal to the 64-row one, with `TF_CUDA_TEST_MLP128` (0 or 1) to pin either for testing (#1730, thanks @Maxrio9).

- CUDA on the RTX PRO 4500 (sm_120): attention takes the MT2 FP8 path from 8,192 cached tokens, token-equal to the default path (#1731, thanks @Maxrio9).

- Flash Next on HIP advises random access over the n-gram table's mapped tensors, so a gather's scattered single-row reads fault only their own pages instead of whole readahead windows: the first request after start decodes at 35.8 tok/s instead of 24.4, and a cold 7K-token prompt starts in 28.9 s instead of 48.4 s on a Strix Halo (gfx1151), with every reply token-equal (#1615, fixes #1602, thanks @gilby).

- `TF_KV_SHARE=n` keeps a paged CUDA family's prompts as its core pool's pages (Gemma 4, Kolibri-1), with replies equal to a fresh prefill; it's opt-in, and costs about 3.5% decode after a short cold prompt on Kolibri-1 (#1755, part of #1657). Thanks to @jschmied.
- Qwen3.5's attention-layer pattern (`layer_types`, `full_attention_interval`) is read by one shared reader on Metal, CUDA, HIP and XPU, so a config gets the same pattern or the same refusal everywhere; a `layer_types` list that contradicts the interval is refused naming both keys, and XPU checks a present `config.json` at load (#2086, fixes #1863).
- Flash Next FP4 on CUDA: drafted rounds send the first token earlier, keep the n-gram table resident, draft deeper on greedy requests and replay rounds through CUDA graphs. One-stream decode on a GB10 is up to 11.5% faster (chat +8.7%, code +6.3% at temperature 0), and every reply is token-equal to main (#1786, thanks @Arminova).

- Docs: the GLM CUDA recipe gives `TF_GLM_OVERLAP_PIECES=8` beside `TF_GLM_CHUNK_SOLO=4096` at TP=3 with `TF_GLM_ROCE_MULTI` (#2067). Since #2116 that's the default there, so the variable only pins it. Thanks to @Bizuayeu.

- Docs: the Flash Next recipe says native Metal takes `--ple-on-ssd`, which reads per-layer embedding rows from the SSD in batches (#2068). Thanks to @swbiggart.

- The cluster planner gives a dense model's rounds zero expert cost instead of dividing zero by zero, so its decode and combined round times are finite (#2070). Thanks to @swbiggart.

- Qwen 3.5 on CUDA: a kept path longer than the 48-row pending buffer is kept at once instead of overrunning it, which fixes the occasional INVALID_VALUE on deferred keeps (#1982). Thanks to @jrb-ai.

- GLM-5.3-Flash on CUDA reads csf0_v1 checkpoints, whose NVFP4 routed experts carry their scales at block 32 as a base plus a 4-bit offset, saving about 13 GiB across a TP=2 pair with the same tokens; block-16 checkpoints are unchanged (#2041). Thanks to @Bizuayeu.

- In one-call tool mode, a stream sends prose held after a hidden call envelope at the finish, so the streamed content equals the whole reply; other streams are unchanged (#2049). Thanks to @devopsfortunato.

- GLM tool calls stream their arguments as the model writes them, as Qwen-style XML calls already do, and the joined pieces equal the whole reply's arguments (#2050). Thanks to @devopsfortunato.
- HIP: Qwen 3.5 loads BF16 checkpoints with their Hugging Face tensor names and runs the BF16 projections through rocBLAS, the base for Clef-Flash. MLX checkpoints are unchanged (#2046). Thanks to @laudney.

- Kolibri-1 on CUDA keeps its full-attention layers' caches in pages of the core KV pool, as Gemma 4 does, with identical tokens when the prompt store is off (#1620, part of #1657). Thanks to @jschmied.

- Qwen 3.5 on CUDA: the mixers' and the dense MLP's scratch rows share one region, freeing 0.46 GiB on the 27B with the same tokens and speed (#1476). Thanks to @diwakergupta.

- The kernel contract checks Qwen3.8-27B's attention_io pack, append and keep at the other two head shapes the family admits, GQA16 (32 query heads over 2) and the 9B's 16 over 4 (#2063). Thanks to @cshintov.

- `kernel-contract` frees each kernel's device buffers once its gates are done, so the full suite fits on a 16 GiB card instead of running out of memory in the MoE backward suite (#2042). Thanks to @BlivionIaG.

- Six remaining fp32-to-bf16 conversions use core's shared rounding: every non-NaN input gives the same bits, and a NaN now stays a NaN where it didn't before (#2052).

- cortheon keeps a source, link, page or query that differs only in spelling (a capital scheme, a query- or fragment-only link, a Windows path, a scheme-relative result), and resolves `.` and `..` without dropping empty path segments, so `/docs/` keeps its slash (#2036). Thanks to @ppardus.

- The Recall tests set their own disk floor, so the full host suite passes on a machine with less than 4 GiB free. The 4 GiB floor itself is unchanged and has its own test (#2051).

- Developers: passing test binaries no longer print, so `zig build test` shows `failed command:` only for a real failure (#2153).

- Community benchmark receipts (v2) can record 1 to 8 loaded ranks and each node's hardware, in separate single-node and multi-node categories. Both the bundled client and one attached to a running server read the running group. v1 is unchanged, and v2 uploads stay off for now (#974).

- Six host-test steps that had gone red on main pass again: their fixtures follow the code they test, and each fix was checked by planting the original break (#2024). Thanks to @jishnuvenugopal.

- Dory on CUDA takes its RoPE tables from core's exact helper, matching transformers' fp32 values (they moved by up to 5 ulp). Replies can change where a token was a near-tie; decode speed is unchanged within 0.3% on a GB10. There's no switch back: the new tables are the reference's (#2035, part of #1852).
- Clusters: the fabric layer can link nodes over ConnectX (RoCE v2), the first step of #1594 (#1957). Thanks to @ThinkOffApp.

- `cluster check`, `cluster status` and `serve`'s preflight now exit 1 when a configured model cannot be planned (#2069). Thanks to @swbiggart.

- DeepSeek-V4.1 on CUDA reads `--prompt-cache-gib` as the kept states' resident limit on every rank (0: none resident in RAM; with `TF_DSV41_SESSION_DISK` an evicted chat is parked on disk): with TP > 1 rank 0 sends its limit to the other ranks at startup. A set `TF_DSV41_SESSION_RESIDENT_GIB` wins on rank 0; set on another rank, it must equal rank 0's limit or every rank refuses to start (#1625).

- On RDNA4 (gfx12), Qwen3-8B's BF16 projections can run on a new WMMA tile, opt-in with `--policy matrix=on`: Qwen3-Embedding-8B embeds 100 short inputs in 0.99 s instead of 2.47 s and Qwen3-Reranker-8B ranks 16 documents in 1.39 s instead of 3.22 s on a Radeon AI PRO R9700. The tile's fp32 sums are less precise than the dot2 chain's (8 to 41 times further from float64 per projection), so the default `matrix=auto` keeps the dot2 tile and its bits; the BF16 projections WMMA writes are as close, and both models pass their oracles on it. Other chips are unchanged, and gfx1200 is unmeasured (part of #1977). Thanks to @laudney.

- `/v1/rerank` sends a request's documents to the engine as one decision, and Qwen3-Reranker-8B on HIP packs them into as few prompt passes as fit instead of one pass a document, with the same responses byte for byte: 16 documents of about 260 tokens take 2.69 s instead of 3.22 s on the dot2 tile and 1.00 s instead of 1.39 s with the gfx12 WMMA tile of #2034 (opt-in, `--policy matrix=on`), on a Radeon AI PRO R9700. The lane host carries `Engine.decide` for a backend that gives it; no other family changes (part of #1977). Thanks to @laudney.

- The native HIP engine serves `Qwen/Qwen3-Embedding-8B` as Qwen publishes it (BF16) through `/v1/embeddings` on a Radeon AI PRO R9700 (gfx1201): BF16 dense weights stay BF16 on a new dot2 GEMM tile, inputs of up to 32,768 tokens batch to the same vectors as alone, and `dimensions` takes any MRL width from 32 to 4,096. Every family's `/v1/embeddings` now refuses a whitespace-only text as SGLang does, and names an input longer than the context with its token count (`context_length_exceeded`) before the batch limit; see [the API notes](docs/api.md#embeddings) (#1978, part of #1977). Thanks to @laudney.

- Eight remaining host BF16 roundings in test oracles (GLM's Metal checks and CUDA FP8 KV reference, Nemotron's training scan, LFM2's DSpark taps, Flash Next's HIP kernel tests) and the XPU FP64 GDN reference use core's shared conversion, so a negative NaN gives a NaN instead of overflowing in Debug builds; every other non-NaN input gives the same bits. No kernel or served path changes (part of #1863).

- Qwen3.8-27B on Metal gains a host GGUF family adapter for metadata, tensor names and exact affine preparation, with the observed qwen35 schema and effective norm gains pinned in host tests; GGUF serving remains unavailable (#1920).

- `POST /v1/rerank` serves `Qwen/Qwen3-Reranker-8B` on HIP in the rerank shape agreed in #1895 (`results` with `index`, `relevance_score` and `document`, `top_n`, `return_documents`, `usage.total_tokens`), taking SGLang's request fields too: each document's prompt is the model card's template token for token, a prompt past `max_length` (default 8192) losing the end of its content as the card's tokenizer cuts it, and its score is sigmoid(yes - no) of the last position's label logits, read from `lm_head` rows in f64. Only an engine that declares a reranker answers; every other server answers the route with HTTP 400, and a reranker server refuses the other model routes by name. The lane host's score queue now stops at a drain's deadline: the score already running finishes and the queued ones fail without running, which also reaches `/v1/decisions` on Flash Next on CUDA (#1981, part of #1977). Thanks to @laudney.

- The Qwen3.8-27B Metal engine serves Qwen3.5-MoE checkpoints (`qwen3_5_moe`): Ornith-1.5-122B on one M5 Ultra from the publisher's own 4-bit MLX weights, with routed and shared experts, 32 query heads over 2 KV heads, its eight-tap DFlash2 drafter, images, the prompt cache and several streams; the dense 27B is unchanged (#1125). Thanks to @BobClawblaw.

- HIP on the gfx1151 iGPU routes a many-expert prompt through a 64-row tile, with up to 12 expert pairs an item and prompt chunks capped at 1,024 rows; other GPUs are unchanged (#1658). Thanks to @gilby.

- PDF documents as input with `--vision --vision-pdf` (#1820). Thanks to @joernchen.

- CUDA loaders gain opt-in checked read-ahead with `TF_LOAD_PREFETCH` on Nemotron, Bolt, MiniMax M3, Kolibri-1, Spark 2.5 and DeepSeek-V4.1 (unset is 0). GLM keeps its legacy default of 16 readers. Upload and existing release order stay unchanged; disabled adapters start no reader, and failed reads name their source (#2003). This extends @jschmied's #1928 mechanism; Gemma 4 and Qwen 3.5 keep its default of 16.

- Living Weights: `tensorfold slide audit` measures each learned fact's pocket neurons against its questions, subject contexts, other facts and 256 fixed general prompts. The explicit graph audit adds firing rates and strongest off-subject prompts without changing weights, lessons or checks (#1968).

- Qwen4 on CUDA drops its checkpoint from the page cache after loading, and tensor-parallel loaders drop theirs only once every rank has finished loading, so ranks on one host no longer evict pages another is still reading (#1868).

## 1.2.2

- CUDA lanes: Gemma 4, MiniMax M3, Kolibri-1, Spark 2.5 and Dory can verify root and context copies as Nemotron does, opt-in: set `TF_GEMMA4_COPIES`, `TF_MINIMAX_COPIES`, `TF_KOLIBRI_COPIES`, `TF_SPARK_COPIES` or `TF_DORY_COPIES` to 1. Replies equal `"draft": false`, and unset is the old policy (#2022, part of #1910).

- The dense Qwen3.5 Metal loader takes its shape from the checkpoint's config rather than the 27B's fixed geometry, checked against the 27B, 9B and 2B with identical tokens (#1817).

- Living Weights on Gemma 4, step 1: the prepared-attention backward on the host, checked against an independent reference; nothing served changes (#2032, part of #1803).

- Docs: the GLM CUDA recipe says what `TF_GLM_CHUNK_SOLO` gains at TP=3 with `TF_GLM_ROCE_MULTI` (a little under half of its gain without it) (#1996). Thanks to @Bizuayeu.

- Living Weights on Qwen4 dense, step 1: the QSA backward pass on the host, checked against a binary64 reference; nothing served changes (#2021, part of #1804).

- CUDA kernels share their duplicated device primitives; the machine code is identical on sm_80, 86, 89, 120 and 121 (#2018, closes #1409).

- Metal: Qwen3.8-27B's keep gathers and scatters the binding's KV heads a row, so a keep stages what the layout holds (#2013). Thanks to @cshintov.

- `TF_LOAD_PREFETCH`: Gemma 4 and Qwen 3.5 on CUDA read each layer ahead while the current one uploads, so the 27B loads about a third faster on a GB10. Replies are unchanged, and `TF_LOAD_PREFETCH=0` is the old path (#1928). Thanks to @jschmied.

- Flash Next on CUDA loads its experts about 12% faster with the direct load: the next chunk is read while the current one packs, and the store switches shards without reopening them. Replies are unchanged (#1960). Thanks to @jschmied.

- Gemma 4 on CUDA keeps its global layers' caches in pages of the core KV pool, every stream on pages, so with `TF_KV_SHARE` its prompts can be kept and resumed as pool pages; unset, tokens and speed are unchanged (#1619, part of #1657). Thanks to @jschmied.

- TensorFold Starlight moves to its own repository, [ashhart/tf-starlight](https://github.com/ashhart/tf-starlight), with its history: it pins the TensorFold release it is tested with and checks the engine contract against that release in CI (#1972, fixes #1970).

- DeepSeek-V4.1 on Metal, part one: its kernels, the decoder over every layer, Engram, prompt chunks and DSpark drafts, each checked against float64 references (#1710-#1719). Serving it on Metal follows with #1720-#1722. Thanks to @gilby.

- Docs: the cache benchmark receipts say which tool-call content they count (#1999). Thanks to @garvrao80.

- `zig build test-info-config` runs info's issue-934 test again: it was built from the CUDA CLI root, which never reached it (#2006). Thanks to @jishnuvenugopal.

- The Nemotron XPU loader and the XPU safetensors reader read shard headers through core, so a malformed or oversized header is refused before anything is read (#1919). Thanks to @jschmied.

- Windows: a cost-cache read that lands mid-replace is counted as a miss (safe: the window is priced again), so its concurrency test no longer flakes (#2001).

- GLM on CUDA with TF_GLM_KV_SHARE no longer reports a balanced idle KV pool as (UNEVEN) at close; nothing else changes (#1993). Thanks to @Bizuayeu.

- lfm3, Spark 2.5 and the GLM vision tower take their RoPE frequency tables from core's exact helper; no value changes (#1995, part of #1852).

- GLM refuses a checkpoint whose `tensorfold_activation_dtype` asks for anything but bfloat16, naming the key, instead of silently serving bf16 activations; docs/api.md says which engines serve `/v1/decisions` (#1987, for #840).

- GLM on CUDA at TP=3 can pack the fused KDA input to an even width with `TF_GLM_KDA_PAD=1`, carried in the fingerprint every rank agrees on; unset, the engine is unchanged (#1545). Thanks to @BadAd84.

- Seven host BF16 conversions on load and inference paths (the XPU loader, the CUDA vision towers, Gemma 4's assistant weights, Step5's vision, DeepSeek's DSpark and Flash Next's PLE scale) round through core's shared conversion, so a negative NaN no longer overflows in Debug builds; every finite value gives the same bits (#1954, part of #1863). Thanks to @jschmied.

- Qwen 3.5 on CUDA rounds its host-side weight copies (norm weights, EXL3 norms and vectors, drafter scales and biases) with core's bf16 rounding, so a NaN weight stays NaN instead of becoming a finite value at load; every other value is bit-identical (#1986, part of #1863).

- Flash Next on HIP routes from the checkpoint's stored BF16 router rows, widened in the router kernel with the FP32 lane order kept, with the same tokens, for about 2.5 MiB more memory a layer (the BF16 rows are held beside the FP32 ones) (#1614). Thanks to @gilby.

- The Metal vision towers' shard reader and the cluster planner's checkpoint reader check a safetensors header's length with core's `headerSize` before reading it, so a malformed or oversized header is refused before any large allocation (#1983, part of #1872).

- `serve --prompt-cache-disk DIR` is a shorter spelling of `--learn-conversations --learn-dir DIR`, with the same limits, spill and start-up line; giving it together with either legacy flag is refused (#1908). Thanks to @garvrao80.

- HIP gate-5 launch records are re-derived with ROCm 10.1's hipcc, now the one reference toolchain; record values only, no kernel or serving change (#1979). Thanks to @laudney.

- GLM on CUDA runs a prompt chunk's MTP pass on its last row past the caches, so cold prompts of 32K to 167K tokens finish 5% to 14% sooner on three GB10s at TP=3, with identical replies (#1577). Thanks to @gilby.

- Flash Next on CUDA runs the MTP head's experts on a call's last row only, the one row that reaches the draft; the head's cache is still written for every row, and replies are unchanged (#1925). Thanks to @jschmied.

- Nemotron on CUDA drafts copies of earlier text in its lanes: a verified copy is tried first inside the existing verify windows with the same acceptance rule, so replies are unchanged and drafts off is main exactly (#1974, part of #1910).

- The XPU build makes one core module per device tree and passes it through the runtime, family and module builders, while the Nemotron XPU device binaries keep their own optimize level (#1918). Thanks to @jschmied.

- On HIP, decode's softmax stats and partial sums load sixteen values ahead and fold them in the same order, with the same bits (#1961). Thanks to @AbrahamCete.

- GLM's CUDA `TF_KV_PAGES` is read through the profile registry, and startup reports the selection it resolved; the value and where it is read are unchanged (#1959, part of #1660). Thanks to @garvrao80.

- GLM on CUDA can keep the KV pages' latent rows in pinned host memory with `TF_GLM_KV_HOST`, for a 1M-token window on two 96 GB cards; unset, the engine is as before, with identical tokens at TP=2 on two GB10s (#1795, part of #1794). Thanks to @russellmm.

- On HIP, the planned decode score takes four keys a warp on long spans and two on short ones, with the same lane order and folds, so decode runs 7.5-18.9% faster on Qwen3.6-35B-A3B and Qwen3.8-27B at 32K and 64K with identical tokens on a gfx1151 (#1932). Thanks to @AbrahamCete.

- Ling's grouped expert routing runs on Metal, checked against a float64 reference: sigmoid scores plus the routing bias, groups picked by their top scores, empty picks handled and both infinities in a group refused (#1836, part of #1247). Thanks to @ThinkOffApp.

- HIP loads a shared expert wider than the routed experts beside the stack instead of refusing it, and keeps the fused stack when the widths match, with the same tokens (#1612). Thanks to @gilby.

- Flash Next on CUDA: prompt passes do less work with the same bits (BF16 projections of 1,024 rows or more on a 128 x 128 tile, the hyper-connection writeback and normalize in one launch, wide route logits read once, the QSA select and the fused DeltaNet gated norm reworked, the prompt scratch sized by use), so cold prompts run about 4-5% faster on a GB10 with identical tokens (#1764, part of #1762). Thanks to @Arminova.

- The proposal and pull-request templates ask for proof and an opt-in for any change to default-path output tokens, planted controls for new kernels, and the `--version` output rather than a release number (#1596). Thanks to @jayleaton.

- On a CUDA card too small for an EXL3 checkpoint's embedding beside its window, the Qwen3.5 engine maps the embedding rows from host memory instead of refusing to start; wherever they fit on the card, they stay there as before (#1678). Thanks to @diwakergupta.

- An Intel XPU server built without DPAS (`-Dxpu-dpas=false`, Gen12LP) serves Nemotron-H and leaves out the Qwen3.8-27B runner and Gemma 4 E4B, refusing them by name, and Gen12LP's Nemotron prompt projection takes 8 x 4 tiles (#1692). Thanks to @jschmied.

- MOSS-Image computes its sampling schedule natively from `scheduler_config.json`, bit for bit what its `MOSSImageScheduler` sets: the shifted float32 grid snapped to the training levels, repeats dropped, sigmas ending at 0 (#1670, part of #1529). Thanks to @BobClawblaw.

- The Qwen3.8-27B recipe gains measurements on a 48 GB M5 Pro Mac mini with 1.2.1: 89.1 tok/s sampled code with DFlash2, against 86.5 on 1.1.0 (#1947). Thanks to @pauldodd123.

- Cache benchmark runs qualify only when a complete stream showed real text or reasoning: a usage-only reply no longer passes as "equal output" (#1967). Thanks to @garvrao80.

- On HIP, an integrated GPU counts the card's own memory only when sysfs confirms the firmware carve-out; every other case, GTT included, keeps the system's count, and the start-up line says which it used (#1963). Thanks to @llstrk.

- The prompt cache keeps exact image identities beside its words (position, rows, grid and a content hash over the image bytes, never a path), so a cached state is never resumed for a different, moved or re-gridded image; image prompts' states are no longer saved to disk or learned. Text-only prompts are unchanged (#1962, part of #1927). Thanks to @laudney.

- Cortheon reads a GitHub file as raw text from `GitHub.com` and `www.github.com` too, keeps search results and seen links whose scheme is written in capitals, and reads a link whose tag breaks after a carriage return (#1949). Thanks to @ppardus.

- Cortheon finds every closing tag around a harness's context blocks in one pass, so a message with many unclosed tags no longer takes time quadratic in its length; the result is unchanged, and the fuzz harness checks it against the old scan (#1898). Thanks to @jishnuvenugopal.

- A request-log test sets its file's mode after creating it, so it passes under any umask (#1900). Thanks to @jishnuvenugopal.

- `zig build -Dgpu=<gpu>` builds one stripped server for one GPU with every model family: about 69% smaller on CUDA (`sm_80`, `sm_86`, `sm_89`, `sm_120`, `sm_121` or `spark`) and 58% on HIP (a `gfx` name), and a CUDA one refuses a GPU of another SM at startup, naming both (#1936, fixes #1934). Thanks to @laudney.

- A lane stream that finishes on its first token is released when its reply ends, in every family (it leaked until the server stopped), and the lane host gains a byte ledger that families can opt into, so a request reserves its prompt and reply and waits when it doesn't fit; no family uses it yet (#1945, part of #1943). Thanks to @laudney.

- On Ryzen AI Max (RDNA 3.5) GPUs, one- and two-row decode launches use narrower stream tiles, so Qwen3.6-35B-A3B chat decodes about 18% faster with drafts on, and Qwen3.8-27B about 5% faster, with the same tokens (#1855). Thanks to @AbrahamCete.

- Flash Next on RDNA 3.5 iGPUs (gfx1150 to gfx1153) fills prompts in 4,096-row chunks by default on one GPU, falling back to 1,024 when the larger chunk would admit fewer streams. Tokens are unchanged. The larger chunk's scratch holds about 4.6 GiB more device memory; `--policy prefill_step=1024` gives it back (#1651, fixes #1649). Thanks to @gilby.

- CUDA's Qwen DFlash2 drafters must declare sliding attention on every layer (`use_sliding_window` and one `sliding_attention` per layer), which the kernels run, so a drafter declaring a full-attention layer is refused before loading instead of running through the sliding ring (#1946, part of #1863). Thanks to @garvrao80.

- Flash Next on HIP gathers n-gram drafts in parallel and releases the checkpoint's weight pages after uploading them, so the page cache falls from 31.5 to 2.8 GiB and the first cold prompt runs in 28.55 s instead of 48.44 s on a gfx1151 (#1611). Thanks to @gilby.

- `serve --profile stable` names today's defaults explicitly, and startup reports the profile and any registered switch set away from it; seven CUDA KV switches (`TF_QWEN_KV_PAGES`, `TF_QWEN_KV_SHARE`, `TF_GLM_KV_SHARE`, `TF_KV_SHARE`, `TF_FLASHNEXT_KV_PAGES` and two `_ALL` checks) are read through one registry, with environment values winning byte for byte; `fast-exact` and `fast` are refused until qualified (#1944, part of #1660). Thanks to @garvrao80.

- Compaction keeps a memory note sent as text parts instead of summarizing the conversation again, and a note or remembered file path that arrived with carriage returns no longer carries them into the next note; Cortheon's checks read a shell command wherever it sits among the call's fields, its search fallback stays within its 120-byte cap, and its text cuts never end in a partial character (#1939). Thanks to @ppardus.

- Qwen3.8-Flash-Next on CUDA serves `nvidia/Qwen3.8-Flash-Next-NVFP4` at `--tp 2` on two GPUs, one rank a host: its BF16 attention, DeltaNet and shared-expert projections, its n-gram table of E4M3 rows with one BF16 scale, and its MTP head's E4M3 experts with 128x128 block scales load as stored. The CUDA server now serves `--tp` (`--rank`, `--master`, `--master-port`); families that open without ranks refuse it by name, and `--idle-unload` is refused with it. The ranks compute and compare a cached digest of every checkpoint byte, so `TENSORFOLD_CHECKPOINT_SHA256` is no longer required (it still overrides the digest). A crash on the first prompt at `--tp 2`, a vtable built on the stack, is fixed. At `--tp 2` each rank owns half the routed experts whole, so replies are one GPU's bytes; on two GB10s drafted replies equal `"draft": false` (6 of 6) and concurrent replies equal their solo runs in 8 of 8 cells (#1771, for #1754). Thanks to @laudney.
- GLM-OCR on HIP: the publisher's BF16 `zai-org/GLM-OCR` served natively, its CogViT image encoder and GLM decoder in Zig and HIP with no Python or libtorch, every reply checked against the model card's Transformers examples (#1940, part of #1927). Thanks to @laudney.
- The dense Qwen Metal loader validates config-derived projection geometry through shared structural checks, retaining the current 9B and 27B qualification policy and the 2B's separate engine (part of #503).

- A model family can declare that its replies never carry a reasoning block, so the whole reply is answer content whatever `enable_thinking` the chat template renders; the OCR models in #1927 are the first to need it (#1933). Thanks to @laudney.

- GLM-5.3-Flash on Macs with `--learn-conversations`: a prompt state restored from disk is charged at its full size (its attention rows as well as its KDA states) before it is chosen, prefetched or recalled, so the prompt cache no longer grows past its budget by about 2.5 GB a restored state at 130,000 tokens (#1818, part of #1572).

- The tokenizer supports NFKC and NFKD normalizers, as Nemotron Parse uses, and a normalized added token decodes as the checkpoint spells it, matching Hugging Face tokenizers. No model we serve decodes differently (#1931, part of #1927). Thanks to @laudney.

- The Flash Next recipe covers native HIP on a Strix Halo iGPU, with speed and quality for three checkpoints (#1747, fixes #1617). Thanks to @gilby.

- HIP gains a shared rocBLAS BF16 projection with FP32 sums, loaded as an optional runtime library. Nothing calls it yet (#1930, part of #1927). Thanks to @laudney.

- The prompt cache can let a family count repeated image or keyed prompts as repeats, under the cache's own key. No family uses it yet (#1929, part of #1927). Thanks to @laudney.

- HIP Qwen3.5-family servers price shared draft rounds at the row bucket they actually run, so Qwen3.6-35B-A3B with MTP at four streams no longer drops into a slower drafting mode, and models load faster. Tokens are unchanged (#1830, fixes #1695). Thanks to @AbrahamCete.

- The Living Weights view reads a shard's header with core's safetensors reader, so a malformed header, length word or offset is refused with an error instead of a panic or a 64 GiB allocation, and its sums and row counts are checked (#1923, part of #1872). Thanks to @jschmied.

- `core/gguf.zig` reads GGUF files with every length and offset checked before use, and repacks GGUF's Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, Q3_K and Q2_K blocks exactly into per-group affine weights. It is the first step toward native GGUF on Apple Silicon (#1926, part of #1920).

- With `TF_NEMOTRON_METAL_KV_PAGES` off, Nemotron on Metal's paged KV checks cost one null check a pass, and debug builds assert that every pass's K/V rows sit inside the lane's pages (#1914, part of #1293). Thanks to @jschmied.

- Flash Next on CUDA reads its experts past the page cache with O_DIRECT, and keeps a load's shards open, so a cold start on a GB10 drops from about 74 s to 48 s with the same replies and the n-gram table still cached. `TF_LOAD_DIRECT=0` keeps the mapped reads (#1901, fixes #1873). Thanks to @jschmied.

- Flash Next on CUDA reads its embedding shard's header with core's safetensors reader, so a malformed header is refused with an error instead of a panic, and a tensor's bytes must match its shape (#1917, part of #1872). Thanks to @jschmied.

- The image session benchmark fails a turn whose stream ends before `[DONE]`, keeps it out of the next turn's history, and exits 1 when any turn failed, instead of reporting a partial reply as a success (#1916, fixes #1915). Thanks to @garvrao80.

- Gemma 4 on Intel GPUs refuses a shard with wrong tensor sizes, overlapping tensors, unknown dtypes or too many dimensions, so a bad file can no longer overrun a device buffer (#1905, part of #1872). Thanks to @jschmied.

- TensorFold Starlight, the enterprise platform, lands in `starlight/`: one governed OpenAI- and Anthropic-compatible endpoint for a fleet of TensorFold engines, with single sign-on (OIDC, SAML and SCIM), per-model access, priorities and budgets, model placement and supervision on Mac, Linux and Windows nodes, an admin console, Prometheus metrics and the TensorFold Connect desktop app (#1922, fixes #1921).

- `TENSORFOLD_PREFILL_ORDER=srpt` lets short replies decode ahead of a long prompt that is still filling, while the prompt keeps at least a tenth of the GPU time and `--decode-share` still applies. Replies are the same tokens. It is off by default (#1876). Thanks to @gilby.

- Qwen3.8-27B with a DFlash drafter on Metal compiles the drafter head's lane pipeline at load, so the first request no longer waits on it, and a head that fails to compile is refused by name at start-up instead of failing that request (#1734). Thanks to @kieranscotney.

- H3 on CUDA finds each tensor by its real header entry, so a look-alike in metadata no longer returns the wrong tensor, spaced JSON headers load, and malformed headers are refused (#1906, part of #1872). Thanks to @jschmied.

- GLM on CUDA refuses a malformed DFlash2 drafter header, or a tensor range past the file, with an error instead of crashing or reading unset memory (#1904, part of #1872). Thanks to @jschmied.

- Drama speech on CUDA reuses a catalog voice's reference prefill across requests, as Metal does: the backbone and tower state after the reference's whole prompt chunks is kept, and a later request in the same voice restores it and prefills only the rest. The cut is a whole chunk, so a cached request makes the chunk cuts an uncached one makes; host tests check this against a fake lane, and it is not yet measured on a GPU. It is opt-in: `TF_DRAMA_REFERENCE_CACHE=N` keeps N voices (1 through 16, each up to 61.4 MiB plus 46 KiB a saved row); the default 0 keeps none, and the default voice is unchanged (#1745). Thanks to @jayleaton.

- `pull` fetches an EXL8 width file's header again when its length word or tensor entries are corrupt, instead of crashing or keeping a bad copy (#1907, part of #1872). Thanks to @jschmied.

- DeepSeek V4.1 packs refuse a shard whose header length runs past its file or past 100 MB before anything is allocated, instead of allocating up to 1 GiB first (#1903, part of #1872). Thanks to @jschmied.

- EAQ containers refuse a shard whose header length runs past its file or past 100 MB before anything is allocated (#1902, part of #1872). Thanks to @jschmied.

- With `TF_NEMOTRON_METAL_KV_PAGES=1`, Nemotron on Metal takes KV pages as a reply grows instead of for its whole context at the start, so a short chat holds 1 page instead of 17. It is off by default (#1837, part of #1293). Thanks to @jschmied.

- Qwen3.5-9B runs on Metal on the 27B's engine, instead of being refused as an unqualified shape (#1826, part of #503). Thanks to @grantoverton.

- The kernel contract computes its affine and GLM float64 references on worker threads, so it finishes sooner, with the same reference bytes (#1776). Thanks to @BlivionIaG.

- Host code: sixteen hand-written fp32-to-bf16 copies now use core's one rounding, identical on every input; a negative NaN no longer panics a Debug build (#1864, fixes #1863). Thanks to @jschmied.

- RoPE: transformers' inverse frequencies are computed in one place (`core.rope_freq`), and Gemma 4 on CUDA uses them, with the same bits (#1853, for #1852). Thanks to @jschmied.

- `tools/bench_image_session.py` plays one chat session that resends the same picture every turn, and records each reply's hash, cached prompt tokens and first-token time, to check that image turns keep and resume their prompt (#1889, for #1828).

- The benchmark clients refuse a malformed `--temperatures` list, or a non-finite entry such as `nan` or `inf`, with a clear error and exit status 2 before any request, instead of a traceback (#1875, fixes #1874). Thanks to @garvrao80.

- LFM3's backends share one copy of the RoPE frequencies and the layer schedule, so every backend reads the same numbers, bit for bit as torch's (#1839). Thanks to @jschmied.

- The CUDA kernel contract's NVFP4 expert-waves control goes red as planted on large GPUs such as the RTX 5090, where its test shape used to fit in one wave (#1832, fixes #1778). Thanks to @rootsbymenda.

- The server's linear-time tests count thread cycles on Windows, so a busy CI runner no longer fails them by chance (#1854). Thanks to @Bizuayeu.

- Nemotron on Intel GPUs refuses a config with an unknown `layers_block_type` name at load, instead of serving that layer as attention (#1865). Thanks to @jschmied.

- GLM-5.3-Flash on CUDA refuses `TF_GLM_CHUNK_SOLO=4096` beside `--drafter` at load with a clear error, instead of holding 1.6 to 1.9 GiB of prompt buffers that no pass uses (#1842). Thanks to @Bizuayeu.

- The Intel GPU host tests build again: `zig build test-slide-xpu-host` no longer stops with a missing kernel image (#1840). Thanks to @jschmied.

- Flash Next on CUDA: `TF_FLASHNEXT_ATTN_LANES=1` (off by default) runs a shared decode round's attention in one launch a kernel, with each row's caches and position in a table uploaded once a round, instead of a launch per stream. Replies are the same tokens; the kernels are bit-equal to the per-stream launches (`tf-cuda-test flashnext-attn-lanes`) (#1739). Thanks to @optimisme.

- GLM, GLM-DSA and Flash Next on Metal drop their checkpoint from the file cache once loaded, keeping the SSD expert and PLE files that serving still reads (#1888, part of #1868).

- On Macs, the q27, Nemotron, LFM2 and Qwen3.5 loaders drop their checkpoint from the file cache once the weights are loaded, so unified memory doesn't hold them twice: on an M5 Ultra a loaded Qwen3.8-27B frees about 14 GiB and Nemotron 3.5 Lightning about 17 GiB, with the same replies (#1885, part of #1868).

- Gemma 4 on CUDA: `TF_KV_SHARE=n` keeps prompts as the KV pool's pages in the shared prompt store, with sliding-window tails kept per stream and the MTP assistant's pages bound to its cache; resumed, concurrent and drafted replies equal their plain runs. Unset, nothing changes (#1883, part of #1657).

- On M5 Macs, opt-in: `TF_ATTENTION_PV_FP32=1` keeps attention's P x V at full fp32 for Nemotron's prompt attention, closer to float64 (output cells off the nearest bf16 value fall from about 1.1M to 1.8K of 8.6M), at a large prompt-speed cost (30% to 87% slower at 2K to 32K). Unset, nothing changes (#1882, part of #1772).

- The benchmark clients refuse a nonpositive `--tokens`, `--reps` or `--levels` entry, or a malformed `--levels` list, with a clear error and exit status 2 before any request, instead of failing mid-run or writing zero-stream cells (#1869, fixes #1867). Thanks to @garvrao80.

- Qwen3.8-27B EXL3 checkpoints on CUDA prefill prompts much faster, with the same tokens: on a GB10, prompt prefill runs 38% to 42% faster from 2K to 64K tokens (2K: 494 to 705 tok/s), and decode gains 5 to 12% (#1467). Thanks to @BHCC2025.

- On CUDA, every family's loader drops its checkpoint from the page cache once the weights are on the GPU, as Flash Next does since 1.2.1, so a GB10's shared memory doesn't hold them twice: a served Qwen3.8-27B frees about 15 GiB. Files read at runtime (PLE rows, DeepSeek's Engram tables) are kept (#1884, part of #1868).

- Lanes: grammar masks for constrained streams are filled in parallel across streams, with the same masks and tokens; one stream is unchanged. On an M5 Ultra, Qwen3.5-2B serves 0.7% to 1.3% faster at 4 to 32 constrained streams (#1881, fixes #1455).

- On Macs, DeepSeek V4.1's CSA2 attention for decode and prompt rows, its bf16 and fp32 matvecs and phase-1 glue, and its registered glue kernels (norms, RoPE, FP8 and FP4 round trips, embed, join, mean) have their own Metal kernels under the kernel contract, ahead of the Mac server (#1709, #1710, #1711, part of #1676). Thanks to @gilby.

- Flash Next on CUDA: `TF_FLASHNEXT_READOUT_ROWS=128` (off by default, 64 unset) moves the hyper-connection read-out's switch to the prompt tiles from 64 rows, so a large shared decode round keeps the row kernels: the same bits, and up to 7% more throughput at 16 streams on a GB10 (#1741, part of #1737). Thanks to @optimisme.

- Flash Next on CUDA (one GPU): a partly kept window's DeltaNet rows are applied at the start of the next window instead of replayed, a lane's fixed buffers are reused across streams, and a pass's shared expert runs on a side stream, all with the same tokens; `TF_FLASHNEXT_KEEP_LATE=0`, `TF_FLASHNEXT_LANE_PARTS=0` and `TF_FLASHNEXT_SHARED_SIDE=0` turn each off (#1763). Thanks to @Arminova.

- HIP: Flash Next loads oMLX checkpoints whose MTP head sits at the top-level `mtp.*`, so their drafts work, and the usual `language_model.mtp.*` layout gives the same tokens as before (#1613, fixes #1598). Thanks to @gilby.

- On Macs, Flash Next with `--ple-on-ssd` reads its per-layer embedding rows from disk in batches instead of one at a time: decode is 36% to 39% faster on an M5 Max (75.8 to 105.1 tok/s on code) and cold prompts take a third of the time at 2K tokens, with the same replies (#1462, fixes #1461). Thanks to @paragontasx.

- Qwen3.5-family models on CUDA can apply a verified window's kept DeltaNet rows inside the next round's slot scan, instead of a separate pass over every layer's state, with the same bits. On a GB10, Qwen3.8-27B MLX 4-bit decodes 1% to 8% faster, most at 16 streams, with none slower, for about 66 MiB more a stream. It is opt-in with `TF_QWEN_KEEP_LATE=1`: on by default it failed some served chats on a GB10 with `CUDA_ERROR_INVALID_VALUE` (8 of 64 requests on Qwen3.8-27B MLX 4-bit), so the separate pass stays the default until that is fixed. Replies are unchanged either way (#1824, #1958). Thanks to @jrb-ai, with code from @fkiene's cuda.fast submission.

## 1.2.1

- Qwen3.8-Flash-Next on CUDA: a loaded server drops its checkpoint from the page cache. On a GB10, whose GPU memory is the system's, about 36 GiB of cached shards had left under 1 GiB free, and long prompts slowed to about 0.73x from some point in a server's life until a restart; they now hold their speed, with the same tokens (fixes #1868).

- GLM-5.3-Flash on CUDA with `TF_GLM_KV_SHARE`: a prompt with images keeps and resumes its prompt state, matched on each image's content, so an agent session that resends a screenshot no longer prefills the whole conversation every turn; a changed image resumes only the text before it (fixes #1828).

- Kernel contract (Metal): the M5's tensor units keep 10 mantissa bits of an fp32 `mma_16x32` operand, cut toward zero. The contract now states that cut for every kernel with an fp32 operand: Gemma 4's, Spark's, Flash Next's, Nemotron's and MiniMax's attention and Nemotron's SSD products. Each has a control that fails without it, and no output changes (#1748, part of #1546). Thanks to @BobClawblaw.

- GLM-5.3-Flash on CUDA: `TF_GLM_CHUNK_SOLO=4096` (opt-in, on every rank) lets a prompt pass take 4,096 rows while no other stream is decoding, with the same bytes. On GB10s a cold prompt's first token comes 6% to 9% sooner from 8K to 128K tokens. It holds 1.6 to 1.9 GiB more for prompt buffers, and a request that arrives during a long pass waits for it to end. Unset, nothing changes (#1823, option C of #1554). Thanks to @Bizuayeu.

- Qwen3.8-27B on M5: checkpoints that mix bit widths in a fused projection (such as the oQ4 and oQ2 quantizations) answer prompts longer than 128 tokens, instead of failing with `BadCoreProjectionRows`; each 128-row piece is the call main already made, so replies keep main's tokens (#1822). Thanks to @grantoverton.

- Windows: the cost cache reads a table through a shared file handle, so a reader that meets a table being replaced sees a miss quietly instead of logging an unexpected `FILE_DELETED` status, which failed a host test (#1809, fixes #1805).

- Qwen3.8-27B and Qwen3.6-35B-A3B on CUDA: with `TF_QWEN_KV_PAGES=n` and `TF_QWEN_KV_PAGES_ALL=1`, the engine's own stream becomes a pool lane too, as on GLM, Gemma 4 and Kolibri-1, so no full-context window sits idle beside the pool (1.90 GiB back for the 27B at `--parallel 8 --context 16384`); without `_ALL` nothing changes (#1750). Thanks to @jschmied.

- Living Weights: a lesson no longer stops at its first step with `LayerRoundsAgain` (2 of 7 Nemotron lessons on a Mac and both on a GB10 before this). New fits are written on the 4-bit grid before they're judged, so rounding a taught layer again keeps its values; a lesson that still can't round says so in words; and the guide explains why the same lesson can take different steps on separate runs (#1810, fixes #1807).

- Lanes, opt-in: an external drafter's own chain chances price its verify rows, the last verified grant bounds the next hold, and measured hold costs filter shared-row reservations, so a stream given no rows is never offered a hold. Defaults are unchanged (#1381). Thanks to @BlivionIaG.

- HIP: prompt projections round to the activation type in their tile instead of writing fp32 and launching a cast, with the same bytes; cold prompts are 2% to 3% faster on Qwen3.8-27B on gfx1151 (#1685). Thanks to @AbrahamCete.

- On Macs, DeepSeek V4.1's routed experts have a grouped prompt GEMM under the kernel contract, bit-equal to the decode rows; nothing serves through it yet (#1708, part of #1676). Thanks to @gilby.

- On Macs, DeepSeek V4.1's MXFP4 experts have their own Metal kernels under the kernel contract, with Kimi's left as they are; nothing serves through them yet (#1707, part of #1676). Thanks to @gilby.

- On Macs, DeepSeek V4.1's FP8 block-scaled rows and tiles have their own Metal kernels under the kernel contract, for the release checkpoint's serving parts; nothing serves through them yet (#1706, part of #1676). Thanks to @gilby.

- LFM3-1.7B and LFM3-VL-2B's official MLX 4-, 6- and 8-bit checkpoints are admitted and checked (geometry, quantization, tensor headers) ahead of their engines, with one table of the decoder's tensor names for every backend (#1513, part of #1491). Thanks to @BlivionIaG and @jschmied.

- Living Weights: `DELETE /v1/slide/facts` takes a list of fact ids and removes them as one reversible change: every graph record publishes together, and a shard write that fails part way restores the whole selection or marks every selected id pending for retry. A single delete is unchanged (#1816, part of #678).

- XPU: Gen12LP GPUs (Iris Xe, DG1) build Nemotron's kernels, with a plain bf16 fallback for its two DPAS products and decode attention through the partial kernels; DPAS builds are unchanged (#1673). Thanks to @jschmied.

- Lane rounds count the time each stream spends in grammar mask fills, per round and per reply, reported with the round metrics; nothing is read when no stream is constrained, and behaviour is unchanged (#1814, part of #1455).

- CUDA tensor parallel: a rank that cannot bind its liveness port (`NCCL_COMM_ID`'s port + 1, often held by an earlier server still exiting) says so, as do a rank that never arrives and a failed accept, instead of reporting a generic "glue kernels" error (#1813, fixes #1761).

- Living Weights: Nemotron Omni registers the shared Nemotron learner on CUDA, reading its widths from the family description and the raw language-model tensors, and a lesson that carries an image or audio is refused by name instead of training through the media towers (#1815, part of #1801).

- Gemma 4 E4B on Intel GPUs: prompts are read 2.0 to 2.6 times faster on an Arc Pro B70 (layers past the KV-owning ones run only for the last row, 1,024-row forwards, the GeGLU fused into the gate_up GEMM), and decode is 4% to 8% faster at 32K to 64K tokens, with the same logits (#1746, part of #1222). Thanks to @Mirrdhyn.

- HIP: the prompt attention tile keeps its probabilities in registers and runs 6 waves a SIMD on gfx1151, with the same output bits; cold first token is 1% to 5% faster from 2K to 64K tokens on Qwen3.8-27B, and the tile is registered with the kernel contract (#1788). Thanks to @AbrahamCete.

- `--slide` on a checkpoint whose family doesn't learn on that backend now stops before loading with "`--slide`: <model type> checkpoints do not learn on <backend> yet", on CUDA, Metal, HIP and XPU, instead of serving and learning nothing. Metal learners register by family, as CUDA's do, and read their widths from the family's Living Weights description, so a new family's learner plugs in without editing the Metal host (#1800).

- Windows: the server loads the grammar library as `tfgrammar.dll`, so structured output and required or named tool calls serve there too (#1796, fixes #1789). Thanks to @wesley4546.

- With `TF_DSV41_PF_4K=1`, `aot-needs` also lists the 2,048-row prefill variants the forward still runs, so a pack built with it no longer fails its first operation with `MissingTritonVariant _kv_store … RING 4096` (#1683). Thanks to @jayleaton.

- DeepSeek-V4.1's prefill planner sizes the indexer's scratch before it shares and prices the workspace, so ragged prompts and the two-batch-overlap branch can no longer plan less than they use. This takes about 128 MiB more per rank on the Flash-1017 pack, and tokens are unchanged (#1681). Thanks to @jayleaton.

- On WSL2, where Intel's Sysman can't run, an Intel GPU's free memory falls back to the device's total less what TensorFold has allocated, so Nemotron with MTP can size its prompt cache there. Native Linux still asks Sysman (#1691). Thanks to @jschmied.

- On Intel GPUs, the engine asks Level Zero how many drivers and devices there are before listing them and skips empty entries, so `tensorfold-xpu devices` and the device tests no longer crash on WSL2 with an Iris Xe (#1672). Thanks to @jschmied.

- With `--cortheon`, text beside an image now counts toward the room left for a fetched page, so a long vision turn no longer gets a full page. Progress lines and the check for a utility request no longer cut a character in half, and a fallback search whose first word doesn't fit is skipped instead of sent empty (#1677). Thanks to @ppardus.

- The CUDA drafter cost-identity host test sets the rewritten file's time itself, so it passes on filesystems with a coarse clock, such as ramfs and NTFS (#1770, fixes #1697). Thanks to @wesley4546.

- `TENSORFOLD_LOG_TIME=1` (or `on`) starts each server log line, the `[tensorfold]` lines and every engine's warnings, with its UTC time to the millisecond, to match the log against other records. It's off by default, and the log is unchanged without it (#1690, fixes #1626). Thanks to @atakann.

- On GB10s, affine-4 (MLX 4-bit) decode and verify projections that read the same input run as one grouped launch, each output byte-equal to its separate launch under the kernel contract, for the dense targets, Nemotron, the DFlash2 drafter, Gemma 4 and Spark. Replies are unchanged. Nemotron 3.5 Lightning: 8 streams +13.3%, 2 streams +5.9%; Qwen3.8-27B with DFlash2: decode after a prompt +3.7% to +5.0%, 1 stream +4.5%, 4 streams +3.5% (#1783).

- Shared draft rounds price each stream's drafts before drafting, and a stream whose drafts would get no verify rows skips the drafter's proposal work, for Nemotron, Qwen's MTP, GLM, Gemma 4 (assistant and DFlash) and attached LFM2 drafters, on CUDA and Metal; each family keeps its head step's cost in the price. Replies are unchanged. Nemotron 3.5 Lightning with its MTP head: on an M5 Ultra, 16-stream chat +9.0% greedy and +4.7% sampled; on a GB10, 16 streams +2.1% and 8 streams +1.1% (#1779).

- MOSS-Image's image decoder (the checkpoint's FLUX.2 small VAE, in float32 as the pipeline runs it) runs on Metal inside the `moss_image` family, and `tf-moss-image` writes `picture.png` itself, so torch no longer decodes (#1687, #1504). Thanks to @BobClawblaw.

- On CUDA, a driver older than CUDA 13.0 is refused at startup with a plain message naming the driver to install (R580 or newer), before any kernel loads; an R550 driver had ended the process with SIGILL inside libcuda on the first module load (#1791, #1724). Thanks to @rootsbymenda.

- Spark 2.5 on CUDA: `TF_KV_SHARE=n` keeps prompts as the KV pool's pages in the shared prompt store, with at most n kept states, so a conversation resumes at the store's last 256-token mark from pages it already holds instead of a full copy. Unset or 0, nothing changes. The native CUDA adapter it uses is family-neutral; Kolibri-1 follows (#1756, part of #1657). Thanks to @jschmied.

- DeepSeek-V4.1 reads the release checkpoint as published (`deepseek-ai/DeepSeek-V4.1-Flash-1017`): FP8 E4M3 blocks with 32x32 E8M0 scales, MXFP4 routed experts and the FP8 Engram table, beside the EXL3 trellises. `tf-dsv41-engram-pack pack` writes the release's Engram tables as row shards with a manifest (the source tensors, each table's SHA-256 and a seeded row sample), `manifest` writes one for tables packed earlier after checking sampled rows against the release, and the CUDA engine refuses a release checkpoint by name before loading it (#1705, for #1676). Thanks to @gilby.

- MOSS-Image's prompt encoder runs on Metal inside the `moss_image` family, in bf16 as published with fp32 sums where the reference sums, under the Metal kernel contract (`tf-kernel-contract --only "moss image"`) (#1512). Thanks to @BobClawblaw.

- GLM-5.3-Flash on CUDA with `--vision`: `TF_GLM_VISION_HOST=1` keeps the vision tower's bf16 weights in pinned host memory mapped into the device and copies each unit into one device slot just before it runs, so rank 0 keeps 240 MiB for the tower instead of 1.05 GiB and every rank gets that room back for KV pages and the prompt cache (#1760). Unset, nothing changes. Thanks to @russellmm.

- A tool marked `"strict": true` is now held to its schema under `tool_choice: "auto"` (or none) on every family with a grammar runtime: the server compiles a grammar that triggers at the call markup, so a strict tool's arguments always validate (#1688). `TENSORFOLD_TOOL_GRAMMAR=1` holds every auto tools request the same way.

- Qwen3.5 and 3.6 on CUDA serve past their trained 262,144 positions with static YaRN rope: `--rope-yarn 4` on the checkpoint as downloaded, or a config whose rope carries a YaRN block (SGLang's staged 1M copy loads as is), raise the window to 1,048,576. On one GB10, `nvidia/Qwen3.6-35B-A3B-NVFP4` prefills a 1,048,512-token prompt in 26 minutes, finds a passphrase placed 5% in, and decodes to the window's last token; drafted, concurrent and resumed replies equal their plain runs. Without YaRN every reply is byte-identical to before. HIP, Metal, the DFlash2 drafters and every other family refuse a scaled rope by name instead of reading it as plain rope (#1675).

- DeepSeek-V4.x requests that omit sampling fields now sample at the release's recommended settings, temperature 1.0 and top_p 1.0 from DeepSeek's model cards, instead of decoding greedily when the checkpoint has no `generation_config.json`. A `generation_config.json`, the serve flags and the request still override them (#1725).

- DeepSeek-V4.x: `tool_choice: "required"` and named tool calls are now constrained to the model's own DSML call markup, so those requests are held to a valid call on DeepSeek too (#1684).

- Qwen3.6-35B-A3B on CUDA: `--drafter` also takes a DFlash2 draft model, chosen by the `dflash_config` in its `config.json`; the checkpoint's own directory still drafts with its MTP layer. With `incoai/Qwen3.6-35B-A3B-DFlash2` over `nvidia/Qwen3.6-35B-A3B-NVFP4`, one GB10 decodes 130 tok/s at one stream on a 4.2K-token prompt (MTP: 99), and every drafted reply equals its plain run (#1674).

- A fresh-start session disk now clears the `.tfs` and `.tmp` files earlier serving processes left behind, under an exclusive lock on its directory, so restarts no longer fill the disk (#1680). Ported by @jayleaton from @mikolaj92's downstream fix, corroborated by @Jenpo.

- GLM-5.3-Flash on CUDA: `TF_GLM_KV_SHARE=n` with `TF_KV_PAGES=on` keeps prompts as the KV pool's pages in the shared prompt store (the one `TF_QWEN_KV_SHARE` uses), with at most n states past them, on one GPU and at tensor parallel. Where main's flat store has no room for a long state (GB10s at TP=3 with the 1M window), a 64k message sent again resumes in about 1.7 s instead of prefilling for 36 s; token-id prompts resume at every width (#1732, #1510). Unset, nothing changes.

- MOSS-Image-Base's transformer runs on CUDA too, through the same test tool (`tf-moss-image`) and cases as on Metal (#1528, for #1518). It runs on an RTX 5090 (sm_120) and a GB10 (sm_121) and refuses other GPUs; on CUDA, the prompt encoder and image decoder still run in torch, and nothing is served yet. Against the publisher's torch bf16 pipeline, every step's cosine is 0.99967 or better, the same step run twice gives the same bytes, and a 1024x1024 picture is 44.0 dB (RTX 5090) and 45.9 dB (GB10) PSNR from torch's. Its 50 guided steps at 1024x1024 take 24.6 s on an RTX 5090 and 69.6 s on a GB10, 1.26x and 2.03x faster than torch. Its four kernels and full attention at head dim 128, a new instance of the vision towers' kernel, pass the kernel contract with five planted controls red; the towers' existing attention kernels compile to the same SASS. See [the recipe](docs/recipes/moss-image-cuda.md). Thanks to @BobClawblaw.

- DeepSeek-V4.x replies carry the Spark recipe's per-response stats (`prompt`, `cached`, `ttft_s`, `finish`, `completion`, `prefill_s`, `decode_s`, `rounds`, `drafted`, `accepted`) in their `tensorfold` block, after its own keys; `TENSORFOLD_WIRE=spark` turns them on for any family and `native` turns them off (#1753).

- DeepSeek-V4.1 on CUDA backs up only the ratio-2 recurrent carries before each window, instead of every layer's sliding-window ring (89 device copies, 6.4 MB a window on Flash-1017); rejected rows still roll back exactly (#1752).

- DeepSeek-V4.1's DSpark drafting on CUDA no longer drains the whole context before each draft pass's staging upload, so a draft pass stops waiting for every side stream (#1751).

- DeepSeek-V4.1 at tensor parallel: every rank's session store now charges disk eviction the same logical size, so the ranks evict the same prefixes and their caps no longer drift apart, and a restore checks the leader's identity before it changes anything (#1679). Adapted by @jayleaton from @florianbrede-ayet's downstream fix, with @rootsbymenda's report of the diverging caps.

- GLM-5.3-Flash on CUDA: `TF_GLM_HOST_CACHE=1` on every rank keeps its prompt states in pinned host memory mapped into the device, sized by `--prompt-cache-gib`, so on discrete cards where the KV pool leaves the device arena a fraction of a GiB a long conversation's state is kept instead of refused and the next turn resumes rather than prefilling the whole conversation (#1736, #1735, #1510). Unset, nothing changes. Thanks to @russellmm.

- EXL3 mixture-of-experts models on CUDA (GLM-5.3-Flash and the Qwen3.5-MoE EXL3 packs): on sm_120 and sm_121 (RTX PRO and RTX 50 Blackwell, GB10), the grouped experts kernel reads a warp's next tile of weights while it decodes the current one (#1505). The decode and the sums are unchanged, so outputs are the same bits; the kernel contract checks them against the previous loop at one to three tiles a warp and at the served models' widths, with two planted load faults that must fail. On a GB10 the kernel ran 11% faster at MiMo-V2.6-Flash's expert shapes (109 to 97 µs a launch), and GLM-5.3-Flash at two GB10s decoded the same tokens, within 1% of its previous speed. Ampere and Ada GPUs build the same code as before. Thanks to @ajensenwaud.

- CUDA graphs: a capture whose end fails because it was invalidated (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`) no longer leaves its stream marked as capturing, which had refused every later capture on that stream and every temporary host upload in the process with `UnownedCaptureSource`. A failed end that leaves the stream capturing, such as one from the wrong thread, still keeps the guard (#1744). Thanks to @jayleaton.

- Chat streams with thinking on: the joined `delta.reasoning_content` pieces now equal the finished reply's `reasoning_content` byte for byte. The stream had kept the whitespace the finished reply strips, such as the newline before `</think>`; it now holds whitespace back until more reasoning follows. A streamed `/v1/responses` reasoning item now matches the non-streamed one too (#1743). Thanks to @jayleaton.

- Spark 2.5 on CUDA keeps its full-attention layers' K/V (every fourth layer) in pages of the core pool, leased per stream as it grows, while the sliding layers keep their rings; streams no longer reserve a full context each up front. On a GB10 tokens are unchanged and speed is level. `TENSORFOLD_SPARK25_KV_PAGES=0` keeps the linear caches (#1653).

- Qwen3.8-27B on M5 Ultra: a prompt chunk of more than 128 rows runs each 4-bit projection in one dispatch instead of 128-row pieces, so each weight matrix is read once per chunk. Replies are token-identical; decode, verify rows and other chips are unchanged (#1738). Thanks to @Nipale-ai.

- The server reads a request's top-level `thinking` switch (`{"type": "disabled"}` or `"enabled"`), as DeepSeek's own API and Hermes Agent's DeepSeek profile send it. Before, it was ignored and the prompt still opened a think block. `chat_template_kwargs` still wins when both are set (#1742).

- MiniMax-M3.1-Flash on CUDA, at tensor parallel over two GB10s (sm_121 and two ranks only; anything else is refused), from Mia-AiLab's published EXL3 pack read as stored: a lossy third-party quantization whose logits against the release, teacher-forced over 2,406 tokens, give a mean KL of 0.0788 and 89.9% top-1 agreement (#1388, after #1235). It serves the release's DSpark drafts, kept prompt states, tools, structured output and images; one stream with drafts ran 24 to 39 tokens a second. The shared EXL3 grouped experts gain a SwiGLU-OAI epilogue (`act_mode` 2); modes 0 and 1 are bit-equal to before in the kernel contract. Thanks to @BobClawblaw.

- Gemma 4 31B on CUDA takes Google's MTP assistant (`google/gemma-4-31B-it-assistant`) as `--drafter`. On a GB10 it decodes German prose at 28.3 tok/s against 12.2 undrafted at one stream, with drafted replies equal to `"draft": false` and four concurrent replies equal to their runs alone. Other assistant types are still refused by type (#1385, for #1383). Thanks to @ThomasHalwax.

- `/metrics` now carries the scheduler load a traffic shaper needs, and a new `GET /v1/load` returns the same values as one JSON snapshot under the `/metrics` access rule (the `read` key scope; `--metrics-open` opens it): the server's identity and process instance, whether `--idle-unload` has parked the engine, model requests in progress by route group, queued prompt tokens and those still filling, requests requeued for memory and background passes that yielded, prompt and decode tokens counted as the work happens with the engine time of their prompt calls, and on CUDA the KV mode, free lanes and GLM-5.3-Flash's page pool in tokens. A value an engine cannot count is left out rather than shown as zero, no engine value is shown while the engine is parked, and `engine_parked` reads 1 only once a request can reload the engine. On the lane host `preemptions_total` now counts those requeues and yields, and the Flash Next host no longer reports it as a constant 0. On a GB10 with Qwen3.6-35B-A3B, replies are token-identical to before and `/v1/load` answers in under a millisecond when idle; scheduling is unchanged (#1655, #1558). Thanks to @laudney.

- Qwen3.8-27B on two Macs, second step: the 27B family can load one rank's half of the checkpoint (12 of 24 attention heads, half of the linear-attention heads, the MLP and the vocabulary head) and run its layers with the pair's row exchanges, while the embedding table stays whole on each Mac and the next token is the better of the two half-vocabulary picks. Each rank keeps the K split one Mac uses for the whole matrix, so the pair's sums are one Mac's. Nothing serves through it yet; on one Mac the split code is compiled out, the attention merges run main's kernels unchanged and every projection keeps main's split, so replies are the same tokens (#1433, after #1432). Thanks to @akol1.

- Flash Next on CUDA: `TF_FLASHNEXT_CHUNK_SOLO` (opt-in, up to 16,384 rows) lets a prompt pass take a larger chunk when no other stream is decoding, and `TF_FLASHNEXT_CHUNK`'s rows otherwise. On a GB10 at 4,096 rows, a lone prompt's first token comes 5-8% sooner, for 2.25 GiB more scratch and up to about 2 s of wait for a request that arrives during a large pass. Unset, nothing changes (#1733, option C of #1554). Thanks to @jschmied.

- Qwen3.8-27B on Metal: the DFlash2 drafter reads its sliding-window keys and values in place in its cache while the context has not wrapped, instead of copying them to scratch for every draft layer each round. Drafts are bit-identical (#1727). Thanks to @kieranscotney.

- Flash Next's MTP head norms its 10,240-value rows with tiled launches on CUDA sm_120 and sm_121 and split wave launches on HIP gfx1201, storing the same bytes as before: the norm kernel runs 1.4x to 2.5x faster on a GB10 and 4x to 5x on a Radeon AI PRO R9700 at 1 to 16 rows (#1702, fixes #1689). Thanks to @laudney.

## 1.2.0

- CUDA EXL3: the dense projections zero their split-K counters only when the buffer grows, on the calls' stream, instead of before every 128-row block of every call; each block's last program already resets its own counter. Replies are token-identical, and on a GB10 with the Qwen3.8-27B 3.50 bpw pack decode is 0.4% to 0.9% faster (#1475). Thanks to @diwakergupta.

- Qwen3.6-35B-A3B on CUDA serves NVIDIA's NVFP4 checkpoint (`nvidia/Qwen3.6-35B-A3B-NVFP4`) as downloaded: W4A16 NVFP4 routed and shared experts, FP8 attention and DeltaNet projections and the NVFP4 head run as stored, and `--drafter` naming the same checkpoint drafts with its bf16 MTP layer, quantized to affine-4 at load. One GB10 decodes 66 tok/s without drafts and 112 to 125 tok/s with them; drafted, concurrent and resumed replies equal their plain runs. The rules come from @gilby's #345.
- Qwen3.8-27B on M5 Macs serves one stream through the 16-row lane loop by default (`TF_QWEN27_LANES=0` keeps the serial loop), and lane rounds shared by several streams price and fill their rows from measured costs. Replies are token-identical. On an M5 Ultra with DFlash2, one stream decodes 2.6% faster on code and 4.6% on chat, and four greedy code streams 18.6% faster; on an M3 Ultra 16 streams are 3.5% faster and one stream 1.9%. Wider 32- and 64-row lone windows stay opt-in (`TF_QWEN27_WINDOW`, `TF_QWEN27_COPIES`) (#1621).

- DeepSeek-V4.1 on CUDA no longer stays up unable to answer after a sticky CUDA context fault: a host monitor sends the fault to fail-fast at once instead of waiting for a health poll, and with `TF_TP_FAILFAST=0` the process exits with status 70 rather than waiting on communicator teardown. Under fail-fast, a round error the ranks cannot recover from (such as `CudaFailed` or `Timeout`) now takes every rank down instead of leaving them out of step. A stall watchdog, off by default, withdraws readiness and exits when a round, snapshot, KV union or follower step makes no progress for `TF_DSV41_STALL_EXIT_S` seconds; off, it costs one null check per launch, and replies are unchanged (#1646). Thanks to @jayleaton.

- LFM2.5-24B-A2B on M1 and M2 Macs: the prompt tiles (`core/tile_qmm`) store their matrix fragments whole instead of element by element, which on an M1 had held every prompt tile to 0.04 TFLOPS. On an M1 Pro the tiles run at 2.4 to 2.7 TFLOPS, a 2,048-row prompt chunk takes 3.9 s instead of 243 s, a cold 930-token prompt's first token comes in 2.0 s instead of 212 s, and 16 greedy streams run at 116 to 123 tok/s together instead of 11 to 17. The tiles' outputs are bit-identical, so replies are the same tokens. M3 and later GPUs compile the tiles exactly as before; the M2 takes the M1's path on its evidence and was not measured (#1590, fixes #1523). Thanks to @Arminova.

- HIP: the matrix-core prompt tile of gfx11 GPUs no longer spills to scratch inside its K loop: a group's index and bounds are a shift and a mask instead of a run-time division, and a group's first product starts from zero on a uniform branch instead of a select per accumulator, so the tile fits in 248 registers. Every sum is the same in the same order, so replies are token-equal to before. On a Ryzen AI Max+ 395 (gfx1151) with Qwen3.8-27B MLX 4-bit, the seven prompt products at 2,048 rows take 63.7 ms instead of 105.8 ms, and cold prefill is 32% faster at 2k tokens (8.60 to 5.83 s) and 24.5% faster at 64k; decode, Metal and CUDA are unchanged (#1663). Thanks to @AbrahamCete.

- The paged prompt store (`prompt_radix`) can now serve a family whose cached rows read tokens past their page, as GLM's MTP rows do: with a lookahead, a page is matched and shared only while a prompt repeats the tokens past it, and a planned family cuts and resumes only on its grid, a whole number of pages. A store given rules it cannot keep (a zero page, or a planned grid that is no multiple of the page) now fails with `error.BadPromptRadixRules` in every build instead of a debug-only assert. No family turns these rules on yet, so HIP's families and every other path keep, resume and evict exactly as before, which a new golden test pins (#1659, part of #1510). Thanks to @Bizuayeu.

- MOSS-Image: `moss_image.noise` makes the pipeline's seeded starting latents (`torch.randn` on a seeded CPU generator) in the family's host code, byte for byte what x86 torch makes, on any CPU. ARM torch, on a Mac or a GB10, gives slightly different bits: for seed 42 at 1024x1024, 200,873 of 524,288 values differ, by at most 2.2e-6. `zig build test` checks 69 cases, 24 seeds and 9 sizes from 16x16 to 2048x2048, against torch 2.13.0's recorded latents, and `moss_image_case.py noise` records them. `tf-moss-image` still reads its latents from the case file, and nothing else changes (#1668, fixes #1530). Thanks to @BobClawblaw.

- Intel Arc Pro B70 (Xe2): the XPU backend serves Gemma 4 E4B (NVIDIA Nemotron 3.5 Bolt E4B) from `mlx_lm.convert` 4-bit checkpoints, with the per-layer input linears kept in BF16 (a checkpoint that quantizes them still loads, with a warning per tensor) and, optionally, 8-bit attention projections. It keeps prompt states in the prompt cache, runs up to 8 streams with shared forwards, draws the sampling top-k on the card and answers `/v1/decisions`; the audio tower and the drafter are not included yet. On the B70 it reads 2,048-token prompts at 6,772 tok/s and decodes one stream at 135.6 tok/s (3,950 and 118.6 with 8-bit attention), and 4 streams reach 426 tok/s together, each reply equal to the stream's reply alone. Qwen3.8 27B and Nemotron 3.5 Lightning on the XPU backend give the same replies at the same speeds (#1552, for #1222). Thanks to @Mirrdhyn.

- Nemotron on CUDA: the router's top-k runs on the shared MoE top-k, which adds the router's slices itself, in place of Nemotron's own `tf_nemo_topk` and `tf_nemo_topk_wide`, and Bolt's latent-expert sum runs on the shared combine. Picks, weights and replies are the same bytes as before. In a GB10 trace of Lightning decoding, the top-k takes 2.74 µs a call instead of 4.37 (#1354, step 2 of #548). Thanks to @jschmied.

- Kernel contract: Nemotron's pocket kernels on CUDA, the learned neurons Living Weights adds beside the shared expert, are now checked against float64: relu^2 up and the down added into the shared expert's slot, one row a block and tiled, with fp32 and bf16 output, and the unused neurons set to NaN so a read of one shows. On a GB10 all six pass gates 1 to 4 and both planted controls go red (a relu by `fmaxf`, which drops a NaN, and a down launched one neuron short), with launch records for sm_80 to sm_121; `tf-cuda-test pocket-contract` runs the set alone (#1629). Inference kernels are unchanged. Thanks to @BobClawblaw.

- Nemotron on CUDA: squared ReLU in the grouped experts (MLX affine, GGUF and NVFP4), the NVFP4 `ck_experts` path and the pocket now keeps a NaN instead of turning it into 0, and an NVFP4 activation block holding a NaN gets a NaN scale instead of a finite one, so a NaN reaches the output rather than vanishing (#1628, step 1 of #1549). Finite values give the same bits as before: only the squared-ReLU functions' SASS changes, every Qwen and Gemma kernel compiles as before, and on a GB10 replies were token-identical on Nemotron 3.5 Lightning (NVFP4 and MLX 4-bit), Bolt and Super VL, greedy, sampled and with an image. Thanks to @BobClawblaw.

- HIP: on gfx11 GPUs the prompt GEMM's matrix-core tile now runs its blocks in bands of four row blocks, as CUDA's tile order does, so neighbouring row blocks read each column block's weights while they are still in cache. Each tile computes the same sums in the same order, so replies are token-identical, and a new kernel test holds the tile's bytes to the dot2 tile's on the 27B's prompt shapes and the band edges. On a Ryzen AI Max+ 395 (gfx1151) with Qwen3.8-27B, a cold prompt takes 7.60 s instead of 8.64 s at 2k tokens (-12.0%), 31.4 s instead of 35.5 s at 8k, and 330.8 s instead of 365.7 s at 64k (-9.5%); decode is unchanged, and other GPUs never run the new order (#1648). Thanks to @AbrahamCete.

- GLM on CUDA loads faster: 8 threads decode and pack each layer's routed experts instead of the loader's one (`TF_GLM_LOAD_PACK`, 1 to 16), the packed experts go up a piece at a time (at most 192 MiB) from pinned memory the GPU copies from, and 16 readers now read ahead of the load instead of 8. GLM-5.3-Flash XSFP4 loads in about 51 s instead of 155 s on two GB10s (TP=2) and 46 s instead of 124 s on three (TP=3), with the same replies and memory free at serving within about 1 GiB (#1636). Thanks to @Bizuayeu.

- `--vision-urls`: an HTTPS image or video download now ends at its deadline however the server at the other end behaves. A timer closes the connection when the deadline passes, so a server that never answers the TLS handshake, or sends bytes slowly enough to keep each read alive, no longer holds the request past its download limit (10 seconds for an image, 20 for a video); the request gets the usual "failed or timed out" refusal. The TLS setup moves to its own file, `media_fetch_tls.zig`, with host tests for silent and malformed peers and for simultaneous first downloads loading the CA certificates once. Replies, model paths and images that download normally are unchanged (#1645). Thanks to @jayleaton.

- DeepSeek V4.1 on CUDA: with both `TF_DSV41_PREPARED` and `TF_DSV41_SESSION_DISK` set, boot now hashes the pack once instead of twice; the session store reuses the content digest the prepared cache was keyed with, so a large pack is no longer read a second time from disk before the server is ready. The identity is the same value as before, and without a prepared cache the store still reads the pack once (#1644). Thanks to @jayleaton.

- DeepSeek V4.1 on CUDA (experimental) copies its weights and persistent forms to the GPU through one reusable 64 MiB pinned stage instead of a host copy of the whole image, so the public Flash-1017 EXL3 pack no longer stops at boot with `CopySourceLimit` at TP=2; with it, a rank loaded all 95.1 GiB of its weights. The bytes on the GPU, their addresses and their digests are unchanged, a failed later chunk keeps the same cleanup, and uploads under 64 MiB make the same single copy as before (#1643). Thanks to @jayleaton.

- Nemotron 3.5 on Metal: the decode attention kernel on M3-class GPUs and the experts' 2-row tiles unroll their fixed loops, with bit-identical outputs. On an M3 Ultra, Nemotron 3.5 Lightning decodes 1.8% faster at one stream (0.8% at 4 streams), and 0.8% and 1.5% faster after 2k and 8k prompts, with the same tokens (#1621).

- DeepSeek-V4.1-Flash on CUDA (experimental) plans EXL3 packs that keep a BF16 `.weight` beside a projection's EXL3 trellis, such as `jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw`, which used to stop at planning with `UnsupportedDenseProjection` (booting that pack also needs the upload-stage change, #1643): the CUDA rank planner takes the stored trellis, still checked by the shared trellis validator, and still refuses dense-only and malformed projections. Metal and every other family keep reading the dense weight first, and packs without a dense sibling plan as before. Thanks to @jayleaton.

- DeepSeek-V4.1 (experimental, behind `TF_EXPERIMENTAL=deepseek`) admits EXL3 checkpoints whose `quantization_config` declares the dsq-1 format, with or without a producer note in brackets, such as the public Flash-1017 2.8 bpw pack, which main refused before reading any tensor. Only the `inst3`, `mcg` and `mul1` codebooks with `out_scales` `always` are admitted; 1.x metadata is admitted as before, and the loader's geometry, trellis, scale and marker checks still apply (#1641). Thanks to @jayleaton.

- DeepSeek-V4.1 (experimental, CUDA): `zig build dsv41-assets` installs two CPU-only tools for preparing its serving assets without a GPU: `tf-dsv41-token-map`, which writes the Engram token map from the release tokenizer and config (it now accepts the official unquantized config for this, while serving still refuses that checkpoint), and `tf-dsv41-assets`, which runs the engine's own AOT planner (`aot-needs`) and lists the Engram shards from `model.safetensors.index.json` alone (`engram-list`). The steps are in the new DeepSeek-V4.1 CUDA recipe; serving is unchanged (#1640). Thanks to @jayleaton.

- Qwen3.8-27B EXL3 on CUDA: the DFlash2 drafter projects only the two vocabulary spans it proposes from, as column views of the stored head, instead of all 248,320 columns, skipping about 545 MB of reads a round at a 6-bit head. Both views plan the head's matrix multiply as the whole head does, so replies are the same tokens as before (144 of 144 equal on an RTX 3090 Ti with the 3.50 bpw pack), and single-stream decode there was 0.5 to 1.8% faster in the cells whose tokens per round matched. `qwen3_5` on CUDA now lists `exl3` among the formats it reads (#1507, after #1465). Thanks to @diwakergupta.

- Dory on CUDA with `TENSORFOLD_DORY_KV_PAGES=1` refuses to attach a context that is not a whole number of 256-position pages, takes its slot count from the round geometry instead of a fixed 8, and loads the paged attention kernels only when paging is on, so the default path no longer loads that module (#1387, for #1284). Thanks to @jschmied.

- Gemma 4 on CUDA serves structured output: `response_format` (`json_schema`, `json_object`) and `tool_choice: "required"` or a named function now constrain its replies, with or without a drafter (#1386, for #1384). Sampled rows are masked on the GPU before the draw; greedy rows pick the first largest allowed token on the host, the same rule as the GPU pick. Requests without a grammar take the old path unchanged. Thanks to @ThomasHalwax.

- Qwen3.6-35B-A3B and Qwen3.8-27B on CUDA: `TF_QWEN_KV_SHARE=n` (off by default) keeps the prompt cache's KV rows as pool pages on the device, which a later turn's stream adopts, instead of copying them to pinned host memory and back; only the recurrent state, at most `n` snapshots of it, still goes through the host (#1357, for #1156). It needs `TF_QWEN_KV_PAGES` set to a divisor of 4096, runs every stream on pool pages, refuses `--learn-conversations`, and warns at startup when `n` is below 4 times `--parallel`. Replies are the same tokens with it on or off. On a GB10 at `--parallel 8` with Qwen3.6-35B-A3B and MTP, a kept 4,096-token state takes about 68 MiB of pinned host memory instead of 148 MiB, a warm 9k-token turn reaches its first token in the same time (0.21 s), and a cold one in 1.68 s against 1.77 s. With every stream on pages (the share, or `TF_QWEN_KV_PAGES_ALL`), `--parallel N` now serves N streams at once where it served N - 1. Thanks to @jschmied.

- Qwen3.8-27B on Metal: the quantized projection tiles (`simd_qmm`'s MMA and chunk-pair kernels) now unroll their fixed loops, which the compiler had left rolled, and a shared round whose drafts get no rows no longer runs the drafter. Outputs are bit-identical. On an M3 Ultra with the MLX 4-bit checkpoint and DFlash2, decode goes from 80.3 to 91.0 tok/s at one stream (+13.4%), +19.0% at 4 streams, +24.9% at 8 and from 145.7 to 209.4 tok/s at 16 (+43.8%); without a drafter, prompts are 17% faster at 2k and 8k tokens and 4 and 16 streams are 14% faster, with prompt-test tokens equal to before. Pre-M5 lane windows can reach 64 rows with head and context copies (`TF_QWEN27_WINDOW`, `TF_QWEN27_COPIES`), off by default until qualified; unknown `TF_QWEN27_*` startup switches are refused by name, and the startup line names the effective loop and lane width. The kernel contract's measured M1/M2 limit records are corrected for three q4 tiles and GLM's row gather (#1621).


- LFM2.5-24B-A2B on Metal can hand an external drafter the rows of the layers it reads (its taps), copied on the GPU by a new kernel, `lfm2_tap`, that the kernel contract checks bit for bit with two planted controls (#1390, part of #1374). A drafting stream resumes a kept prompt state only when it also holds the drafter's context. Nothing uses the taps yet: without a drafter none are kept and every forward runs as before. Thanks to @BlivionIaG.

- Nemotron on CUDA: the MoE residual's norm, Mamba's group norm, the MTP input norms and the bf16 token rows run on the shared norm and embedding kernels instead of Nemotron's own copies, which are removed. The shared kernels keep Nemotron's summation order, so replies are the same tokens as before, and one fewer CUDA image loads (#1353, step 1 of #548). Thanks to @jschmied.

- Qwen3.8-27B on two Macs, first step: the pair seam's Metal kernels, which exchange whole output rows between the two Macs as bf16 copies, so every sum stays on one Mac and no partial sum crosses the link, plus a one-Mac loopback peer and six kernel-contract controls that each go red as planted (#1432). Nothing serves through it yet, and one Mac's projections keep the same reduction order, so replies are byte-identical. Thanks to @akol1.

- CUDA: with DFlash2 beside the dense Qwen3.8-27B, a context copy now needs 20 matching tokens instead of 5, so short matches no longer take the place of DFlash2's drafts. On an RTX 3090 Ti with the MLX 4-bit 27B, decode after two long prompts is up 12.8% sampled and 18.5% greedy, and up to 6.3% on the code and chat cells; replies are token-identical to before and to `"draft": false`. Without a draft model the 27B keeps 5 and runs as before (#1477). Thanks to @diwakergupta.

- Qwen3.8-27B on CUDA reads turboderp's dense EXL3 checkpoints (`mul1` codebook, such as `turboderp/Qwen3.8-27B-exl3`), which used to be refused at load, and drafts them with DFlash2 as it drafts the MLX checkpoint. The dense MLP runs on the same EXL3 `Dense` projections the MoE checkpoints use, with no kernel changes; MoE EXL3 checkpoints still refuse a drafter, and the MLX and MoE EXL3 paths are unchanged. On a GB10 at 4.00 bpw, drafted replies equal plain ones and decode runs at 45.8 tok/s for chat and 71.9 for code with DFlash2, against 15.3 and 15.5 without it; prompts run at 338 to 359 tok/s from 2k to 64k tokens (#1465). Thanks to @BHCC2025, and to @alexderz for the RTX 3090 run.

- HIP: on RDNA4 GPUs (gfx1201 and gfx1200) short routed prompts take the K-parallel tile further, bounded by the plan's items (now 507) and, new, its pairs per item (3.5). On an R9700, cold prefill on Qwen3.6-35B-A3B is 0.53 to 0.80 times main's time from 33 to 128 tokens, and on Nemotron 3.5 Lightning 0.82 to 0.97 times from 129 to 180 tokens; longer prompts and decode stay within 1% and replies are token-equal. Other GPUs' tables set no pairs bound, so their choices are unchanged; `hip-affine-tune` now also times Nemotron's and Flash Next's routed plans (#1607, fixes #1557). Thanks to @laudney.

- CUDA: tree attention (head dims 128 and 256) and head dim 512 window decode and prompt rows gain kernels that read committed keys from a page table, so a family can keep its full-attention caches in the shared KV page pool (#1618; Gemma 4 and Kolibri-1 adopt them in follow-ups). Nothing launches them yet: the existing kernels' SASS is unchanged on sm_80, 86, 89, 120 and 121, and paged results are byte-equal to the contiguous cache over shuffled pages of 16, 37 and 256 positions. Thanks to @jschmied.

- Nemotron 3.5 on Metal: `TF_NEMOTRON_METAL_KV_PAGES=1` (off by default) keeps each stream's keys and values in pages of the shared KV pool instead of a buffer per layer, with paged versions of Nemotron's six K/V kernels for decode, MTP drafts, trees, GPU rounds and prompts, and admission counted in pages (#1537, step 3 of #1293). Each paged kernel is bit-equal to the one it replaces, and the kernel contract now checks Nemotron's attention kernels against float64, with launch records for the 288 to 512 thread partials. Replies are the same tokens with the setting on or off and as before: on an M3 Pro and an M5 Max, at 1 and 4 streams, prompts up to 20k tokens, greedy and sampled, drafted and plain, and on resumed turns; on the M3 Pro, prompt and decode speed were within 0.3%. A stream still takes pages for its whole context when it opens, so it saves no memory yet. Thanks to @jschmied and @swbiggart.

- Vulkan: the first part of an opt-in Vulkan backend for Linux GPUs, built only with `-Dvulkan=true`: the runtime (loader, device, buffers, fenced submits) and one kernel, the MLX affine 4-bit group-64 projection, registered in the kernel contract with its float64 cases, planted controls and measured per-chip limits. `zig build vulkan-kernel-contract -Dvulkan=true` runs it; the author's runs pass gates 1 to 4 and catch all 8 controls on Apple M1, M1 Max and M2 Max under Mesa Honeykrisp. No model runs on it yet, and default builds are unchanged (#1526, for #1420). Thanks to @joshuaswarren.

- Multirank RoCE (`TF_COMM_BACKEND=roce` at three ranks or more) now connects over a switchless ring: each peer's stripes go over the ports on the /24 it shares, so three GB10s cabled in a ring connect where they used to refuse, and pass `tf-tp-test suite` exactly, with an 8 KiB exchange taking 10.7 µs against NCCL's 26.3 µs. A switch pairs the ports as before, two ranks are unchanged, and a ring of four still refuses and falls back to NCCL (#1516, for #1288). Thanks to @Bizuayeu.

- GLM-5.3-Flash on CUDA at three ranks or more takes `TF_GLM_ROCE_MULTI=1` (off by default; it needs `TF_TP_MASTER` and `TF_TP_KERNEL_IMAGE`), which sends the main stream's small all-gathers over the multirank RoCE wire instead of NCCL. Every rank first checks one gather on the wire against NCCL's, byte for byte, and the wire is used only when all agree; otherwise every rank stays on NCCL. On three GB10s in a ring at TP=3, decode went from 51.2 to 52.6 tok/s (+2.57%) with the drafts and +1.13% without, with the same tokens, but a cold 136k-token prompt is 1.2 to 1.6% slower with the wire open, so NCCL stays the default (#1524, part of #1288). Thanks to @Bizuayeu.

- Gemma 4: Metal and CUDA now prepare images from one shared source file instead of a CUDA copy kept equal by a test (#1589). The patches and image hashes each step produces are unchanged. Thanks to @jschmied.

- Qwen-VL pictures (Qwen3.8-27B, Qwen3.6-35B-A3B and Qwen3.8 Flash Next, on Metal and CUDA) are resized and normalized as the checkpoints' named processor, transformers' `Qwen2VLImageProcessorFast`, does: torchvision's antialiased bicubic and its fused normalization in float32, the same `pixel_values` bits on five test pictures. Pillow's bicubic was used before; on a 941×1672 photo it differed in 0.1% of pixel values, by at most 2 levels, so replies to pictures can differ slightly. Video frames already took these settings, and text-only replies are unchanged (#1586). Thanks to @jschmied.

- LFM2-VL's picture sizing uses the shared `smart_resize` in `core/picture`, as a variant that keeps each side at least one 32-pixel step and has no aspect bound, instead of its own copy (#1593). Every picture from 1 to 4,000 pixels on each side gets the same size as before, so image tokens and replies are unchanged. Thanks to @jschmied.

- Picture preparation for vision towers (smart_resize, pixel bounds, normalization and merge-order patches) moves from the Qwen-VL code to a shared `core/picture/patches.zig`, so other Qwen-VL-style processors such as LFM3-VL's can use it (#1584). Qwen-VL images are prepared to the same bits as before. Two options, both off by default, match transformers' torchvision backend: its antialiased bicubic resize and its fused normalization; together they give the same `pixel_values` bits as transformers 5.19's LFM3-VL processor on five test pictures. Thanks to @jschmied.

- Conversation states saved to disk: a save that fits the staging permit (1 GiB and eight jobs) on its own now waits for a write in flight to finish instead of being skipped, and a block still being written is charged and copied once, however many saves name it (#1548, for #1538). A save larger than the permit is still skipped. So is a waiting save whose state leaves RAM, unless `--spill-gib` copies it then; that copy, like any made at an eviction, is skipped when the permit is full. Thanks to @wojo.

- Qwen3.8-27B on CUDA: ModelOpt checkpoints (NVFP4 and FP8), such as `nvidia/Qwen3.8-27B-NVFP4`, now serve on cards from sm_80 to sm_89, such as a Jetson AGX Orin, where decode stopped at the first projection with `MissingLanePart`. The engine reserves the projections' partial sums for windows under 256 rows (about 21 MB for the 27B, and only on those cards) and reduces them without clusters, as Nemotron's ModelOpt path does; a card older than sm_80 is refused at load with the reason. It has not run on an sm_80 to sm_89 card yet. On sm_90 and newer, including the GB10, nothing changes (#1470, for #1452). Thanks to @javierbits and @valacroix.

- More CUDA kernels take their `ldmatrix` loads and f16 `mma` from one home each: the prompt attention and Qwen 3.5 tensor-core attention take `qmm_frag`'s transposed load, the EXL3 grouped-expert, dense prompt and prompt kernels take its `ldmatrix4` and `ldmatrix2`, and the grouped EXL3 experts take `exl3_decode.cuh`'s f16 `mma16816`. All 1,935 kernel functions in the 14 CUDA images this reaches compile to the same code as before at sm_80, sm_86, sm_89, sm_120 and sm_121 (#1591). Thanks to @jschmied.

- The CUDA training kernels for Living Weights' and 4-bit training's gated attention backward now compile from one shared source, `zig/kernels/cuda/attn_back.cuh`, instead of two copies; both images keep their kernel names, and every function in them compiles to the same code as before at sm_80, 86, 89, 120 and 121 (#1565, for #1409). Thanks to @jschmied.

- The GLM CUDA matrix multiply kernels take their copy, ldmatrix, mma and nibble helpers from the shared `zig/kernels/cuda/common/qmm_frag.cuh`, and their 117-line private copy of it is gone. All 110 functions in the `glm_mm` image compile to the same code as before at sm_80, sm_86, sm_89, sm_120 and sm_121 (#1564). Thanks to @jschmied.

- Kernel contract: the CUDA runner now loads fp8_experts' image, so a `zig build kernel-contract` run on a CUDA machine no longer stops at its first fp8_experts record (about 90 minutes in on a GB10) and makes all 1,325 live-limit checks through to its `SUMMARY` line. A host test now fails if any registered CUDA launch or control names an image the runner cannot load (#1579). Inference kernels are unchanged. Thanks to @Bizuayeu.

- Chat templates that build the prompt by appending to a string with other joins in between, as DeepSeek-V4.1's does, no longer copy the whole prompt at every turn, so memory grows with the prompt's length instead of its square: a 2,000-message chat through DeepSeek-V4.1's template peaks at 15 MB instead of 154 MB (53 MB instead of 152 MB with tools). Every rendered prompt is byte-identical to before (#1633). Thanks to @fche for the report.

- Cortheon: a research planner query longer than 200 bytes is now cut on a character boundary, so a multi-byte character at the cap is left out whole instead of being split into invalid UTF-8 in the search query (#1610). Thanks to @ppardus.

- With `--cortheon`, the task goal that the search planner and web reads see is cut at 4,000 bytes on a character boundary. A character straddling the cut was split, so the planner's prompt was not valid UTF-8; now a character that does not fit is left out whole, as page titles and search queries already were (#1575). Without `--cortheon` nothing changes. Thanks to @ppardus.

- [The Flash Next CUDA recipe](docs/recipes/flash-next-cuda.md#prompt-chunk-size) documents `TF_FLASHNEXT_CHUNK` (1 to 16,384 prompt rows a pass, default 1,536): a larger chunk takes a long prompt in fewer passes, each reading its routed experts' weights once, while a stream decoding during a pass waits up to the whole pass and the prompt scratch grows with the chunk (#1553). Thanks to @jschmied.

- Qwen-VL video, LFM2, GLM5-Next and Step-5 pictures now turn each pixel's value into the vision tower's input through one shared table a channel, in `core/picture/normalize.zig`, instead of each family's own arithmetic (#1592). The tests check every value of every channel against each family's old expression, so the vision inputs are the same bits as before. Thanks to @jschmied.

- Nemotron on M1-family Macs: routed experts in windows of 16 or more rows (shared rounds of concurrent streams, a 16-row prompt chunk, wide copy windows) take two members a pass at 8 rows a simdgroup, through two new instantiations of the existing kernel (#1466). Each output keeps the same summation order, so replies are the same tokens. On an M1 Max the routed-expert kernels take 4 to 5% less time at 16 rows and 14 to 20% less at 64 to 128 rows; served Nemotron 3.5 Lightning shows no regression there, and M3 Ultra and M5 Ultra, which keep their kernels, measured the same as before. Thanks to @mrv777.

- CUDA EXL3: the trellis weight loads in the expert, linear, segmented and unpack kernels skip L1 allocation (`ld.global.nc.L1::no_allocate`), since each word is read once a token; the values read are the same, so replies are unchanged. On three GB10s at TP=3 with GLM-5.3-Flash EXL3 4bpw, decode rose 2.3% on one stream and 0.5 to 2.3% from 2 to 8 streams (#1534). Thanks to @BadAd84.

- Qwen3.8-27B on Metal keeps 16-row windows and lane copies off by default. On pre-M5 chips, `TF_QWEN27_WINDOW=32` or `64` opts into a wider lane window, and `TF_QWEN27_COPIES=1` opts into lane context copies; served qualification remains pending (#1621).

- The native server takes `--decode-share X` (0 to under 1) instead of refusing it: while a long prompt fills beside running replies, the replies get X of each fill step's time, the same time share `TENSORFOLD_PREFILL_SHARE` sets from the prompt's side (#1327, for #840). It works on the CUDA, Metal and HIP lane servers; DeepSeek, Spark-X on Metal, Qwen4 and the XPU servers ignore it. No model's prompt chunks change size: the flag can also cap a filling prompt's chunks at 512 rows, but that stays off for every model until it is shown to give the same tokens at any cut. Without the flag nothing changes. Thanks to @javierbits and @simon-lin88.

- MiniMax-M3.1-Flash on Metal: the new `minimax_m3` family serves MiniMax's 428B-A23B model (`minimax_m3_vl`, its own `config.json`) on one 256 GB M5 Ultra, with tools, structured output, thinking on or off, images, the prompt cache, two streams and DFlash2 or DSpark drafts that keep replies equal to plain decoding. It reads two community checkpoints, lossy conversions of MiniMax's NVFP4 release, labelled with their cost against it (mean KL 0.244 on mix34, 0.295 on mix256k); everything from those weights on is exact, with a bf16 key/value cache (147,456 tokens on mix256k, 32,768 on mix34). Measured on the M5 Ultra with DFlash2, one stream decodes 44-58 tokens a second on mix256k. `MINIMAX_LOAD_LIMIT_GB` sets the memory budget and `MINIMAX_DRAFTER` picks the drafter (#1235, for #1519). Thanks to @BobClawblaw.

- [The runbook](RUNBOOK.md#prompt-prefill-share-and-budget) documents `TENSORFOLD_PREFILL_SHARE` and `TENSORFOLD_PREFILL_BUDGET`: what each accepts, its default, and how it trades a long prompt's finish against decoding streams' next rounds.

- Flash Next on CUDA drafts a shared round's sampled streams in the batched MTP head pass too, each lane's draft drawn on the host under its own keyed rule from its own row of the pass, the draw its lone pick makes, so drafts and replies are unchanged. On a GB10 at 5 x 131,072, sampled code and chat requests decode 24% and 19% faster in aggregate at 5 streams (15% and 19% at 4), greedy cells and time to first token unchanged, every concurrent reply equal to its lone run. `TF_FLASHNEXT_SAMPLED_HEADS=0` restores drafting those streams one at a time. The parity checker covers sampled and mixed shapes. Receipt: `docs/receipts/flashnext-cuda-sampled-heads/`.

- Kernel contract: Qwen4's dense activation check carries FP32 rounding through the stored BF16 SiLU before the up product, avoiding a false shape failure near a BF16 midpoint (#1547). Inference kernels are unchanged.

- HIP: `-Dhip-relax-branches=true` (off by default) lets gfx1100 build the dot2 prompt tiles with Ubuntu's clang 21, which otherwise stops on a branch out of range (#1542, for #1521). It has not run on a gfx1100 GPU yet; with it off, every compile command is the same as before. Thanks to @AbrahamCete.

- Ling 3.1: a host-only family package that the Metal and CUDA engines will share: config parsing and its refusals, the weight map checked against the published index, a float64 reference and small fixtures (#1541, for #1247). Nothing is served yet. Thanks to @ThinkOffApp and @jschmied.

- `tools/zig/cuda_sass_identity.py` builds an image that differs again from both trees, one build at a time. A function counts as changed only if it differs again, and a tree whose two builds disagree fails the check as unstable instead of passing or failing by chance (#1532). `--no-recheck` keeps the old count. Thanks to @jschmied.

- The CUDA kernels' `cp.async` group commit and wait come from one shared header, `zig/kernels/cuda/common/cp_async.cuh`, instead of eleven local copies under five names, in the attention, expert, EXL3 prompt and GLM indexer kernels. All 2,245 kernel functions in the 27 CUDA images it reaches compile to the same code as before at sm_80, sm_86, sm_89, sm_120 and sm_121 (#1588). Thanks to @jschmied.

- More CUDA kernels take their helpers from the shared headers in `zig/kernels/cuda/common`: the warp sum in the Flash Next and Qwen4 gated delta kernels, the FP4 and FP8 pair conversions, and the activations (`sigmoid`, `softplus`, `silu` and its gradient). All 1,866 kernel functions in the 104 CUDA images compile to the same code as before at sm_121 and sm_80 (#1550, #1555, #1556). Thanks to @jschmied.

- After a tool reply, the prompt cache resumes the next turn at the part of the conversation that turn repeats. A chat template that closes a tool reply's turn only when nothing follows (Gemma 4's) used to put the resume point past it, so the next turn resumed one turn late; on an Arc Pro B70 with Gemma 4 E4B, those turns reached their first token in 0.8 to 1.4 s instead of 1.7 to 2.7 s (#1551). Other templates resume exactly as before. Thanks to @Mirrdhyn.

- A conversation state on disk that its model refuses while reading it back is forgotten after one recall, as a file that fails its own checks already is, so later turns stop reading it again (#1543, for #1539). Thanks to @wojo.

- Qwen3.6-35B-A3B on CUDA drafts every stream of a shared round together with its MTP layer: absorb passes, then passes of up to 16 rows a draft level for all streams, each stream's drafts unchanged, and the drafter's ring attention runs ahead of its loads (#1562, for #1560). On one GB10 with a 4.2K-token prompt, aggregate output at 2, 4 and 8 users is 137, 182 and 220 tok/s, against 116, 138 and 159 tok/s with `TF_MTP_BATCH=0` (one stream at a time) on the same build. Thanks to @laudney.

- GLM on CUDA: as the load packs each layer, 8 reader threads read that layer's weights into the page cache ahead of it, so GLM-5.3-Flash XSFP4 loads in about 155 s instead of 327 s on two GB10s (TP=2) and 123 s instead of 268 s on three (TP=3), with the same replies and the same memory free at serving (#1616). A rank now reads its layers whole from the disk, its unkept expert rows included, and still drops each layer's pages once it is packed. `TF_GLM_LOAD_PREFETCH` sets the count (`0` turns the readers off, at most 16); they run only on Linux. Thanks to @Bizuayeu.

## 1.1.9


- Qwen3.8-27B on Metal: `TF_Q27_PROFILE=N` logs each kernel class's GPU time, the DFlash2 drafter's included, every N decode rounds (`1` means every 64), and adds `gpu_seconds` to replies: the GPU time of the forward passes the reply ran in, with a round shared by several replies counted in each (#1215). It is a diagnostic and off by default: without it replies are byte-identical to before, and a value it cannot read leaves it off with a warning. Timing every round is slow: on an M5 Ultra one stream decodes at about a ninth of its usual rate with it on, so use it to diagnose, not to serve. Thanks to @grantoverton.

- Qwen3.8 Flash Next on Metal: when a reply repeats text from its prompt or from earlier in the reply (edits, rewrites, renames, quotes), its copied drafts now verify in windows as wide as their measured acceptance pays for, about 11 rows where they stopped at about 4, and a copy skips the MTP head's later draft steps it replaces. Requests whose prompt plus max_tokens passes about 2,000 tokens, which includes every request at the server's default max_tokens of 4,096, now get the GPU gate that sets each round's width from the head's confidence, as shorter requests already did. On an M3 Ultra, copy-heavy edits ran 212 to 257 tok/s and rewrites 222 to 300; at the default max_tokens, prose ran 138 to 142 and long documents 173 to 179. Every reply is the same tokens as before, drafted or not.

- Qwen3.8 Flash Next on Metal, sampled replies: the checkpoint samples by default (temperature 1.0, top_k 20, top_p 0.95), and a sampled reply now gets the GPU gate, MTP drafts drawn under the reply's own keyed rule (as shared rounds' drafts already were), and a faster draw. For top_k from 1 to 64 the GPU keeps each of 32 slices' top_k, then merges and races the row's top_k in a second pass: about 29 µs where the whole-vocabulary draw took about 700 µs, with the same sums, cuts and tie rule, so it picks the same token at every seed and position. On an M3 Ultra, sampled prose ran 122.5 to 136.0 tok/s, code 180 to 194 and hard prompts 133 to 146, each reply the same tokens as before; greedy replies are unchanged.

- Nemotron 3.5 on Metal: a GPU round prices its draft head's depth from the backend's window-cost table, where it used to choose between all of the head's levels and a fixed 4, and it no longer steps back up to a level that has shown it doesn't land (#1115). Replies are the same tokens. With Nemotron 3.5 Lightning 30B-A3B (MLX 4-bit), decode after 2k to 64k-token prompts is 1.2 to 2.4% faster on an M3 Ultra and within 0.3% of before on an M5 Ultra. Thanks to @jishnuvenugopal.

- Qwen3.8 Flash Next on CUDA with NVFP4 experts: at decode, the routed experts' work goes out in equal waves instead of leaving a short last wave, and when every half unit fits the GPU at once, two warps share a unit (#1351). Each output keeps the same multiply-adds in the same order, so replies are the same tokens, and the expert kernels Nemotron and GLM use compile to the same code as before. On a GB10 with Qwen3.8-Flash-Next FP4, served decode is 0.6% faster at one stream and 0.2% at two. Thanks to @jschmied.

- Qwen3.8-27B on CUDA: prompts on the default path (without `TF_QWEN_KV_PAGES`) take whole-window attention kernels without page-table arguments again, as before #1373, which wins back the prefill that change cost at long prompts (#1458). Replies are the same tokens. On a GB10, prefill is 0.5% faster at 32k tokens and 0.8% at 64k.

- Tests: `kernel-contract-limits` no longer reports Qwen's prompt partials as launched past their bounds on every SM. Two stale 1,024-thread entries sat beside their 128-thread records.

- Release builds: the release path check no longer refuses a clean binary whose address tables happen to hold the bytes `5c 2e 6f 00`, which read as `\.o` (#1533). A path's extension now needs a file name before it; every other rule is unchanged.

- Qwen3.5-2B on Metal: with `--prompt-cache-gib N`, a later request that extends a conversation resumes from its deepest kept prompt state, held in RAM (recurrence, convolution tails, attention K/V and logits), instead of prefilling it again (#505). Without the flag nothing changes: the same tokens and speed as before. On an M5 Ultra, the next turn of a 28.5k-token chat resumed with 28,544 tokens cached, and on an M5 Ultra and an M3 Ultra every resumed reply equalled a fresh one. Thanks to @akol1.

- Qwen3.8-27B on Metal requests 2,048 prompt rows on M5 when no other request is decoding, and caps each chunk at 512 beside a decoder (#1478). The larger frame is used only when the configured context, every stream and full prompt scratch fit beside the weights and the existing 8 GiB margin; otherwise M5 also keeps 512 rows. Startup now reserves those scratch allocations. Earlier chips keep the 512-row default. `TF_PROMPT_ROWS` overrides the requested width within the existing frame and attention limits.

- Qwen3.8-27B on CUDA: a prompt runs up to four of its projections in one launch, each keeping the shipped BF16 kernel's own block order, and takes a 2x2-warp tile at the measured large-prompt widths (#1439, for #1398). Outputs are the same bytes. On a GB10 with the MLX 4-bit checkpoint, prefill is 10.8% faster at 2k tokens, 9.7% at 8k and 7.8% at 32k. Thanks to @yguanrong.

- GLM-5.3-Flash on CUDA: at long positions a decode step's DSA select gives each row a 1,024-thread block where it used 128 threads, for up to 48 rows over buckets of 4,096 pools or more, with the same picks in the same order (#1424). On three GB10s at TP=3, decode after a 136k-token prompt ran 1.8% faster; on two GB10s at TP=2 with a 65k window, tokens match main and speed is unchanged. Thanks to @Bizuayeu.

- GLM-5.3-Flash on CUDA with two or more ranks splits a prompt chunk's DSA indexer across the ranks: from position 2,048 on, each rank scores and picks the pools for its share of the chunk's rows, and an all-gather gives every rank every row's picks (#1404). A row's picks come from the same kernels on the same inputs whichever rank computes them, so replies are the same tokens, and decode never splits. `TF_GLM_INDEX_SPLIT` sets the position it starts from, and 0 turns it off. On two GB10s at TP=2, prefill at 128k tokens was 1 to 2% faster, with the same tokens at every length. Thanks to @Bizuayeu.

- With `TENSORFOLD_PREFILL_SHARE` below 1, GLM-5.3-Flash on CUDA ends a long prompt's step on the GPU before it returns when the lane host bills that step to decoding streams (#1352). The step used to return with part of its GPU work still queued, so that tail ran in the decoders' next round and counted against their share. At the default of 1, or with nothing decoding, no step waits and nothing changes; replies are the same tokens. Thanks to @Bizuayeu.

- The serving line names what can propose drafts, for example `drafts: copies only`, `drafts: MTP + copies` or `drafts: DFlash2 (z-lab/Qwen3.8-27B-DFlash2) + copies`, where it used to say `drafts: on` whether or not a drafter was loaded (#1515). `drafts: off` is unchanged, and drafting itself doesn't change. Thanks to @Deesha08.

- HIP: AMD RDNA4 cards (gfx1201) take a tuning table of their own, which keeps gfx1100's bounds for now, so no kernel choice changes; `zig build hip-affine-tune -Dhip=true` times the MLX affine prompt tiles on a card to fill it (#1527). Thanks to @laudney.

- CUDA kernels take their shared helpers from one set of headers; every kernel compiles to the same code as before (#1440). Thanks to @jschmied.

- Qwen3.8-27B on Metal: a follow-up turn resumes from the state its conversation's last reply left in the server's first slot when the prompt cache holds nothing as far, so on a Mac whose cache cannot keep a long conversation's state (a 32 GB Mac) the next turn reads only its new tokens instead of its whole prompt (#1197). Where the cache cannot keep a whole context, a turn whose template drops the last reply's reasoning goes back to that slot's state at the last prompt's history end, which the server keeps in one extra buffer. The first prompt chunk stays in the ordinary cache even when its background warm pass steps aside, so an interleaved conversation can still resume that checkpoint. Replies are the same tokens as a fresh prompt pass, and `TF_QWEN27_SLOT_RESUME=0` turns it off. Thanks to @RS91Dev.

- `--compact-at auto` on a context window smaller than its 16,384-token reserve reserves fifteen percent of the window instead, so a short request on `--context 8192` is no longer compacted and the memory note's budget fits inside the window (#1536). Larger windows keep the 16,384-token reserve. Thanks to @ppardus.

## 1.1.8

- Qwen3.5 and 3.6 on HIP (Linux): when a checkpoint shard can't be memory-mapped and is read into memory instead (for example on some network or FUSE mounts), releasing an uploaded layer no longer zeroes that copy. Before, it also zeroed neighbouring tensors not yet uploaded: 269 of 690 layer tensors on Qwen3.5-2B, so the model ran on wrong weights or the loader panicked at close. A malformed shard header is now refused by name. Loads that map their shards are byte-identical to before.

- Nemotron on CUDA counts the checkpoint's bytes for admission from the shard headers alone, without reading the weights. The estimate is unchanged, and with memory mapping off the count no longer reads each shard into memory.

- Tests: the CUDA kernel contract runs to completion in a ReleaseSafe build. Three test adapters did things a safe build stops on: freeing or copying memory wrongly, and walking two operands of different lengths together (#1428). Thanks to @Bizuayeu.

- The KV pool keeps its counts of free, kept and leased pages up to date as pages change hands, so an admission check no longer scans every page: about 55 µs at 262k pages before, now constant. Admission decisions and page order are unchanged, and on a GB10 with Nemotron's paged CUDA cache replies are the same tokens.

- API keys can be limited to some routes: in the `--api-key-file`, `app[inference]: KEY` reaches only model requests, `grafana[read]: KEY` only listings and metrics, and `ops[admin]: KEY` everything. A scoped key is refused with 403 before its body is read. Keys without brackets, from `--api-key` or `TENSORFOLD_API_KEY`, keep every route as before, and a file that gives one key two different scope sets is rejected. See [key scopes](docs/api.md#key-scopes).

- Opt-in admission limits for a shared server: `--max-inflight N` and `--max-inflight-per-key N` cap model requests in progress, overall or per key, and `--requests-per-second N` with `--request-burst N` limits how fast a key may start them. A refused request answers 429 with `Retry-After`. All four are off by default, and with none set every reply is byte-identical to before. See [admission](docs/api.md#admission).

- Loading an MLX affine checkpoint checks each projection's geometry before it is read, cut or uploaded (bits, group size, packed words, scales and biases shapes, extents), and refuses a malformed one by name. On HIP, a MoE checkpoint whose shared expert stores its tables in another dtype than the routed experts now converts each from its own dtype, where it used to read the wrong bytes or overrun its buffer. Every published checkpoint we load gives the same bytes as before.

- A streamed chat reply with tools now carries the same content as the whole reply: an unoffered or malformed tool call the whole reply keeps as text reaches the stream too, including around accepted calls and before later prose (#1365, #1429). A reply whose answer starts with whitespace sends its text when the reply ends, because only the final parse decides whether that whitespace stays. Inference tokens are unchanged. Thanks to @plotarmordev.

- On Linux, `/metrics` reports the server's resident memory: `tensorfold:process_resident_bytes` and its anonymous, file and shared parts, plus the kernel's lifetime peak, read from `/proc/self/status` at scrape time (#1422). On a GB10 the CPU and GPU share one memory pool, so host memory a rank adds counts against the same limit as the weights. Thanks to @nuk3s.

- Compaction reads a message's text parts as the chat template does, so a turn sent as an array of text parts is no longer treated as empty when the memory note and the kept span are chosen (#1406). Replies are unchanged. Thanks to @ppardus.

- Tests: the LoRA symlink cases skip on Windows when the account may not create symlinks (#1405). Thanks to @BobClawblaw.

- Speech: the speech lane is released as soon as the engine finishes a request, and before any error reply, so a client that sends one request after another no longer gets 503 "speech output capacity is full" for a reply it has already received in full (#1438). Thanks to @plotarmordev.

- CUDA servers with two or more ranks log, at startup, each GPU's peer-to-peer read and write status and PCIe path to the others, the driver, CUDA and NCCL versions, the P2P-related NCCL and CUDA settings in effect, and the transport NCCL chose. Nothing changes peer access or NCCL settings, nothing here can fail startup, and replies are unchanged; `TENSORFOLD_P2P_DIAGNOSTICS=0` turns the report off and `TENSORFOLD_P2P_PROBE` times a small all-gather.

- CI builds the release server and runs the release path check on it: for every pull request on Linux x86_64, and for every push to main on each platform, so a private path in a built binary fails CI instead of the release (#1472).

- With `--cortheon`, a turn sent as an array of parts (a vision turn) gives Cortheon its text parts joined, as the chat template reads them, so its checks see the turn's words and two different vision turns no longer share a key (#1469). Without `--cortheon` nothing changes. Thanks to @ppardus.

- Structured output: with `"grammar_budget": true` beside a grammar or a required tool call, each token is drawn so the reply can still finish the grammar within `max_tokens`, so a tool call is no longer cut off unparsed (#1473). It is off by default, where nothing changes; drafted, plain and concurrent replies stay equal with it on, and a grammar it cannot follow (token-level, budgeted, captured or left-recursive rules) is refused with HTTP 400. The first request with a new grammar explores it once, about 60 ms for one tool and 640 ms for twelve.

- `--loop-guard-stop` (opt-in, implies `--loop-guard`): when the reasoning-loop guard fires, the reply ends there instead of closing the reasoning and answering, so an agent no longer gets tool calls that repeat finished work (#1463). The reply keeps its reasoning and `tensorfold.loop` with `"stopped": true`, carries `finish_reason: "content_filter"` and an error with code `reasoning_loop` (`/v1/messages`: `stop_reason: "refusal"`), and the stream's lane is released at once. Without the flag nothing changes. Thanks to @kesslerio for the report.

- The prompt cache finds a conversation's kept state through an index kept longest first, instead of comparing the prompt with every kept state, so a conversation that keeps a state each turn no longer pays a full prefix compare per kept state on every lookup. Every resume, its position, ties and evictions are the same as before; on an M5 Ultra the Qwen3.8-27B cells give the same tokens at the same speed.

- On macOS, engines served through the shared lane host (Flash Next with `--parallel` above one, Qwen3.8-27B and GLM) watch the system's memory pressure (#1454); see [memory pressure](RUNBOOK.md#memory-pressure-on-a-shared-mac). At warning, a queued request starts only when no other stream runs and the prompt cache lets kept states go in its usual eviction order, one a round, down to half its budget (`TENSORFOLD_PRESSURE_CACHE_PERCENT`), writing each one to disk first with `--spill-gib`; at critical, new requests are refused with 503 and a message naming the pressure, while queued and running requests continue. Replies are the same tokens either way, and at normal pressure nothing changes; on an M5 Ultra the Qwen3.8-27B cells give the same tokens at the same speed. `TENSORFOLD_MEMORY_PRESSURE=0` turns it off.

- Flash Next on CUDA with NVFP4 experts loads them on several threads straight from the checkpoint's mapped planes, staged through the store's pinned buffer, so a server starts in about 83 s instead of about 200 s on a GB10 with `Qwen3.8-Flash-Next-FP4-Local` (#1368). Replies are the same tokens, and prompt and decode speed are unchanged or slightly faster. Thanks to @Arminova.

- GLM-5.3-Flash on CUDA can score a prompt chunk's DSA indexer in two kernels of its own with `TF_GLM_INDEX_CUDA=1` (opt-in; #1449, for #1430). They read the same MMA fragments and epilogue as the kernel they replace and give the same bytes: on a GB10 the kernel contract finds them byte-equal and stable across repeats, and replies are the same tokens. The first token of a cold prompt came 2% sooner at 131k tokens, 9% at 524k and 16% at 1M. Without the variable nothing changes. Thanks to @wojo.

- With `--spill-gib`, a state the prompt cache pushes out is written to disk by one I/O worker from its own staged copy, instead of on the engine thread, so decoding streams no longer pause for the write (#1453). A request waiting in the queue has its conversation's state read ahead from disk on another I/O thread while other streams decode, waiting for that state's write if it is still in flight (#1464; `TENSORFOLD_RESTORE_PREFETCH=0` turns the read-ahead off). On an M5 Ultra with Qwen3.8-27B, four streams of ~14k-token conversations through two RAM slots, returning conversations came back in 0.6-1.0 s instead of up to 21 s, the returning wave decoded at 149 tok/s instead of 61, and the longest pause between tokens fell from 499 to 417 ms; replies are the same tokens.

- With `--spill-gib`, conversations held only in RAM are written to disk when the server unloads for idleness, so the next turn after an idle unload resumes instead of starting over (#1471). On an M5 Ultra a 32k-token Nemotron conversation woke in 1.3 s instead of 4.9 s, with the same reply, and shutdown stays under half a second.

- Opt-in: `TENSORFOLD_PREFILL_BUDGET=N` bounds how many short-prompt rows go in beside decoding streams before their next round. Short prompts (up to 512 uncached rows) normally go in back to back, so a burst that arrives while streams decode pauses them for up to a lane count of 512-row prompts; with a budget, a fill visit ends once N rows have been filled while a stream could decode, and the decoders' round goes first. Prompts admitted together while nothing decodes still go in together, and a long prompt that has waited its turn is not starved by short arrivals. In the host tests, a decoder beside 24 queued 512-row prompts waits on at most four 128-row calls with a 512-row budget instead of sixteen. Unset or 0, the fill order is unchanged. On an M5 Ultra with Qwen3.8-27B, replies are the same tokens with the budget on or off, and throughput at 1 to 8 streams is unchanged.

- Tools: `tools/bench_cache.py` takes `--api-key-env NAME`, as the other benchmark clients do, so its probes can reach a server started with `--api-key`. The key is read from that variable alone and is never printed or written into a report; without the flag the probes send no key, as before (#1483). Thanks to @Maxrio9.

- LFM2.5-24B-A2B on Metal decodes several streams in shared rounds: each layer's short convolution and attention run once over all of a round's streams, in kernels bit-equal to the per-stream ones, and a verify's greedy rows take the GPU's pick of their logits (#1377). On an M5 Ultra with the 4-bit export, aggregate decode rose 20% at 4 streams, 34% at 8 and 31% at 16, and one stream about 2%; every reply is the same tokens as before, alone or beside others, and the exactness checks pass on the 4-, 6- and 8-bit exports. [The recipe](docs/recipes/lfm2.5-24b-a2b.md) has the author's own receipt, measured against an earlier main. Thanks to @BlivionIaG.

- Kernel contract: Nemotron's routed-expert Metal pipelines are checked from one row to 128 on the engine's route tables: the default window kernels against float64, and the rows2, rows4 and geometry pipelines bit-equal to them, with two planted controls (#1494). One known finding is recorded: a NaN dot in `tf_xup_w3` comes out as 0, as Metal's `max` returns the other operand.

- Tests: Gemma 4's E-series per-layer inputs have a host reference that equals MLX 0.32.4 bit for bit at one row, checked against MLX's own SHA-256 prints at the E4B's shapes (an M4 Max and an M5 Ultra give the same prints) (#1402). The E-series stays refused, and nothing served changes. Thanks to @Mirrdhyn.

- MOSS-Image-Base's 5B transformer runs on Metal as a test tool (`tf-moss-image`), from the released bf16 checkpoint as published, on cases `tools/zig/moss_image_case.py` makes with the publisher's own pipeline, which also encodes the prompt and decodes the image (#1457). It needs the M5's tensor units and refuses other GPUs, and nothing is served yet. On an M5 Ultra, a 1024x1024 case's worst step has a cosine of 0.9997982 against the publisher's torch bf16 pipeline, the same step run twice gives the same bytes, its 50 steps take 47.3 s, and the decoded picture is 31.9 dB PSNR from torch's. Its eleven kernels pass the kernel contract, with eight planted controls red. See [the recipe](docs/recipes/moss-image.md). Thanks to @BobClawblaw.
- GLM-5.3-Flash on CUDA: the DFlash2 drafter's loader sizes its 4-bit scratch for the largest linear at the server's rank count. It held only the two-rank case's, so on one rank or three or more, loading the drafter at 4 bits (the default) wrote about 9.4 MB past its end (#1496). At two ranks it allocates the same bytes as before. Thanks to @BadAd84.
- Qwen3.6-35B-A3B on CUDA with `TF_QWEN_KV_PAGES` keeps each stream's MTP draft ring and state in that stream's pool pages instead of a fixed allocation per stream. A prompt filled in steps now opens its drafter after the target's first step, so MTP and DFlash2 drafts use the stream's pool lane, where before every stream past the first allocated a ring of its own (#1356). On a GB10 with Qwen3.8-27B served as by default, replies are the same tokens at the same speed. Thanks to @jschmied.

- Opt-in, per CUDA family: the prompt cache can keep the state where a new prompt parts from one it has already kept, so when a long document is sent again with a different question, the third and later questions resume after the document instead of prefilling it again (#1291). No family has it on yet (each turns it on after its own device check), so kept states, cached-token counts and replies are unchanged. Thanks to @paragontasx.

- `--learn-conversations` keeps Qwen3.8-27B's and Qwen3.6-35B-A3B's conversation states on disk on CUDA too, as it does Nemotron's, so a turn after a restart reads its state back where that is quicker than prefilling (#1358). Before, CUDA refused the flag for these models. Rows two states share are stored once, every piece read is checked against its hash, and `--learn-gib`, `--learn-daily-gib` and the free-disk floor apply. A state whose prompt holds images stays in RAM. Thanks to @jschmied and @gilby.

- `--learn-conversations` keeps Flash Next's conversation states on disk on CUDA too, so a turn after a restart reads its state back where that is quicker than prefilling (#1359). A state on disk holds everything its resume depends on: the attention and indexer rows, the DeltaNet states, the PLE tail, the n-gram history and, when the server drafts, the MTP head's rows; a state kept from a stream that did not draft stays in RAM. The flag is refused with `--tp`, since each rank keeps its own states. Thanks to @jschmied and @gilby.

- CUDA serves `--spill-gib` and `--checkpoint-slots` for Nemotron, Qwen3.8-27B, Qwen3.6-35B-A3B and Flash Next, as a Mac does (#1413): a state the prompt cache pushes out of RAM is staged on the engine thread, written to disk by an I/O thread, and read back when its conversation returns, after a restart too. `--checkpoint-slots N` sets how many states stay in RAM beside the store and needs `--spill-gib` or `--learn-conversations`. Other CUDA families refuse `--spill-gib` at startup, and without either flag nothing changes. See [conversations on disk](docs/api.md#conversations-on-disk). Thanks to @jschmied and @gilby.

## 1.1.7

- Flash Next on a Mac can leave its routed experts and n-gram tables in the checkpoint and read them from SSD with `--ssd-experts GIB` and `--ple-on-ssd`, so a checkpoint larger than the Mac's memory can be served (#1039, see [Exact Fluid Subgraph Residency](docs/recipes/fluid-subgraph-residency.md)). Neither is on by default. Replies are the same tokens as with everything in memory; a budget too small for one layer's experts refuses at startup instead of changing the model's settings. On an M5 Ultra the 6-bit MTP checkpoint, which does not fit in memory, serves with a 16 GiB expert budget; on an M3 Ultra with the 4-bit checkpoint, servers without the flags are unchanged in prefill and decode.

- Flash Next on Macs with `--parallel` above one take `--spill-gib` and `--checkpoint-slots` too (#1349): shared sessions write a state the prompt cache pushes out to disk and read it back when the conversation returns. `--learn` and `--learn-conversations` are still refused there. On an M3 Ultra with the 4-bit MTP checkpoint and 4 sessions, a returning 16k-token conversation that was pushed to disk starts its next turn after 0.9 s instead of 11.9 s for a fresh pass, with the same reply; decode without the flags is unchanged. Thanks to @gilby.

- The NVFP4 prompt GEMM gains a persistent 128x256 tile with a decoupled copy ring (#1350). It does the same arithmetic in the same order, so prompt tokens are unchanged. On a GB10, Qwen3.8-27B NVFP4 prefill rises 4.2% to 6.3% from 2k to 64k tokens; Nemotron and Flash Next NVFP4 are unchanged. Thanks to @jschmied.

- Qwen3.8-27B on CUDA with `TF_QWEN_KV_PAGES`: a stream on pages takes the same tensor-core attention as a whole window, so its reply beside other paged streams equals its reply alone (#1373; before, 6 to 8 of 12 replies differed on an RTX 5090). On a GB10, prefill and decode are unchanged. Thanks to @BobClawblaw.

- With `--prompt-precision int8` on CUDA, a prompt chunk that falls back to bf16 is counted and logged once, so a slow prefill can be told from a fallback (#1371). Thanks to @BobClawblaw.

- Gemma 4 E4B (preparation): the family config admits the E4B geometry and each backend refuses it by name until it computes it, and the tensor plan names the E4B's per-layer inputs, global value projections and shared layers, with mlx-lm's BF16 and 8-bit per-module overrides (#1400, #1401, part of #1222). Existing Gemma 4 models are unchanged. Thanks to @Mirrdhyn.

- Lanes: two opt-in hooks for the LFM2 drafter work (#1378, #1380, part of #1374): a target that reads held drafts on the device gets them without a host read-back, and a family may price copy windows from each stream's own copy history. Nothing sets either yet, so every model is unchanged. Thanks to @BlivionIaG.

- The kernel contract records nvcc 13.3's register and shared-memory limits for the 206 launches it compiles differently (#1397). Thanks to @BobClawblaw.

- A streamed tool call cut by `max_tokens` while an XML value's opening newline was still held no longer sends that framing newline as an extra `\n` in the argument; the stream now matches the whole reply (#1408, item 3 of #1396). Thanks to @Maxrio9.

- Qwen3.8-27B and Qwen3.5-MoE on CUDA apply `presence_penalty` and `frequency_penalty` (-2 to 2; 400 outside), which the server used to ignore silently, as the Qwen3.8 model card's thinking-off settings use (#1346). A drafted row is penalized exactly as the plain step at its position, so drafted replies equal undrafted ones; engines that don't apply them say so once in the log. On a GB10 with Qwen3.8-27B, requests without penalties give the same tokens as before, and prefill and decode are unchanged. Thanks to @BobClawblaw.

- A request's grammar compiles once: the lane thread builds its matcher from the HTTP thread's compiled handle instead of compiling the same text again under the compiler lock (#1362). Replies are unchanged; on M3 Ultras with Qwen3.8-27B, decode at 1 to 8 streams is unchanged. Thanks to @jschmied.

- The prompt cache keeps, for each entry, links to the entries that extend it, so choosing an eviction victim no longer compares every pair of entries (#1363). Choices are the same (a randomized test checks them against the old scans); in a 220-turn forked conversation with 200 slots, choosing a victim takes about 0.5 ms instead of 20 ms. Thanks to @jschmied.

- Flash Next CUDA NVFP4 next-chunk file hints are opt-in with `TF_FLASHNEXT_WARM_AHEAD=1`; unset or `0` leaves them off (#1292).

- Flash Next on Macs: concurrent chats share each round's work more fully (#1338, #1339, #1340, #1341, #1342, #1343). A shared round samples every sampled stream's draft and target rows in one dispatch, drafts every stream in one packed MTP pass, copies kept states in the verify command buffer, runs the sessions' recurrences in one verify without barriers between them, and prices a batched head's extra drafts as the deepest stream's steps. On an M3 Ultra with the 4-bit MTP checkpoint, decode across 2, 4 and 8 concurrent streams rises 6.9%, 8.8% and 11.8% (4.5% and 10.7% at 2 and 4 streams on an M5 Ultra); one stream and prefill are unchanged, and prompt tokens equal the previous build's. `FZ_VERIFY_OVERLAP=0`, `FZ_PRICE_BATCHED=0` and `FZ_KEYED_DRAFTS=0` restore the previous paths. Thanks to @gilby.

- Qwen4 dense (preparation, ahead of a released checkpoint): the native binary admits `qwen4` and `qwen4_text` configs on Metal and CUDA, sharing Flash Next's layer graph with a dense feed-forward, and refuses any geometry its kernels do not cover before allocating (#1244). Weights stay as stored (BF16, MLX affine, ModelOpt FP8 or NVFP4). Synthetic device gates (`zig build test-qwen4-metal`, `tf-cuda-test qwen4`) load nonzero checkpoints through the serving engine and match a binary64 reference with no tolerance; no real checkpoint, speed or memory fit is claimed yet. Existing models are unchanged.

## 1.1.6

- DeepSeek-V4.1 (experimental): a tool call in the model's spaced DSML spelling now comes back in `tool_calls`, streamed and whole, instead of in the reply's text, and a lone DSML opener inside a `<tool_call>` no longer crashes the parser; the `TF_DSV41_ASSETS` refusal names the file boot actually opens (`engram-token-map.bin`) (#1324).

- Host sampling with no rank cut (`top_k: 0`, `top_p: 1`) finds the sampled token by a keyed race over the candidates instead of sorting the whole vocabulary (#1337). The draw is the same: a host test compares it with the full sort over 483,840 sampler settings, ties and extreme values included, and on M3 Ultras with Qwen3.8-27B and Nemotron sampled replies equal the previous build's token for token. Thanks to @ajensenwaud.

- Dory on CUDA can keep its full-attention caches in pages of the core pool with `TENSORFOLD_DORY_KV_PAGES=1` (#1284); off by default. On a GB10 with Dory 1.0 4B NVFP4, pages give the same tokens as the contiguous cache, the shared window kernels compile to identical machine code, and the default path is unchanged. Thanks to @jschmied.

- GLM-5.3-Flash images: a chat's later turns no longer run the vision tower again for images the kept prompt state already holds (Metal), and the engine reads the vision tower of MLX conversions that name it `vision_tower.` with channels-last convolution kernels, which it used to refuse with `NoVisionTower`. On an M3 Ultra with the oQ4 checkpoint, a follow-up image turn's first token comes about 90 ms sooner (0.13 s instead of 0.22 s on the second turn), and every reply equals a fresh server's. `tools/image_cache_turns.py` measures multi-turn image chats' cached tokens, first-token times and replies against a served model.

- Drama speech: `POST /v1/audio/speech` takes per-request `temperature`, `top_p`, `top_k` (1 to 64) and an exact 64-bit `seed`, whole and streamed (#1277, fixing #1273), and accepts dialogue with speaker tags `<|speaker:0|>` through `<|speaker:15|>` (#1278, fixing #1274). A request without them sounds exactly as before: on an M3 Ultra with Drama 3, default requests give the same PCM as the previous build byte for byte, repeats and concurrent requests included. Thanks to @krunkosaurus.

- On Macs, `--spill-gib N` writes a conversation state that the prompt cache pushes out of memory to disk (under `~/.cache/tensorfold/spilled`, at most N GiB, least recently used first), and reads it back when the conversation returns; `--checkpoint-slots N` bounds how many states stay in memory (#840, from Python #68). Qwen3.8-27B, Nemotron, Flash Next and GLM-5.3-Flash take both; neither is on by default. With one slot in memory and a second 16k-token conversation in between, a returning conversation's next turn starts after 2.6 s instead of 56 s with Qwen3.8-27B on an M3 Ultra (1.8 s instead of 31 s with GLM-5.3-Flash), with the same reply as a fresh pass. Thanks to @gilby.

- Requests that arrive together on a new shared prefix no longer each prefill it when prompts fill a chunk a step: a request whose prefix another prompt is still filling waits in the queue for that prompt's cut, then resumes it (#1321, fixing #1310). `TENSORFOLD_PREFIX_WAIT_MS` bounds the wait (60000 by default, 0 turns it off). On a GB10 with eight chat requests arriving together on a new 28k-token system prompt, the last first token comes after 31 s instead of 201 s with Qwen3.8-27B, and after 25 s instead of 146 s with Flash Next FP4. Prefill and decode speeds are unchanged on the GB10 and on an M3 Ultra, and every reply equals its run alone. Thanks to @wojo.

- DeepSeek-V4.1 (experimental native engine): `zig build dsv41-serve` installs runnable `tensorfold-dsv41` and `tf-dsv41` binaries on real CUDA images (#1325), and packs that store the router bias or the DSpark confidence head in F16 load them widened exactly to F32 (#1326). Thanks to @ecohash-co.

- CUDA on unified memory (GB10): the start-up budget counts the free pages the kernel parks on its per-CPU lists, which MemAvailable leaves out, so a start sized right after another process freed memory admits what actually fits (#1323, fixing #1311). Measured on a GB10 right after a 30B model freed its memory: about 690 MiB parked, still about 180 MiB 30 s later. Thanks to @wojo.

- GLM-5.3-Flash on CUDA takes KV pages from the core pool by default from two ranks, not only from three, and the plan funds the window (#1322, fixing #1313). At TP=2 on two GB10s with the NVFP4 checkpoint, tokens are identical; two streams serve 14% more tokens/s for chat and code, and prefill is flat to +17%. Thanks to @wojo.

- Spark-X2.5/2.6-4B also runs on Metal, from the bf16 release as published (#1320). On an M3 Ultra it prefills 2,840 tokens/s at 2k and decodes 69 tokens/s on one stream; repeats and concurrent streams equal their runs alone, and its tokens match the CUDA engine's on the same prompt. Thanks to @BobClawblaw.

- GLM-5.3: a tool call that `max_tokens` cuts comes back in `tool_calls` with `finish_reason: "length"`, streamed and whole, instead of leaking its `<tool_call>` markup into the reply's text (#1298; thanks @Boscoeuk for the report).

- `tensorfold plan MODEL [--memory-gb GIB] [--ram GB]...` says, before any download or weight load, the memory CUDA admission would grant and whether the checkpoint's weights fit each budget, and which setting would make them fit (#1319, from Python #281; thanks @akol1).

- `--name-priority ID=background` gives a served name or alias a background default: a request naming it without its own `priority` yields to foreground work (#1317, from Python #445).

- Qwen3.8-27B on Metal fills prompts a chunk at a time with the other streams' rounds between chunks, so streams already decoding keep going while a long prompt fills (#1303), and kept states reuse spare buffers with their save blits off the engine thread (#1304). On an M3 Ultra, with an 8.7k-token prompt arriving while two streams decode, the decoding streams' longest pause goes from 28.6 s to 0.47 s; the long prompt's first token comes 7.9% later (28.6 s to 30.9 s), and every reply is unchanged.

- Gemma 4 on CUDA with a drafter keeps its prompt-cache mark after the drafter absorbs a stepped prompt chunk, so a chat's next turn resumes instead of prefilling again: with Gemma 4 31B and its DFlash drafter on a GB10, a 30k-token chat's next turn starts after 3.6 s instead of 45 s, with the same reply (#1286).

- New family: Spark-X2.5/2.6-4B on CUDA (`spark2_5`), from MLX affine checkpoints converted by `tools/zig/spark25_convert.py`, on the shared CUDA operators with an exact GELU and its head gate under the kernel contract (#1297). On a GB10 with Spark-X2.6-4B at 8 bits it prefills 5,800 tokens/s at 2k and decodes 160 tokens/s on one stream; repeats, concurrent streams and resumed chats equal their runs alone. Thanks to @BobClawblaw.

- Metal: Qwen3.5, Nemotron, LFM2 and Gemma 4 admit their streams at load under the load limit: an unnamed `--parallel` is lowered to the count that fits, and a named one that does not fit is refused at startup instead of failing under load (#1302).

- Lane host: a burst of short prompts ends after a lane count of them while streams decode, short prompts are picked among themselves rather than behind an aged long one, and decode debt owed to streams that all ended is dropped (#1287). Tokens and speed are unchanged.

- Qwen3.5 on Metal can keep its streams' keys and values in pages of the core pool with `TF_QWEN35_METAL_KV_PAGES=1`, planned under the load limit at load; the paged kernels' arithmetic is the contiguous ones', and the default path is unchanged (#1294).

- Tests and builds: the f64 reference keeps signed zeros at interval endpoints, so x86_64 hosts accept an exact +0 (#1309, #1315); the memory HTTP test reads the fake server's port line without buffering past it (#1306, #1308); `zig build server-cpu` builds the CPU fake server and engine-less CLI through the shared build graph (#1305, #1314).

- `tensorfold pull` on macOS finishes a connection a signal interrupts instead of crashing (`ISCONN` after a retried connect).

## 1.1.5

- Flash Next on Macs without tensor units decodes faster with the same tokens: the hyper-connection act pair runs for windows of two rows and more, the HC down projection runs without staging or barriers at four outputs a threadgroup, rounds are gated on the GPU with a GPU-only event between them, drafts are gated at tau 0.4, the decode and draft argmaxes take two launches, and wide windows use grouped experts and exact tree windows. On an M3 Ultra with Qwen3.8 Flash Next MLX 4-bit MTP, the served sweep goes from 120.2 to 147.6 tokens/s on one stream (+22.7%), 129.7 to 155.3 at two (+19.8%), 143.8 to 167.0 at four (+16.2%) and 154.4 to 176.1 at eight (+14.1%); prefill is unchanged and every token is identical.

- Flash Next on CUDA with NVFP4 checkpoints prefills 24-29% faster with the same tokens: NVFP4 experts on prompt tiles with the decode form's sums (#1143), no zero fills the one-rank ModelOpt pass reads back (#1147), the hyper-connection normalize over 256-column slices and the dense mix a warp a 32-column group (#1164), and half tiles for decode calls whose 64-column grid leaves SMs idle (#1171). On a GB10 with Qwen3.8 Flash Next FP4-Local, prefill at 16k goes from 1,357 to 1,748 tokens/s and decode gains 1-2%. Attention caches are readable through a page table and the streams beside the first take KV pages from the core pool (#1169, #1170).

- GLM-5.3-Flash on Metal decodes faster with the same tokens by reading the weights once per decode window (#1130): on an M3 Ultra, +2.6% at one stream and +10.2%, +6.4% and +6.3% at two, four and eight.

- A follow-up that sends the same long document again with a new question, in one user message, resumes from a kept prompt state instead of prefilling the document again, on Flash Next CUDA and Qwen3.8-27B on Metal; the reply equals a fresh run's. With a 35k-token document, the follow-up's first token comes after 2.8 s instead of 25.7 s with Qwen3.8 Flash Next FP4-Local on a GB10, and after 9.0 s instead of 125.9 s with Qwen3.8-27B on an M3 Ultra (#1262).

- Drama speech on Metal reuses a catalog voice's reference prefill across requests on the same lane, so a request with the same voice prefills only its own text. Audio is byte-identical to an uncached request. On an M3 Ultra with Drama 3 and a catalog voice, the first audio byte of a streamed reply comes 90% sooner (15.6 s to 1.5 s over four texts) and whole replies take about half as long; the default voice and the first request with a voice are unchanged (#1237).

- A CUDA request cancelled during its prompt keeps the prefill it had completed in the prompt cache, so retrying the same prompt resumes from there instead of starting over; repeated cancellations add up, and an uncancelled request's path is unchanged. On a GB10 with Qwen3.8-27B, a request cancelled 2 s into a 32k prompt kept 1,792 tokens, and its retry's reply equalled an uncancelled run's (#1228). Thanks to @Sapphire-Rapids.

- Lane host: prompt steps share wall time with decoding streams, and the Qwen CUDA prompt pass runs a cut at a time, so a long prompt no longer stalls the streams already decoding; short prompts still go in whole (#1188).

- Flash Next on CUDA: a lone stream's copy window widens to 31 drafts after a copy that lands whole (#1252).

- Tokenizer calls start their scratch on the stack, recovering a small one-stream decode loss in 1.1.4 (#1248).

- experts3 CUDA has a split decode pair form (#1139); radio CUDA reuses one pinned pixel stage for every image (#1140).

- New family: Nemotron-Labs Dory 1.0 4B on CUDA (`dory`), served at 1 to 4 loops with kept prompt states, from ModelOpt W4A16 NVFP4 checkpoints (#1256-#1261). On a GB10 at 2 loops it decodes 44 tokens/s on one stream and 299 aggregate at eight, with exact repeats and resumed == fresh.

- Qwen3.8-27B on CUDA serves ModelOpt and compressed-tensors NVFP4/FP8 exports such as nvidia/Qwen3.8-27B-NVFP4, with bf16 activations against the stored weights (#1145), and ExLlamaV3 (EXL3) Qwen3.5-MoE checkpoints such as Ornith-1.5-122B (#1150). A checkpoint-math mode that would quantize activations is not included.

- GLM-5.3-Flash on CUDA loads nvidia/GLM-5.3-Flash-FP4-Local as stored (#1050), takes its streams' KV pages from the core pool with kept prompt states sized by their position (#1163), and puts the engine's own sequence on pages too (#1211). Tokens and speed at TP=2 are unchanged.

- Gemma 4 runs on Macs before the M5, with its dense projections on the row path; the first token comes from the last prompt row.

- Drama 3 on Metal reads the pruned NVFP4 checkpoint's packed E2M1 experts with their block-16 E4M3 and F32 global scales as stored, without requantizing or expanding them, and routes over the experts it keeps. Opt-in semantic classifier-free guidance (`--cfg-gamma`, `--cfg-allow-request-gamma`, `--cfg-force-guided`, `--cfg-batched`) keeps a conditional and an unconditional backbone state over shared weights; admission counts the extra state, and without it the default and catalog voices' audio is byte-identical to before (#1268). CUDA guidance is not part of this change. Thanks to @jerryfane.

- Qwen3.8-27B on CUDA: the streams beside the first take KV pages from the core pool (`TF_QWEN_KV_PAGES`), with flat batches keeping the contiguous tensor-core attention; tokens and speed are unchanged (#1170).

- GLM-5.3-Flash on CUDA: rank 0's NCCL id goes over the ranks' own lines, so a /32 ring's followers bootstrap (#1178); the RoCE transport serves 2 to 16 ranks (#1209); a request waits in the queue when the shared context pool is momentarily full instead of failing (#1263).

- Gemma 4 on CUDA: admission counts the vision workspace and the drafter, a failed start frees all its buffers, and a malformed DFlash config is refused instead of crashing (#1245).

- LFM2 tool calls: arguments take the type the schema names, the JSON list form is parsed, each call streams as soon as it ends, and required or named calls work through a grammar (#1265).

- Drama voice catalogs can carry a description, listed by `/v1/voices` (#1270); the docs explain why Drama 3 needs the silence guard, with a starting configuration (#1276).

- Server: a tool call that `max_tokens` cuts is returned in `tool_calls` with `finish_reason: "length"` in non-streamed replies too, as it already streamed (#1238, #1299); `return_token_ids` puts the reply's token ids in the response's `tensorfold` block, the same tokens `token_sha` hashes (#1290); GLM-5.3's chat template receives `clear_thinking` from `TF_GLM_CLEAR_THINKING` or the request, so a checkpoint whose template cleared earlier reasoning by default keeps it and reuses those turns' prefill (#1300).

- `--prompt-precision int8` (opt-in): a prompt chunk with a non-finite input now falls back to bf16 as intended; in 1.1.4 the quantizer never raised its fallback flag.

- The CUDA kernel contract's gates pass on a GB10 again (all 338 registered kernels pass gates 1-4 and all 92 planted controls are caught); its resource-limit check still reports 52 problems in Qwen3.5 lane kernels' records (stale or missing register counts), being completed.

- Tensor-parallel ranks join their failure hooks before releasing the hooks' owners, fixing a rare crash at shutdown after a rank fails.

- CUDA builds on Windows with MSVC as the host compiler again: explicit kernel instantiations spell their definitions' `__restrict__` (#1296).

- Desktop app: an agent run whose question passes its deadline stops before another tool-enabled model call.
