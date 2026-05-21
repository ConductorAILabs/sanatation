# Streaming / state-carryover plan

The single biggest differentiator this repo can offer beyond junafinity's
renderer is **state carryover between walk steps** — each chunk continues
the world established by the previous chunk instead of starting from
scratch with just the last frame as a seed.

This is what makes a walk-around feel like exploring a coherent space
rather than dreaming through disconnected vignettes. junafinity explicitly
lists it as future work; we plan to ship it.

## The good news

The model already supports it. Upstream code already in the patched repo:

- **`SanaModelWrapper.forward(...)`** in
  `diffusion/scheduler/longlive_flow_euler_sampler.py:272` accepts
  `save_kv_cache=True` and a `kv_cache=...` kwarg and returns the
  updated cache as the third return value.
- **`LongLiveFlowEuler`** in the same file (line 317) is a multi-chunk
  sampler that internally builds a `kv_cache: list[list[list[3]]]`
  (one entry per chunk, per transformer block, with three slots for
  `[cum_vk, cum_k_sum, last_k]`) and accumulates it across chunks via
  `_accumulate_kv_cache(...)`.
- **`CachedCausalAttention`** and **`CachedGLUMBConvTemp`** are the
  module types that consume / produce the cache. `_initialize_cached_modules`
  walks the model's `blocks` and collects them.

The catch: `LongLiveFlowEuler.sample()` is designed to process a single
multi-chunk batch in one call. The `kv_cache` lives inside that call's
stack frame; it's not exposed to the caller.

## The work

To get state carryover across separate `sample()` calls (the walk-step
boundary), expose the cache as durable state on a new sampler. Sketch:

```python
class StreamingSanaSampler:
    """Like LongLiveFlowEuler but persists kv_cache across .step() calls."""

    def __init__(self, model_fn, condition, model_kwargs, *, base_chunk_frames=8, **kwargs):
        # same setup as LongLiveFlowEuler.__init__ — wrap model, collect
        # cached modules, configure scheduler + denoising steps.
        ...
        self.kv_cache_history: list[list[list]] = []   # one entry per chunk so far

    @torch.no_grad()
    def step(self, noise_for_new_frames: torch.Tensor) -> torch.Tensor:
        """Generate the next chunk continuing from prior state.

        Args:
            noise_for_new_frames: (1, C, F_new, H, W) — fresh noise for the
                                  next chunk's latents; the seed frame is
                                  already encoded into the first latent slot.
        Returns:
            (1, C, F_new, H, W) refined latent for the new chunk.
        """
        # 1. Build chunk_kv_cache by accumulating over self.kv_cache_history
        #    (same logic as LongLiveFlowEuler._accumulate_kv_cache, but
        #     operating on history instead of a per-call kv_cache list).
        # 2. Run denoising loop over self.denoising_step_list with
        #    save_kv_cache=False at intermediate steps, then a final pass
        #    with save_kv_cache=True at timestep=0 to extract the cache for
        #    this chunk.
        # 3. Append the new cache to self.kv_cache_history; trim if
        #    num_cached_blocks > 0.
        # 4. Return the denoised latent.

    def reset(self) -> None:
        """Drop all accumulated state — start a fresh scene."""
        self.kv_cache_history.clear()
```

`walk.py` would then:

```python
sampler = StreamingSanaSampler(pipeline.model, condition=cond, ...)
for key in keypress_stream:
    cam_action = KEY_TO_ACTION[key] + f"-{CHUNK_FRAMES}"
    c2w = action_string_to_c2w(cam_action, start_pose=current_pose)
    # ... build noise tensor for new frames, optionally seed slot 0 from
    #     current_image (only on first call), or splice in the last
    #     decoded latent as slot 0 (continuation across chunks).
    new_latent = sampler.step(noise)
    video = vae_decode_chunk(new_latent)
    write_chunk_to_mp4(video)
    current_pose = c2w[-1]
```

The trickiest piece is the `noise_for_new_frames` shape and how to fold
the previous chunk's last latent into the new chunk's first slot — this is
exactly what `_sample_stage1` does for the seed image today (encoding the
PIL frame through the Sana VAE encoder). For continuation we want the
VAE-encoded last frame of the previous chunk's MP4, or even better, the
last latent of the previous chunk directly (skipping the encode/decode
round-trip).

## Estimated work

- Build `StreamingSanaSampler`: ~150 LOC borrowing heavily from
  `LongLiveFlowEuler.sample()`.
- Wire it into `walk.py` behind a `--streaming` flag (default off until
  validated): ~80 LOC changes.
- Validate against the non-streaming path: a 4-step walk forward should
  produce a visibly more coherent result vs the current frame-seed-only
  approach.
- First-pass failure modes: kv_cache shape mismatches (chunk boundaries
  off by 1), state leak between unrelated runs, MPS-specific dtype issues
  with the cached tensors.

## Why this is hard to do cleanly

`LongLiveFlowEuler` assumes a batched multi-chunk view of the world — the
entire trajectory is known up front, chunks are processed in order, kv_cache
indexes by absolute chunk index. Streaming is the same loop with one chunk
per outer call, and chunk indices are monotonic across calls. The math is
the same; the bookkeeping isn't. Easy to get subtly wrong.

The lazy fallback if streaming proves too finicky: render the FULL planned
trajectory in advance (e.g., 80 frames = 10 chunks of 8) using stock
`LongLiveFlowEuler`, then play it back chunk-by-chunk in the UI. You lose
true interactivity (the player can't change direction mid-clip) but get the
visual coherence for free. That's a 50-line `walk_pre_rendered.py` and
might be the right thing to ship first.
