# Timelapse videos (for non-technical viewers)

Added 2026-10-10. These videos show how each condition learned and how the
finished policies compare. They are made **after training**, from the
snapshots every run already saves. Nothing extra has to be switched on while
training runs.

## How it works

- Every training run saves a snapshot of its policy every 250 iterations
  (`logs/rsl_rl/<experiment>/<run>/model_N.pt`). A 3000-iteration run keeps 13
  snapshots; a 1500-iteration warm start keeps 7.
- `src/microduck_pretrain/runs.py` finds every run on disk and rebuilds a
  playable policy from any snapshot. It uses the same math as the official
  exporter; `tests/test_runs.py` checks that the two agree.
- `scripts/make_timelapse.py` replays those policies in the CPU simulator (the
  one `md-eval` uses). Every policy is filmed from the same camera, on the same
  course, with the same random seed, so the clips can be compared side by side.

## The three videos

| Video | File | What it shows |
| --- | --- | --- |
| Learning | `videos/learning_seed1.mp4` | C2, C3 and C4 side by side at each snapshot. A banner shows the training iteration and how much simulated practice that is: 3000 iterations is about 68 days of walking. |
| Exam | `videos/exam_seed1.mp4` | C1 (vendor) and the final policy of every condition, facing six challenges in turn: home turf, slippery floor, shoves, a heavy backpack (drawn as an orange block), a bumpy floor, and everything at once. Each panel keeps a running fall count. |
| Seeds | `videos/seeds.mp4` | For each condition, its three training seeds on "everything at once". It shows how repeatable each recipe is. |

Each video opens with a title card written for viewers who have not seen the
project before.

## Commands (desktop, inside WSL)

```bash
cd ~/microduck-pretraining
git pull
uv run python scripts/make_timelapse.py --list                 # what can be filmed
uv run python scripts/make_timelapse.py all --quick            # small test versions -> videos/quick/
uv run python scripts/make_timelapse.py all                    # full versions -> videos/
uv run python scripts/make_timelapse.py learning --seed 2      # one video, another seed
uv run python scripts/make_timelapse.py exam --conditions c3 c5 c3x
```

Options:

- `--fast` turns off shadows. It roughly halves render time without a GPU.
- `--workers N` films that many clips in parallel. The default is half the
  CPU cores, up to 8.
- `--out DIR` changes where the videos are written.

The first line of output names the renderer. "egl (GPU)" or "glfw (GPU)" is
fast. "osmesa (software)" or "egl (software)" means the CPU is drawing every
frame, which is slower.

To watch a video from Windows, copy it out of WSL:
`cp videos/*.mp4 /mnt/c/Users/jsmcl/Videos/`, or open `\\wsl$\Ubuntu\home\...`
in Explorer.

## Time and size

Measured without a GPU (cloud container, software rendering):

- Software rendering takes about 0.28 s per frame with shadows and about
  0.10 s without them.
- The full set of three videos, with every condition and seed, is roughly
  11,000 panel-frames. That is about 50 minutes on one core, and much less
  with several workers or a GPU.
- The `--quick` set takes about 2.5 minutes.
- The finished videos are small, a few MB each.
- Videos go in `videos/` and are not committed. They can be rebuilt at any
  time from `logs/`.

## Keeping every training filmable (the rule for the rest of the project)

1. **Never delete `logs/rsl_rl/`.** The snapshots are the only record of how
   each policy learned. Each snapshot is about 5 MB; the whole project is
   about 1-2 GB.
2. **Name runs `<key>_seed<N>`.** The train scripts already do this. If a short
   test run shares a name with a real run, the run with the most training
   wins. Warm-start staging folders (`warmstart_*`) are skipped.
3. **New kinds of training show up automatically**, labeled with their folder
   name. To give one a friendly title and caption in the videos, add one line
   to `EXPERIMENTS` in `src/microduck_pretrain/runs.py`.
4. **Moving runs between machines:** copy the whole run folder, including
   `params/`. Only the `model_N.pt` files are needed for filming.
5. **More frames of learning:** snapshots every 250 iterations give 13 steps
   per run. For a smoother learning video in a future run, pass
   `--agent.save-interval 100` to the train command. This changes nothing about
   training except disk use.
