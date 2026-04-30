"""M121 / M121b one-shot vectorized-vs-serial perf comparison.

Two scenarios, both built through the public ``plotsim.builder`` API
(`create_from_yaml` / `create`) so the perf check exercises the
production-facing surface, not the engine-internal ``PlotsimConfig``
constructor:

* ``baseline`` — ``create_from_yaml('plotsim/configs/new/saas_template.yaml')``.
  95 entities across 6 segments, 6 metrics, 24 monthly periods, 1
  declared connection. Auto would route this to ``vectorized`` since
  95 ≥ ``_VECTORIZED_AUTO_THRESHOLD`` (50); we force both modes for
  the comparison.

* ``stress`` — ``create(...)`` with 1,020 entities (3 segments of
  ~340), 20 metrics, 4 ``connections``. Mirrors the mission AC's
  "≥10× at 500+ entities per archetype" gate. The operator's check
  threshold is 5×.

Reports two sections:

1. **Wall-clock** — ``generate_tables`` serial vs vectorized for each
   scenario. Stress speedup gates the script's exit code (default
   5×, AC target 10×).

2. **Memory (M121b)** — stress config with ``output.format=parquet``,
   measures peak memory of ``write_tables`` for streaming
   (vectorized + parquet, triggers ``_write_streaming_parquet_facts``)
   vs non-streaming (serial + parquet, single-shot ``to_parquet``).
   Peak measured via ``tracemalloc`` (Python-side allocations) and
   ``psutil`` RSS delta (captures pyarrow's C++-side buffers when
   psutil is installed; the streaming win is largely in the pyarrow
   buffer space, so RSS deltas matter more than tracemalloc here).
   Reports % reduction against the 25% AC threshold.

Run:
    python analysis/perf/m121_vectorized.py
    python analysis/perf/m121_vectorized.py --gate 5
    python analysis/perf/m121_vectorized.py --skip-memory
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover — psutil is optional
    psutil = None
    _HAS_PSUTIL = False

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from plotsim.builder import create, create_from_yaml  # noqa: E402
from plotsim.config import PlotsimConfig  # noqa: E402
from plotsim.output import write_tables  # noqa: E402
from plotsim.tables import generate_tables  # noqa: E402


# --- Subprocess child entry point ------------------------------------------


def _child_main() -> int:
    """Child-process entry: load fixture + write_tables under RSS sampler.

    Reads ``--fixture`` (pickle of ``(config, tables)``) and ``--mode``
    ("streaming" or "non-streaming") from argv, runs one
    ``write_tables`` call, prints peak-RSS-above-baseline (bytes) +
    tracemalloc peak (bytes) + wall seconds as JSON to stdout, exits.

    Subprocess isolation gives each write a clean pyarrow memory pool
    so the streaming-vs-non-streaming RSS comparison isn't biased by
    cross-call pool retention. The parent invokes the script's child
    mode via ``sys.executable`` to inherit the same Python env.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", required=True)
    p.add_argument("--mode", choices=["streaming", "non-streaming"], required=True)
    a = p.parse_args(sys.argv[2:])  # argv[1] is "--child"
    with open(a.fixture, "rb") as f:
        cfg, tables = pickle.load(f)
    if a.mode == "streaming":
        cfg = cfg.model_copy(update={"generation_mode": "vectorized"})
    else:
        cfg = cfg.model_copy(update={"generation_mode": "serial"})
    process = psutil.Process() if _HAS_PSUTIL else None
    baseline = process.memory_info().rss if process else 0
    peak = baseline
    stop = threading.Event()

    def _sample():
        nonlocal peak
        while not stop.is_set():
            if process is not None:
                cur = process.memory_info().rss
                if cur > peak:
                    peak = cur
            stop.wait(0.005)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    with tempfile.TemporaryDirectory() as tmp:
        tracemalloc.start()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_tables(tables, cfg, output_dir=tmp)
        wall = time.perf_counter() - t0
        _, tm_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    stop.set()
    t.join(timeout=1.0)
    if process is not None:
        cur = process.memory_info().rss
        if cur > peak:
            peak = cur
    print(json.dumps({
        "tm_peak": tm_peak,
        "rss_peak_above_baseline": peak - baseline,
        "wall": wall,
    }))
    return 0


# --- Stress config builder --------------------------------------------------


def _memory_stress_config() -> PlotsimConfig:
    """Larger config sized so the pyarrow-buffer transient peak dominates
    measurement noise.

    The wall-clock stress config (1,020 entities × 24 periods × 20 metrics
    ≈ 5 MB unified fact DataFrame) is too small for the streaming Parquet
    memory win to surface above pandas/pyarrow constant overheads — at
    ~5 MB, pool allocator behavior and per-row-group metadata swamp the
    chunk-vs-unified buffer-size difference. Scale up to 8,000 entities
    × 60 monthly periods × 20 metrics ≈ 84 MB unified fact DataFrames
    so the pyarrow conversion peak is the dominant transient. 480K cell-
    count stays just below the engine's 500K-cell warning threshold so
    the load-time gate doesn't print noise to stderr.

    Generation runs once in vectorized mode (~130s on the reference
    machine). The memory comparison then writes the same in-memory
    tables twice — once with streaming dispatch enabled and once with
    it disabled — so only the *write* phase is sampled. Three trials
    each, median reported, via the ``_RssSampler`` 5 ms-interval
    polling thread.
    """
    metrics: list[dict] = []
    for j in range(5):
        metrics.append({"name": f"score_{j}", "type": "score", "polarity": "positive"})
    for j in range(5):
        metrics.append({
            "name": f"amount_{j}", "type": "amount",
            "polarity": "positive", "range": [0.0, 1000.0],
        })
    for j in range(5):
        metrics.append({"name": f"count_{j}", "type": "count", "polarity": "positive"})
    for j in range(5):
        metrics.append({
            "name": f"index_{j}", "type": "index",
            "polarity": "positive", "range": [0.0, 100.0],
        })
    return create(
        about="m121b memory stress",
        unit="entity",
        # 5 years monthly = 60 periods. Wider window than the wall-
        # clock stress so the per-fact row count crosses ~300K rows
        # and the unified pyarrow buffer comfortably exceeds the
        # ~10 MB noise floor of psutil RSS sampling.
        window=("2020-01", "2024-12", "monthly"),
        metrics=metrics,
        segments=[
            {"name": "growers", "archetype": "growth", "count": 2700},
            {"name": "decliners", "archetype": "decline", "count": 2700},
            {"name": "flats", "archetype": "flat", "count": 2600},
        ],
        connections=[
            "score_0 mirrors amount_0",
            "score_1 driven_by index_0",
            "amount_1 opposes count_0",
            "index_1 related count_1",
        ],
    )


def _stress_config() -> PlotsimConfig:
    """Builder-API stress config: 1,020 entities × 20 metrics × 4 connections.

    Three segments of 340 entities each across three archetype shapes
    (growth / decline / flat) so the vectorized dispatcher exercises
    its multi-archetype grouping. 20 metrics span the four
    ``MetricInput.type`` flavors (score / amount / count / index) so
    the batched samplers cover the dispatch cases that actually occur
    in production. Four ``connections`` give the batched copula a
    non-trivial Cholesky workload — the copula is one of the heaviest
    per-cell scalar costs in the serial loop and the largest single
    speedup lever.
    """
    metrics: list[dict] = []
    # 5 score metrics — bounded [0,1], beta under the hood.
    for j in range(5):
        metrics.append({
            "name": f"score_{j}", "type": "score", "polarity": "positive",
        })
    # 5 amount metrics — lognorm with explicit range.
    for j in range(5):
        metrics.append({
            "name": f"amount_{j}", "type": "amount",
            "polarity": "positive", "range": [0.0, 1000.0],
        })
    # 5 count metrics — poisson, no range.
    for j in range(5):
        metrics.append({
            "name": f"count_{j}", "type": "count", "polarity": "positive",
        })
    # 5 index metrics — beta with declared range.
    for j in range(5):
        metrics.append({
            "name": f"index_{j}", "type": "index",
            "polarity": "positive", "range": [0.0, 100.0],
        })
    return create(
        about="m121 perf stress",
        unit="entity",
        window=("2024-01", "2025-12", "monthly"),
        metrics=metrics,
        segments=[
            {"name": "growers", "archetype": "growth", "count": 340},
            {"name": "decliners", "archetype": "decline", "count": 340},
            {"name": "flats", "archetype": "flat", "count": 340},
        ],
        connections=[
            "score_0 mirrors amount_0",
            "score_1 driven_by index_0",
            "amount_1 opposes count_0",
            "index_1 related count_1",
        ],
    )


# --- Timing -----------------------------------------------------------------


def _time_one(cfg: PlotsimConfig) -> float:
    """One ``generate_tables`` call timed in seconds."""
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_tables(cfg, rng)
    return time.perf_counter() - t0


def _force_mode(cfg: PlotsimConfig, mode: str) -> PlotsimConfig:
    """Override ``generation_mode`` so each timed run is unambiguous.

    Builder configs default to ``"auto"``; for a clean comparison the
    perf script overrides to ``"serial"`` then ``"vectorized"`` so
    neither timing measurement depends on the auto-resolver.
    """
    return cfg.model_copy(update={"generation_mode": mode})


def _run_scenario(name: str, cfg: PlotsimConfig) -> tuple[float, float]:
    """Time serial then vectorized; return (serial_s, vectorized_s)."""
    cfg_s = _force_mode(cfg, "serial")
    cfg_v = _force_mode(cfg, "vectorized")
    s = _time_one(cfg_s)
    v = _time_one(cfg_v)
    speedup = (s / v) if v > 0 else float("inf")
    n_entities = len(cfg.entities)
    n_metrics = len(cfg.metrics)
    n_periods = cfg.time_window.period_count()
    print(
        f"  {name}: entities={n_entities:>5d}  metrics={n_metrics:>2d}  "
        f"periods={n_periods:>3d}  serial={s:7.3f}s  "
        f"vectorized={v:7.3f}s  speedup={speedup:5.2f}x"
    )
    return s, v


# --- Memory (M121b 25% AC) --------------------------------------------------


class _RssSampler:
    """Background thread that polls ``Process.memory_info().rss`` and
    tracks the maximum.

    Single before/after RSS snapshots miss the peak entirely on a
    short-lived call: pyarrow allocates a transient buffer, then frees
    it before the call returns, so the post-call RSS may be lower than
    the during-call peak. A 5 ms poll interval gives ~200 samples per
    second — fast enough to catch the buffer at its high-water mark on
    sub-second writes without measurably stealing CPU from the call
    being measured.

    Usage: ``with _RssSampler() as s: ...; peak = s.peak_rss``.
    Falls through to ``baseline_rss`` (the pre-start sample) when
    psutil is unavailable.
    """

    def __init__(self, interval_s: float = 0.005):
        self._proc = psutil.Process() if _HAS_PSUTIL else None
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_rss: int = 0
        self.peak_rss: int = 0

    def __enter__(self) -> "_RssSampler":
        if self._proc is None:
            return self
        self.baseline_rss = self._proc.memory_info().rss
        self.peak_rss = self.baseline_rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        # Final sample in case the thread missed a last spike.
        if self._proc is not None:
            current = self._proc.memory_info().rss
            if current > self.peak_rss:
                self.peak_rss = current

    def _run(self) -> None:
        while not self._stop.is_set():
            current = self._proc.memory_info().rss
            if current > self.peak_rss:
                self.peak_rss = current
            self._stop.wait(self._interval_s)


def _measure_write_peak_subprocess(
    fixture_path: Path,
    mode: str,
) -> tuple[int, int, float]:
    """Spawn a child Python process to measure one write's peak memory.

    Each subprocess starts with a clean pyarrow memory pool so the
    measured RSS peak isn't biased by allocator state from a previous
    write. Three trials per scenario; the median across trials is
    returned to bound noise.

    Returns ``(tracemalloc_peak_bytes, rss_peak_above_baseline_bytes,
    wall_seconds)``.
    """
    tm_peaks: list[int] = []
    rss_peaks: list[int] = []
    wall_seconds: list[float] = []
    for _trial in range(3):
        proc = subprocess.run(
            [
                sys.executable, "-W", "ignore",
                str(Path(__file__).resolve()),
                "--child", "--fixture", str(fixture_path), "--mode", mode,
            ],
            capture_output=True, text=True, check=True,
        )
        # Filter to the LAST stdout line; engine summary lines from the
        # child's config_load may print earlier.
        out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        payload = json.loads(out_lines[-1])
        tm_peaks.append(int(payload["tm_peak"]))
        rss_peaks.append(int(payload["rss_peak_above_baseline"]))
        wall_seconds.append(float(payload["wall"]))
    return (
        int(statistics.median(tm_peaks)),
        int(statistics.median(rss_peaks)),
        float(statistics.median(wall_seconds)),
    )


def _measure_write_peak(
    cfg: PlotsimConfig,
    tables: dict,
) -> tuple[int, int, float]:
    """Measure ``write_tables`` peak memory.

    Returns ``(tracemalloc_peak_bytes, rss_peak_above_baseline_bytes,
    wall_seconds)``. ``tracemalloc`` captures Python-side peak —
    precise, but does not see pyarrow's C++ memory pool. The RSS
    measurement uses a 5 ms-interval sampling thread (``_RssSampler``)
    so the peak transient pyarrow buffer is captured, not just the
    pre/post sandwich. The reported RSS value is
    ``peak_during_call - baseline_before_call`` — the call's
    additional peak above its starting RSS. ``-1`` when psutil is
    unavailable.

    Each call runs in a fresh tempdir; ``gc.collect()`` runs before
    each measurement to reset the Python heap baseline. Three trials
    per scenario; the median across trials is returned to bound noise
    from background OS activity.
    """
    tm_peaks: list[int] = []
    rss_peaks: list[int] = []
    wall_seconds: list[float] = []
    for _trial in range(3):
        gc.collect()
        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tracemalloc.start()
                with _RssSampler() as sampler:
                    t0 = time.perf_counter()
                    write_tables(tables, cfg, output_dir=tmp)
                    wall = time.perf_counter() - t0
                _, tm_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if _HAS_PSUTIL:
                    rss_peak = sampler.peak_rss - sampler.baseline_rss
                else:
                    rss_peak = -1
        tm_peaks.append(tm_peak)
        rss_peaks.append(rss_peak)
        wall_seconds.append(wall)
    return (
        int(statistics.median(tm_peaks)),
        int(statistics.median(rss_peaks)),
        float(statistics.median(wall_seconds)),
    )


def _mb(b: int) -> float:
    """Bytes → MB (decimal, matching how pyarrow reports buffer sizes)."""
    return b / 1_000_000.0


def _run_memory_comparison(stress_cfg: PlotsimConfig) -> tuple[int, int, int, int]:
    """Generate the stress config once, then measure streaming and
    non-streaming Parquet write peak memory in fresh subprocesses.

    Returns ``(streaming_tm, streaming_rss, non_streaming_tm,
    non_streaming_rss)`` — bytes for each. Generation runs once
    (vectorized mode produces the tables both writers consume) and
    pickles the result to a tempfile; each subprocess loads the
    pickle, runs one write, reports peak memory, exits. Subprocess
    isolation gives each measurement a clean pyarrow memory pool so
    the comparison isn't biased by cross-call allocator retention
    (the in-process measurement saw ~64% bias depending on call
    order — pyarrow's pool monotonically grows with each
    ``pa.Table.from_pandas`` call and never returns memory to the OS
    within a single Python process).

    Path selection: the streaming dispatch in
    ``plotsim.output._streaming_parquet_eligible`` triggers when
    ``config.generation_mode`` resolves to ``"vectorized"`` AND
    ``output.format == "parquet"``. The child process toggles
    ``generation_mode`` per ``--mode`` argument; the underlying
    tables are unchanged.
    """
    parquet_cfg = stress_cfg.model_copy(update={
        "output": stress_cfg.output.model_copy(update={"format": "parquet"}),
        "generation_mode": "vectorized",
    })
    rng = np.random.default_rng(parquet_cfg.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = generate_tables(parquet_cfg, rng)

    # Pickle (config, tables) to a single tempfile each child loads.
    # Tables can be large (~100 MB on the memory-stress config) so
    # the tempfile lives only for the duration of the comparison.
    fixture_dir = Path(tempfile.mkdtemp(prefix="m121b_perf_"))
    fixture_path = fixture_dir / "fixture.pkl"
    try:
        with open(fixture_path, "wb") as f:
            pickle.dump((parquet_cfg, tables), f, protocol=pickle.HIGHEST_PROTOCOL)
        # Free the parent's reference to ``tables`` so its memory
        # doesn't bias children spawned next (the child has its own
        # process, but on Windows the parent's RSS is still part of
        # the system's memory pressure).
        del tables
        gc.collect()

        n_tm, n_rss, n_wall = _measure_write_peak_subprocess(
            fixture_path, "non-streaming",
        )
        s_tm, s_rss, s_wall = _measure_write_peak_subprocess(
            fixture_path, "streaming",
        )
    finally:
        try:
            fixture_path.unlink()
            fixture_dir.rmdir()
        except OSError:
            pass

    rss_label_s = (
        f"RSS peak={_mb(s_rss):7.2f} MB" if _HAS_PSUTIL else "RSS=     n/a"
    )
    rss_label_n = (
        f"RSS peak={_mb(n_rss):7.2f} MB" if _HAS_PSUTIL else "RSS=     n/a"
    )
    print(
        f"  streaming    : tracemalloc peak={_mb(s_tm):7.2f} MB  "
        f"{rss_label_s}  write={s_wall:5.2f}s"
    )
    print(
        f"  non-streaming: tracemalloc peak={_mb(n_tm):7.2f} MB  "
        f"{rss_label_n}  write={n_wall:5.2f}s"
    )
    return s_tm, s_rss, n_tm, n_rss


# --- Entry point ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate", type=float, default=5.0,
        help="Stress-scenario speedup floor; exits non-zero below this. "
             "Default 5.0× (mission AC target is 10×; 5× is the operator's "
             "check threshold).",
    )
    parser.add_argument(
        "--memory-gate", type=float, default=25.0,
        help="Streaming Parquet memory-reduction floor in percent. M121b AC "
             "target is 25%. Reported but not gated on the script's exit "
             "code — wall-clock gate is the primary CI signal.",
    )
    parser.add_argument(
        "--skip-memory", action="store_true",
        help="Skip the M121b tracemalloc + RSS memory comparison.",
    )
    args = parser.parse_args()

    print("M121 vectorized vs serial wall-clock — one-shot timing")
    print("=" * 64)

    print("Baseline (saas_template.yaml via create_from_yaml):")
    baseline_path = ROOT / "plotsim" / "configs" / "new" / "saas_template.yaml"
    cfg_baseline = create_from_yaml(baseline_path)
    _run_scenario("baseline", cfg_baseline)

    print()
    print("Stress (create(): 1,020 entities, 20 metrics, 4 connections):")
    cfg_stress = _stress_config()
    s, v = _run_scenario("stress", cfg_stress)
    speedup = (s / v) if v > 0 else float("inf")

    # M121b memory comparison — uses a SEPARATE, larger config sized so
    # pyarrow's transient buffer dominates measurement noise. The
    # wall-clock stress (1,020 × 24 × 20 ≈ 5 MB unified DF) is too
    # small for the streaming win to clear pyarrow's pool-allocator
    # constants; the memory-stress config (5,000 × 60 × 20 ≈ 50 MB)
    # sizes the unified DF an order of magnitude up so the AC's 25%
    # threshold is meaningful. Generation runs once in vectorized mode
    # (~80s); the same tables write twice (streaming + non-streaming)
    # for the comparison.
    if not args.skip_memory:
        print()
        print("Memory (M121b: 8,000 entities × 60 periods × 20 metrics, parquet output):")
        if not _HAS_PSUTIL:
            print(
                "  (psutil not installed; only tracemalloc Python-side peak "
                "is reported. The streaming Parquet win is largely in "
                "pyarrow's C++ buffer space, which tracemalloc does not see. "
                "`pip install psutil` to surface the RSS-peak signal.)"
            )
        cfg_mem = _memory_stress_config()
        s_tm, s_rss, n_tm, n_rss = _run_memory_comparison(cfg_mem)
        print()
        if n_tm > 0:
            tm_reduction_pct = 100.0 * (1.0 - s_tm / n_tm)
            print(
                f"  tracemalloc reduction: "
                f"{_mb(n_tm):7.2f} MB -> {_mb(s_tm):7.2f} MB  "
                f"({tm_reduction_pct:+.1f}%)"
            )
        if _HAS_PSUTIL and n_rss > 0:
            rss_reduction_pct = 100.0 * (1.0 - s_rss / n_rss)
            print(
                f"  RSS delta reduction:  "
                f"{_mb(n_rss):7.2f} MB -> {_mb(s_rss):7.2f} MB  "
                f"({rss_reduction_pct:+.1f}%)"
            )
            verdict = (
                "PASS" if rss_reduction_pct >= args.memory_gate else "FAIL"
            )
            print(
                f"  {verdict} — RSS-based reduction "
                f"{rss_reduction_pct:+.1f}% vs M121b AC target "
                f"{args.memory_gate:.0f}%"
            )

    print()
    if speedup >= args.gate:
        print(f"PASS — stress speedup {speedup:.2f}x >= gate {args.gate:.2f}x")
        return 0
    print(f"FAIL — stress speedup {speedup:.2f}x < gate {args.gate:.2f}x")
    return 1


if __name__ == "__main__":
    # Subprocess mode: ``python m121_vectorized.py --child --fixture <path>
    # --mode <streaming|non-streaming>`` runs ``_child_main`` to write
    # the fixture once and report peak RSS as JSON. The parent process
    # spawns one of these per measurement to isolate pyarrow's memory
    # pool between runs.
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(_child_main())
    sys.exit(main())
