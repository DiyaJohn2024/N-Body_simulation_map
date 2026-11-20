# N-Body Simulation – Cluster E (Teams 17, 18, 19 & 20)

**Our team: Team 20 – Problem 20: Precision, Accuracy and Integration**

> Team members: `Disha B H`, `Diya Anna John`
> Course: `MULTICORE ARCHITECTURE PROGRAMMING` 
---

## Quick start for the reader

* **Looking for the final code and results?** Use **`MAP-Version2/`**. It is the final, corrected version.
* **Looking for our first attempt?** That is `MAP-Version1/` (everything on Google Colab, smaller and quicker runs).
* **Want the main conclusions fast?** Jump to [Section 8 – Results](#8-results-and-findings-final-version-version-2).
* **Want to re-run the code?** See [Section 6 – How to run](#6-how-to-run-the-code).

---

## 1. What is this project?

The project simulates the **gravitational N-body problem**: N particles attract each other, and we compute how they move over time. The same simulation is implemented on four different platforms, and **Team 20 compares them for speed and numerical accuracy** using three floating-point precision modes (FP64, FP32, mixed).

| Team | Problem | Platform | Program |
|---|---|---|---|
| 17 | P17 | Shared-memory CPU: sequential, OpenMP, Pthreads | `nbody_smp.cpp` |
| 18 | P18 | Distributed CPU: MPI | `nbody_mpi.cpp` |
| 19 | P19 | GPU: CUDA | `nbody_cuda.cu` |
| **20** | **P20** | **Runs all of the above in FP64 / FP32 / mixed for 10,000 steps and analyses accuracy** | `team20_run.py`, `team20_analyze.py` |

All programs include one shared header, `nbody_common.hpp`, so every platform uses exactly the same physics, initial conditions, integrator and output format. This is what makes the comparison fair.

---

## 2. Repository structure

This repo contains **two versions** of the work. **Version 2 is the final version.**

```
.
├── README.md                             
│
├── MAP-Version1/                          <- VERSION 1 (first attempt, Colab only)
│   ├── NBody_ClusterE_Teams17-20_Colab_updated.ipynb   (one notebook, everything in Colab)
│   ├── nbody_acc (1)/acc/                 (Team 20 accuracy-study output of Version 1)
│   └── nbody_results/results/             (Team 17/18/19 performance output of Version 1)
│
└── MAP-Version2/                          <- VERSION 2 (FINAL: local PC + Colab)
    ├── NBody_1_LOCAL_Teams17-18_and_CPU_runs.ipynb      (run on a local PC / WSL)
    ├── NBody_2_COLAB_Team19_and_Team20_final.ipynb      (run in Google Colab, T4 GPU)
    ├── nbody_work/                        (local PC working folder: CPU results + CPU accuracy runs)
    ├── colab_results/results/             (performance CSVs and charts: Teams 17, 18, 19)
    └── colab_acc/acc/                     (final Team 20 accuracy study: CPU + GPU, plots, summary tables)
```

> The two notebooks of Version 2 are the final code. The source code of every C++/CUDA/Python program is written out by the notebooks themselves (`%%writefile` cells), so the notebooks *are* the code.

---

## 3. Version 1 vs Version 2 – what is the difference?

| | **Version 1** | **Version 2 (final)** |
|---|---|---|
| Notebooks | 1 notebook (everything in Colab) | 2 notebooks (LOCAL PC + COLAB) |
| Where the CPU code runs | Colab (only ~2 vCPUs) | Our own PC through WSL (more cores) |
| Where the GPU code runs | Colab T4 | Colab T4 |
| OpenMP / Pthreads threads | 1, 2, 4 | 1, 2, 4, 8, 16 |
| MPI ranks | 1, 2, 4, 8, 16 (mostly oversubscribed) | 1, 2, 4, 8, 16 |
| CPU benchmark sizes N | 4,096 / 8,192 / 16,384 | 4,096 / 16,384 / 65,536 |
| CPU benchmark steps | 10 (to save time) | **100** (as required by the brief) |
| GPU benchmark sizes N | up to 262,144 | up to **1,048,576** |
| **Team 20 accuracy study** | N = **1,024**, MPI 2 ranks, 10,000 steps | N = **4,096**, MPI 4 ranks, 10,000 steps |
| Accuracy tolerances | energy 1e-5 | energy 1e-5 **and** position 1e-6 |
| Final combined CPU + GPU analysis | all on one Colab machine | CPU runs (own PC) and GPU runs (Colab) combined in one analysis |

**Why Version 2 is better:** Version 1 was executed entirely in Google Colab, which provided limited CPU resources(2 CPU cores). Consequently, higher thread counts and MPI ranks could lead to CPU oversubscription, affecting the representativeness of the performance measurements. Version 2 addresses this by moving CPU experiments to a local multi-core machine while retaining GPU experiments in Colab. Version 2 runs the CPU part on a real multi-core PC with the full 100 steps, uses a larger accuracy-study problem (N = 4,096), and merges CPU and GPU accuracy runs into one final analysis.

---

## 4. Physics and numerical setup (identical on every platform)

* **Force law:** softened gravity, `G = 1`, softening `ε = 0.05`, all masses `m_i = 1/N`
* **Initial conditions:** deterministic random generator (**seed 42**); positions in [-1, 1]³, velocities in [-0.5, 0.5]³. Every platform starts from the exact same particles.
* **Integrator:** velocity-Verlet (kick-drift-kick), fixed **dt = 1e-3**
* **Precision modes**
  * `fp64`: everything in double precision (this is the reference)
  * `fp32`: everything in single precision
  * `mixed`: positions, velocities and accumulation in double; the pair-force arithmetic in float
* **Diagnostics:** energy and verification checks are *always* computed in FP64.
* **Verification:** every program compares its first-step forces with the sequential FP64 reference and prints a `VERIFY` line (pass limit **1e-5**).
* **FLOP convention:** 20 flop per particle-pair interaction (used for GFLOP/s numbers).

---

## 5. Programs and variants

**Team 17 – `nbody_smp.cpp`** (`--variant`)
`seq`, `seq_aos` (sequential, structure-of-arrays vs array-of-structures) · `omp_full` · `omp_full_aos` · `omp_sym_priv` (Newton's 3rd law, per-thread private buffers + reduction) · `omp_sym_atomic` (Newton's 3rd law with `omp atomic`) · `pthreads` · `pthreads_sym` (Pthreads version of the best OpenMP variant)

**Team 18 – `nbody_mpi.cpp`** (`--variant`)
`allgather` (baseline) · `ring` (blocks passed around a ring with `MPI_Sendrecv_replace`) · `ring_overlap` (non-blocking sends/receives overlapped with computation). N must be divisible by the number of ranks.

**Team 19 – `nbody_cuda.cu`** (`--variant`)
`naive` (all reads from global memory) · `tiled` (shared-memory tiles, `--block` = 64/128/256). Options: `--fast 1` (fast reciprocal square root), `--profile 1` (registers/occupancy), `-DUNROLL=1/4/8` (loop unrolling).

**Team 20 – `team20_run.py` and `team20_analyze.py`**
* `team20_run.py` runs every platform in FP64/FP32/mixed for 10,000 steps and saves the total energy and all positions every 100 steps.
* `team20_analyze.py` creates the energy-drift, trajectory-error, Pareto (accuracy vs speed) and CPU-vs-GPU plots and the summary/recommendation tables.

Platforms in the Team 20 study (each in `fp64`, `fp32`, `mixed`):

| Tag | Program | Configuration |
|---|---|---|
| `seq` | `nbody_smp` | sequential (FP64 = reference) |
| `openmp` | `nbody_smp` | `omp_full`, all cores |
| `mpi` | `nbody_mpi` | `ring`, 4 ranks (Version 1: 2 ranks) |
| `cuda` | `nbody_cuda` | `tiled`, block 128, IEEE sqrt |
| `cuda_rsqrt` | `nbody_cuda` | same, with fast `rsqrt` |

---

## 6. How to run the code

### Version 2 (final) – recommended

**Requirements**
* Local PC: Linux, or Windows with **WSL2**, with `g++`, `openmpi-bin`, `libopenmpi-dev`, `python3`, `pandas`, `matplotlib`, `numpy`
  (Ubuntu/WSL: `sudo apt install g++ openmpi-bin libopenmpi-dev python3-pip` then `pip install pandas matplotlib numpy`)
* Google Colab with a **T4 GPU** (*Runtime → Change runtime type → T4 GPU*)

**Steps**
1. Open `NBody_1_LOCAL_...ipynb` on the local PC (Jupyter / VS Code) and run all cells top to bottom. This covers setup, Team 17 (Part B), Team 18 (Part C) and the CPU half of Team 20 (Part E). Runtime: roughly 1–2 hours.
   * ⚠️ The local notebook contains **hard-coded Windows/WSL paths** (e.g. `/mnt/c/Users/.../map code/nbody_work`). Change these to your own folder before running.
2. The last cell creates **`local_results.zip`** (this is the content of `MAP-Version2/nbody_work/` results).
3. Open `NBody_2_COLAB_...ipynb` in Google Colab with a T4 GPU and run all cells top to bottom. Upload `local_results.zip` when asked (step A3). This covers Team 19 (Part D), the GPU half of Team 20 and the final analysis of all platforms.
4. The last cell downloads `colab_results.zip` and `colab_acc.zip` (these are `colab_results/` and `colab_acc/` in this repo).

### Version 1

Open the single notebook in Colab with a T4 GPU runtime and run all cells from top to bottom (Parts A–E). The last cell downloads `nbody_results.zip` and `nbody_acc.zip`.

### Running a single program by hand (examples)

```bash
g++ -O3 -march=native -fno-math-errno -fopenmp -std=c++17 nbody_smp.cpp -o nbody_smp -lpthread
./nbody_smp --variant omp_sym_priv --prec fp64 --n 4096 --steps 100 --runs 5 --threads 4 --csv results/test.csv

mpicxx -O3 -march=native -fno-math-errno -fopenmp-simd -std=c++17 nbody_mpi.cpp -o nbody_mpi
mpirun -np 4 ./nbody_mpi --variant ring --prec fp64 --n 4096 --steps 100 --csv results/test_mpi.csv

nvcc -O3 -std=c++17 -arch=sm_75 nbody_cuda.cu -o nbody_cuda
./nbody_cuda --variant tiled --block 128 --prec fp32 --n 16384 --steps 20 --csv results/test_gpu.csv
```

---

## 7. Where to find the results (Version 2)

| What | Folder / file |
|---|---|
| Team 17 final benchmark (shared memory) | `MAP-Version2/colab_results/results/team17_corrected.csv` (also in `nbody_work/results/`), chart `team17_corrected_charts.png` |
| Team 18 strong scaling / weak scaling (MPI) | `MAP-Version2/colab_results/results/team18.csv`, `team18_weak.csv`, chart `team18_charts.png` |
| Team 19 GPU benchmark | `team19.csv` (tile/size sweep), `MAP-Version2/colab_results/results/team19_rsqrt.csv` (fast rsqrt), `team19_unroll.csv` (unrolling), chart `team19_tile_chart.png` |
| Team 20 accuracy data | `MAP-Version2/colab_acc/acc/<platform>_<precision>_energy.csv`, `_results.csv`, `_traj.bin` (CPU-only copies in `nbody_work/acc/`) |
| Team 20 summaries | `MAP-Version2/colab_acc/acc/summary_all_runs.csv`, `summary_recommendation.csv` |
| Team 20 plots | `plot_energy_drift.png`, `plot_trajectory_error.png`, `plot_pareto.png`, `plot_particles.png` (in `colab_acc/acc/`) |

Notes:
* The `results/` folders also hold small scratch files from development (`fixed_test.csv`, `stability_test.csv`, `python_loop_test.csv`, `test_mpi.csv`, `test_write.txt`, `v18.csv`). They were used to check that the programs and the notebooks work and are **not** part of the final results.
* `team17_corrected.csv` is the final Team 17 data (100 steps, 1–16 threads).
* Result CSVs use the course's common column format:
  `cluster, problem, platform, implementation, dataset, size, threads_or_ranks, run_id, total_ms, kernel_ms, transfer_ms, throughput, error, power_w`
  (`throughput` = interactions per second; for MPI `transfer_ms` = communication time, for CUDA `transfer_ms` = host↔device copies, `threads_or_ranks` for CUDA = block size.)
* Version 1's results are in `MAP-Version1/nbody_results/results/` and `MAP-Version1/nbody_acc (1)/acc/` and have the same file layout. They were produced with the smaller settings listed in Section 3.

---

## 8. Results and findings (final version, Version 2)

All numbers below are taken from the CSV files in `MAP-Version2`. "G int/s" means 10⁹ particle-pair interactions per second (higher is better).

### 8.1 Team 17 – shared memory (OpenMP / Pthreads), 100 steps, FP64

| N | Sequential | Best OpenMP (`omp_sym_priv`) | Speedup |
|---|---|---|---|
| 4,096 | 0.71 G int/s | 2.55 G int/s (4 threads) | **≈ 3.6×** |
| 16,384 | 0.61 G int/s | 2.86 G int/s (16 threads; 2.81 with 4 threads) | **≈ 4.7×** |

* **Newton's 3rd law with private buffers** (`omp_sym_priv`) is the best variant: it halves the number of force evaluations. The plain `omp_full` reaches only ≈ 1.5 G int/s (≈ 2.5×).
* The Pthreads version of the symmetric method (`pthreads_sym`, ≈ 2.76 G int/s) is practically as fast as OpenMP.
* **`omp_sym_atomic` is much slower than sequential** (0.04–0.15 G int/s): atomic updates are far too expensive.
* **SoA is slightly faster than AoS** (0.61 vs 0.59 G int/s sequentially at N = 16,384).
* **FP32 is about 3.3× faster than FP64** sequentially (2.04 vs 0.61 G int/s at N = 16,384) thanks to wider SIMD.
* Speedup stops growing at about 4 threads (4→16 threads gives almost no gain); with more threads than physical cores, performance stays flat or even drops (at N = 4,096). The same happens for MPI below.
* At N = 65,536 only the sequential baseline (0.49 G int/s) is stored in the final CSV (the parallel runs were skipped there because they take very long), so scaling conclusions are based on N = 4,096 and 16,384.

### 8.2 Team 18 – distributed memory (MPI), FP64, 100 steps

Strong scaling at N = 16,384 (total time per run):

| Ranks | allgather | ring | ring_overlap | comm/comp ratio (ring) |
|---|---|---|---|---|
| 1 | 43.4 s | 45.6 s | 47.1 s | 0 |
| 2 | 27.0 s | 29.3 s | 28.7 s | 0.006 |
| 4 | 19.3 s | 18.8 s | 19.6 s | 0.06 |
| 8 | 20.8 s | 25.7 s | 20.9 s | 0.51 |
| 16 | 33.0 s | 35.3 s | 30.8 s | 0.84 |

* The best result is at **4 ranks (≈ 2.3× faster than 1 rank)**. Beyond that the ranks are **oversubscribed** (more ranks than physical cores), so the time rises again and the communication-to-computation ratio explodes (0.0001 → ~0.8 for ring at 16 ranks).
* `ring_overlap` helps when communication matters (8 and 16 ranks: 20.9 s vs 25.7 s and 30.8 s vs 35.3 s compared with the plain `ring`).
* **Weak scaling** (work per rank constant, N = 2048·√p) is poor: the time grows from ≈ 0.6–0.8 s (1 rank) to ≈ 8.6–10.1 s (16 ranks), where the ideal curve would stay flat. Again the oversubscription of cores and the growing communication are the cause.

### 8.3 Team 19 – CUDA on a Colab T4 GPU

| N | Naive FP32 | Tiled FP32 (block 128) | Tiled FP64 (block 128) |
|---|---|---|---|
| 4,096 | 522 GFLOP/s | 703 GFLOP/s | 70 GFLOP/s |
| 16,384 | 1,487 GFLOP/s | 1,514 GFLOP/s | 70 GFLOP/s |
| 65,536 | 1,688 GFLOP/s | 1,783 GFLOP/s | 87 GFLOP/s |
| 262,144 | 1,787 GFLOP/s | 1,882 GFLOP/s | (skipped) |
| 1,048,576 | – | **1,900 GFLOP/s** | (skipped) |

* **Tiling beats the naive kernel** (about +35 % at N = 4,096, +5 % at large N). Block sizes 64/128/256 differ by only a few percent.
* **GPU performance grows with N** and saturates near ≈ 1.9 TFLOP/s (FP32) because small problems cannot fill the GPU.
* **FP64 is about 20–25× slower than FP32** on the T4 (FP64 runs at 1/32 rate).
* **Loop unrolling (1, 4, 8)** makes no measurable difference (≈ 1,427–1,429 GFLOP/s).
* **Fast `rsqrt`** (N = 16,384, tiled): FP32 ≈ 1.9× faster (≈ 81 → ≈ 147 G int/s) and FP64 ≈ 1.4× faster, while the first-step force error changes only marginally (FP32: 6.34e-6 → 6.40e-6).
* In the combined accuracy study (N = 4,096) the GPU is **≈ 6× faster than the sequential CPU in FP64** (3.5 vs 0.55 G int/s) and **≈ 26× faster in FP32** (35.2 vs 1.37 G int/s); with fast rsqrt FP32 the GPU reaches 120 G int/s (≈ 88× the sequential FP32 CPU).

### 8.4 Team 20 – precision, accuracy and integration (N = 4,096, 10,000 steps)

Reference: `seq_fp64`. Values from `colab_acc/acc/summary_all_runs.csv`.

| Platform | Precision | Speed (G int/s) | Max abs. energy drift | Final RMS position error | First-step force error |
|---|---|---|---|---|---|
| seq | fp64 | 0.547 | 1.25e-6 | 0 (reference) | 0 |
| seq | fp32 | 1.368 | 1.18e-6 | 4.4e-5 | 5.8e-7 |
| seq | mixed | 0.420 | 1.26e-6 | 1.7e-7 | 2.2e-7 |
| openmp | fp64 | 0.783 | 1.25e-6 | 0 | 0 |
| openmp | fp32 | 2.683 | 1.18e-6 | 4.4e-5 | 5.8e-7 |
| openmp | mixed | 1.268 | 1.26e-6 | 1.7e-7 | 2.2e-7 |
| mpi (ring, 4 ranks) | fp64 | 1.359 | 1.25e-6 | 1.7e-14 | 1.9e-15 |
| mpi | fp32 | 4.295 | 1.21e-6 | 4.5e-5 | 4.0e-7 |
| mpi | mixed | 1.828 | 1.26e-6 | 1.7e-7 | 2.2e-7 |
| cuda | fp64 | 3.512 | 1.25e-6 | 2.3e-14 | 8.0e-16 |
| cuda | fp32 | 35.17 | 1.22e-6 | 4.5e-5 | 3.5e-6 |
| cuda | mixed | 6.762 | 1.26e-6 | 1.7e-7 | 2.2e-7 |
| cuda_rsqrt | fp64 | 4.966 | 1.25e-6 | 2.2e-14 | 1.1e-15 |
| cuda_rsqrt | fp32 | **120.0** | 1.24e-6 | 4.5e-5 | 3.5e-6 |
| cuda_rsqrt | mixed | 6.908 | 1.28e-6 | 2.4e-7 | 2.4e-7 |

**What we learned**

1. **Energy drift does not reveal the precision.** All 15 runs have the same max |ΔE/E₀| of about 1.2–1.3·10⁻⁶. This drift is dominated by the integrator error of velocity-Verlet with dt = 1e-3, not by round-off.
2. **Trajectory error does reveal the precision.** FP64 platforms agree with the reference to ~10⁻¹⁴ (RMS), **mixed** to ~10⁻⁷, and **FP32** to ~4–5·10⁻⁵ (max error ~8·10⁻⁴). Gravity is chaotic, so small differences grow with time (see `plot_trajectory_error.png`).
3. **Different platforms in the same precision behave almost the same:** sequential and OpenMP produce identical results (error exactly 0 in FP64, same FP32 values), and MPI and CUDA FP64 agree with the reference to ~1e-14 (only the summation order differs).
4. **Speed:**  FP32 gives the largest speedup. On the CPU, it achieves approximately 2.5–3.4× the FP64 performance. On the standard CUDA configuration, it achieves approximately 10× the FP64 performance. With fast rsqrt, the GPU FP32 speedup reaches approximately 24× compared with its FP64 counterpart. **Mixed** precision is only worthwhile where the pair-force math dominates: it was *slower* than FP64 on the sequential CPU (0.42 vs 0.55), ~1.6× faster than FP64 on OpenMP, ~1.35× on MPI and ~1.9× on the T4 GPU.
5. **Fast `rsqrt` on the GPU** gives a large speedup (FP32: 35 → 120 G int/s, FP64: 3.5 → 5.0) at no visible accuracy loss on the trajectory.
6. **Recommended configuration per platform** (rule: fastest precision with max |ΔE/E₀| ≤ 1e-5 **and** final RMS position error ≤ 1e-6; from `summary_recommendation.csv`):

| Platform | Recommended precision | Speed (G int/s) | Position error |
|---|---|---|---|
| seq | fp64 | 0.547 | 0 |
| openmp | mixed | 1.268 | 1.7e-7 |
| mpi | mixed | 1.828 | 1.7e-7 |
| cuda | mixed | 6.762 | 1.7e-7 |
| cuda_rsqrt | mixed | 6.908 | 2.4e-7 |

   **FP32 is the fastest everywhere but fails the position tolerance (4.5e-5 > 1e-6)**, so it is only suitable if a short simulation or a statistical result (e.g. energy) is enough. If a looser position tolerance is acceptable, `cuda_rsqrt` in FP32 is by far the fastest option.

7. **Limits of the study:** one problem size (N = 4,096), one random seed, one local PC and one Colab T4 GPU, a fixed dt = 1e-3, and a non-trivial amount of oversubscription for MPI/OpenMP on our PC (see 8.1 and 8.2). A smaller dt would be needed to see round-off in the energy drift. Absolute speeds depend on our hardware: CPU `<your CPU model / cores>`, GPU NVIDIA T4 (Colab).

---

## 9. Notes and known limitations

* Optional extras from the brief (Barnes–Hut tree code for Team 17, 2-D force decomposition for Team 18, cell-list Lennard-Jones code for Team 19) were **not** implemented.
* Nsight Compute profiling (step D7) is optional and often blocked on Colab; the built-in `PROFILE` output (registers per thread, shared memory, occupancy) is used instead.
* The local notebook contains test and debugging cells (checking WSL, paths, small test runs) in addition to the main experiment cells. The main steps are labelled A1, B1–B7, C1–C7, E1–E2 in the markdown cells.
* Version 1 is kept for completeness and to show how the work evolved. For grading and conclusions, please use Version 2.

---

## 10. Tools used

C++17 (`g++ -O3 -march=native`), OpenMP, Pthreads, OpenMPI, CUDA (`nvcc`), Python (pandas, numpy, matplotlib), Jupyter / Google Colab, WSL2.
