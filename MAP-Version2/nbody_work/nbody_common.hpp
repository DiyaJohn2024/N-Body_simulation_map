// =============================================================================
//  nbody_common.hpp  --  shared by Team 17 (OpenMP/Pthreads), Team 18 (MPI),
//  Team 19 (CUDA) and used by Team 20 (precision study).
//
//  EVERY platform uses exactly the same:
//    * physics   : softened gravity, G = 1, eps = 0.05, masses m_i = 1/N
//    * initial conditions : deterministic RNG (seed 42) -> identical on all platforms
//    * integrator: velocity-Verlet (kick-drift-kick), dt = 1e-3
//    * output    : results CSV (common course format), energy CSV, trajectory file
//
//  Precision modes (template "Mode" types):
//      fp64  : state S=double, pair-force arithmetic F=double
//      fp32  : state S=float , pair-force arithmetic F=float
//      mixed : state S=double, pair-force arithmetic F=float, accumulation in double
//              (dx = xj-xi is formed in double, THEN cast to float; the float pair
//               force is added to a double accumulator)
//  Diagnostics (energy, verification) are ALWAYS computed in FP64.
// =============================================================================
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

#if defined(__CUDACC__)
  #define NB_PRAGMA(x)
#else
  #define NB_PRAGMA(x) _Pragma(x)
#endif

namespace nb {

// ----------------------------- physics constants ----------------------------
constexpr double G    = 1.0;
constexpr double EPS2 = 0.05 * 0.05;   // softening length squared
constexpr double DT   = 1.0e-3;        // fixed time step
constexpr double V0   = 0.5;           // initial velocity half-range
constexpr uint64_t SEED = 42;

// ----------------------------- precision modes ------------------------------
struct M64  { typedef double S; typedef double F; static const char* name() { return "fp64";  } };
struct M32  { typedef float  S; typedef float  F; static const char* name() { return "fp32";  } };
struct MMix { typedef double S; typedef float  F; static const char* name() { return "mixed"; } };

inline std::string lower(std::string s) { for (auto& c : s) c = (char)tolower(c); return s; }

// ----------------------------- timing ---------------------------------------
inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ----------------------------- argument parsing -----------------------------
struct Args {
    std::vector<std::pair<std::string, std::string>> kv;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) == 0) {
                k = k.substr(2);
                std::string v = "1";
                if (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) v = argv[++i];
                kv.push_back({k, v});
            }
        }
    }
    bool has(const std::string& k) const { for (auto& p : kv) if (p.first == k) return true; return false; }
    std::string str(const std::string& k, const std::string& d) const { for (auto& p : kv) if (p.first == k) return p.second; return d; }
    int    i(const std::string& k, int d)    const { return has(k) ? atoi(str(k, "").c_str()) : d; }
    double d(const std::string& k, double dv) const { return has(k) ? atof(str(k, "").c_str()) : dv; }
};

// ----------------------------- initial conditions ---------------------------
struct Particles {
    int n = 0;
    std::vector<double> x, y, z, vx, vy, vz, m;
};

inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
inline double u01(uint64_t& s) { return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0); }

// positions uniform in [-1,1]^3, velocities uniform in [-V0,V0]^3, m = 1/N.
inline Particles make_ic(int n, uint64_t seed = SEED) {
    Particles p; p.n = n;
    p.x.resize(n); p.y.resize(n); p.z.resize(n);
    p.vx.resize(n); p.vy.resize(n); p.vz.resize(n); p.m.assign(n, 1.0 / n);
    uint64_t s = seed;
    for (int i = 0; i < n; ++i) {
        p.x[i]  = 2.0 * u01(s) - 1.0;  p.y[i]  = 2.0 * u01(s) - 1.0;  p.z[i]  = 2.0 * u01(s) - 1.0;
        p.vx[i] = V0 * (2.0 * u01(s) - 1.0);
        p.vy[i] = V0 * (2.0 * u01(s) - 1.0);
        p.vz[i] = V0 * (2.0 * u01(s) - 1.0);
    }
    return p;
}

// ----------------------------- the force kernel (host) ----------------------
// Adds to (ax,ay,az) the acceleration on particle i (position xi,yi,zi) caused by
// the nj source particles xj,yj,zj,mj.  Self-interaction is harmless (dx=0).
template <class M>
inline void accel_one(typename M::S xi, typename M::S yi, typename M::S zi, int nj,
                      const typename M::S* xj, const typename M::S* yj, const typename M::S* zj,
                      const typename M::S* mj,
                      typename M::S& ax, typename M::S& ay, typename M::S& az) {
    typedef typename M::S S; typedef typename M::F F;
    S sx = 0, sy = 0, sz = 0;
    NB_PRAGMA("omp simd reduction(+:sx,sy,sz)")
    for (int j = 0; j < nj; ++j) {
        F dx = (F)(xj[j] - xi), dy = (F)(yj[j] - yi), dz = (F)(zj[j] - zi);
        F r2  = dx * dx + dy * dy + dz * dz + (F)EPS2;
        F inv = (F)1 / std::sqrt(r2);
        F s   = (F)mj[j] * (inv * inv * inv);
        sx += (S)(s * dx); sy += (S)(s * dy); sz += (S)(s * dz);
    }
    ax += sx; ay += sy; az += sz;
}

// ----------------------------- FP64 diagnostics -----------------------------
struct Energy { double ke, pe, etot; };

inline Energy energy(const Particles& p) {
    const int n = p.n;
    double ke = 0, pe = 0;
    for (int i = 0; i < n; ++i) ke += 0.5 * p.m[i] * (p.vx[i]*p.vx[i] + p.vy[i]*p.vy[i] + p.vz[i]*p.vz[i]);
    for (int i = 0; i < n; ++i) {
        double s = 0;
        NB_PRAGMA("omp simd reduction(+:s)")
        for (int j = i + 1; j < n; ++j) {
            double dx = p.x[j]-p.x[i], dy = p.y[j]-p.y[i], dz = p.z[j]-p.z[i];
            s += p.m[j] / std::sqrt(dx*dx + dy*dy + dz*dz + EPS2);
        }
        pe -= G * p.m[i] * s;
    }
    return {ke, pe, ke + pe};
}

// Reference accelerations: sequential, FP64.
inline void ref_accel(const Particles& p, std::vector<double>& ax, std::vector<double>& ay, std::vector<double>& az) {
    const int n = p.n;
    ax.assign(n, 0); ay.assign(n, 0); az.assign(n, 0);
    for (int i = 0; i < n; ++i)
        accel_one<M64>(p.x[i], p.y[i], p.z[i], n, p.x.data(), p.y.data(), p.z.data(), p.m.data(), ax[i], ay[i], az[i]);
}

// max over particles of |a - a_ref| / |a_ref|
inline double max_rel_force_error(const std::vector<double>& ax, const std::vector<double>& ay, const std::vector<double>& az,
                                  const std::vector<double>& rx, const std::vector<double>& ry, const std::vector<double>& rz) {
    double worst = 0;
    for (size_t i = 0; i < ax.size(); ++i) {
        double ex = ax[i]-rx[i], ey = ay[i]-ry[i], ez = az[i]-rz[i];
        double num = std::sqrt(ex*ex + ey*ey + ez*ez);
        double den = std::sqrt(rx[i]*rx[i] + ry[i]*ry[i] + rz[i]*rz[i]);
        worst = std::max(worst, num / (den > 0 ? den : 1.0));
    }
    return worst;
}

// ----------------------------- results CSV (course format) ------------------
inline void csv_row(const std::string& path, int problem, const std::string& platform, const std::string& impl,
                    const std::string& dataset, int size, int par, int run_id,
                    double total_ms, double kernel_ms, double transfer_ms, double thr, double err) {
    bool exists = false;
    if (FILE* t = fopen(path.c_str(), "r")) { fseek(t, 0, SEEK_END); exists = ftell(t) > 0; fclose(t); }
    FILE* f = fopen(path.c_str(), "a");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return; }
    if (!exists) fprintf(f, "cluster,problem,platform,implementation,dataset,size,threads_or_ranks,run_id,total_ms,kernel_ms,transfer_ms,throughput,error,power_w\n");
    char e[64] = ""; if (err >= 0) snprintf(e, sizeof e, "%.3e", err);
    fprintf(f, "E,%d,%s,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6e,%s,\n", problem, platform.c_str(), impl.c_str(),
            dataset.c_str(), size, par, run_id, total_ms, kernel_ms, transfer_ms, thr, e);
    fclose(f);
}

// ----------------------------- the driver -----------------------------------
// A "Sim" backend must provide:
//   bool is_root(); std::string impl(); int par();
//   void load(const Particles&);      // set state + compute initial accelerations, reset timers
//   void advance(int steps);          // velocity-Verlet steps
//   void sync(); void finalize();     // barrier / copy-back (finalize = timed D2H for GPU)
//   Timing timing();                  // kernel_ms<0 means "same as step wall time"
//   bool gather(Particles&);          // full FP64 state on root (returns true on root)
//   bool gather_accel(vector<double>&,vector<double>&,vector<double>&);
//   std::string notes();
struct Timing { double kernel_ms = -1, transfer_ms = 0, extra_ms = 0; };

template <class Sim>
int drive(Sim& sim, const Args& a, int problem, const std::string& platform, const char* prec_name) {
    const int n = a.i("n", 4096), steps = a.i("steps", 100), runs = a.i("runs", 5);
    const int eevery = a.i("energy-every", 0);
    const std::string csv = a.str("csv", "results.csv"), out = a.str("out", "run");
    const bool root = sim.is_root();
    const std::string impl = sim.impl();
    Particles ic = make_ic(n);

    // ---- first-step verification against the sequential FP64 reference ----
    double err = -1;
    if (a.i("verify", 1) && n <= a.i("verify-max", 65536)) {
        sim.load(ic);
        std::vector<double> ax, ay, az;
        bool r = sim.gather_accel(ax, ay, az);
        if (r) {
            std::vector<double> rx, ry, rz; ref_accel(ic, rx, ry, rz);
            err = max_rel_force_error(ax, ay, az, rx, ry, rz);
            printf("VERIFY  impl=%s  N=%d  max_rel_force_error_vs_seq_fp64 = %.3e\n", impl.c_str(), n, err);
        }
    }

    if (eevery <= 0) {
        // ------------------------- performance mode -------------------------
        std::vector<double> tot, thr;
        for (int r = 0; r <= runs; ++r) {            // r = 0 is the warm-up run (discarded)
            sim.load(ic); sim.sync();
            double t0 = now_ms();
            sim.advance(steps);
            sim.sync();
            double step_ms = now_ms() - t0;
            sim.finalize();
            Timing t = sim.timing();
            double total = step_ms + t.extra_ms;
            double kern  = t.kernel_ms >= 0 ? t.kernel_ms : step_ms - t.transfer_ms;
            double ips   = (double)n * n * steps / (step_ms * 1e-3);   // N^2 pair interactions / s
            if (r > 0 && root) {
                csv_row(csv, problem, platform, impl, "gravity_steps" + std::to_string(steps), n, sim.par(), r, total, kern, t.transfer_ms, ips, err);
                tot.push_back(total); thr.push_back(ips);
            }
        }
        if (root) {
            double mt = 0, mi = 0; for (double v : tot) mt += v; for (double v : thr) mi += v;
            mt /= tot.size(); mi /= thr.size();
            double st = 0, si = 0; for (double v : tot) st += (v-mt)*(v-mt); for (double v : thr) si += (v-mi)*(v-mi);
            st = std::sqrt(st / std::max<size_t>(1, tot.size()-1)); si = std::sqrt(si / std::max<size_t>(1, thr.size()-1));
            printf("RESULT  %-28s N=%-7d steps=%-5d par=%-3d total_ms=%.2f+-%.2f  Ginteractions/s=%.4f+-%.4f  GFLOP/s(20 flop/int)=%.2f  [%d timed runs]\n",
                   impl.c_str(), n, steps, sim.par(), mt, st, mi/1e9, si/1e9, 20.0*mi/1e9, (int)tot.size());
            std::string nt = sim.notes(); if (!nt.empty()) printf("NOTES   %s\n", nt.c_str());
            printf("CSV rows appended to %s\n", csv.c_str());
        }
        return 0;
    }

    // ------------------------- accuracy / energy mode -----------------------
    sim.load(ic);
    FILE* fe = nullptr; std::vector<double> snaps; int nsnap = 0; double E0 = 0, maxdrift = 0, adv_ms = 0;
    if (root) { fe = fopen((out + "_energy.csv").c_str(), "w"); fprintf(fe, "step,time,ke,pe,etot,rel_drift\n"); }
    auto sample = [&](int step) {
        Particles s; if (!sim.gather(s)) return;
        Energy e = energy(s);
        if (step == 0) E0 = e.etot;
        double drift = (e.etot - E0) / std::fabs(E0); maxdrift = std::max(maxdrift, std::fabs(drift));
        fprintf(fe, "%d,%.9g,%.17g,%.17g,%.17g,%.6e\n", step, step * DT, e.ke, e.pe, e.etot, drift);
        snaps.insert(snaps.end(), s.x.begin(), s.x.end());
        snaps.insert(snaps.end(), s.y.begin(), s.y.end());
        snaps.insert(snaps.end(), s.z.begin(), s.z.end());
        ++nsnap;
    };
    sample(0);
    for (int s = 0; s < steps; s += eevery) {
        int k = std::min(eevery, steps - s);
        sim.sync(); double t0 = now_ms();
        sim.advance(k);
        sim.sync(); adv_ms += now_ms() - t0;
        sample(s + k);
    }
    if (root) {
        fclose(fe);
        FILE* ft = fopen((out + "_traj.bin").c_str(), "wb");      // header: int64 n, int64 nsnap; then nsnap*(x[n],y[n],z[n]) doubles
        int64_t h[2] = {n, nsnap}; fwrite(h, sizeof(int64_t), 2, ft);
        fwrite(snaps.data(), sizeof(double), snaps.size(), ft); fclose(ft);
        double ips = (double)n * n * steps / (adv_ms * 1e-3);
        csv_row(csv, problem, platform, impl, "accuracy_steps" + std::to_string(steps), n, sim.par(), 1, adv_ms, adv_ms, 0, ips, err);
        printf("ACCURACY impl=%s N=%d steps=%d  advance_time=%.1f ms  Ginteractions/s=%.4f  max|dE/E0|=%.3e\n",
               impl.c_str(), n, steps, adv_ms, ips / 1e9, maxdrift);
        printf("wrote %s_energy.csv and %s_traj.bin\n", out.c_str(), out.c_str());
    }
    (void)prec_name;
    return 0;
}

}  // namespace nb
