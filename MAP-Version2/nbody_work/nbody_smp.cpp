// =============================================================================
//  Team 17 / Problem 17 : Shared-memory N-body  (sequential, OpenMP, Pthreads)
//
//  variants (--variant):
//    seq             sequential all-pairs, structure-of-arrays (SoA)   <- baseline
//    seq_aos         sequential, array-of-structures layout
//    omp_full        OpenMP over particles, each thread computes the FULL force on its own particles (SoA)
//    omp_full_aos    same, AoS layout
//    omp_sym_priv    Newton's 3rd law, per-thread private force buffers + reduction
//    omp_sym_atomic  Newton's 3rd law, '#pragma omp atomic' updates
//    pthreads        Pthreads version of omp_full (row blocks, create/join every force evaluation)
//    pthreads_sym    Pthreads version of omp_sym_priv (the fastest OpenMP variant): Newton's 3rd law,
//                    per-thread private buffers, 2nd create/join phase does the reduction
//  precision (--prec): fp64 | fp32 | mixed
// =============================================================================
#include "nbody_common.hpp"
#include <omp.h>
#include <pthread.h>

using namespace nb;

template <class S> struct Body { S x, y, z, m; };

template <class M>
struct SmpSim {
    typedef typename M::S S; typedef typename M::F F;
    int n, T; std::string variant; bool use_omp;
    std::vector<S> x, y, z, vx, vy, vz, m, ax, ay, az, priv;
    std::vector<Body<S>> aos;
    std::vector<double> md;
    double t_create = 0, t_join = 0;

    SmpSim(int n_, int T_, const std::string& v) : n(n_), T(T_), variant(v) {
        static const char* ok[] = {"seq","seq_aos","omp_full","omp_full_aos","omp_sym_priv","omp_sym_atomic","pthreads","pthreads_sym"};
        bool found = false; for (auto s : ok) if (v == s) found = true;
        if (!found) { fprintf(stderr, "unknown variant '%s'\n", v.c_str()); exit(1); }
        use_omp = v.rfind("omp", 0) == 0;
        x.resize(n); y.resize(n); z.resize(n); vx.resize(n); vy.resize(n); vz.resize(n); m.resize(n);
        ax.resize(n); ay.resize(n); az.resize(n);
        if (v == "seq_aos" || v == "omp_full_aos") aos.resize(n);
        if (v == "omp_sym_priv" || v == "pthreads_sym") priv.resize((size_t)T * 3 * n);
    }
    bool is_root() const { return true; }
    std::string impl() const { return variant + "_" + M::name(); }
    int par() const { return T; }
    void sync() {}
    void finalize() {}
    Timing timing() { return Timing(); }
    std::string notes() {
        if (variant.rfind("pthreads", 0) != 0) return "";
        char b[160]; snprintf(b, sizeof b, "pthreads: total thread-create time %.3f ms, total join/wait time %.3f ms (last run)", t_create, t_join);
        return b;
    }

    void load(const Particles& ic) {
        for (int i = 0; i < n; ++i) {
            x[i] = (S)ic.x[i]; y[i] = (S)ic.y[i]; z[i] = (S)ic.z[i];
            vx[i] = (S)ic.vx[i]; vy[i] = (S)ic.vy[i]; vz[i] = (S)ic.vz[i]; m[i] = (S)ic.m[i];
        }
        md = ic.m;
        compute_forces();
        t_create = t_join = 0;
    }

    // ---------------- force kernels ----------------
    void range_full(int i0, int i1) {                         // SoA, full force on particles [i0,i1)
        const S *X = x.data(), *Y = y.data(), *Z = z.data(), *Mm = m.data();
        for (int i = i0; i < i1; ++i) {
            S a = 0, b = 0, c = 0;
            accel_one<M>(X[i], Y[i], Z[i], n, X, Y, Z, Mm, a, b, c);
            ax[i] = a; ay[i] = b; az[i] = c;
        }
    }
    void forces_full_soa(bool par) {
        const S *X = x.data(), *Y = y.data(), *Z = z.data(), *Mm = m.data();
        #pragma omp parallel for schedule(static) num_threads(T) if(par)
        for (int i = 0; i < n; ++i) {
            S a = 0, b = 0, c = 0;
            accel_one<M>(X[i], Y[i], Z[i], n, X, Y, Z, Mm, a, b, c);
            ax[i] = a; ay[i] = b; az[i] = c;
        }
    }
    void forces_full_aos(bool par) {
        for (int i = 0; i < n; ++i) aos[i] = Body<S>{x[i], y[i], z[i], m[i]};   // O(N) repack (negligible vs O(N^2))
        const Body<S>* B = aos.data();
        #pragma omp parallel for schedule(static) num_threads(T) if(par)
        for (int i = 0; i < n; ++i) {
            S xi = B[i].x, yi = B[i].y, zi = B[i].z, sx = 0, sy = 0, sz = 0;
            #pragma omp simd reduction(+:sx,sy,sz)
            for (int j = 0; j < n; ++j) {
                F dx = (F)(B[j].x - xi), dy = (F)(B[j].y - yi), dz = (F)(B[j].z - zi);
                F r2  = dx * dx + dy * dy + dz * dz + (F)EPS2;
                F inv = (F)1 / std::sqrt(r2);
                F s   = (F)B[j].m * (inv * inv * inv);
                sx += (S)(s * dx); sy += (S)(s * dy); sz += (S)(s * dz);
            }
            ax[i] = sx; ay[i] = sy; az[i] = sz;
        }
    }
    void forces_sym_priv() {
        const S *X = x.data(), *Y = y.data(), *Z = z.data(), *Mm = m.data();
        #pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num(), nt = omp_get_num_threads();
            S* bx = priv.data() + (size_t)t * 3 * n; S* by = bx + n; S* bz = by + n;
            for (int i = 0; i < 3 * n; ++i) bx[i] = 0;                      // own buffer only -> no race
            #pragma omp for schedule(dynamic, 8)                           // triangular loop -> dynamic
            for (int i = 0; i < n - 1; ++i) {
                S xi = X[i], yi = Y[i], zi = Z[i], mi = Mm[i], sx = 0, sy = 0, sz = 0;
                #pragma omp simd reduction(+:sx,sy,sz)
                for (int j = i + 1; j < n; ++j) {
                    F dx = (F)(X[j] - xi), dy = (F)(Y[j] - yi), dz = (F)(Z[j] - zi);
                    F r2  = dx * dx + dy * dy + dz * dz + (F)EPS2;
                    F inv = (F)1 / std::sqrt(r2);
                    F s   = inv * inv * inv;
                    F fx = s * dx, fy = s * dy, fz = s * dz;
                    sx += (S)((F)Mm[j] * fx); sy += (S)((F)Mm[j] * fy); sz += (S)((F)Mm[j] * fz);
                    bx[j] -= (S)((F)mi * fx); by[j] -= (S)((F)mi * fy); bz[j] -= (S)((F)mi * fz);
                }
                bx[i] += sx; by[i] += sy; bz[i] += sz;
            }                                                              // implicit barrier
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {                                  // reduction over thread buffers
                S a = 0, b = 0, c = 0;
                for (int q = 0; q < nt; ++q) {
                    const S* p = priv.data() + (size_t)q * 3 * n;
                    a += p[i]; b += p[n + i]; c += p[2 * n + i];
                }
                ax[i] = a; ay[i] = b; az[i] = c;
            }
        }
    }
    void forces_sym_atomic() {
        const S *X = x.data(), *Y = y.data(), *Z = z.data(), *Mm = m.data();
        S *AX = ax.data(), *AY = ay.data(), *AZ = az.data();
        for (int i = 0; i < n; ++i) { AX[i] = 0; AY[i] = 0; AZ[i] = 0; }
        #pragma omp parallel for schedule(dynamic, 8) num_threads(T)
        for (int i = 0; i < n - 1; ++i) {
            S xi = X[i], yi = Y[i], zi = Z[i], mi = Mm[i], sx = 0, sy = 0, sz = 0;
            for (int j = i + 1; j < n; ++j) {
                F dx = (F)(X[j] - xi), dy = (F)(Y[j] - yi), dz = (F)(Z[j] - zi);
                F r2  = dx * dx + dy * dy + dz * dz + (F)EPS2;
                F inv = (F)1 / std::sqrt(r2);
                F s   = inv * inv * inv;
                F fx = s * dx, fy = s * dy, fz = s * dz;
                sx += (S)((F)Mm[j] * fx); sy += (S)((F)Mm[j] * fy); sz += (S)((F)Mm[j] * fz);
                S ux = (S)((F)mi * fx), uy = (S)((F)mi * fy), uz = (S)((F)mi * fz);
                #pragma omp atomic
                AX[j] -= ux;
                #pragma omp atomic
                AY[j] -= uy;
                #pragma omp atomic
                AZ[j] -= uz;
            }
            #pragma omp atomic
            AX[i] += sx;
            #pragma omp atomic
            AY[i] += sy;
            #pragma omp atomic
            AZ[i] += sz;
        }
    }
    struct Job { SmpSim* s; int i0, i1; };
    static void* worker(void* p) { Job* j = (Job*)p; j->s->range_full(j->i0, j->i1); return nullptr; }
    void forces_pthreads() {
        std::vector<pthread_t> th(T); std::vector<Job> jobs(T);
        double t0 = now_ms();
        for (int t = 0; t < T; ++t) {
            jobs[t] = Job{this, (int)((long)n * t / T), (int)((long)n * (t + 1) / T)};
            pthread_create(&th[t], nullptr, &SmpSim::worker, &jobs[t]);
        }
        double t1 = now_ms();
        for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
        double t2 = now_ms();
        t_create += t1 - t0; t_join += t2 - t1;
    }
    // ---- pthreads_sym: thread t takes rows t, t+T, t+2T... (round-robin balances the triangular loop) ----
    void sym_rows(int t) {
        const S *X = x.data(), *Y = y.data(), *Z = z.data(), *Mm = m.data();
        S* bx = priv.data() + (size_t)t * 3 * n; S* by = bx + n; S* bz = by + n;
        for (int i = 0; i < 3 * n; ++i) bx[i] = 0;                         // own buffer only -> no race
        for (int i = t; i < n - 1; i += T) {
            S xi = X[i], yi = Y[i], zi = Z[i], mi = Mm[i], sx = 0, sy = 0, sz = 0;
            #pragma omp simd reduction(+:sx,sy,sz)
            for (int j = i + 1; j < n; ++j) {
                F dx = (F)(X[j] - xi), dy = (F)(Y[j] - yi), dz = (F)(Z[j] - zi);
                F r2  = dx * dx + dy * dy + dz * dz + (F)EPS2;
                F inv = (F)1 / std::sqrt(r2);
                F s   = inv * inv * inv;
                F fx = s * dx, fy = s * dy, fz = s * dz;
                sx += (S)((F)Mm[j] * fx); sy += (S)((F)Mm[j] * fy); sz += (S)((F)Mm[j] * fz);
                bx[j] -= (S)((F)mi * fx); by[j] -= (S)((F)mi * fy); bz[j] -= (S)((F)mi * fz);
            }
            bx[i] += sx; by[i] += sy; bz[i] += sz;
        }
    }
    void sym_reduce(int t) {                                               // sum the T private buffers
        for (int i = (int)((long)n * t / T); i < (int)((long)n * (t + 1) / T); ++i) {
            S a = 0, b = 0, c = 0;
            for (int q = 0; q < T; ++q) { const S* p = priv.data() + (size_t)q * 3 * n; a += p[i]; b += p[n + i]; c += p[2 * n + i]; }
            ax[i] = a; ay[i] = b; az[i] = c;
        }
    }
    struct SymJob { SmpSim* s; int t, phase; };
    static void* sym_worker(void* p) { SymJob* j = (SymJob*)p; if (j->phase == 0) j->s->sym_rows(j->t); else j->s->sym_reduce(j->t); return nullptr; }
    void forces_pthreads_sym() {
        std::vector<pthread_t> th(T); std::vector<SymJob> jobs(T);
        for (int phase = 0; phase < 2; ++phase) {                          // phase 0: forces, phase 1: reduction
            double t0 = now_ms();
            for (int t = 0; t < T; ++t) { jobs[t] = SymJob{this, t, phase}; pthread_create(&th[t], nullptr, &SmpSim::sym_worker, &jobs[t]); }
            double t1 = now_ms();
            for (int t = 0; t < T; ++t) pthread_join(th[t], nullptr);
            t_create += t1 - t0; t_join += now_ms() - t1;
        }
    }
    void compute_forces() {
        if      (variant == "seq")            forces_full_soa(false);
        else if (variant == "seq_aos")        forces_full_aos(false);
        else if (variant == "omp_full")       forces_full_soa(true);
        else if (variant == "omp_full_aos")   forces_full_aos(true);
        else if (variant == "omp_sym_priv")   forces_sym_priv();
        else if (variant == "omp_sym_atomic") forces_sym_atomic();
        else if (variant == "pthreads_sym")   forces_pthreads_sym();
        else                                  forces_pthreads();
    }

    // ---------------- velocity-Verlet ----------------
    void advance(int steps) {
        const S dt = (S)DT, hdt = (S)(0.5 * DT);
        const bool par = use_omp;
        for (int s = 0; s < steps; ++s) {
            #pragma omp parallel for schedule(static) num_threads(T) if(par)
            for (int i = 0; i < n; ++i) {
                vx[i] += hdt * ax[i]; vy[i] += hdt * ay[i]; vz[i] += hdt * az[i];
                x[i] += dt * vx[i];   y[i] += dt * vy[i];   z[i] += dt * vz[i];
            }
            compute_forces();
            #pragma omp parallel for schedule(static) num_threads(T) if(par)
            for (int i = 0; i < n; ++i) { vx[i] += hdt * ax[i]; vy[i] += hdt * ay[i]; vz[i] += hdt * az[i]; }
        }
    }

    bool gather(Particles& o) {
        o.n = n; o.m = md;
        o.x.assign(x.begin(), x.end()); o.y.assign(y.begin(), y.end()); o.z.assign(z.begin(), z.end());
        o.vx.assign(vx.begin(), vx.end()); o.vy.assign(vy.begin(), vy.end()); o.vz.assign(vz.begin(), vz.end());
        return true;
    }
    bool gather_accel(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c) {
        a.assign(ax.begin(), ax.end()); b.assign(ay.begin(), ay.end()); c.assign(az.begin(), az.end()); return true;
    }
};

template <class M>
int run(const Args& a) {
    std::string v = a.str("variant", "omp_full");
    int n = a.i("n", 4096);
    int T = a.i("threads", omp_get_max_threads());
    if (v.rfind("seq", 0) == 0) T = 1;
    if (T < 1) T = 1;
    SmpSim<M> sim(n, T, v);
    std::string plat = v.rfind("seq", 0) == 0 ? "cpu_seq" : (v.rfind("pthreads", 0) == 0 ? "cpu_pthreads" : "cpu_openmp");
    printf("CONFIG  variant=%s prec=%s N=%d steps=%d threads=%d (hardware threads visible: %d)\n",
           v.c_str(), M::name(), n, a.i("steps", 100), T, omp_get_max_threads());
    return drive(sim, a, 17, plat, M::name());
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string p = lower(a.str("prec", "fp64"));
    if (p == "fp64")  return run<M64>(a);
    if (p == "fp32")  return run<M32>(a);
    if (p == "mixed") return run<MMix>(a);
    fprintf(stderr, "--prec must be fp64, fp32 or mixed\n"); return 1;
}
