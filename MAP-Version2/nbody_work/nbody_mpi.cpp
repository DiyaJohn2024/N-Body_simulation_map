// =============================================================================
//  Team 18 / Problem 18 : Distributed N-body with MPI
//
//  Each rank owns n = N/p particles (N must be divisible by p).
//  variants (--variant):
//    allgather     MPI_Allgather of all positions every step               <- baseline
//    ring          ring exchange: blocks are passed to the next rank p-1 times
//                  (MPI_Sendrecv_replace), forces accumulated as blocks arrive
//    ring_overlap  ring with double buffering: MPI_Isend/Irecv of the NEXT block
//                  overlaps with the force computation on the CURRENT block
//  precision (--prec): fp64 | fp32 | mixed
//
//  Timing columns written to the CSV:
//     kernel_ms   = force-computation time (max over ranks)
//     transfer_ms = MPI communication time (max over ranks); for ring_overlap this is only the
//                   *exposed* (non-hidden) communication = posting + waiting time
// =============================================================================
#include "nbody_common.hpp"
#include <mpi.h>

using namespace nb;

template <class S> MPI_Datatype mpi_type();
template <> MPI_Datatype mpi_type<float>()  { return MPI_FLOAT;  }
template <> MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }

template <class M>
struct MpiSim {
    typedef typename M::S S;
    int N, n, rank, p; std::string variant; MPI_Comm comm = MPI_COMM_WORLD;
    std::vector<S> x, y, z, vx, vy, vz, ax, ay, az;   // local particles
    std::vector<S> mall;                               // masses of ALL particles (replicated, never changes)
    std::vector<S> blk0, blk1, posall;                 // 3n-long position blocks [x(n) y(n) z(n)]; posall = p*3n
    double t_comp = 0, t_comm = 0;

    MpiSim(int N_, const std::string& v) : N(N_), variant(v) {
        MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &p);
        if (v != "allgather" && v != "ring" && v != "ring_overlap") {
            if (rank == 0) fprintf(stderr, "unknown variant '%s'\n", v.c_str());
            MPI_Abort(comm, 1);
        }
        if (N % p != 0) {
            if (rank == 0) fprintf(stderr, "N=%d must be divisible by the number of ranks %d\n", N, p);
            MPI_Abort(comm, 1);
        }
        n = N / p;
        x.resize(n); y.resize(n); z.resize(n); vx.resize(n); vy.resize(n); vz.resize(n);
        ax.resize(n); ay.resize(n); az.resize(n); mall.resize(N);
        blk0.resize(3 * n); blk1.resize(3 * n); posall.resize((size_t)3 * N);
    }
    bool is_root() const { return rank == 0; }
    std::string impl() const { return variant + "_" + M::name(); }
    int par() const { return p; }
    void sync() { MPI_Barrier(comm); }
    void finalize() {}
    std::string notes() {
        char b[200]; snprintf(b, sizeof b, "MPI ranks=%d, particles per rank=%d, comm/comp ratio (last run, max over ranks) = %.3f", p, n, last_ratio);
        return b;
    }
    double last_ratio = 0;
    Timing timing() {
        double mine[2] = {t_comp, t_comm}, mx[2] = {0, 0};
        MPI_Reduce(mine, mx, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
        Timing t; t.kernel_ms = mx[0]; t.transfer_ms = mx[1];
        last_ratio = mx[0] > 0 ? mx[1] / mx[0] : 0;
        return t;
    }

    void load(const Particles& ic) {
        const int o = rank * n;
        for (int i = 0; i < n; ++i) {
            x[i] = (S)ic.x[o + i]; y[i] = (S)ic.y[o + i]; z[i] = (S)ic.z[o + i];
            vx[i] = (S)ic.vx[o + i]; vy[i] = (S)ic.vy[o + i]; vz[i] = (S)ic.vz[o + i];
        }
        for (int i = 0; i < N; ++i) mall[i] = (S)ic.m[i];
        md = ic.m;
        compute_forces();
        t_comp = t_comm = 0;
    }
    std::vector<double> md;

    void pack(std::vector<S>& b) {
        for (int i = 0; i < n; ++i) { b[i] = x[i]; b[n + i] = y[i]; b[2 * n + i] = z[i]; }
    }
    // add the pull of one source block (3n positions, n masses) onto all local particles
    void accumulate(const S* bp, const S* bm) {
        for (int i = 0; i < n; ++i) {
            S a = ax[i], b = ay[i], c = az[i];
            accel_one<M>(x[i], y[i], z[i], n, bp, bp + n, bp + 2 * n, bm, a, b, c);
            ax[i] = a; ay[i] = b; az[i] = c;
        }
    }
    void zero_acc() { for (int i = 0; i < n; ++i) { ax[i] = 0; ay[i] = 0; az[i] = 0; } }

    void compute_forces() {
        const MPI_Datatype T = mpi_type<S>();
        const int right = (rank + 1) % p, left = (rank - 1 + p) % p;
        zero_acc();
        if (variant == "allgather") {
            pack(blk0);
            double t0 = now_ms();
            MPI_Allgather(blk0.data(), 3 * n, T, posall.data(), 3 * n, T, comm);
            double t1 = now_ms();
            for (int b = 0; b < p; ++b) accumulate(posall.data() + (size_t)b * 3 * n, mall.data() + (size_t)b * n);
            double t2 = now_ms();
            t_comm += t1 - t0; t_comp += t2 - t1;
        } else if (variant == "ring") {
            pack(blk0);
            for (int k = 0; k < p; ++k) {
                const int src = ((rank - k) % p + p) % p;           // owner of the block we hold now
                double t0 = now_ms();
                accumulate(blk0.data(), mall.data() + (size_t)src * n);
                double t1 = now_ms(); t_comp += t1 - t0;
                if (k < p - 1) {
                    MPI_Sendrecv_replace(blk0.data(), 3 * n, T, right, 7, left, 7, comm, MPI_STATUS_IGNORE);
                    t_comm += now_ms() - t1;
                }
            }
        } else {  // ring_overlap
            std::vector<S>* buf[2] = {&blk0, &blk1};
            pack(blk0);
            for (int k = 0; k < p; ++k) {
                std::vector<S>& cur = *buf[k & 1]; std::vector<S>& nxt = *buf[(k & 1) ^ 1];
                const int src = ((rank - k) % p + p) % p;
                MPI_Request rq[2]; double tc = 0;
                if (k < p - 1) {
                    double t0 = now_ms();
                    MPI_Irecv(nxt.data(), 3 * n, T, left, 8, comm, &rq[0]);
                    MPI_Isend(cur.data(), 3 * n, T, right, 8, comm, &rq[1]);
                    tc += now_ms() - t0;
                }
                double t1 = now_ms();
                accumulate(cur.data(), mall.data() + (size_t)src * n);      // compute while messages are in flight
                double t2 = now_ms(); t_comp += t2 - t1;
                if (k < p - 1) { MPI_Waitall(2, rq, MPI_STATUSES_IGNORE); tc += now_ms() - t2; }
                t_comm += tc;
            }
        }
    }

    void advance(int steps) {
        const S dt = (S)DT, hdt = (S)(0.5 * DT);
        for (int s = 0; s < steps; ++s) {
            for (int i = 0; i < n; ++i) {
                vx[i] += hdt * ax[i]; vy[i] += hdt * ay[i]; vz[i] += hdt * az[i];
                x[i] += dt * vx[i];   y[i] += dt * vy[i];   z[i] += dt * vz[i];
            }
            compute_forces();
            for (int i = 0; i < n; ++i) { vx[i] += hdt * ax[i]; vy[i] += hdt * ay[i]; vz[i] += hdt * az[i]; }
        }
    }

    void gather_one(const std::vector<S>& src, std::vector<double>& dst) {
        std::vector<double> tmp(n); for (int i = 0; i < n; ++i) tmp[i] = (double)src[i];
        if (rank == 0) dst.resize(N);
        MPI_Gather(tmp.data(), n, MPI_DOUBLE, dst.data(), n, MPI_DOUBLE, 0, comm);
    }
    bool gather(Particles& o) {
        o.n = N; o.m = md;
        gather_one(x, o.x); gather_one(y, o.y); gather_one(z, o.z);
        gather_one(vx, o.vx); gather_one(vy, o.vy); gather_one(vz, o.vz);
        return rank == 0;
    }
    bool gather_accel(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c) {
        gather_one(ax, a); gather_one(ay, b); gather_one(az, c);
        return rank == 0;
    }
};

template <class M>
int run(const Args& a) {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string v = a.str("variant", "ring");
    int n = a.i("n", 4096);
    MpiSim<M> sim(n, v);
    if (rank == 0) printf("CONFIG  variant=%s prec=%s N=%d steps=%d ranks=%d\n", v.c_str(), M::name(), n, a.i("steps", 100), sim.p);
    return drive(sim, a, 18, "cpu_mpi", M::name());
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Args a(argc, argv);
    std::string p = lower(a.str("prec", "fp64"));
    int rc = 1;
    if (p == "fp64")       rc = run<M64>(a);
    else if (p == "fp32")  rc = run<M32>(a);
    else if (p == "mixed") rc = run<MMix>(a);
    else fprintf(stderr, "--prec must be fp64, fp32 or mixed\n");
    MPI_Finalize();
    return rc;
}
