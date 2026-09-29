"""
Warm starts for the lifted (force-state) transcription, holding the physical (q, v, u)
guess FIXED and varying only how the force states are initialized.

The force dynamics is exactly affine in lambda: ABA is affine in the joint torque and
lambda enters the torque through J^T, so

    lambda_{t+1}^pred = A_t lambda_t + b_t ,   A_t = I + dt * dlambda_dot/dlambda ,

with A_t read from the model's own Fx block and b_t evaluated at lambda_t = 0. The
multiple-shooting force defect is therefore affine in the force trajectory, and

    min_{lambda_1..N}  sum_t ||A_t lambda_t + b_t - lambda_{t+1}||^2
                     + eps * sum_t ||lambda_t - lambda_t^anchor||^2      (lambda_0 fixed)

is a linear least-squares problem. eps -> inf returns the anchor-map guess used so far;
eps -> 0 returns the exactly dynamically consistent forward rollout. Intermediate eps
trades force-defect against how far the forces stray from the physically plausible
anchor values, which is what makes the initialization effect measurable rather than
all-or-nothing.
"""
import numpy as np


def affine_force_dynamics(models, xs, us, nf):
    """(A_t, b_t) per node, from the model's own derivatives"""
    A, b = [], []
    for m, x, u in zip(models, xs, us):
        d = m.createData()
        x0 = np.array(x, dtype=float).copy()
        x0[-nf:] = 0.0
        m.calc(d, x0, u)
        b.append(np.array(d.xnext)[-nf:].copy())
        m.calcDiff(d, x0, u)
        A.append(np.array(d.Fx)[-nf:, -nf:].copy())
    return A, b


def rollout_forces(models, xs, us, lam0, nf):
    """exactly dynamically consistent force trajectory (zero force defect)"""
    A, b = affine_force_dynamics(models, xs, us, nf)
    lam = [np.array(lam0, dtype=float).copy()]
    for t in range(len(us)):
        lam.append(A[t] @ lam[t] + b[t])
    return lam


def ls_forces(models, xs, us, lam0, lam_ref, nf, eps):
    """least-squares force trajectory between the anchor guess (eps large) and the
    exact rollout (eps -> 0); lambda_0 is fixed to the measured force"""
    N = len(us)
    A, b = affine_force_dynamics(models, xs, us, nf)
    n = nf * N                                    # unknowns: lambda_1 .. lambda_N
    H = np.zeros((n, n)); g = np.zeros(n)
    def sl(t):                                    # slice of lambda_t in the unknowns
        return slice((t - 1) * nf, t * nf)
    for t in range(N):
        # residual r_t = A_t lambda_t + b_t - lambda_{t+1}
        Jt = np.zeros((nf, n)); rt = b[t].copy()
        if t == 0:
            rt = rt + A[0] @ np.asarray(lam0, dtype=float)
        else:
            Jt[:, sl(t)] += A[t]
        Jt[:, sl(t + 1)] -= np.eye(nf)
        H += Jt.T @ Jt; g += Jt.T @ rt
    for t in range(1, N + 1):                     # eps * ||lambda_t - lambda_t^ref||^2
        H[sl(t), sl(t)] += eps * np.eye(nf)
        g[sl(t)] -= eps * np.asarray(lam_ref[t], dtype=float)
    sol = np.linalg.solve(H, -g)
    return [np.asarray(lam0, dtype=float).copy()] + [sol[sl(t)].copy() for t in range(1, N + 1)]


def defect(models, state, xs, us, nf):
    """(total, q/v, force) 1-norm of the multiple-shooting defect, and max|force defect|"""
    tot = qv = ff = 0.0; mx = 0.0
    for t, (m, x, u) in enumerate(zip(models, xs, us)):
        d = m.createData(); m.calc(d, x, u)
        gg = np.array(state.diff(xs[t + 1], np.array(d.xnext)))
        tot += np.abs(gg).sum(); qv += np.abs(gg[:-nf]).sum(); ff += np.abs(gg[-nf:]).sum()
        mx = max(mx, np.abs(gg[-nf:]).max())
    return tot, qv, ff, mx
