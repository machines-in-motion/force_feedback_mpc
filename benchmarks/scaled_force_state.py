"""
Nondimensionalization of the lifted force state: lambda_bar = lambda / F_s.

This is an exact linear change of variables applied to the PRODUCTION models, not a
reformulation: the force block of the state is Euclidean, so `integrate`/`diff` are
unchanged and the production state object can be reused as-is. Everything else is the
similarity transform with S = blkdiag(I_2nv, F_s * I_nf):

    x_phys = S x_bar            xnext_bar = S^-1 xnext_phys
    Fx_bar = S^-1 Fx S          Fu_bar    = S^-1 Fu
    Lx_bar = S^T Lx             Lxx_bar   = S^T Lxx S        Lxu_bar = S^T Lxu
    Gx_bar = Gx S               cost, g, Lu, Luu, Gu unchanged

The NLP, its stationary points and its constraint set are therefore identical; only the
numerical representation handed to the solver changes.
"""
import numpy as np
import crocoddyl


class IAMScaledForce(crocoddyl.ActionModelAbstract):
    def __init__(self, inner, Fs, nf=15):
        crocoddyl.ActionModelAbstract.__init__(
            self, inner.state, inner.nu, inner.nr, inner.ng, inner.nh
        )
        self.inner = inner
        self.Fs = float(Fs)
        self.nf = nf
        self.ndx_x = inner.state.ndx - nf          # 2 nv
        if inner.ng:
            self.g_lb = np.array(inner.g_lb)
            self.g_ub = np.array(inner.g_ub)
        s = np.ones(inner.state.ndx)
        s[self.ndx_x:] = self.Fs
        self.s = s                                  # diag(S)

    # ------------------------------------------------------------------ helpers
    def _to_phys(self, x):
        y = np.array(x, dtype=float).copy()
        y[-self.nf:] *= self.Fs
        return y

    def _from_phys(self, x):
        y = np.array(x, dtype=float).copy()
        y[-self.nf:] /= self.Fs
        return y

    # --------------------------------------------------------------------- calc
    def calc(self, data, x, u=None):
        di = data.inner
        if u is None:
            self.inner.calc(di, self._to_phys(x))
        else:
            self.inner.calc(di, self._to_phys(x), u)
        data.xnext[:] = self._from_phys(np.array(di.xnext))
        data.cost = di.cost
        if self.ng:
            data.g[:] = np.array(di.g)

    def calcDiff(self, data, x, u=None):
        di = data.inner
        if u is None:
            self.inner.calcDiff(di, self._to_phys(x))
        else:
            self.inner.calcDiff(di, self._to_phys(x), u)
        s = self.s
        data.Fx[:, :] = (np.array(di.Fx) * s[None, :]) / s[:, None]
        data.Lx[:] = np.array(di.Lx) * s
        data.Lxx[:, :] = np.array(di.Lxx) * s[None, :] * s[:, None]
        if self.ng:
            data.Gx[:, :] = np.array(di.Gx) * s[None, :]
        if self.nu:
            data.Fu[:, :] = np.array(di.Fu) / s[:, None]
            data.Lu[:] = np.array(di.Lu)
            data.Luu[:, :] = np.array(di.Luu)
            data.Lxu[:, :] = np.array(di.Lxu) * s[:, None]
            if self.ng:
                data.Gu[:, :] = np.array(di.Gu)

    def createData(self):
        d = crocoddyl.ActionModelAbstract.createData(self)
        d.inner = self.inner.createData()
        return d


def scale_state(x, Fs, nf=15):
    y = np.array(x, dtype=float).copy()
    y[-nf:] /= Fs
    return y


def unscale_state(x, Fs, nf=15):
    y = np.array(x, dtype=float).copy()
    y[-nf:] *= Fs
    return y
