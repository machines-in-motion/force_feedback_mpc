import numpy as np
import pinocchio as pin

# Numerical difference function
def numdiff(f,x0,h=1e-6):
    '''
    f : vector space -> vector space
    '''
    f0 = f(x0).copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T

def numdiff_vfu_dam_cost(f,x0,h=1e-6):
    '''
    Numdiff in tangent space for Differential Action models cost (v)
    f : vector -> scalar
    '''
    f0 = f(x0)
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T

def numdiff_q_dam_dyn(f, q0, model, h=1e-6):
    '''
    Numdiff in tangent space for Differential Action models dynamics
    f : manifold -> vector
    '''
    f0 = f(q0).copy()
    q = q0.copy()
    # Create perturbation vector
    v = np.zeros(model.nv)
    Fq = []
    for iv in range(model.nv):
        # Apply perturbation to the iv-th component
        v[iv] = h
        q = pin.integrate(model, q0, v)
        # Compute finite difference
        Fq.append((f(q) - f0) / h)
        # Reset perturbation
        v[iv] = 0.0
    return np.array(Fq).T

def numdiff_q_dam_cost(f, q0, model, h=1e-6):
    '''
    Numdiff in tangent space for Differential Action models cost (q)
    f : manifold -> scalar
    '''
    f0 = f(q0)
    q = q0.copy()
    # Create perturbation vector
    v = np.zeros(model.nv)
    Fq = []
    for iv in range(model.nv):
        # Apply perturbation to the iv-th component
        v[iv] = h
        q = pin.integrate(model, q0, v)
        # Compute finite difference
        Fq.append((f(q) - f0) / h)
        # Reset perturbation
        v[iv] = 0.0
    return np.array(Fq).T


def numdiff_x_iam_dyn(f, x0, state, h=1e-6):
    '''
    Numdiff in tangent space for Integrated Action models (dynamics Fx)
    partial of dynamics (xnext) w.r.t. state x
    '''
    f0 = f(x0).copy()
    x = x0.copy()
    # Create perturbation vector
    dx = np.zeros(state.ndx)
    Fx = []
    for idx in range(state.ndx):
        # Apply perturbation to the iv-th component
        dx[idx] = h
        x = state.integrate(x0, dx)
        # Compute finite difference
        dx_h = state.diff(f0, f(x))
        Fx.append(dx_h / h)
        # Reset perturbation
        dx[idx] = 0.0
    return np.array(Fx).T

def numdiff_u_iam_dyn(f, u0, state, h=1e-6):
    '''
    Numdiff in tangent space for Integrated Action models
    partial of dynamics (xnext) w.r.t. control u
    '''
    f0 = f(u0).copy()
    u = u0.copy()
    # Create perturbation vector
    du = np.zeros_like(u0) 
    Fu = []
    for idu in range(len(du)):
        # Apply perturbation to the iv-th component
        u[idu] += h
        # Compute finite difference
        Fu.append(state.diff(f0, f(u)) / h)
        # Reset perturbation
        u[idu] -= h
    return np.array(Fu).T


def numdiff_x_iam_cost(f, x0, state, h=1e-6):
    '''
    Numdiff in tangent space for Integrated Action models
    partial of cost (scalar) w.r.t. control u
    '''
    f0 = f(x0)
    x = x0.copy()
    # Create perturbation vector
    dx = np.zeros(state.ndx)
    Fx = []
    for idx in range(state.ndx):
        # Apply perturbation to the iv-th component
        dx[idx] = h
        x = state.integrate(x0, dx)
        # Compute finite difference
        Fx.append( ( f(x) - f0 )/ h)
        # Reset perturbation
        dx[idx] = 0.0
    return np.array(Fx).T

def numdiff_u_iam_cost(f, u0, h=1e-6):
    '''
    Numdiff in tangent space for Integrated Action models
    partial of cost (scalar) w.r.t. control u
    '''
    f0 = f(u0)
    u = u0.copy()
    # Create perturbation vector
    du = np.zeros_like(u0) 
    Fu = []
    for idu in range(len(du)):
        # Apply perturbation to the iv-th component
        u[idu] += h
        # Compute finite difference
        Fu.append( ( f(u) - f0 )/ h)
        # Reset perturbation
        u[idu] -= h
    return np.array(Fu).T


def numdiff_x_iam_cstr(f, x0, state, h=1e-6):
    '''
    Numdiff in tangent space for Integrated Action models (constraint Gx)
    partial of constraint residual (g) w.r.t. state x
    '''
    f0 = f(x0).copy()
    x = x0.copy()
    # Create perturbation vector
    dx = np.zeros(state.ndx)
    Fx = []
    for idx in range(state.ndx):
        # Apply perturbation to the iv-th component
        dx[idx] = h
        x = state.integrate(x0, dx)
        # Compute finite difference
        Fx.append( ( f(x) - f0 )/ h)
        # Reset perturbation
        dx[idx] = 0.0
    return np.array(Fx).T

def numdiff_u_iam_cstr(f, u0, h=1e-6):
    '''
    Numdiff in tangent space for Integrated Action models (constraint Gx)
    partial of constraint residual (g) w.r.t. state x
    '''
    f0 = f(u0).copy()
    u = u0.copy()
    # Create perturbation vector
    du = np.zeros_like(u0) 
    Fu = []
    for idu in range(len(du)):
        # Apply perturbation to the iv-th component
        u[idu] += h
        # Compute finite difference
        Fu.append( ( f(u) - f0 )/ h)
        # Reset perturbation
        u[idu] -= h
    return np.array(Fu).T

# Finite differences
def get_xdot(dam, dad, q, v, f, u=None):
    x = np.concatenate([q,v])
    if(u is not None):
        dam.calc(dad, x, f, u)
    else:
        dam.calc(dad, x, f)
    return dad.xout 

def get_fdot(dam, dad, q, v, f, u=None):
    x = np.concatenate([q,v])
    if(u is not None):
        dam.calc(dad, x, f, u)
    else:
        dam.calc(dad, x, f)
    return dad.fout

def get_dam_cost(dam, dad, q, v, f, u=None):
    x = np.concatenate([q,v])
    if(u is not None):
        dam.calc(dad, x, f, u)
    else:
        dam.calc(dad, x, f)
    return dad.cost

def get_ynext(iam, iad, q, v, f, u=None):
    y = np.concatenate([q,v,f])
    if(u is not None):
        iam.calc(iad, y, u)
    else:
        iam.calc(iad, y)
    return iad.xnext 

def get_ynext_y(iam, iad, y, u=None):
    if(u is not None):
        iam.calc(iad, y, u)
    else:
        iam.calc(iad, y)
    return iad.xnext 

def get_iam_cost(iam, iad, y, u=None):
    if(u is not None):
        iam.calc(iad, y, u)
    else:
        iam.calc(iad, y)
    return iad.cost

def get_cstr(iam, iad, y, u=None):
    if(u is not None):
        iam.calc(iad, y, u)
    else:
        iam.calc(iad, y)
    return iad.g 