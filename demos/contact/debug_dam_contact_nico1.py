import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm,inv,pinv,svd,eig
import crocoddyl

# Set seed to 0 so that all executions are the same
np.random.seed(1)
pin.seed(1)
np.set_printoptions(precision=3, linewidth=300, suppress=True,threshold=10000)

# Load Talos arm in model/data
r = robex.load('talos_arm')
model = r.model
data = model.createData()
model.q0 = np.array([.5,-1,1.5,0,0,-0.5,0])

# Choose a contact point, cid is the frame id, jid is the corresponding joint id
cid = model.getFrameId('gripper_left_fingertip_1_link')
jid = model.frames[cid].parent

# Visu with Gepetto viewer.
# viz = pin.visualize.GepettoVisualizer(model,r.collision_model,r.visual_model)
# viz.initViewer(loadModel=True)
# viz.display(model.q0)

# Size of the main spaces
nq,nv = model.nq,model.nv
nx = nq+nv
ndx = 2*nv
nu = nv
nc = 3  # number of contacts

# Choose arbitrary variables for q,v and u
q = model.q0.copy()
v =np.random.rand(nv) #(np.random.rand(nv)*2-1)
tau = np.random.rand(nv) #*2-1
x = np.concatenate([q,v])
print("x0 = ", x)
print("tau = ", tau)

# --- CROCO baseline
# Build a DAM for asserting the values and derivatives later manually computed.
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)
cost = crocoddyl.CostModelSum(state) # dummy cost model (no cost is used)
contact = crocoddyl.ContactModel3D(state, cid, np.zeros(3))
contacts = crocoddyl.ContactModelMultiple(state,actuation.nu)
contacts.addContact('c3d',contact)
dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state,actuation,contacts,cost,0,True)
dad = dam.createData()

dam.calc(dad,x,tau)
dam.calcDiff(dad,x,tau)
croco_a = dad.xout
croco_phi = model.frames[cid].placement.inverse()*dad.multibody.contacts.contacts['c3d'].f
assert(norm(croco_phi.angular)<1e-6)
croco_f= croco_phi.linear
croco_J = dad.multibody.contacts.Jc

# --- MANUALLY RECOMPUTE FDYN IN LOCAL --------------------
'''
a,f = fdyn(x,u) = [MJ'J0]^-1 [ tau-b;-a0 ]
with a the configuration acc, f the local force, b the bias, a0 the cartesian coriolis acc.
'''
pin.computeAllTerms(model,data,q,v)
pin.forwardKinematics(model,data,q,v,v*0)
pin.updateFramePlacements(model,data)
M = data.M
J = pin.getFrameJacobian(model,data,cid,pin.LOCAL)[:3,:]
b = data.nle
a0 = pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL).linear
a0_check = (model.frames[cid].placement.inverse()*data.a[jid]).linear
# a0 = alpha.l + wxv
nu = model.frames[cid].placement.inverse()*data.v[jid]
a0_check += np.cross(nu.angular,nu.linear)
assert(norm(a0-a0_check)<1e-6)

K = np.block([ [M,J.T],[J,np.zeros([nc,nc])] ])
k = np.concatenate([ tau-b, -a0 ])

af = inv(K)@k
a = af[:nv]
f = af[nv:] # same as lamda_c = lagrange multipliers (here LOCAL force)
phi = model.frames[cid].placement*pin.Force(f,np.zeros(3)) # joint force
print("FEXT = ", phi)
print("fwd dyn LOCAL : acc,f = \n", af)

# Assert that the fdyn is the same as crocoddyl
assert( norm(f+croco_f)<1e-6 )
assert( norm(a-croco_a)<1e-6 ) # Checkout forces are oposite in the manual model

# --- MANUALLY RECOMPUTE D FDYN IN LOCAL -----------------------------
'''
fdyn = K^-1 k with K=MJ'J0 and k=b-tau;-a0
d fdyn = -K^-1 [d rnea; d a0 ]
with drnea given by Pinocchio and d a0:
d a0 = da[:3] + wx dnu[:3] - vx dnu[3:]
with a0 = a[:3] + wxv, a the spatial acceleration and nu=v;w
'''

# rnea = rnea(q,v,a,f) must be evaluated with local forces stored as stdVec_force
fs = pin.StdVec_Force()
for i,j in enumerate(model.joints):
    if model.frames[cid].parent == i:
        fs.append(-phi)
    else:
        fs.append(pin.Force.Zero())
# pin.computeForwardKinematicsDerivatives(model,data,q,v,a)
pin.computeRNEADerivatives(model,data,q,v,a,fs)
assert( norm(data.dtau_dq-dad.multibody.pinocchio.dtau_dq)<1e-6 )
drnea_dx = np.hstack([data.dtau_dq,data.dtau_dv])
dnu_dq,da_dq,da_dv,da_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL)
da_dx = np.hstack([ da_dq,da_dv ])
dnu_dx = np.hstack([ dnu_dq,da_da ])
da0_dx = da_dx[:3,:] - pin.skew(nu.linear)@dnu_dx[3:,:] + pin.skew(nu.angular)@dnu_dx[:3,:]

# print("fext = \n", phi)
print("d(contact) LOCAL : da0_dx = \n", da0_dx)

# Assert a0 derivatives is the same as crocoddyl.
cdata=dad.multibody.contacts.contacts['c3d']
assert(norm(da0_dx-cdata.da0_dx))

# Assert fdyn derivatives are the same as crocoddy
# print("DRNEA_DX = \n", drnea_dx)
daf_dx = -inv(K)@np.vstack([ drnea_dx, da0_dx ])
assert(norm(daf_dx[:nv,:]-dad.Fx)<1e-6)
assert(norm(daf_dx[nv:,:]+cdata.df_dx)<1e-6)


print("d(fwd dyn) LOCAL : d(acc,f) = \n", daf_dx)

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

'''
Ok, now entering the fun part: let's check the derivatives of fdyn with world forces.
We now write the contact model in world axes, i.e. J is world aligned, and a0 is expressed in 
world coordinates.

First establish the numdiff routines, then compute the new jacobian, finally assert.
'''

def numdiff(f,x0,h=1e-6):
    f0 = f(x0).copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T

# Encapsulation of the computations previously written
def fdyn(x,u):
    '''
    fdyn(x,u) = forward contact dynamics(q,v,tau) 
    returns the concatenation of configuration acceleration and contact forces expressed in local
    coordinates.
    The computations below correponds to what has been asserted above.
    '''
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model,data,q,v)
    pin.forwardKinematics(model,data,q,v,v*0)
    pin.updateFramePlacements(model,data)
    M = data.M
    J = pin.getFrameJacobian(model,data,cid,pin.LOCAL)[:3,:]
    b = data.nle
    a0 = pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL).linear
    K = np.block([ [M,J.T],[J,np.zeros([nc,nc])] ])
    k = np.concatenate([ tau-b, -a0 ])
    af = inv(K)@k
    return af

# Sanity check: this matches the previous computation, and numdiff is ok.
fdyn(x,tau)
Fx_nd=numdiff(lambda x_:fdyn(x_,tau),x)
assert(norm(Fx_nd[:-3]-dad.Fx)<1e-3)
assert(norm(Fx_nd[-3:]+cdata.df_dx)<1e-3)

# Forward dynamics rewritten with forces in world coordinates.
def fdynw(x,u):
    '''
    fwdyn(x,u) = forward contact dynamics(q,v,tau) 
    returns the concatenation of configuration acceleration and contact forces expressed in world
    coordinates.
    '''
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model,data,q,v)
    pin.forwardKinematics(model,data,q,v,v*0)
    pin.updateFramePlacements(model,data)
    M = data.M
    J = pin.getFrameJacobian(model,data,cid,pin.LOCAL_WORLD_ALIGNED)[:3,:]
    b = data.nle
    a0 = pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED).linear
    K = np.block([ [M,J.T],[J,np.zeros([nc,nc])] ])
    k = np.concatenate([ tau-b, -a0 ])
    af = inv(K)@k
    # print("inv KKT = \n", inv(K))
    return af # in world frame!

awf = fdynw(x,tau)
wFx_nd=numdiff(lambda x_:fdynw(x_,tau),x)
assert(norm(wFx_nd[:-3]-dad.Fx)<1e-3)
assert(norm(data.oMf[cid].rotation@af[-3:]-awf[-3:])<1e-6)

print("fwd dyn WORLD : acc,f = \n", awf)

### Now computing the wfdyn derivatives (with force world coordinates)
'''
a = fdyn(x,u) = wfdyn(x,u)
a' is the same
'''
dawf_dx = daf_dx.copy()
'''
wf = R F(x)
wf. = R. F + R F. = wx (R F) + R F. = R F. - (RF)x w
wf' = R' F(x) + R F'(x) = -(R F(x))x Jw + R F'
'''
Jw = pin.getFrameJacobian(model,data,cid,pin.LOCAL_WORLD_ALIGNED)[3:,:]
# print(Jw)
# print("SKEW = ", pin.skew(awf[-3:]))
# print("df_dx LOCAL : \n", dawf_dx[-3:])
dawf_dx[-3:] = data.oMf[cid].rotation@daf_dx[-3:]
dawf_dx[-3:,:nv] -= pin.skew(awf[-3:])@Jw
assert(norm(dawf_dx-wFx_nd)<1e-3)

print("d(fwd dyn) WORLD : d(acc,f) = \n", dawf_dx)