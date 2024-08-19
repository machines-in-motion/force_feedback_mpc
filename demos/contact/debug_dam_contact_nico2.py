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
f = af[nv:]
phi = model.frames[cid].placement*pin.Force(f,np.zeros(3))

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
pin.computeForwardKinematicsDerivatives(model,data,q,v,a)
pin.computeRNEADerivatives(model,data,q,v,a,fs)
assert( norm(data.dtau_dq-dad.multibody.pinocchio.dtau_dq)<1e-6 )
drnea_dx = np.hstack([data.dtau_dq,data.dtau_dv])
dnu_dq,da_dq,da_dv,da_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL)
da_dx = np.hstack([ da_dq,da_dv ])
dnu_dx = np.hstack([ dnu_dq,da_da ])
da0_dx = da_dx[:3,:] - pin.skew(nu.linear)@dnu_dx[3:,:] + pin.skew(nu.angular)@dnu_dx[:3,:]

# Assert a0 derivatives is the same as crocoddyl.
cdata=dad.multibody.contacts.contacts['c3d']
assert(norm(da0_dx-cdata.da0_dx))

# Assert fdyn derivatives are the same as crocoddy
daf_dx = -inv(K)@np.vstack([ drnea_dx, da0_dx ])
assert(norm(daf_dx[:nv,:]-dad.Fx)<1e-6)
assert(norm(daf_dx[nv:,:]+cdata.df_dx)<1e-6)


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
    return af # in world frame!

awf = fdynw(x,tau)
wFx_nd=numdiff(lambda x_:fdynw(x_,tau),x)
assert(norm(wFx_nd[:-3]-dad.Fx)<1e-3)
assert(norm(data.oMf[cid].rotation@af[-3:]-awf[-3:])<1e-6)

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
dawf_dx[-3:] = data.oMf[cid].rotation@daf_dx[-3:]
dawf_dx[-3:,:nv] -= pin.skew(awf[-3:])@Jw
assert(norm(dawf_dx-wFx_nd)<1e-3)

#############################################################################################
### calcdiff from scratch ###################################################################
#############################################################################################
pin.computeAllTerms(model,data,q,v)
pin.forwardKinematics(model,data,q,v,v*0)
pin.updateFramePlacements(model,data)

# Compute LOCAL drift, Jacobian and KKT
M = data.M
b = data.nle
lJ = pin.getFrameJacobian(model,data,cid,pin.LOCAL)[:3,:]
la0 = pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL).linear
print("la0 = \n", la0)
lK = np.block([ [M,lJ.T],[lJ,np.zeros([nc,nc])] ])
# print('Jc = \n', lJ)
lk = np.concatenate([ tau-b, -la0 ])
# Compute WORLD drift, Jacobian and KKT
wJ = pin.getFrameJacobian(model,data,cid,pin.LOCAL_WORLD_ALIGNED)[:3,:]
wa0 = pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED).linear
wK = np.block([ [M,wJ.T],[wJ,np.zeros([nc,nc])] ])
wk = np.concatenate([ tau-b, -wa0 ])
# Check 
R=data.oMf[cid].rotation
IR = np.eye(nv+nc)
IR[nv:,nv:] = R
assert(norm(IR@lK@IR.T-wK)<1e-6)
assert(norm(IR@lk-wk)<1e-6)
lKi = inv(lK)
wKi = inv(wK)
assert(norm(IR@lKi@IR.T-wKi)<1e-6)
laf = fdyn(x,tau)
waf = fdynw(x,tau)
assert(norm(IR@laf-waf)<1e-6)

# # rnea derivatives
# fs[jid] = -(model.frames[cid].placement*pin.Force(R.T@waf[-nc:],np.zeros(3)))
# print(fs[jid])
# pin.computeForwardKinematicsDerivatives(model,data,q,v,a)
# pin.computeRNEADerivatives(model,data,q,v,a,fs)
# drnea_dx = np.hstack([data.dtau_dq,data.dtau_dv])

# # Local acc drift derivatives
# fJf = pin.getFrameJacobian(model,data,cid,pin.LOCAL)
# parendJointId = model.frames[cid].parent   
# v_partial_dq, a_partial_dq, a_partial_dv, _ = pin.getJointAccelerationDerivatives(model, data, parendJointId, pin.LOCAL) 
# lnu = pin.getFrameVelocity(model,data,cid,pin.LOCAL)
# print("lnu = ", lnu)
# print(v_partial_dq)
# vv_skew = pin.skew(lnu.linear)
# vw_skew = pin.skew(lnu.angular)
# fXj = model.frames[cid].placement.actionInverse
# lda0_dx = np.zeros((3,nx))
# lda0_dx[:,:nv] = (fXj @ a_partial_dq)[:3,:]
# lda0_dx[:,:nv] += pin.skew(lnu.angular) @ (fXj @ v_partial_dq)[:3,:]
# lda0_dx[:,:nv] -= pin.skew(lnu.linear) @ (fXj @ v_partial_dq)[3:,:]
# lda0_dx[:,nv:] = (fXj @ a_partial_dv)[:3,:]
# lda0_dx[:,nv:] += pin.skew(lnu.angular) @ fJf[:3,:] 
# lda0_dx[:,nv:] -= pin.skew(lnu.linear) @ fJf[3:,:]
# def F_lacc(qva):
#     pin.forwardKinematics(model,data,qva[:nq],qva[nq:nq+nv],qva[-nv:])
#     pin.updateFramePlacements(model,data)
#     return pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL).linear
# NDlacc=numdiff(F_lacc,np.concatenate([x,a]))[:,:ndx]
# assert(norm(NDlacc-lda0_dx)<1e-3)
# print("la0_dx = \n", lda0_dx)



# # World drift derivatives
# wda0_dx = np.zeros((3,nx))
# lda0_dx_temp = lda0_dx.copy()
# Jw = pin.getFrameJacobian(model,data,cid,pin.LOCAL_WORLD_ALIGNED)[3:,:]
# wda0_dx = R @ lda0_dx
# # ????? why wrong here but ok in C++ ?
# # wda0_dx[:,nv:] = R @ lda0_dx_temp[:,nv:] - pin.skew(R@la0) @ Jw 
# # wda0_dx[:,:nv] = R @ lda0_dx_temp[:,:nv]  
# def F_wacc(qva):
#     pin.forwardKinematics(model,data,qva[:nq],qva[nq:nq+nv],qva[-nv:])
#     pin.updateFramePlacements(model,data)
#     return pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED).linear
# NDwacc=numdiff(F_wacc,np.concatenate([x,a]))[:,:ndx]
# # print(NDwacc)
# # print(wda0_dx)
# assert(norm(NDwacc-wda0_dx)<1e-3)

# wda0_dx[:,:nv] = R @ lda0_dx[:,:nv]

# # Assert LOCAL fdyn derivatives are the same as crocoddy
# ldaf_dx = -lKi@np.vstack([ drnea_dx, lda0_dx ])
# assert(norm(ldaf_dx[:nv,:]-dad.Fx)<1e-6)
# assert(norm(ldaf_dx[nv:,:]+cdata.df_dx)<1e-6)
# lNDaf_dx = numdiff( lambda _x: fdyn(_x,tau),x)

# # Compute fdyn derivatives in WORLD
# wdaf_dx = -wKi@np.vstack([ drnea_dx, wda0_dx ])
# wNDaf_dx = numdiff( lambda _x: fdynw(_x,tau),x)
# print(wdaf_dx)
# print(wNDaf_dx)
# assert(norm(wdaf_dx - wNDaf_dx)<1e-6)

lnuc = model.frames[cid].placement.inverse()*data.v[jid]
lnu = pin.getFrameVelocity(model,data,cid,pin.LOCAL)
wnu = pin.getFrameVelocity(model,data,cid,pin.LOCAL_WORLD_ALIGNED)

fs[jid] = -(model.frames[cid].placement*pin.Force(R.T@waf[-nc:],np.zeros(3)))
pin.computeForwardKinematicsDerivatives(model,data,q,v,a)
pin.computeRNEADerivatives(model,data,q,v,a,fs)
# print("fext = \n", fs.tolist())
# print('xout = \n', a)
# print("drnea_dq= \n", data.dtau_dq)

drnea_dx = np.hstack([data.dtau_dq,data.dtau_dv])

ldnu_dq,lda_dq,lda_dv,lda_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL)
lda_dx = np.hstack([ lda_dq,lda_dv ])
ldnu_dx = np.hstack([ ldnu_dq,lda_da ])
lda0_dx = lda_dx[:3,:] - pin.skew(lnu.linear)@ldnu_dx[3:,:] + pin.skew(lnu.angular)@ldnu_dx[:3,:]
# print("ldnu_dq = \n", ldnu_dq)
print("lda0_dx = \n", lda0_dx)
wdnu_dq,wda_dq,wda_dv,wda_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL_WORLD_ALIGNED)
# wdnu_dq,wda_dq,wda_dv,wda_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL)
wda_dx = np.hstack([ wda_dq,wda_dv ])
wdnu_dx = np.hstack([ wdnu_dq,wda_da ])
wda0_dx = wda_dx[:3,:] - pin.skew(wnu.linear)@wdnu_dx[3:,:] + pin.skew(wnu.angular)@wdnu_dx[:3,:]

# Assert fdyn derivatives are the same as crocoddy
ldaf_dx = -lKi@np.vstack([ drnea_dx, lda0_dx ])
assert(norm(ldaf_dx[:nv,:]-dad.Fx)<1e-6)
assert(norm(ldaf_dx[nv:,:]+cdata.df_dx)<1e-6)
lNDaf_dx = numdiff( lambda _x: fdyn(_x,tau),x)
print(ldaf_dx)
wdaf_dx = -wKi@np.vstack([ drnea_dx, wda0_dx ])
wNDaf_dx = numdiff( lambda _x: fdynw(_x,tau),x)
### not working ... WHY!
###### likely because wda_dx is strange, not sure what Pinocchio is computing for acc-diff in LWA.

#############################################################################################
# Checking what is inside lda_dx ... just the derivatives of local acc.l
def F_lal(qva):
    pin.forwardKinematics(model,data,qva[:nq],qva[nq:nq+nv],qva[-nv:])
    pin.updateFramePlacements(model,data)
    return pin.getFrameAcceleration(model,data,cid,pin.LOCAL).linear.copy()
NDlal = numdiff(F_lal,np.concatenate([x,a]))
assert(norm(NDlal[:,:nv*2]-lda_dx[:3])<1e-3)

# Checking what is inside wda_dx .... I am expecting the derivatives of WLA acc.l, but no
# wa = R00R la
# wa.l = R la.l
la = pin.getFrameAcceleration(model,data,cid,pin.LOCAL)
wa = pin.getFrameAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED)
# dwal = R dlal - lalx dR
dwal = R@lda_dx[:3]
dwal[:,:nv] -= pin.skew(wa.linear)@R@lda_da[3:]

def F_wal(qva):
    pin.forwardKinematics(model,data,qva[:nq],qva[nq:nq+nv],qva[-nv:])
    pin.updateFramePlacements(model,data)
    return pin.getFrameAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED).linear
NDwal = numdiff(F_wal,np.concatenate([x,a]))
assert(norm(NDwal[:,:nv*2]-dwal)<1e-3)

wda0_dx = R@lda0_dx
wa = pin.getFrameClassicalAcceleration(model,data,cid).linear
assert(norm(wa)<1e-6)

def F_wacc(qva):
    pin.forwardKinematics(model,data,qva[:nq],qva[nq:nq+nv],qva[-nv:])
    pin.updateFramePlacements(model,data)
    return pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED).linear
def F_lacc(qva):
    pin.forwardKinematics(model,data,qva[:nq],qva[nq:nq+nv],qva[-nv:])
    pin.updateFramePlacements(model,data)
    return pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL).linear

NDwacc=numdiff(F_wacc,np.concatenate([x,a]))[:,:ndx]
NDlacc=numdiff(F_lacc,np.concatenate([x,a]))[:,:ndx]
assert(norm(NDwacc-wda0_dx)<1e-3)
# print(NDwacc)
# print(wda0_dx)
# That should be it ....
wdaf_dx = -wKi@np.vstack([ drnea_dx, wda0_dx  ])
# print(wdaf_dx)
# Still not!

#############################################################################################
#############################################################################################
#############################################################################################
# assert( norm( (wK@dawf_dx)[-3:]+wda0_dx )<1e-3 )

# def F_wrnea(x,a,wf):
#     q=x[:nq]
#     v=x[nq:]
#     pin.framesForwardKinematics(model,data,q)
#     fs[jid] = -(model.frames[cid].placement*pin.Force(R.T@wf,np.zeros(3)))
#     return pin.rnea(model,data,q,v,a,fs)

# ND_wrnea = numdiff(lambda _x: F_wrnea(_x,awf[:nv],awf[nv:]),x)
# assert( norm( (-wK@dawf_dx)[:nv] - ND_wrnea )<1e-3 )

# dwrnea_dx = drnea_dx
# dwrnea_dx[:,:nv] += lJ.T@pin.skew(laf[-3:])@lda_da[3:] # second method : rnea in WORLD, adding the term in Jacobian + skew rotated force
# assert(norm(dwrnea_dx-ND_wrnea)<1e-3)

#############################################################################################
#############################################################################################
#############################################################################################
pin.computeAllTerms(model,data,q,v)
pin.forwardKinematics(model,data,q,v,v*0)
pin.updateFramePlacements(model,data)

M = data.M
b = data.nle

R=data.oMf[cid].rotation
wJ = pin.getFrameJacobian(model,data,cid,pin.LOCAL_WORLD_ALIGNED)[:3,:]
wa0 = pin.getFrameClassicalAcceleration(model,data,cid,pin.LOCAL_WORLD_ALIGNED).linear
wK = np.block([ [M,wJ.T],[wJ,np.zeros([nc,nc])] ])
wk = np.concatenate([ tau-b, -wa0 ])

wKi = inv(wK)

waf = wKi@wk

wnu = pin.getFrameVelocity(model,data,cid,pin.LOCAL_WORLD_ALIGNED)

fs[jid] = -(model.frames[cid].placement*pin.Force(R.T@waf[-nc:],np.zeros(3)))

pin.computeForwardKinematicsDerivatives(model,data,q,v,a)
pin.computeRNEADerivatives(model,data,q,v,a,fs)
ldnu_dq,lda_dq,lda_dv,lda_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL)
wdnu_dq,wda_dq,wda_dv,wda_da = pin.getFrameAccelerationDerivatives(model,data,cid,pin.LOCAL_WORLD_ALIGNED)

dwrnea_dx = np.hstack([data.dtau_dq,data.dtau_dv])
dwrnea_dx[:,:nv] += lJ.T@pin.skew(laf[-3:])@lda_da[3:]
# -------


lda_dx = np.hstack([ lda_dq,lda_dv ])
ldnu_dx = np.hstack([ ldnu_dq,lda_da ])
lda0_dx = lda_dx[:3,:] - pin.skew(lnu.linear)@ldnu_dx[3:,:] + pin.skew(lnu.angular)@ldnu_dx[:3,:]
# print("local = \n",lda0_dx)

# wda_dx = np.hstack([ wda_dq,wda_dv ])
# wdnu_dx = np.hstack([ wdnu_dq,wda_da ])
# wda0_dx = wda_dx[:3,:] - pin.skew(wnu.linear)@wdnu_dx[3:,:] + pin.skew(wnu.angular)@wdnu_dx[:3,:]
# print("world 1 = \n",wda0_dx)
wda0_dx = R @ lda0_dx
print("nd world = \n", NDwacc)
# Jw = pin.getFrameJacobian(model,data,cid,pin.LOCAL_WORLD_ALIGNED)[3:,:]
# wda0_dx[:,:nv] -= pin.skew(R@la0) @ Jw 
# print(pin.getFrameClassicalAcceleration(model, data, cid, pin.LOCAL_WORLD_ALIGNED))
print("world 2 = \n",wda0_dx)

# Assert fdyn derivatives are the same as crocoddy
# print("drnea_dx = \n", drnea_dx)
ldaf_dx = -lKi@np.vstack([ drnea_dx, lda0_dx ])
# print("local : \n", drnea_dx)
assert(norm(ldaf_dx[:nv,:]-dad.Fx)<1e-6)
assert(norm(ldaf_dx[nv:,:]+cdata.df_dx)<1e-6)
lNDaf_dx = numdiff( lambda _x: fdyn(_x,tau),x)

# wdaf_dx = -wKi@np.vstack([ drnea_dx, wda0_dx ])


# Final solution for WORLD
wdaf_dx = ldaf_dx
# print(ldaf_dx)
wdaf_dx[-3:,:] = R @ ldaf_dx[-3:,:]
wdaf_dx[-3:, :nv] -= pin.skew(R @ laf[-3:])@Jw

wNDaf_dx = numdiff( lambda _x: fdynw(_x,tau),x)
print("computed \n", wdaf_dx)
print("numdiff \n", wNDaf_dx)
# print(wda0_dx)
assert(norm(wNDaf_dx- wdaf_dx)<1e-3)

### not working ... WHY!
###### likely because wda_dx is strange, not sure what Pinocchio is computing for acc-diff in LWA.
