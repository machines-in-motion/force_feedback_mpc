"""
Per end-effector contact force logging split by contacting body (PyBullet only)

mim_robots' end_effector_forces() sums all contact points of an end-effector link,
whatever the other body is (ground, wall, ...). This helper computes the same
quantity (same formula and sign convention: force applied by the environment on
the robot, in WORLD frame) restricted to contacts with one given body.
"""

import numpy as np
import pybullet


def endeff_forces_by_body(robot, other_body_id):
    """
    Returns (forces, npoints) for each end-effector in robot.bullet_endeff_ids
      forces  : (n_endeff, 3) contact force on the robot from other_body_id (WORLD frame)
      npoints : (n_endeff,) number of contact points with other_body_id
    """
    n = len(robot.bullet_endeff_ids)
    forces = np.zeros((n, 3))
    npoints = np.zeros(n, dtype=int)
    for ci in pybullet.getContactPoints(bodyA=robot.robot_id, bodyB=other_body_id):
        if ci[3] not in robot.bullet_endeff_ids:
            continue
        k = list(robot.bullet_endeff_ids).index(ci[3])
        forces[k] += (
            ci[9] * np.array(ci[7])
            - ci[10] * np.array(ci[11])
            - ci[12] * np.array(ci[13])
        )
        npoints[k] += 1
    return forces, npoints


def ocp_horizon_forces_classical(mpc, datas=None):
    """
    Contact forces predicted by the classical OCP at the accepted solver iterate,
    for each running node (the terminal node has no contact constraint).
    Recomputed with separate action datas: after an early solver exit, the problem
    datas may hold a rejected line-search trial instead of the accepted iterate.
    Returns (forces, datas) with forces of shape (T, n_contacts, 3), LOCAL_WORLD_ALIGNED
    """
    models = list(mpc.solver.problem.runningModels)
    if datas is None:
        datas = [m.createData() for m in models]
    xs = [np.array(x).copy() for x in mpc.solver.xs]
    us = [np.array(u).copy() for u in mpc.solver.us]
    forces = np.zeros((len(models), len(mpc.ee_frame_names), 3))
    for t, (m, d) in enumerate(zip(models, datas)):
        m.calc(d, xs[t], us[t])
        for k, fname in enumerate(mpc.ee_frame_names):
            forces[t, k] = d.differential.multibody.contacts.contacts[
                fname + "_contact"
            ].f.linear
    return forces, datas


def ocp_horizon_forces_soft(mpc):
    """
    Contact forces predicted by the force-feedback OCP (force states of the accepted
    iterate), for all nodes 0..T. Node 0 is the measured force (not a decision variable).
    Returns an array of shape (T+1, n_contacts, 3), LOCAL_WORLD_ALIGNED
    """
    n = mpc.rmodel.nq + mpc.rmodel.nv
    return np.array([np.array(x)[n:].copy() for x in mpc.solver.xs]).reshape(
        -1, len(mpc.ee_frame_names), 3
    )


def set_endeff_lateral_friction(robot, coef, n_feet=4):
    """Sets the lateral friction of the first n_feet end-effector links (the feet)"""
    for bullet_id in robot.bullet_endeff_ids[:n_feet]:
        pybullet.changeDynamics(robot.robot_id, bullet_id, lateralFriction=coef)
