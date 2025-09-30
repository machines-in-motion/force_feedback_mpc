#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <stdexcept>

#include "force_feedback_mpc/softcontact/multicontact.hpp"

namespace force_feedback_mpc {
    namespace softcontact {

// ViscoElasticContact3D implementation
ViscoElasticContact3D::ViscoElasticContact3D(pinocchio::Model& model, 
                                             pinocchio::FrameIndex frameId,
                                             const Eigen::Vector3d& oPc,
                                             double Kp,
                                             double Kv,
                                             pinocchio::ReferenceFrame pinRef) :
    active(true), nc(3), name(""), model(model), frameId(frameId), 
    oPc(oPc), Kp(Kp), Kv(Kv), pinRef(pinRef) {
    
    parentId = model.frames[frameId].parent;
    jMf = model.frames[frameId].placement;
    
    fext = pinocchio::Force::Zero();
    fout = Eigen::Vector3d::Zero();
    fout_copy = Eigen::Vector3d::Zero();
    
    dABA_df = Eigen::MatrixXd::Zero(model.nv, nc);
    dfdt_dx = Eigen::MatrixXd::Zero(nc, model.nv*2);
    dfdt_du = Eigen::MatrixXd::Zero(nc, model.nv);
    dfdt_df = Eigen::MatrixXd::Zero(nc, nc);
    
    ldfdt_dx_copy = Eigen::MatrixXd::Zero(nc, model.nv*2);
    ldfdt_du_copy = Eigen::MatrixXd::Zero(nc, model.nv);
    ldfdt_df_copy = Eigen::MatrixXd::Zero(nc, nc);
}

pinocchio::Force ViscoElasticContact3D::calc(pinocchio::Data& data, const Eigen::Vector3d& f) {
    pinocchio::SE3 oMf = data.oMf[frameId];
    Eigen::Matrix3d oRf = oMf.rotation();
    
    if(pinRef == pinocchio::LOCAL) {
        fext = jMf.act(pinocchio::Force(f, Eigen::Vector3d::Zero()));
    } else {
        fext = jMf.act(pinocchio::Force(oRf.transpose() * f, Eigen::Vector3d::Zero()));
    }
    
    return fext;
}

Eigen::Vector3d ViscoElasticContact3D::calc_fdot(pinocchio::Data& data) {
    pinocchio::Motion v = pinocchio::getFrameVelocity(model, data, frameId, pinocchio::LOCAL);
    pinocchio::Motion a = pinocchio::getFrameAcceleration(model, data, frameId, pinocchio::LOCAL);
    
    fout = -Kp * v.linear() - Kv * a.linear();
    fout_copy = fout;
    
    if(pinRef != pinocchio::LOCAL) {
        pinocchio::Motion v_world = pinocchio::getFrameVelocity(model, data, frameId, pinocchio::LOCAL_WORLD_ALIGNED);
        pinocchio::Motion a_world = pinocchio::getFrameAcceleration(model, data, frameId, pinocchio::LOCAL_WORLD_ALIGNED);
        fout = -Kp * v_world.linear() - Kv * a_world.linear();
    }
    
    return fout;
}

void ViscoElasticContact3D::update_ABAderivatives(pinocchio::Data& data, const Eigen::Vector3d& f) {
    Eigen::MatrixXd lJ = pinocchio::getFrameJacobian(model, data, frameId, pinocchio::LOCAL);
    pinocchio::SE3 oMf = data.oMf[frameId];
    Eigen::Matrix3d oRf = oMf.rotation();
    
    dABA_df = data.Minv * lJ.topRows(3).transpose() * model.frames[frameId].placement.rotation() * Eigen::Matrix3d::Identity();
    
    if(pinRef != pinocchio::LOCAL) {
        Eigen::Matrix3d f_skew = pinocchio::skew(oRf.transpose() * f);
        data.Fx.topRows(model.nv) += data.Minv * lJ.topRows(3).transpose() * f_skew * lJ.bottomRows(3);
        dABA_df = dABA_df * oRf.transpose();
    }
}

void ViscoElasticContact3D::calcDiff(pinocchio::Data& data) {
    pinocchio::SE3 oMf = data.oMf[frameId];
    Eigen::Matrix3d oRf = oMf.rotation();
    
    // Get derivatives in LOCAL frame
    pinocchio::Motion v = pinocchio::getFrameVelocity(model, data, frameId, pinocchio::LOCAL);
    Eigen::MatrixXd v_dq, v_dv;
    pinocchio::getFrameVelocityDerivatives(model, data, frameId, pinocchio::LOCAL, v_dq, v_dv);
    
    Eigen::MatrixXd a_dq, a_dv, a_da;
    pinocchio::getFrameAccelerationDerivatives(model, data, frameId, pinocchio::LOCAL, a_dq, a_dv, a_da);
    
    Eigen::MatrixXd da_dx = Eigen::MatrixXd::Zero(3, 2*model.nv);
    da_dx.leftCols(model.nv) = a_dq.topRows(3) + a_da.topRows(3) * data.Fx.topRows(model.nv);
    da_dx.rightCols(model.nv) = a_dv.topRows(3) + a_da.topRows(3) * data.Fx.bottomRows(model.nv);
    
    Eigen::MatrixXd da_du = a_da.topRows(3) * data.Fu;
    Eigen::MatrixXd da_df = a_da.topRows(3) * dABA_df;
    
    dfdt_dx = -Kp * v_dq.topRows(3) - Kv * da_dx;
    dfdt_du = -Kv * da_du;
    dfdt_df = -Kv * da_df;
    
    ldfdt_dx_copy = dfdt_dx;
    ldfdt_du_copy = dfdt_du;
    ldfdt_df_copy = dfdt_df;
    
    if(pinRef != pinocchio::LOCAL) {
        Eigen::MatrixXd oJ = pinocchio::getFrameJacobian(model, data, frameId, pinocchio::LOCAL_WORLD_ALIGNED);
        dfdt_dx.leftCols(model.nv) = oRf * ldfdt_dx_copy.leftCols(model.nv) - pinocchio::skew(oRf * fout_copy) * oJ.bottomRows(3);
        dfdt_dx.rightCols(model.nv) = oRf * ldfdt_dx_copy.rightCols(model.nv);
        dfdt_du = oRf * ldfdt_du_copy;
        dfdt_df = oRf * ldfdt_df_copy;
    }
}

// ViscoElasticContact3dMultiple implementation
ViscoElasticContact3dMultiple::ViscoElasticContact3dMultiple(pinocchio::Model& model,
                                                             const std::vector<std::shared_ptr<ViscoElasticContact3D>>& contacts) :
    model(model), contacts(contacts), nv(model.nv), nc(contacts.size()) {
    
    nc_tot = 0;
    for(auto& ct : contacts) {
        nc_tot += ct->getNc();
        if(ct->isActive()) {
            active = true;
        }
    }
    
    Jc = Eigen::MatrixXd::Zero(nc_tot, nv);
    fext.resize(model.njoints, pinocchio::Force::Zero());
    fext_copy.resize(model.njoints, pinocchio::Force::Zero());
    fout = Eigen::VectorXd::Zero(nc_tot);
    fout_copy = Eigen::VectorXd::Zero(nc_tot);
}

std::vector<pinocchio::Force> ViscoElasticContact3dMultiple::calc(pinocchio::Data& data, const Eigen::VectorXd& f) {
    int nc_i = 0;
    std::fill(fext.begin(), fext.end(), pinocchio::Force::Zero());
    
    for(auto& ct : contacts) {
        if(ct->isActive()) {
            fext[ct->getParentId()] = ct->calc(data, f.segment(nc_i, ct->getNc()));
        }
        nc_i += ct->getNc();
    }
    
    return fext;
}

Eigen::VectorXd ViscoElasticContact3dMultiple::calc_fdot(pinocchio::Data& data) {
    int nc_i = 0;
    fout.setZero();
    
    for(auto& ct : contacts) {
        if(ct->isActive()) {
            fout.segment(nc_i, ct->getNc()) = ct->calc_fdot(data);
        }
        nc_i += ct->getNc();
    }
    
    fout_copy = fout;
    return fout;
}

void ViscoElasticContact3dMultiple::update_ABAderivatives(pinocchio::Data& data, const Eigen::VectorXd& f) {
    int nc_i = 0;
    data.dABA_df.setZero();
    
    for(auto& ct : contacts) {
        if(ct->isActive()) {
            ct->update_ABAderivatives(data, f.segment(nc_i, ct->getNc()));
            data.dABA_df.block(0, nc_i, nv, ct->getNc()) = ct->get_dABA_df();
        }
        nc_i += ct->getNc();
    }
}

void ViscoElasticContact3dMultiple::calcDiff(pinocchio::Data& data) {
    int nc_i = 0;
    data.dfdt_dx.setZero();
    data.dfdt_du.setZero();
    data.dfdt_df.setZero();
    
    for(auto& ct : contacts) {
        if(ct->isActive()) {
            ct->calcDiff(data);
            data.dfdt_dx.block(nc_i, 0, ct->getNc(), 2*nv) = ct->get_dfdt_dx();
            data.dfdt_du.block(nc_i, 0, ct->getNc(), nv) = ct->get_dfdt_du();
            data.dfdt_df.block(nc_i, nc_i, ct->getNc(), ct->getNc()) = ct->get_dfdt_df();
        }
        nc_i += ct->getNc();
    }
}

// FrictionConeConstraint implementation
FrictionConeConstraint::FrictionConeConstraint(pinocchio::FrameIndex frameId, double coef) :
    active(true), nc(3), nr(1), frameId(frameId), coef(coef) {
    
    residual = 0.0;
    residual_df = Eigen::MatrixXd::Zero(nr, nc);
    lb = 0.0;
    ub = std::numeric_limits<double>::infinity();
}

double FrictionConeConstraint::calc(const Eigen::Vector3d& f) {
    if(f.norm() > 1e-3) {
        residual = coef * f[2] - std::sqrt(f[0]*f[0] + f[1]*f[1]);
    } else {
        residual = 0.0;
    }
    return residual;
}

Eigen::MatrixXd FrictionConeConstraint::calcDiff(const Eigen::Vector3d& f) {
    if(f.norm() > 1e-3) {
        residual_df(0, 0) = -f[0] / std::sqrt(f[0]*f[0] + f[1]*f[1]);
        residual_df(0, 1) = -f[1] / std::sqrt(f[0]*f[0] + f[1]*f[1]);
        residual_df(0, 2) = coef;
    } else {
        residual_df.setZero();
    }
    return residual_df;
}

// ForceBoxConstraint implementation
ForceBoxConstraint::ForceBoxConstraint(pinocchio::FrameIndex frameId, 
                                     const Eigen::Vector3d& lb, 
                                     const Eigen::Vector3d& ub) :
    active(true), nc(3), nr(3), frameId(frameId), lb(lb), ub(ub) {
    
    residual = Eigen::Vector3d::Zero();
    residual_df = Eigen::Matrix3d::Identity();
}

Eigen::Vector3d ForceBoxConstraint::calc(const Eigen::Vector3d& f) {
    residual = f;
    return residual;
}

Eigen::MatrixXd ForceBoxConstraint::calcDiff(const Eigen::Vector3d& f) {
    return residual_df;
}

// ForceConstraintManager implementation
ForceConstraintManager::ForceConstraintManager(
    const std::vector<std::shared_ptr<FrictionConeConstraint>>& frictionConstraints,
    const std::vector<std::shared_ptr<ForceBoxConstraint>>& boxConstraints,
    const std::shared_ptr<ViscoElasticContact3dMultiple>& contacts) :
    frictionConstraints(frictionConstraints),
    boxConstraints(boxConstraints),
    contacts(contacts) {
    
    // Initialize mappings
    for(auto& ct : contacts->getContacts()) {
        contact_to_friction_map[ct->getFrameId()] = {};
        contact_to_box_map[ct->getFrameId()] = {};
        contact_to_nr_map[ct->getFrameId()] = 0;
    }
    
    // Populate friction constraints
    for(auto& fc : frictionConstraints) {
        bool found = false;
        for(auto& ct : contacts->getContacts()) {
            if(ct->getFrameId() == fc->getFrameId()) {
                contact_to_friction_map[fc->getFrameId()].push_back(fc);
                contact_to_nr_map[fc->getFrameId()] += fc->getNr();
                found = true;
                break;
            }
        }
        if(!found) {
            throw std::runtime_error("No contact model found for friction constraint on frame " + std::to_string(fc->getFrameId()));
        }
    }
    
    // Populate box constraints
    for(auto& bc : boxConstraints) {
        bool found = false;
        for(auto& ct : contacts->getContacts()) {
            if(ct->getFrameId() == bc->getFrameId()) {
                contact_to_box_map[bc->getFrameId()].push_back(bc);
                contact_to_nr_map[bc->getFrameId()] += bc->getNr();
                found = true;
                break;
            }
        }
        if(!found) {
            throw std::runtime_error("No contact model found for box constraint on frame " + std::to_string(bc->getFrameId()));
        }
    }
    
    // Calculate total dimensions
    nr = 0;
    nc = 0;
    for(auto& ct : contacts->getContacts()) {
        nc += ct->getNc();
        nr += contact_to_nr_map[ct->getFrameId()];
    }
    
    has_force_constraint = !frictionConstraints.empty() || !boxConstraints.empty();
    
    // Initialize bounds
    lb.resize(nr);
    ub.resize(nr);
    int idx = 0;
    for(auto& fc : frictionConstraints) {
        lb.segment(idx, fc->getNr()) << fc->getLowerBound();
        ub.segment(idx, fc->getNr()) << fc->getUpperBound();
        idx += fc->getNr();
    }
    for(auto& bc : boxConstraints) {
        lb.segment(idx, bc->getNr()) = bc->getLowerBound();
        ub.segment(idx, bc->getNr()) = bc->getUpperBound();
        idx += bc->getNr();
    }
    
    residual.resize(nr);
    residual_df.resize(nr, nc);
}

Eigen::VectorXd ForceConstraintManager::calc(const Eigen::VectorXd& f) {
    int nc_i = 0;
    residual.setZero();
    
    for(auto& ct : contacts->getContacts()) {
        int nr_i = 0;
        
        // Process friction constraints
        for(auto& fc : contact_to_friction_map[ct->getFrameId()]) {
            residual.segment(nr_i, fc->getNr()) << fc->calc(f.segment(nc_i, fc->getNc()));
            nr_i += fc->getNr();
        }
        
        // Process box constraints
        for(auto& bc : contact_to_box_map[ct->getFrameId()]) {
            residual.segment(nr_i, bc->getNr()) = bc->calc(f.segment(nc_i, bc->getNc()));
            nr_i += bc->getNr();
        }
        
        nc_i += ct->getNc();
    }
    
    return residual;
}

Eigen::MatrixXd ForceConstraintManager::calcDiff(const Eigen::VectorXd& f) {
    int nc_i = 0;
    residual_df.setZero();
    
    for(auto& ct : contacts->getContacts()) {
        int nr_i = 0;
        
        // Process friction constraints
        for(auto& fc : contact_to_friction_map[ct->getFrameId()]) {
            residual_df.block(nr_i, nc_i, fc->getNr(), fc->getNc()) = fc->calcDiff(f.segment(nc_i, fc->getNc()));
            nr_i += fc->getNr();
        }
        
        // Process box constraints
        for(auto& bc : contact_to_box_map[ct->getFrameId()]) {
            residual_df.block(nr_i, nc_i, bc->getNr(), bc->getNc()) = bc->calcDiff(f.segment(nc_i, bc->getNc()));
            nr_i += bc->getNr();
        }
        
        nc_i += ct->getNc();
    }
    
    return residual_df;
}

// ForceCost implementation
ForceCost::ForceCost(pinocchio::Model& model,
                    pinocchio::FrameIndex frameId,
                    const Eigen::Vector3d& f_des,
                    const Eigen::Matrix3d& f_weight,
                    pinocchio::ReferenceFrame pinRef) :
    frameId(frameId), model(model), f_des(f_des), f_weight(f_weight), pinRef(pinRef), nr(3), nc(3) {
    
    f_residual = Eigen::Vector3d::Zero();
    f_residual_x = Eigen::MatrixXd::Zero(3, 2*model.nv);
    f_cost = 0.0;
    Lf = Eigen::Vector3d::Zero();
    Lff = Eigen::Matrix3d::Zero();
}

double ForceCost::calc(pinocchio::Data& data, const Eigen::Vector3d& f, pinocchio::ReferenceFrame pinRefDyn) {
    pinocchio::SE3 oMf = data.oMf[frameId];
    Eigen::Matrix3d oRf = oMf.rotation();
    
    if(pinRef != pinRefDyn) {
        if(pinRef == pinocchio::LOCAL) {
            f_residual = oRf.transpose() * f - f_des;
        } else {
            f_residual = oRf * f - f_des;
        }
    } else {
        f_residual = f - f_des;
    }
    
    f_cost = 0.5 * f_residual.transpose() * f_weight * f_residual;
    return f_cost;
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ForceCost::calcDiff(pinocchio::Data& data, const Eigen::Vector3d& f, pinocchio::ReferenceFrame pinRefDyn) {
    pinocchio::SE3 oMf = data.oMf[frameId];
    Eigen::Matrix3d oRf = oMf.rotation();
    Eigen::MatrixXd lJ = pinocchio::getFrameJacobian(model, data, frameId, pinocchio::LOCAL);
    
    if(pinRef != pinRefDyn) {
        if(pinRef == pinocchio::LOCAL) {
            f_residual = oRf.transpose() * f - f_des;
            Lf = f_residual.transpose() * f_weight * oRf.transpose();
            f_residual_x.leftCols(model.nv) = pinocchio::skew(oRf.transpose() * f) * lJ.bottomRows(3);
            data.Lx += f_residual.transpose() * f_weight * f_residual_x;
            Lff = f_weight * oRf * oRf.transpose();
        } else {
            f_residual = oRf * f - f_des;
            Lf = f_residual.transpose() * f_weight * oRf;
            Eigen::MatrixXd oJ = pinocchio::getFrameJacobian(model, data, frameId, pinocchio::LOCAL_WORLD_ALIGNED);
            f_residual_x.leftCols(model.nv) = pinocchio::skew(oRf * f) * oJ.bottomRows(3);
            data.Lx += f_residual.transpose() * f_weight * pinocchio::skew(oRf * f) * f_residual_x;
            Lff = f_weight * oRf.transpose() * oRf;
        }
    } else {
        f_residual = f - f_des;
        Lf = f_residual.transpose() * f_weight;
        Lff = f_weight;
    }
    
    return std::make_pair(Lf, Lff);
}

// ForceCostManager implementation
ForceCostManager::ForceCostManager(const std::vector<std::shared_ptr<ForceCost>>& forceCosts,
                                 const std::shared_ptr<ViscoElasticContact3dMultiple>& contacts) :
    forceCosts(forceCosts), contacts(contacts) {
    
    // Initialize mappings
    for(auto& ct : contacts->getContacts()) {
        contact_to_cost_map[ct->getFrameId()] = {};
        contact_to_nr_map[ct->getFrameId()] = 0;
    }
    
    // Populate force costs
    for(auto& fc : forceCosts) {
        bool found = false;
        for(auto& ct : contacts->getContacts()) {
            if(ct->getFrameId() == fc->getFrameId()) {
                contact_to_cost_map[fc->getFrameId()].push_back(fc);
                contact_to_nr_map[fc->getFrameId()] += fc->getNr();
                found = true;
                break;
            }
        }
        if(!found) {
            throw std::runtime_error("No contact model found for force cost on frame " + std::to_string(fc->getFrameId()));
        }
    }
    
    // Calculate total dimensions
    nr = 0;
    nc = 0;
    for(auto& ct : contacts->getContacts()) {
        nc += ct->getNc();
        nr += contact_to_nr_map[ct->getFrameId()];
    }
    
    cost = 0.0;
    Lf.resize(nc);
    Lff.resize(nr, nc);
}

double ForceCostManager::calc(pinocchio::Data& data, const Eigen::VectorXd& f) {
    int nc_i = 0;
    cost = 0.0;
    
    for(auto& ct : contacts->getContacts()) {
        pinocchio::ReferenceFrame pinRefDyn = ct->getPinRef();
        
        for(auto& cost : contact_to_cost_map[ct->getFrameId()]) {
            cost += cost->calc(data, f.segment(nc_i, cost->getNc()), pinRefDyn);
        }
        
        nc_i += ct->getNc();
    }
    
    return cost;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ForceCostManager::calcDiff(pinocchio::Data& data, const Eigen::VectorXd& f) {
    int nc_i = 0;
    Lf.setZero();
    Lff.setZero();
    
    for(auto& ct : contacts->getContacts()) {
        pinocchio::ReferenceFrame pinRefDyn = ct->getPinRef();
        
        for(auto& cost : contact_to_cost_map[ct->getFrameId()]) {
            auto [Lf_ct, Lff_ct] = cost->calcDiff(data, f.segment(nc_i, cost->getNc()), pinRefDyn);
            Lf.segment(nc_i, cost->getNc()) = Lf_ct;
            Lff.block(nc_i, nc_i, cost->getNr(), cost->getNc()) = Lff_ct;
        }
        
        nc_i += ct->getNc();
    }
    
    return std::make_pair(Lf, Lff);
}

}  // namespace softcontact
}  // namespace force_feedback_mpc