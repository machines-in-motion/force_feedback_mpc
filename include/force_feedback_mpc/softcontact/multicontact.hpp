#ifndef VISCOELASTIC_CONTACT_HPP
#define VISCOELASTIC_CONTACT_HPP

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/force.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>

namespace force_feedback_mpc {
    namespace softcontact {

// Forward declarations
class ViscoElasticContact3D;
class ViscoElasticContact3dMultiple;
class FrictionConeConstraint;
class ForceBoxConstraint;
class ForceConstraintManager;
class ForceCost;
class ForceCostManager;

// 3D soft contact model
class ViscoElasticContact3D {
public:
    ViscoElasticContact3D(pinocchio::Model& model, 
                         pinocchio::FrameIndex frameId, 
                         const Eigen::Vector3d& oPc = Eigen::Vector3d::Zero(),
                         double Kp = 10.0, 
                         double Kv = 0.0,
                         pinocchio::ReferenceFrame pinRef = pinocchio::LOCAL_WORLD_ALIGNED);
    
    // Getters
    bool isActive() const { return active; }
    int getNc() const { return nc; }
    std::string getName() const { return name; }
    pinocchio::FrameIndex getFrameId() const { return frameId; }
    pinocchio::ReferenceFrame getPinRef() const { return pinRef; }
    
    // Setters
    void setActive(bool active_) { active = active_; }
    void setName(const std::string& name_) { name = name_; }
    
    // Core methods
    pinocchio::Force calc(pinocchio::Data& data, const Eigen::Vector3d& f);
    Eigen::Vector3d calc_fdot(pinocchio::Data& data);
    void update_ABAderivatives(pinocchio::Data& data, const Eigen::Vector3d& f);
    void calcDiff(pinocchio::Data& data);
    
    // Access to derivatives
    const Eigen::MatrixXd& get_dfdt_dx() const { return dfdt_dx; }
    const Eigen::MatrixXd& get_dfdt_du() const { return dfdt_du; }
    const Eigen::MatrixXd& get_dfdt_df() const { return dfdt_df; }
    const Eigen::MatrixXd& get_dABA_df() const { return dABA_df; }
    
private:
    bool active;
    int nc;
    std::string name;
    pinocchio::Model& model;
    pinocchio::FrameIndex frameId;
    pinocchio::JointIndex parentId;
    pinocchio::SE3 jMf;
    Eigen::Vector3d oPc;
    double Kp, Kv;
    pinocchio::ReferenceFrame pinRef;
    
    pinocchio::Force fext;
    Eigen::Vector3d fout;
    Eigen::Vector3d fout_copy;
    
    Eigen::MatrixXd dABA_df;
    Eigen::MatrixXd dfdt_dx;
    Eigen::MatrixXd dfdt_du;
    Eigen::MatrixXd dfdt_df;
    
    Eigen::MatrixXd ldfdt_dx_copy;
    Eigen::MatrixXd ldfdt_du_copy;
    Eigen::MatrixXd ldfdt_df_copy;
};

// 3D Soft contact models stack
class ViscoElasticContact3dMultiple {
public:
    ViscoElasticContact3dMultiple(pinocchio::Model& model, 
                                 const std::vector<std::shared_ptr<ViscoElasticContact3D>>& contacts);
    
    bool isActive() const { return active; }
    int getNc() const { return nc; }
    int getNcTot() const { return nc_tot; }
    
    std::vector<pinocchio::Force> calc(pinocchio::Data& data, const Eigen::VectorXd& f);
    Eigen::VectorXd calc_fdot(pinocchio::Data& data);
    void update_ABAderivatives(pinocchio::Data& data, const Eigen::VectorXd& f);
    void calcDiff(pinocchio::Data& data);
    
private:
    pinocchio::Model& model;
    std::vector<std::shared_ptr<ViscoElasticContact3D>> contacts;
    int nv, nc, nc_tot;
    bool active;
    
    Eigen::MatrixXd Jc;
    std::vector<pinocchio::Force> fext;
    std::vector<pinocchio::Force> fext_copy;
    Eigen::VectorXd fout;
    Eigen::VectorXd fout_copy;
};

// 3D Friction cone constraint
class FrictionConeConstraint {
public:
    FrictionConeConstraint(pinocchio::FrameIndex frameId, double coef);
    
    bool isActive() const { return active; }
    int getNc() const { return nc; }
    int getNr() const { return nr; }
    pinocchio::FrameIndex getFrameId() const { return frameId; }
    
    double calc(const Eigen::Vector3d& f);
    Eigen::MatrixXd calcDiff(const Eigen::Vector3d& f);
    
    double getLowerBound() const { return lb; }
    double getUpperBound() const { return ub; }
    
private:
    bool active;
    int nc, nr;
    pinocchio::FrameIndex frameId;
    double coef;
    double residual;
    Eigen::MatrixXd residual_df;
    double lb, ub;
};

// 3D force box constraint
class ForceBoxConstraint {
public:
    ForceBoxConstraint(pinocchio::FrameIndex frameId, 
                      const Eigen::Vector3d& lb, 
                      const Eigen::Vector3d& ub);
    
    bool isActive() const { return active; }
    int getNc() const { return nc; }
    int getNr() const { return nr; }
    pinocchio::FrameIndex getFrameId() const { return frameId; }
    
    Eigen::Vector3d calc(const Eigen::Vector3d& f);
    Eigen::MatrixXd calcDiff(const Eigen::Vector3d& f);
    
    const Eigen::Vector3d& getLowerBound() const { return lb; }
    const Eigen::Vector3d& getUpperBound() const { return ub; }
    
private:
    bool active;
    int nc, nr;
    pinocchio::FrameIndex frameId;
    Eigen::Vector3d residual;
    Eigen::MatrixXd residual_df;
    Eigen::Vector3d lb, ub;
};

// 3D force constraint manager
class ForceConstraintManager {
public:
    ForceConstraintManager(const std::vector<std::shared_ptr<FrictionConeConstraint>>& frictionConstraints,
                          const std::vector<std::shared_ptr<ForceBoxConstraint>>& boxConstraints,
                          const std::shared_ptr<ViscoElasticContact3dMultiple>& contacts);
    
    bool hasForceConstraint() const { return has_force_constraint; }
    int getNr() const { return nr; }
    int getNc() const { return nc; }
    
    Eigen::VectorXd calc(const Eigen::VectorXd& f);
    Eigen::MatrixXd calcDiff(const Eigen::VectorXd& f);
    
    const Eigen::VectorXd& getLowerBound() const { return lb; }
    const Eigen::VectorXd& getUpperBound() const { return ub; }
    
private:
    std::vector<std::shared_ptr<FrictionConeConstraint>> frictionConstraints;
    std::vector<std::shared_ptr<ForceBoxConstraint>> boxConstraints;
    std::shared_ptr<ViscoElasticContact3dMultiple> contacts;
    
    bool has_force_constraint;
    int nr, nc;
    
    Eigen::VectorXd residual;
    Eigen::MatrixXd residual_df;
    Eigen::VectorXd lb, ub;
    
    std::map<pinocchio::FrameIndex, std::vector<std::shared_ptr<FrictionConeConstraint>>> contact_to_friction_map;
    std::map<pinocchio::FrameIndex, std::vector<std::shared_ptr<ForceBoxConstraint>>> contact_to_box_map;
    std::map<pinocchio::FrameIndex, int> contact_to_nr_map;
};

// 3D Force cost
class ForceCost {
public:
    ForceCost(pinocchio::Model& model,
              pinocchio::FrameIndex frameId,
              const Eigen::Vector3d& f_des,
              const Eigen::Matrix3d& f_weight,
              pinocchio::ReferenceFrame pinRef);
    
    int getNr() const { return nr; }
    int getNc() const { return nc; }
    pinocchio::FrameIndex getFrameId() const { return frameId; }
    
    double calc(pinocchio::Data& data, const Eigen::Vector3d& f, pinocchio::ReferenceFrame pinRefDyn);
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> calcDiff(pinocchio::Data& data, const Eigen::Vector3d& f, pinocchio::ReferenceFrame pinRefDyn);
    
private:
    pinocchio::FrameIndex frameId;
    pinocchio::Model& model;
    Eigen::Vector3d f_des;
    Eigen::Matrix3d f_weight;
    Eigen::Vector3d f_residual;
    Eigen::MatrixXd f_residual_x;
    double f_cost;
    Eigen::Vector3d Lf;
    Eigen::Matrix3d Lff;
    pinocchio::ReferenceFrame pinRef;
    int nr, nc;
};

// 3D Force cost manager
class ForceCostManager {
public:
    ForceCostManager(const std::vector<std::shared_ptr<ForceCost>>& forceCosts,
                    const std::shared_ptr<ViscoElasticContact3dMultiple>& contacts);
    
    int getNr() const { return nr; }
    int getNc() const { return nc; }
    
    double calc(pinocchio::Data& data, const Eigen::VectorXd& f);
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> calcDiff(pinocchio::Data& data, const Eigen::VectorXd& f);
    
private:
    std::vector<std::shared_ptr<ForceCost>> forceCosts;
    std::shared_ptr<ViscoElasticContact3dMultiple> contacts;
    
    int nr, nc;
    double cost;
    Eigen::VectorXd Lf;
    Eigen::MatrixXd Lff;
    
    std::map<pinocchio::FrameIndex, std::vector<std::shared_ptr<ForceCost>>> contact_to_cost_map;
    std::map<pinocchio::FrameIndex, int> contact_to_nr_map;
};

}  // namespace softcontact
}  // namespace force_feedback_mpc

#endif // VISCOELASTIC_CONTACT_HPP