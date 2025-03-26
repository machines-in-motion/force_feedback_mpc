#include "viscoelastic_contact.hpp"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/numpy.hpp>
#include <pinocchio/bindings/python/utils/namespace.hpp>

namespace force_feedback_mpc {
    namespace softcontact {
    
namespace bp = boost::python;
namespace np = boost::python::numpy;

// Helper functions to convert between Eigen and numpy
template<typename Matrix>
np::ndarray eigenToNumpy(const Matrix& matrix) {
    bp::tuple shape = bp::make_tuple(matrix.rows(), matrix.cols());
    np::dtype dtype = np::dtype::get_builtin<typename Matrix::Scalar>();
    np::ndarray result = np::zeros(shape, dtype);
    std::copy(matrix.data(), matrix.data() + matrix.size(), reinterpret_cast<typename Matrix::Scalar*>(result.get_data()));
    return result;
}

template<typename Matrix>
Matrix numpyToEigen(const np::ndarray& array) {
    int rows = bp::extract<int>(array.attr("shape")[0]);
    int cols = bp::extract<int>(array.attr("shape")[1]);
    Matrix matrix(rows, cols);
    std::copy(reinterpret_cast<typename Matrix::Scalar*>(array.get_data()),
              reinterpret_cast<typename Matrix::Scalar*>(array.get_data()) + matrix.size(),
              matrix.data());
    return matrix;
}

// Pinocchio Force to/from numpy
np::ndarray forceToNumpy(const pinocchio::Force& f) {
    Eigen::Matrix<double,6,1> v = f.toVector();
    return eigenToNumpy<Eigen::Matrix<double,6,1>>(v);
}

pinocchio::Force numpyToForce(const np::ndarray& array) {
    Eigen::Matrix<double,6,1> v = numpyToEigen<Eigen::Matrix<double,6,1>>(array);
    return pinocchio::Force(v);
}

// Vector of Force to/from numpy
bp::list forceVectorToList(const std::vector<pinocchio::Force>& forces) {
    bp::list l;
    for(const auto& f : forces) {
        l.append(forceToNumpy(f));
    }
    return l;
}

std::vector<pinocchio::Force> listToForceVector(const bp::list& l) {
    std::vector<pinocchio::Force> forces;
    for(int i = 0; i < bp::len(l); ++i) {
        forces.push_back(numpyToForce(bp::extract<np::ndarray>(l[i])));
    }
    return forces;
}

BOOST_PYTHON_MODULE(viscoelastic_contact) {
    Py_Initialize();
    np::initialize();
    
    // Expose pinocchio reference frame enum
    bp::enum_<pinocchio::ReferenceFrame>("ReferenceFrame")
        .value("LOCAL", pinocchio::LOCAL)
        .value("LOCAL_WORLD_ALIGNED", pinocchio::LOCAL_WORLD_ALIGNED)
        .value("WORLD", pinocchio::WORLD);
    
    // ViscoElasticContact3D
    bp::class_<ViscoElasticContact3D, std::shared_ptr<ViscoElasticContact3D>>("ViscoElasticContact3D", 
        bp::init<pinocchio::Model&, pinocchio::FrameIndex, const Eigen::Vector3d&, double, double, pinocchio::ReferenceFrame>(
            (bp::arg("model"), bp::arg("frameId"), bp::arg("oPc")=Eigen::Vector3d::Zero(), 
            bp::arg("Kp")=10.0, bp::arg("Kv")=0.0, bp::arg("pinRef")=pinocchio::LOCAL_WORLD_ALIGNED)))
        .def("calc", &ViscoElasticContact3D::calc, (bp::arg("data"), bp::arg("f")))
        .def("calc_fdot", &ViscoElasticContact3D::calc_fdot, bp::arg("data"))
        .def("update_ABAderivatives", &ViscoElasticContact3D::update_ABAderivatives, (bp::arg("data"), bp::arg("f")))
        .def("calcDiff", &ViscoElasticContact3D::calcDiff, bp::arg("data"))
        .def("isActive", &ViscoElasticContact3D::isActive)
        .def("setActive", &ViscoElasticContact3D::setActive, bp::arg("active"))
        .def("getName", &ViscoElasticContact3D::getName)
        .def("setName", &ViscoElasticContact3D::setName, bp::arg("name"))
        .def("getFrameId", &ViscoElasticContact3D::getFrameId)
        .def("getPinRef", &ViscoElasticContact3D::getPinRef)
        .def("get_dfdt_dx", &ViscoElasticContact3D::get_dfdt_dx, bp::return_value_policy<bp::reference_existing_object>())
        .def("get_dfdt_du", &ViscoElasticContact3D::get_dfdt_du, bp::return_value_policy<bp::reference_existing_object>())
        .def("get_dfdt_df", &ViscoElasticContact3D::get_dfdt_df, bp::return_value_policy<bp::reference_existing_object>())
        .def("get_dABA_df", &ViscoElasticContact3D::get_dABA_df, bp::return_value_policy<bp::reference_existing_object>());
    
    // ViscoElasticContact3dMultiple
    bp::class_<ViscoElasticContact3dMultiple, std::shared_ptr<ViscoElasticContact3dMultiple>>("ViscoElasticContact3dMultiple", 
        bp::init<pinocchio::Model&, const std::vector<std::shared_ptr<ViscoElasticContact3D>>&>(
            (bp::arg("model"), bp::arg("contacts"))))
        .def("calc", &ViscoElasticContact3dMultiple::calc, (bp::arg("data"), bp::arg("f")))
        .def("calc_fdot", &ViscoElasticContact3dMultiple::calc_fdot, bp::arg("data"))
        .def("update_ABAderivatives", &ViscoElasticContact3dMultiple::update_ABAderivatives, (bp::arg("data"), bp::arg("f")))
        .def("calcDiff", &ViscoElasticContact3dMultiple::calcDiff, bp::arg("data"))
        .def("isActive", &ViscoElasticContact3dMultiple::isActive)
        .def("getNc", &ViscoElasticContact3dMultiple::getNc)
        .def("getNcTot", &ViscoElasticContact3dMultiple::getNcTot);
    
    // FrictionConeConstraint
    bp::class_<FrictionConeConstraint, std::shared_ptr<FrictionConeConstraint>>("FrictionConeConstraint", 
        bp::init<pinocchio::FrameIndex, double>((bp::arg("frameId"), bp::arg("coef"))))
        .def("calc", &FrictionConeConstraint::calc, bp::arg("f"))
        .def("calcDiff", &FrictionConeConstraint::calcDiff, bp::arg("f"))
        .def("isActive", &FrictionConeConstraint::isActive)
        .def("getFrameId", &FrictionConeConstraint::getFrameId)
        .def("getLowerBound", &FrictionConeConstraint::getLowerBound)
        .def("getUpperBound", &FrictionConeConstraint::getUpperBound);
    
    // ForceBoxConstraint
    bp::class_<ForceBoxConstraint, std::shared_ptr<ForceBoxConstraint>>("ForceBoxConstraint", 
        bp::init<pinocchio::FrameIndex, const Eigen::Vector3d&, const Eigen::Vector3d&>(
            (bp::arg("frameId"), bp::arg("lb"), bp::arg("ub"))))
        .def("calc", &ForceBoxConstraint::calc, bp::arg("f"))
        .def("calcDiff", &ForceBoxConstraint::calcDiff, bp::arg("f"))
        .def("isActive", &ForceBoxConstraint::isActive)
        .def("getFrameId", &ForceBoxConstraint::getFrameId)
        .def("getLowerBound", &ForceBoxConstraint::getLowerBound, bp::return_value_policy<bp::reference_existing_object>())
        .def("getUpperBound", &ForceBoxConstraint::getUpperBound, bp::return_value_policy<bp::reference_existing_object>());
    
    // ForceConstraintManager
    bp::class_<ForceConstraintManager, std::shared_ptr<ForceConstraintManager>>("ForceConstraintManager", 
        bp::init<const std::vector<std::shared_ptr<FrictionConeConstraint>>&, 
                 const std::vector<std::shared_ptr<ForceBoxConstraint>>&,
                 const std::shared_ptr<ViscoElasticContact3dMultiple>&>(
            (bp::arg("frictionConstraints"), bp::arg("boxConstraints"), bp::arg("contacts")))))
        .def("calc", &ForceConstraintManager::calc, bp::arg("f"))
        .def("calcDiff", &ForceConstraintManager::calcDiff, bp::arg("f"))
        .def("hasForceConstraint", &ForceConstraintManager::hasForceConstraint)
        .def("getLowerBound", &ForceConstraintManager::getLowerBound, bp::return_value_policy<bp::reference_existing_object>())
        .def("getUpperBound", &ForceConstraintManager::getUpperBound, bp::return_value_policy<bp::reference_existing_object>());
    
    // ForceCost
    bp::class_<ForceCost, std::shared_ptr<ForceCost>>("ForceCost", 
        bp::init<pinocchio::Model&, pinocchio::FrameIndex, const Eigen::Vector3d&, const Eigen::Matrix3d&, pinocchio::ReferenceFrame>(
            (bp::arg("model"), bp::arg("frameId"), bp::arg("f_des"), bp::arg("f_weight"), bp::arg("pinRef"))))
        .def("calc", &ForceCost::calc, (bp::arg("data"), bp::arg("f"), bp::arg("pinRefDyn")))
        .def("calcDiff", &ForceCost::calcDiff, (bp::arg("data"), bp::arg("f"), bp::arg("pinRefDyn")))
        .def("getFrameId", &ForceCost::getFrameId);
    
    // ForceCostManager
    bp::class_<ForceCostManager, std::shared_ptr<ForceCostManager>>("ForceCostManager", 
        bp::init<const std::vector<std::shared_ptr<ForceCost>>&, const std::shared_ptr<ViscoElasticContact3dMultiple>&>(
            (bp::arg("forceCosts"), bp::arg("contacts"))))
        .def("calc", &ForceCostManager::calc, (bp::arg("data"), bp::arg("f")))
        .def("calcDiff", &ForceCostManager::calcDiff, (bp::arg("data"), bp::arg("f")));
    
    // Register Eigen vector conversions
    bp::class_<std::vector<pinocchio::Force>>("ForceVector")
        .def(bp::vector_indexing_suite<std::vector<pinocchio::Force>>());
    
    // Register conversions between Eigen and numpy
    bp::to_python_converter<Eigen::Vector3d, eigenToNumpy<Eigen::Vector3d>>();
    bp::to_python_converter<Eigen::Matrix3d, eigenToNumpy<Eigen::Matrix3d>>();
    bp::to_python_converter<Eigen::MatrixXd, eigenToNumpy<Eigen::MatrixXd>>();
    bp::to_python_converter<std::vector<pinocchio::Force>, forceVectorToList>();
    
    bp::converter::registry::push_back(
        &numpyToEigen<Eigen::Vector3d>,
        bp::type_id<Eigen::Vector3d>());
    bp::converter::registry::push_back(
        &numpyToEigen<Eigen::Matrix3d>,
        bp::type_id<Eigen::Matrix3d>());
    bp::converter::registry::push_back(
        &numpyToEigen<Eigen::MatrixXd>,
        bp::type_id<Eigen::MatrixXd>());
    bp::converter::registry::push_back(
        &listToForceVector,
        bp::type_id<std::vector<pinocchio::Force>>());
}


}  // namespace softcontact
}  // namespace force_feedback_mpc
