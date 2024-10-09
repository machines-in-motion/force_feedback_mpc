
import crocoddyl
import pinocchio as pin
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper
import force_feedback_mpc


robot_name = 'iiwa'

robot = load_pinocchio_wrapper(robot_name) #, locked_joints=['A7'])
print("Loaded ", robot_name)

DAM_TYPE = '1D' # '3D'

print("Construct OCP with DAM_"+str(DAM_TYPE))

# minimal croco problem
nx = robot.model.nq + robot.model.nv
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
costModel = crocoddyl.CostModelSum(state, actuation.nu)
costUreg = crocoddyl.ResidualModelControl(state, actuation.nu)
costXreg = crocoddyl.ResidualModelState(state, np.zeros(nx), actuation.nu)
costModel.addCost("ureg", crocoddyl.CostModelResidual(state, costUreg), 0.1)
costModel.addCost("xreg", crocoddyl.CostModelResidual(state, costXreg), 0.1)
frameId = robot.model.getFrameId('contact')
# instantiate custom DAM
oPc = np.ones(3)
if(DAM_TYPE == '1D'):
    nc = 1
    Kp = np.ones(1)
    Kv = np.ones(1)
    mask = force_feedback_mpc.Vector3MaskType.z
    DAM = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, actuation, costModel, frameId, Kp, Kv, oPc, mask)
if(DAM_TYPE == '3D'):
    nc = 3
    Kp = np.ones(3)
    Kv = np.ones(3)
    DAM = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(state, actuation, costModel, frameId, Kp, Kv, oPc)

print("Created DAM "+str(nc)+"D")

# dummy state
x0 = np.zeros(nx)
f0 = np.zeros(1)
y0 = np.concatenate([x0, f0])
u0 = np.zeros(actuation.nu)

print('x0 = ', x0.shape)
print('f0 = ', f0.shape)
print('y0 = ', y0.shape)
print('u0 = ', u0.shape)

# calc DAM
DAD = DAM.createData()
weho
# DAD = crocoddyl.DifferentialActionDataAbstract(DAM)
DAM.calc(DAD, x0, f0, u0)

# # custom IAM
# dt=0.1
# iam = force_feedback_mpc.IAMSoftContactAugmented(DAM , dt)

# I am trying to debug the following error occurring when I try to access the attribute of a python-binded derived data class : 
# ArgumentError                         
# ----> 1 DAD.my_attribute
# ArgumentError: Python argument types in
#     None.None(DADSoftContact1DAugmentedFwdDynamics)
# did not match C++ signature:
#     None(force_feedback_mpc::softcontact::DADSoftContact1DAugmentedFwdDynamics {lvalue})

# Here is the base abstract data class definition : 
# struct DADSoftContactAbstractAugmentedFwdDynamics : 
#     public crocoddyl::DifferentialActionDataAbstract {
#   template <class DAModel>
#   explicit DADSoftContactAbstractAugmentedFwdDynamics(DAModel* const model)
#       : DADBase(model) { // constructor implementation }

# where model a base abstract class that contains the createData member function : 
# virtual boost::shared_ptr<DifferentialActionDataAbstract> createData() {
# return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
# }
# which is binded by : 
#       .def("createData", &DAMSoftContactAbstractAugmentedFwdDynamics_wrap::createData,
#            &DAMSoftContactAbstractAugmentedFwdDynamics_wrap::default_createData,
#            bp::args("self"), "Comment")
# and the wrapper function is defined as 
#   boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> createData() {
#     crocoddyl::enableMultithreading() = false;
#     if (boost::python::override createData = this->get_override("createData")) {
#       return bp::call<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> >(createData.ptr());
#     }
#     return DAMSoftContactAbstractAugmentedFwdDynamics::createData();
#   }
#   boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> default_createData() {
#     return this->DAMSoftContactAbstractAugmentedFwdDynamics::createData();
#   }


# Now, here is the derived data class definition :
# struct DADSoftContact1DAugmentedFwdDynamics 
#     : public DADSoftContactAbstractAugmentedFwdDynamics {
#   template <class Model>
#   explicit DADSoftContact1DAugmentedFwdDynamics(Model* const model)
#       : Base(model) { // constructor implementation }
# and its corresponding base class' createData function : 
# virtual boost::shared_ptr<DifferentialActionDataAbstract> createData(){
# return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
# }
# which is binded by :
#       .def("createData", &DAMSoftContact1DAugmentedFwdDynamics::createData,
#            bp::args("self"), "Create the forward dynamics differential action data.")

# Can you help me understand why this code generate the python error when I try to access the derived class attribute ?


# *** stack smashing detected ***: terminated

# Thread 1 "python" received signal SIGABRT, Aborted.
# __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
# 50	../sysdeps/unix/sysv/linux/raise.c: No such file or directory.
# (gdb) bt
# #0  __GI_raise (sig=sig@entry=6) at ../sysdeps/unix/sysv/linux/raise.c:50
# #1  0x00007ffff7c67859 in __GI_abort () at abort.c:79
# #2  0x00007ffff7cd226e in __libc_message (action=action@entry=do_abort, fmt=fmt@entry=0x7ffff7dfc08f "*** %s ***: terminated\n")
#     at ../sysdeps/posix/libc_fatal.c:155
# #3  0x00007ffff7d74cda in __GI___fortify_fail (msg=msg@entry=0x7ffff7dfc077 "stack smashing detected") at fortify_fail.c:26
# #4  0x00007ffff7d74ca6 in __stack_chk_fail () at stack_chk_fail.c:24
# #5  0x00007fffa8b0a8d9 in boost::python::detail::caller_arity<5u>::impl<void (force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics::*)(boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&), boost::python::default_call_policies, boost::mpl::vector6<void, force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics&, boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&> >::operator() (this=<optimized out>, args_=<optimized out>)
#     at /home/skleff/miniconda3/envs/force_feedback/include/boost/python/converter/arg_from_python.hpp:106
# #6  0x00007fffb5926597 in boost::python::objects::function::call(_object*, _object*) const ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libboost_python312.so.1.86.0
# #7  0x00007fffb5926779 in boost::detail::function::void_function_ref_invoker<boost::python::objects::(anonymous namespace)::bind_return, void>::invoke(boost::detail::function::function_buffer&) ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libboost_python312.so.1.86.0
# #8  0x00007fffb592b4bb in boost::python::detail::exception_handler::operator()(boost::function_n<void> const&) const ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libboost_python312.so.1.86.0
# #9  0x00007fffb1fea77c in boost::python::detail::translate_exception<crocoddyl::Exception, void (*)(crocoddyl::Exception const&)>::operator() (
#     this=0x55555640ea18, handler=..., f=..., translate=0x7fffb1fe8370 <crocoddyl::python::translateException(crocoddyl::Exception const&)>)
#     at /home/skleff/miniconda3/envs/force_feedback/include/boost/python/detail/translate_exception.hpp:46
# #10 0x00007fffb1fea08a in boost::_bi::list<boost::arg<1>, boost::arg<2>, boost::_bi::value<void (*)(crocoddyl::Exception const&)> >::call_impl<bool, boost::python::detail::translate_exception<crocoddyl::Exception, void (*)(crocoddyl::Exception const&)>, boost::_bi::rrlist<boost::python::detail::exception_handler const&, boost::function_n<void> const&>, 0ul, 1ul, 2ul> (this=0x55555640ea20, f=..., a2=...)
#     at /home/skleff/miniconda3/envs/force_feedback/include/boost/bind/bind.hpp:182
# #11 0x00007fffb1fe9bf5 in boost::_bi::list<boost::arg<1>, boost::arg<2>, boost::_bi::value<void (*)(crocoddyl::Exception const&)> >::operator()<bool, boost::python::detail::translate_exception<crocoddyl::Exception, void (*)(crocoddyl::Exception const&)>, boost::_bi::rrlist<boost::python::detail::exception_handler const&, boost::function_n<void> const&> > (this=0x55555640ea20, f=..., a2=...)
#     at /home/skleff/miniconda3/envs/force_feedback/include/boost/bind/bind.hpp:208
# #12 0x00007fffb1fe98a6 in boost::_bi::bind_t<bool, boost::python::detail::translate_exception<crocoddyl::Exception, void (*)(crocoddyl::Exception const&)>, boost::_bi::list<boost::arg<1>, boost::arg<2>, boost::_bi::value<void (*)(crocoddyl::Exception const&)> > >::operator()<boost::python::detail::exception_handler const&, boost::function_n<void> const&> (this=0x55555640ea18)
#     at /home/skleff/miniconda3/envs/force_feedback/include/boost/bind/bind.hpp:321
# #13 0x00007fffb1fe961f in boost::detail::function::function_obj_invoker<boost::_bi::bind_t<bool, boost::python::detail::translate_exception<crocoddyl::Exception, void (*)(crocoddyl::Exception const&)>, boost::_bi::list<boost::arg<1>, boost::arg<2>, boost::_bi::value<void (*)(crocoddyl::Exception const&)> > >, bool, boost::python::detail::exception_handler const&, boost::function_n<void> const&>::invoke (function_obj_ptr=..., 
#     a#0=..., a#1=...) at /home/skleff/miniconda3/envs/force_feedback/include/boost/function/function_template.hpp:79
# #14 0x00007fffb5c06da4 in boost::detail::function::function_obj_invoker<boost::_bi::bind_t<bool, boost::python::detail::translate_exception<eigenpy::Exception, void (*)(eigenpy::Exception const&)>, boost::_bi::list<boost::arg<1>, boost::arg<2>, boost::_bi::value<void (*)(eigenpy::Exception const&)> > >, bool, boost::python::detail::exception_handler const&, boost::function_n<void> const&>::invoke(boost::detail::function::function_buffer&, boost::python::detail::exception_handler const&, boost::function_n<void> const&) ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libeigenpy.so
# #15 0x00007fffb592b3cd in boost::python::handle_exception_impl(boost::function_n<void>) ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libboost_python312.so.1.86.0
# #16 0x00007fffb59233b3 in function_call ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libboost_python312.so.1.86.0
# #17 0x000055555576109b in _PyObject_MakeTpCall (tstate=0x555555bf3058 <_PyRuntime+459704>, callable=0x5555564b0d20, args=<optimized out>, 
#     nargs=5, keywords=0x0) at /usr/local/src/conda/python-3.12.5/Include/object.h:704
# #18 0x0000555555667f65 in _PyEval_EvalFrameDefault (tstate=<optimized out>, frame=0x7ffff7fbe020, throwflag=<optimized out>)
#     at Python/bytecodes.c:2714
# #19 0x000055555581bece in PyEval_EvalCode (co=0x555555ca99b0, globals=<optimized out>, locals=0x7ffff7451700)
#     at /usr/local/src/conda/python-3.12.5/Python/ceval.c:578
# #20 0x0000555555840d9a in run_eval_code_obj (tstate=0x555555bf3058 <_PyRuntime+459704>, co=0x555555ca99b0, globals=0x7ffff7451700, 
#     locals=0x7ffff7451700) at /usr/local/src/conda/python-3.12.5/Python/pythonrun.c:1722
# #21 0x000055555583bf4b in run_mod (mod=<optimized out>, filename=0x7ffff73fa330, globals=0x7ffff7451700, locals=0x7ffff7451700, 
#     flags=0x7fffffffc4c0, arena=0x7ffff7373c90) at /usr/local/src/conda/python-3.12.5/Python/pythonrun.c:1743
# #22 0x0000555555854bd0 in pyrun_file (fp=0x555555c2e190, filename=0x7ffff73fa330, start=<optimized out>, globals=0x7ffff7451700, 
#     locals=0x7ffff7451700, closeit=1, flags=0x7fffffffc4c0) at /usr/local/src/conda/python-3.12.5/Python/pythonrun.c:1643
# #23 0x000055555585420e in _PyRun_SimpleFileObject (fp=0x555555c2e190, filename=0x7ffff73fa330, closeit=1, flags=0x7fffffffc4c0)
# --Type <RET> for more, q to quit, c to continue without paging--
#     at /usr/local/src/conda/python-3.12.5/Python/pythonrun.c:433
# #24 0x0000555555853ee4 in _PyRun_AnyFileObject (fp=0x555555c2e190, filename=0x7ffff73fa330, closeit=1, flags=0x7fffffffc4c0)
#     at /usr/local/src/conda/python-3.12.5/Python/pythonrun.c:78
# #25 0x000055555584cf42 in pymain_run_file_obj (skip_source_first_line=0, filename=0x7ffff73fa330, program_name=0x7ffff7387b10)
#     at /usr/local/src/conda/python-3.12.5/Modules/main.c:360
# #26 pymain_run_file (config=0x555555b95c38 <_PyRuntime+77720>) at /usr/local/src/conda/python-3.12.5/Modules/main.c:379
# #27 pymain_run_python (exitcode=0x7fffffffc494) at /usr/local/src/conda/python-3.12.5/Modules/main.c:633
# #28 Py_RunMain () at /usr/local/src/conda/python-3.12.5/Modules/main.c:713
# #29 0x00005555558047e7 in Py_BytesMain (argc=<optimized out>, argv=<optimized out>) at /usr/local/src/conda/python-3.12.5/Modules/main.c:767
# #30 0x00007ffff7c69083 in __libc_start_main (main=0x555555804720 <main>, argc=2, argv=0x7fffffffc6f8, init=<optimized out>, 
#     fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7fffffffc6e8) at ../csu/libc-start.c:308
# #31 0x0000555555804681 in _start () at /usr/local/src/conda/python-3.12.5/Parser/parser.c:41555
# (gdb) frame 6
# #6  0x00007fffb5926597 in boost::python::objects::function::call(_object*, _object*) const ()
#    from /home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/pinocchio/../../../libboost_python312.so.1.86.0
# (gdb) info locals
# No symbol table info available.
# (gdb) frame 5
# #5  0x00007fffa8b0a8d9 in boost::python::detail::caller_arity<5u>::impl<void (force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics::*)(boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&), boost::python::default_call_policies, boost::mpl::vector6<void, force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics&, boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&> >::operator() (this=<optimized out>, args_=<optimized out>)
#     at /home/skleff/miniconda3/envs/force_feedback/include/boost/python/converter/arg_from_python.hpp:106
# 106	struct arg_rvalue_from_python
# (gdb) info locals
# inner_args = <optimized out>
# c0 = <optimized out>
# c1 = {<boost::python::converter::arg_rvalue_from_python<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> > const&>> = {
#     m_data = {<boost::python::converter::rvalue_from_python_storage<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> > const&>> = {stage1 = {convertible = 0x7fffffffbc50, construct = 0x7fffb1de36a2
#      <boost::python::converter::shared_ptr_from_python<crocoddyl::DifferentialActionDataAbstractTpl<double>, boost::shared_ptr>::construct(_object*, boost::python::converter::rvalue_from_python_stage1_data*)>}, storage = {data = {data_ = {buf = "P\203MVUU\000\000\360\tKVUU\000", 
#               align_ = {<No data fields>}}}, bytes = "P\203MVUU\000\000\360\tKVUU\000"}}, <No data fields>}, 
#     m_source = 0x7fffa8b98cf0}, <No data fields>}
# c2 = {<boost::python::converter::arg_rvalue_from_python<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&>> = {
#     m_data = {<boost::python::converter::rvalue_from_python_storage<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&>> = {stage1 = {convertible = 0x7fffffffbc80, 
#           construct = 0x7fffb5e0c460 <void eigenpy::eigen_from_py_construct<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const>(_object*, boost::python::converter::rvalue_from_python_stage1_data*)>}, storage = {data = {data_ = {
#               buf = "\240\334LVUU\000\000\016\000\000\000\000\000\000\000\060", '\000' <repeats 22 times>, align_ = {<No data fields>}}}, 
#           bytes = "\240\334LVUU\000\000\016\000\000\000\000\000\000\000\060", '\000' <repeats 22 times>}}, <No data fields>}, 
#     m_source = 0x7fffa8b817d0}, <No data fields>}
# c3 = {<boost::python::converter::arg_rvalue_from_python<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&>> = {
#     m_data = {<boost::python::converter::rvalue_from_python_storage<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&>> = {stage1 = {convertible = 0x7fffa8b817d0, construct = 0x0}, storage = {data = {data_ = {
#               buf = "\200\274\377\377\377\177\000\000\001\000\000\000\000\000\000\000\060\274\377\377\377\177", '\000' <repeats 17 times>, 
#               align_ = {<No data fields>}}}, 
#           bytes = "\200\274\377\377\377\177\000\000\001\000\000\000\000\000\000\000\060\274\377\377\377\177", '\000' <repeats 17 times>}}, <No data fields>}, m_source = 0x7fffa8b81830}, <No data fields>}
# c4 = {<boost::python::converter::arg_rvalue_from_python<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&>> = {
#     m_data = {<boost::python::converter::rvalue_from_python_storage<Eigen::Ref<Eigen::Matrix<double, -1, 1, 0, -1, 1> const, 0, Eigen::InnerStride<1> > const&>> = {stage1 = {convertible = 0x7fffa8b81830, construct = 0x0}, storage = {data = {data_ = {
#               buf = "\300\274\377\377\377\177\000\000\a\000\000\000\000\000\000\000\060", '\000' <repeats 22 times>, 
#               align_ = {<No data fields>}}}, 
#           bytes = "\300\274\377\377\377\177\000\000\a\000\000\000\000\000\000\000\060", '\000' <repeats 22 times>}}, <No data fields>}, 
#     m_source = 0x7fffa8b818f0}, <No data fields>}
# result = <optimized out>
