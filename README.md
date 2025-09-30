# force_feedback_mpc
Optimal control toolbox to achieve force feedback control in MPC. This library is basically an extension of the Crocoddyl optimal control library: it implements custom action models in C++ with unittests and python bindings. In particular, it contains the core classes used in MPC experiments of the following papers
- S. Kleff et. al, "Introducing Force Feedback in Model-Predictive Control", IROSS 2022. [PDF](https://hal.science/hal-03594295/document)
- S. Kleff et. al, "Force feedback in Model-Predictive Control: A Soft Contact Approach" (under prepation) 

# Dependencies
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) (<=2.0)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [boost](https://www.boost.org/)
- [eigenpy](https://github.com/stack-of-tasks/eigenpy) (>=2.7.10)
Optionally [OpenMP](https://www.openmp.org/) if Crocoddyl was built with the multi-threading option.

## For the Python demo scripts
- [croco_mpc_utils](https://github.com/machines-in-motion/mim_robots)
- [mim_robots](https://github.com/machines-in-motion/mim_robots)
- [PyBullet](https://pybullet.org/wordpress/)  
- PyYAML
- importlib_resources
- matplotlib

# How to use it
Simply prototype your OCP using Crocoddyl as you would normally do, but use the custom integrated action models provided in this library (`IntegratedActionModelLPF` and `IAMSoftContactAugmented`).

# Installation
```
git clone --recursive https://github.com/machines-in-motion/force_feedback_mpc.git
mkdir build && cd build
cmake .. 
make -j6 && sudo make install
```
