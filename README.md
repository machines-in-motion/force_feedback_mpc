# force_feedback_mpc
Optimal control toolbox to achieve force feedback control in MPC. This library is basically an extension of the Crocoddyl optimal control library: it implements custom action models in C++ with unittests and python bindings. In particular, it contains the core classes used in MPC experiments of the following papers
- S. Kleff, et. al, "Introducing Force Feedback in Model-Predictive Control", IROSS 2022. [PDF](https://hal.science/hal-03594295/document)
- S. Kleff, et. al, "Force Feedback in Model-Predictive Control: A Soft Contact Approach" [PDF](https://hal.science/hal-04572399/) (under review)

# Dependencies
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) (>=3.0)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [boost](https://www.boost.org/)
- [eigenpy](https://github.com/stack-of-tasks/eigenpy) (>=2.7.10)
- [Optional] [OpenMP](https://www.openmp.org/) if Crocoddyl was built from source with the multi-threading option.

## For the Python demo scripts
- [croco_mpc_utils](https://github.com/machines-in-motion/mim_robots)
- [mim_robots](https://github.com/machines-in-motion/mim_robots)
- [PyBullet](https://pybullet.org/wordpress/)  
- PyYAML
- importlib_resources
- matplotlib

# Installation
```
git clone --recursive https://github.com/machines-in-motion/force_feedback_mpc.git
mkdir build && cd build
cmake .. 
make -j6 && sudo make install
```

# How to use it
Simply prototype your OCP using Crocoddyl as you would normally do, but use the custom integrated action models provided in this library (`IntegratedActionModelLPF` and `IAMSoftContactAugmented`). Example python scripts can be found in the `demos` directory (simulated force and polishing tasks).

In the `python` directory, OCP utilities are implemented. These are simplified interfaces to Crocoddyl's API ; they allow the quick proto-typing of OCPs from templated YAML files (i.e. this is an extension of [croco_mpc_utils](https://github.com/machines-in-motion/mim_robots) to force-feedback OCPs).

# Citing this work
```
@unpublished{kleff:hal-04572399,
  TITLE = {{Force Feedback in Model-Predictive Control: A Soft Contact Approach}},
  AUTHOR = {Kleff, S{\'e}bastien and Jordana, Armand and Khorrambakht, Rooholla and Mansard, Nicolas and Righetti, Ludovic},
  URL = {https://hal.science/hal-04572399},
  NOTE = {working paper or preprint},
  HAL_LOCAL_REFERENCE = {Rapport LAAS n{\textdegree} 24093},
  YEAR = {2025},
  MONTH = Jun,
  PDF = {https://hal.science/hal-04572399v2/file/force_feedback_article_second_submission.pdf},
  HAL_ID = {hal-04572399},
  HAL_VERSION = {v2},
}
```
