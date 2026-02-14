# Force Feedback MPC

Optimal control toolbox to achieve force feedback control in MPC. This library is basically an extension of the Crocoddyl optimal control library: it implements custom action models in C++ with unittests and python bindings. In particular, it contains the core classes used in MPC experiments of the following papers:
- S. Kleff, et. al, "Introducing Force Feedback in Model-Predictive Control", IROS 2022. [PDF](https://hal.science/hal-03594295/document)
- S. Kleff, et. al, "Force Feedback in Model-Predictive Control: A Soft Contact Approach" [PDF](https://hal.science/hal-04572399/) (under review)

The code to reproduce our experiments (i.e. real-time implementation of the force-feedback MPC), along with our experimental data are available in [this separate repository](https://github.com/machines-in-motion/force_feedback_dgh).

## Dependencies

**Python compatibility:** 3.10-3.13

### Core dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [Pinocchio](https://github.com/stack-of-tasks/pinocchio) >= 3.7.0 | Robot dynamics & kinematics |
| [Crocoddyl](https://github.com/lariodante/crocoddyl) | >= 3.2.0 | Optimal control library |
| [mim-solvers](https://github.com/machines-in-motion/mim_solvers) | 0.2.0 | Optimization solvers |
| CMake | >= 3.10 | Build system |

### Demo dependencies

**For force tracking & polishing demos (Kuka Iiwa):**
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/)
- [PyBullet](https://pybullet.org/) (physics simulation)

**For Go2 multi-contact demos (requires above + specific deps below):**
- [MuJoCo](https://mujoco.org/) (physics engine)
- [meshcat-python](https://github.com/rdeits/meshcat-python) (3D visualization)
- [OpenCV](https://opencv.org/) (computer vision - Go2Py dependency)
- [Go2Py](https://github.com/machines-in-motion/Go2Py/tree/mpc) (Go2 quadruped interface)

## Installation

Using the provided conda environment file:

```bash
# 1. Create environment from file
conda env create -f environments/force_feedback_mpc.yml
conda activate force_feedback_mpc

# 2. Build and install force_feedback_mpc from source
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j4
make install

# 3. Install required Python packages (from workspace root)
cd ..
pip install -e ./croco_mpc_utils --no-deps 
pip install -e ./mim_robots --no-deps

# 4. [Optional] Install Go2Py for Go2 demos
pip install -e /path/to/Go2Py
```

## Running the Demos

The demos are located in the `demos/` directory. First, activate the environment:

```bash
conda activate force_feedback_mpc
```

### Classical MPC (rigid contact force model)

```bash
python demos/force_tracking/classical/force_tracking_classical_mpc.py
```

### Force-feedback MPC (soft contact force model)

```bash
python demos/force_tracking/soft/force_tracking_soft_mpc.py
```

### Polishing Task (apply constant normal force while tracking end-effector circle)

```bash
python demos/polishing/classical/polishing_classical_mpc.py
python demos/polishing/soft/polishing_soft_mpc.py
```

### Go2 Quadruped (whole-body multi-contact MPC)

Requires [Go2Py](https://github.com/machines-in-motion/Go2Py/tree/mpc) to be installed.

```bash
python demos/go2arm/Go2MPC_demo_classical.py
python demos/go2arm/Go2MPC_demo_soft.py
```

## Citing this work

```bibtex
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
