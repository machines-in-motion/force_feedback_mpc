# Force Feedback MPC

Optimal control toolbox to achieve force feedback control in MPC. This library is basically an extension of the Crocoddyl optimal control library: it implements custom action models in C++ with unittests and python bindings. In particular, it contains the core classes used in MPC experiments of the following papers:
- S. Kleff, et. al, "Introducing Force Feedback in Model-Predictive Control", IROS 2022. [PDF](https://hal.science/hal-03594295/document)
- S. Kleff, et. al, "Force Feedback in Model-Predictive Control: A Soft Contact Approach" [PDF](https://hal.science/hal-04572399/) (under review)

The code to reproduce our experiments (i.e. real-time implementation of the force-feedback MPC), along with our experimental data are available in [this separate repository](https://github.com/machines-in-motion/force_feedback_dgh).

## Installation

We recommend using **Conda** to manage dependencies. This library is compatible with Python 3.10-3.13 (Python 3.14 is currently incompatible with MuJoCo).

### 1. Create a Conda Environment

```bash
conda create -n force_feedback python=3.12
conda activate force_feedback
```

### 2. Install Dependencies

Install Core dependencies from conda-forge:

```bash
conda install -c conda-forge pinocchio crocoddyl mim-solvers
conda install -c conda-forge numpy matplotlib pyyaml importlib_resources pybullet
```

Install additional Python dependencies:

```bash
pip install .
```

> **Note:** The `croco_mpc_utils` and `mim_robots` packages are required for the demos. Install them via
```bash
pip install . --no-deps
```

### 3. Build and Install

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j4
make install
```

## Running the Demos

The demos are located in the `demos/` directory.

### Classical Force Control (KUKA iiwa)
These demos use the classical force control formulation (hard contact).

```bash
python demos/force_tracking/classical/force_tracking_classical_mpc.py
```

### Soft Contact MPC (Force Tracking)
These demos use the soft contact formulation for force tracking.

```bash
python demos/force_tracking/soft/force_tracking_soft_mpc.py
```

### Polishing Task
Demos for the polishing task (circular motion while maintaining force).

```bash
python demos/polishing/classical/polishing_classical_mpc.py
```

### Go2 Quadruped (MuJoCo)
**Requires MuJoCo.** Ensure `mujoco` is installed (`conda install -c conda-forge mujoco-python`).
Note: Go2Py assets must be available.

```bash
python demos/go2arm/Go2MPC_demo_classical.py
```

## Dependencies Table

| Package | Version | Source |
|---------|---------|--------|
| Crocoddyl | >= 3.0 | conda-forge |
| Pinocchio | >= 3.0 | conda-forge |
| mim-solvers | >= 0.0.5 | conda-forge |
| eigenpy | >= 3.0 | conda-forge |
| Python | 3.10 - 3.13 | conda-forge |

**Known Issues:**
- Python 3.14 is currently incompatible with `mujoco-python`. Use Python 3.12 or 3.13.

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
