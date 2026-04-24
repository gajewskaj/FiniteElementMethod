# FiniteElementMethod
**Table of Contents**
- [FiniteElementMethod](#finiteelementmethod)
  - [Program Purpose and Output](#program-purpose-and-output)
  - [Example Temperature Simulation](#example-temperature-simulation)
  - [How to run](#how-to-run)
  - [How it works](#how-it-works)
  - [Notes for further development](#notes-for-further-development)
    - [How to update requirements list](#how-to-update-requirements-list)
    - [How to create/update the documentation](#how-to-createupdate-the-documentation)

## Program Purpose and Output
The program is designed to generate data depicting the temperature distribution over time within a 2D element.
The program generates `.vtk` files, which can be used to create simulations, for example, in the ParaView application.

## Example Temperature Simulation
Check out a ParaView simulation created using the output from this program:
![ParaViewAnimation](img/ParaView_example_animation.gif)

## How to run
### Docker (recommended)
```bash
docker compose build
docker compose run --rm fem python main.py
```

### Local Python
```bash
pip install -r requirements.txt
python main.py
```

By default, it runs with:
- `--mesh input/10x10_tri.msh`
- `--data input/global_data.json`

You can override them, e.g.:
```bash
python main.py --mesh input/100x100_quad.msh --data input/global_data.json
```

## How it works
Program calculates temperatures $\{ t_1 \}$ by solving the below equation:
$$
\left( [H] + \dfrac{[C]}{ \Delta \tau} \right) \{ t_1 \} -
\left( \dfrac{[C]}{ \Delta \tau} \right) \{ t_0 \} + \{P\} = 0
$$

Where:
$$[H] = \int k
\left(
    \left\{{ \dfrac{ \partial \{N\}}{ \partial x}} \right \}
    \left\{{ \dfrac{ \partial \{N\}}{ \partial x}} \right \}^T +
    \left\{{ \dfrac{ \partial \{N\}}{ \partial y}} \right \}
    \left\{{ \dfrac{ \partial \{N\}}{ \partial y}} \right \}^T
\right) \mathrm{d}V +
\int \limits_S \alpha \{N\}\{N\}^T\mathrm{d}S
$$

$$
\{ P \} = - \int \limits_S \alpha \{N\}t_{ \infin} \mathrm{d}S
$$

$$
[C] = \int \limits_V \rho c \{N\}\{N\}^T \mathrm{d}V
$$

## Notes for further development
### How to update requirements list
```ps
pipreqs . --force
```

