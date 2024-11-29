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
1. Clone this repository.
2. In the main folder, open cmd and type: `pip install -r requirements.txt`.
3. After installing the necessary packages, double click on the `temperature_simulation.py` file or type: `py temperature_simulation.py` in the cmd.
4. ...

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

### How to create/update the documentation
If you want to start from scratch:
```ps
pip install Sphinx
mkdir docs
cd docs
sphinx-quickstart
cd ..
sphinx-apidoc -o docs .
```

The docs/conf.py file should look something like this:
```python
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'FiniteElementMethod'
copyright = '2024, Julia Bahyrycz'
author = 'Julia Bahyrycz'
release = 'v1'

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
```

Make sure that all of the modules have `docs/{module_name}.rst` files generated and that `docs/modules.rst` contains the list of all the modules that you want to document.
`docs/modules.rst`
```rst
FiniteElementMethod
===================

.. toctree::
   :maxdepth: 4

   main
   src
```

Make sure that `docs/index.rst` contains `modules`:
```
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```

```ps
pip install sphinx-rtd-theme
cd docs
make html
```

If you already have the docs folder and just want to update:
```ps
cd docs
make clean
make html
```