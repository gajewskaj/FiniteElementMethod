# FiniteElementMethod
## Program Purpose and Output
The program is designed to generate data depicting the temperature distribution over time within a 2D element.
The program generates `.vtk` files, which can be used to create simulations, for example, in the ParaView application.
## Example Temperature Simulation
Check out a ParaView simulation created using the output from my program:

![ParaViewAnimation](Images/ParaView_example_animation.gif)
## How to run
1. Clone this repository.
2. In the main folder, open cmd and type: `pip install -r requirements.txt`.
3. After installing the necessary packages, double click on the `temperature_simulation.py` file or type: `py temperature_simulation.py` in the cmd.
4. ...

### How it works
Program calculates temperatures $\{ t_1 \}$ by solving the below equation:
$$
\left( [H] + \dfrac{[C]}{ \Delta \tau} \right) \{ t_1 \} -
\left( \dfrac{[C]}{ \Delta \tau} \right) \{ t_0 \} + \{P\} = 0
$$

Where:
$$[H] = \int k(t)
\left(
    \Bigl\{{ \dfrac{ \partial \{N\}}{ \partial x}} \Bigr\}
    \Bigl\{{ \dfrac{ \partial \{N\}}{ \partial x}} \Bigr\}^T +
    \Bigl\{{ \dfrac{ \partial \{N\}}{ \partial y}} \Bigr\}
    \Bigl\{{ \dfrac{ \partial \{N\}}{ \partial y}} \Bigr\}^T
\right) \mathrm{d}V +
\int \limits_S \alpha \{N\}\{N\}^T\mathrm{d}S
$$

$$
[P] = \int \limits_S \alpha \{N\}t_{ot} \mathrm{d}V
$$

$$
[C] = \int \limits_S \rho c_p \{N\}\{N\}^T \mathrm{d}V
$$

### How to run tests
`py test.py -v`
