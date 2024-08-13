# FiniteElementMethod
## Program Purpose and Output
The program is designed to generate data depicting the temperature distribution over time within a 2D element.
The program generates `.vtk` files, which can be used to create simulations, for example, in the ParaView application.
## Example Temperature Simulation
Check out a ParaView simulation created using the output from my program:

![ParaViewAnimation](https://github.com/jbahyrycz/FiniteElementMethod/assets/86531146/b84f004f-6249-489a-96fd-256bcdac9256)
## How to run
1. Clone this repository.
2. In the main folder open the command prompt and type: `python.exe -m pip install -r requirements.txt`.
3. After installing the necessary packages, double click on the `temperature_simulation.py` file, or type: `python.exe temperature_simulation.py` in the cmd.
4. Once the window opens, locate and select the file containing your grid data (ensuring that the data format matches the one in the `example_grid.txt` file):

![FemOpening](https://github.com/jbahyrycz/FiniteElementMethod/assets/86531146/d9a48502-2555-41b2-a2ce-d9aa872df077)
5. Output files will be generated:

![FemOutput](https://github.com/jbahyrycz/FiniteElementMethod/assets/86531146/e4064025-ac1b-46e6-9a73-7849da33a6c4)

### How to run tests
`py test.py -v`
