# 🚀 GPU-Accelerated Finite Element Method (FEM) Solver

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12x-76B900.svg)
![cuDSS](https://img.shields.io/badge/cuDSS-0.7.1-76B900.svg)

## 📌 Overview & Purpose
The program is designed for modeling transient thermal phenomena in an object's cross-section during heating. It utilizes the **Finite Element Method (FEM)** applied to Fourier's heat conduction equation, incorporating Newton's boundary conditions.

**Developing a basic FEM solver was not the primary goal.** The main objective was to design, implement, and benchmark multiple variants of the code to deeply understand performance bottlenecks, memory access patterns, and the impact of different matrix reordering algorithms.

### 🎥 Simulation Output
The program generates `.vtk` files, which can be visualized in tools like ParaView.

![ParaViewAnimation](img/ParaView_example_animation.gif)
*Example of a transient thermal simulation visualized in ParaView*

---

## ⚡ Key Features & Technologies
* **Custom CUDA Kernels:** Implemented custom kernels for the calculation and initial assembly of element matrices, achieving a **speedup of several dozen times** compared to the CPU implementation.
* **Multithreaded CPU Fallback:** A fully functional CPU-based solver for performance baseline comparison.
* **Multiple Sparse Linear Solvers:** Integration and benchmarking of various sparse linear system solvers:
    * `SciPy` (CPU-based computations)
    * `CuPy` (solve phase GPU-accelerated)
    * `cuDSS` (factorization & solve phases GPU-accelerated)
* **Profiling:** Analyzed GPU kernels using **Nvidia Nsight Systems** and **Nvidia Nsight Compute**.

---

## 📊 Performance Analysis & Results

Below is a summary of the core results. The full thesis containing detailed metrics will be published here following the standard 6-month academic embargo.

### 1. Matrix Assembly: CPU vs. GPU
By shifting the calculation and initial assembly of element matrices to the GPU using custom CUDA kernels, the program achieved significant performance gains. The results were tested on 2 computers: (🟦 Aorus, 🟥 Estera) with different 🔴 CPUs and ⭕ GPUs. The performance was evaluated across meshes of varying densities (*DOF - degrees of freedom*).

![Matrices Calculation Time Chart](img/charts/matrices_calculation_time.png)
*Time needed to calculate and agregate initially the element matrices, logarithmic scale*

### 2. Linear Solvers Comparison
#### Factorization phase
Factorization time includes symbolic analysis and decomposition:
- LU for SciPy and CuPy,
- Cholesky for cuDSS.

![Factorization Time](img/charts/factorization_time.png)
*Time needed to factorize matrix*

#### Solve phase
Solve phase consists of forward and backward substitution steps.

![Solving Time](img/charts/solving_time.png)
*Time needed to solve the sparse linear system of equations, logarithmic scale*

---

## 🛠️ How to Run
### 1. Start the Environment
It is highly advised to run the program inside the provided Docker container:

```bash
docker compose run --rm fem bash
```
If `Dockerfile`, `docker-compose.yml` or `requirements.txt` were modified since the last usage, rebuild the image first:

```bash
docker compose build --no-cache --pull fem
```

### 2. Run the program

The general usage syntax is:

```bash
python main.py --mesh <mesh_file> --data <data_file> --mc {cpu,gpu} --solver <solver_name>
```

**Quick Start:**

To quickly test the program with default parameters, simply run:

```bash
python main.py
```

This is equivalent to running:

```bash
python main.py --mesh input/100x100_quad.msh --data input/global_data.json --mc gpu --solver cudss_v4
```

---

## 🧮 Mathematical Background (How it works)
This section will only brush the surface of mathematical foundations for this project. A detailed description will available in my Master's thesis following the expiration of the standard 6-month university publication embargo.

### FEM in heat transfer problem
The mathematical model implemented in this solver is based on the transient heat conduction formulation described in [[1]](#1). The program determines the vector of nodal temperatures $\lbrace t_1 \rbrace$ at the next time step $\Delta \tau$ by solving the following linear system of equations (implicit Euler scheme):

$$
\left( [H] + [H_{BC}] + \frac{[C]}{\Delta \tau} \right) \lbrace t_1 \rbrace = \lbrace P \rbrace + \frac{[C]}{\Delta \tau} \lbrace t_0 \rbrace
$$

#### Notation:
- $\Delta \tau$ - time step $[s]$
- $t_0$ - start temperature $[K]$

Where the components are defined as:

$$
[H] = \int_S k \left( \left\lbrace \frac{\partial \lbrace N \rbrace}{\partial x} \right\rbrace \left\lbrace \frac{\partial \lbrace N \rbrace}{\partial x} \right\rbrace^T + \left\lbrace \frac{\partial \lbrace N \rbrace}{\partial y} \right\rbrace \left\lbrace \frac{\partial \lbrace N \rbrace}{\partial y} \right\rbrace^T \right) \mathrm{d}S
$$

$$
[H_{BC}] = \int_l \alpha \lbrace N \rbrace \lbrace N \rbrace^T \mathrm{d}l
$$

$$
\lbrace P \rbrace = \int_l \alpha \lbrace N \rbrace t_{\infty} \mathrm{d}l
$$

$$
[C] = \int_S \rho c \lbrace N \rbrace \lbrace N \rbrace^T \mathrm{d}S
$$

#### Notation:
- $k$ - thermal conductivity $\left[ \frac{W}{m \cdot K} \right]$
- $\alpha$ - convective heat transfer coefficient $\left[ \frac{W}{m^2 \cdot K} \right]$
- $t_\infty$ - ambient fluid temperature $[K]$
- $\rho$ - density $\left[ \frac{kg}{m^3} \right]$
- $c$ - specific heat capacity $\left[ \frac{J}{kg \cdot K} \right]$
- $\lbrace N \rbrace$ - shape functions vector

## 📚 References
<a id="1">[1]</a> A. Milenin, *Podstawy Metody Elementów Skończonych*, Kraków: AGH, 2010