from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, Viewer, ImplicitSourceTerm
import numpy
import matplotlib.pyplot as plt
from fipy import FaceVariable


# Define 2D mesh
nx, ny = 350, 350
dx, dy = 0.01, 0.01
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# Define phase field variable (e.g., phi from 0 to 1)
phase = CellVariable(name=r"$\phi$", mesh=mesh, hasOld=True)

# Define temperature field variable
temp = CellVariable(name="T", mesh=mesh, hasOld=False)

# Set initial conditions for phase field
x, y = mesh.cellCenters
phase.setValue(0.0)
mask = ((x - nx * dx / 2) ** 2 + (y - ny * dy / 2) ** 2) < (5 * dx) ** 2
phase.setValue(1.0, where=mask)

# Set initial conditions for temperature field
x, y = mesh.cellCenters
temp.setValue(-20.0)
mask = ((x - nx * dx / 2) ** 2 + (y - ny * dy / 2) ** 2) < (5 * dx) ** 2
temp.setValue(0.0, where=mask)

# Define all parameters required for phase field equation to be solved.
delta = 0.3
gamma = 0.1
epsilon_tilda = 0.01
alpha = .9
sym_order_j = 6
latent_heat = 334
theta_not = 0
Cp = 4.2199
dt = .2
steps = 4000
alpha_t = 30 * epsilon_tilda**2

# Theta field solver to define the angle variation throughout the phase field
x0 = nx * dx / 2.0
y0 = ny * dy / 2.0

# Equation used in (Staroselsky et al. 2023) paper

Xf, Yf = mesh.faceCenters
arg = sym_order_j * (numpy.arctan2(Yf - y0, Xf - x0) - theta_not)

a = epsilon_tilda * (1 + delta * numpy.cos(arg))
ap = -sym_order_j * epsilon_tilda * delta * numpy.sin(arg)
e2 = a**2
cross = a * ap

Gamma = FaceVariable(mesh=mesh, rank=2, value=numpy.array([[e2, -cross], [cross, e2]]))

source = phase.old * (1 - phase.old) * (phase.old - 0.5 + alpha*numpy.arctan(gamma*(0 - temp)))
eq1 = TransientTerm(var=phase) == DiffusionTerm(coeff=Gamma, var=phase) + source

'''
# Anisotropic Epsilon Calculation
ep_squared = (epsilon_tilda + (epsilon_tilda * delta * numerix.cos((sym_order_j * theta) - (sym_order_j * theta_not))))**2
ep_epprime = (-sym_order_j * epsilon_tilda * delta * numerix.sin((sym_order_j * theta) - (sym_order_j * theta_not))) * (epsilon_tilda + (epsilon_tilda * delta * numerix.cos((sym_order_j * theta) - (sym_order_j * theta_not))))

# Phase field governing equation
phi_grad = phase.grad
eq1 = TransientTerm(var=phase) == -(phi_grad[1]*ep_epprime).grad[0] + (phi_grad[0]*ep_epprime).grad[1] + (phi_grad[0]*ep_squared).grad[0] + (phi_grad[1]*ep_squared).grad[1] + (phase.old * (1 - phase.old) * (phase.old - 0.5 + alpha*numpy.arctan(gamma*(0 - temp))))
'''

# Temperature field governing equation
eq2 = TransientTerm(var=temp) == DiffusionTerm(var=temp, coeff=alpha_t) + ((latent_heat / Cp) * (phase - phase.old) / dt)

# Set up viewer and time stepping
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
viewer1 = Viewer(vars=phase, axes = ax1, datamin=0.0, datamax=1.0, title = 'PHASE FIELD')
viewer2 = Viewer(vars=temp, axes = ax2, datamin=-40.0, datamax=0.0, title = 'TEMPERATURE FIELD')
plt.tight_layout()

for step in range(steps):
    phase.updateOld()

    # Equation solver
    eq1.solve(var=phase, dt=dt)
    eq2.solve(var=temp, dt=dt)

    if step % 5 == 0:
        viewer1.plot()
        viewer2.plot()

plt.show()