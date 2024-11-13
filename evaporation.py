from dolfinx import mesh, fem, default_scalar_type, io
import ufl
import basix
from dolfinx.cpp.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.utils import symmetric_tensor_to_vector
import toml
from adaptive_solver import adaptive_solver
from boundary_conditions import evap_heat, evap_fluid, heat_convection
from vtx_utils import VTX_init, VTX_write
from mesh import progressive_mesh


### Basic Setup ###
thickness = 15e-3 # Mesh thickness

progressive_mesh(thickness) # create mesh

domain, _, _ = io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=2)

# External temperature and RH
T_ext = fem.Constant(domain, default_scalar_type(295.15))
Rh_ext = fem.Constant(domain, default_scalar_type(0.5))
### ----------- ###

material = MFrontMaterial(
    "src/libBehaviour.dylib",
    "THM2X_h",
    hypothesis="plane_strain",
    material_properties=toml.load("material_properties.toml"),
)

external_state_variables = {
    "Displacement": domain.geometry.dim,
    "LiquidPressure": 1,
    "Temperature": 1,
}

disp_el = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(2,))
fluid_el = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
temp_el = basix.ufl.element("Lagrange", domain.basix_cell(), 1)

V = fem.functionspace(domain, basix.ufl.mixed_element([disp_el, fluid_el, temp_el]))
v = fem.Function(V, name="Unknown")


def bottom(x):
    return np.isclose(x[1], 0)


def left(x):
    return np.isclose(x[0], 0)


def evap_boundary(x):
    return np.isclose(x[0], thickness)


V_ux, _ = V.sub(0).sub(0).collapse()
V_uy, _ = V.sub(0).sub(1).collapse()
bottom_facets = mesh.locate_entities_boundary(domain, 1, bottom)
bottom_dofs_y = fem.locate_dofs_topological((V.sub(0).sub(1), V_uy), 1, bottom_facets)
left_facets = mesh.locate_entities_boundary(domain, 1, left)
left_dofs_x = fem.locate_dofs_topological((V.sub(0).sub(0), V_ux), 1, left_facets)
bcs = [
    fem.dirichletbc(fem.Function(V_uy), bottom_dofs_y, V.sub(0).sub(1)),
    fem.dirichletbc(fem.Function(V_ux), left_dofs_x, V.sub(0).sub(0)),
]
evap_boundary_facets = mesh.locate_entities_boundary(domain, 1, evap_boundary)
facet_tags = mesh.meshtags(
    domain, 1, evap_boundary_facets, np.full_like(evap_boundary_facets, 1)
)

ds = ufl.Measure(
    "ds", domain=domain, subdomain_data=facet_tags, metadata={"quadrature degree": 2}
)

qmap = QuadratureMap(domain, 2, material)
(u, pl, tem) = ufl.split(v)
Tref = fem.Constant(domain, default_scalar_type(295.15))
qmap.register_gradient("Strain", symmetric_tensor_to_vector(ufl.sym(ufl.grad(u))))
qmap.register_external_state_variable("LiquidPressure", pl)
qmap.register_gradient("LiquidPressureGradient", ufl.grad(pl))
qmap.register_external_state_variable("Temperature", tem + Tref)
qmap.register_gradient("TemperatureGradient", ufl.grad(tem))

sig = qmap.fluxes["Stress"]
w = qmap.fluxes["FluidFlux"]
j = qmap.fluxes["HeatFlux"]
ml = qmap.internal_state_variables["FluidMass"]
S = qmap.internal_state_variables["Entropy"]
sf = qmap.internal_state_variables["FluidEntropy"]
xi_1 = qmap.internal_state_variables["HydrationExtent_1"]
xi_2 = qmap.internal_state_variables["HydrationExtent_2"]

ml_n = ml.copy()
S_n = S.copy()
xi_1_n = xi_1.copy()
xi_2_n = xi_2.copy()

dt = fem.Constant(domain, default_scalar_type(0.0))

v_ = ufl.TestFunction(V)
(u_, pl_, tem_) = ufl.split(v_)
dv = ufl.TrialFunction(V)


mech_res = ufl.dot(sig, symmetric_tensor_to_vector(ufl.sym(ufl.grad(u_)))) * qmap.dx
fluid_res = (
    ((ml - ml_n) / dt + 0.684 * (xi_1 - xi_1_n) / dt + 0.13 * (xi_2 - xi_2_n) / dt)
    * pl_
    - ufl.dot(w, ufl.grad(pl_))
) * qmap.dx + evap_fluid(pl, tem + Tref, T_ext, Rh_ext) * pl_ * ds(1)
tem_res = (
    (Tref * ((S - S_n) / dt) * tem_ + ufl.dot(Tref * sf * w - j, ufl.grad(tem_)))
    * qmap.dx
    + heat_convection(tem + Tref, T_ext) * tem_ * ds(1)
    + evap_heat(pl, tem + Tref, T_ext, Rh_ext) * tem_ * ds(1)
)

F = mech_res + fluid_res + tem_res
Jac = qmap.derivative(F, v, dv)
problem = NonlinearMaterialProblem(qmap, F, Jac, v, bcs)

# Temperature initialisation
def T0_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 8
    return values

v.sub(2).interpolate(T0_init)

# Pressure initialisation for an inital saturation < 1
def pl_init(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = -350e3
    return values

v.sub(1).interpolate(pl_init)

qmap.update()
ml.vector.copy(ml_n.vector)
S.vector.copy(S_n.vector)
xi_1.vector.copy(xi_1_n.vector)
xi_2.vector.copy(xi_2_n.vector)

newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-7
newton.atol = 1e-7
newton.convergence_criterion = "incremental"
newton.report = True
newton.max_it = 25

# Save these functions to VTX
vtx, save_functions = VTX_init(
    domain,
    "evaporation",
    material,
    [
        "Displacement",
        "LiquidPressure",
        "Temperature",
        "HydrationExtent_1",
        "HydrationExtent_2",
        "FluidMass",
        "SaturationDegree",
        "RelativeHumidity",
        "Permeability",
        "EffectivePermeability",
    ],
)

# To write after convergence of time step
def callback(t):
    VTX_write(t, vtx, save_functions, v, list(external_state_variables.keys()), qmap)


t = 0.0
t_goals = (
    np.concatenate(
        ([i for i in range(100, 4000, 100)], [i for i in range(4000, 90000, 1000)])
    )
    / 3600 # Time is in hours
)

adaptive_solver(
    newton,
    problem,
    v,
    qmap,
    t,
    t_goals,
    1000 / 3600,
    1e-4,
    1,
    dt,
    material,
    update_variables={ml_n: ml, S_n: S, xi_1_n: xi_1, xi_2_n: xi_2},
    callback=callback,
)

vtx.close()
