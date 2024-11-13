from dolfinx import fem
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx.cpp.nls.petsc import NewtonSolver
from typing import Callable, List, Dict, Any

from numpy.typing import NDArray
from numpy import float_

def none_callback(t):
    return None

def adaptive_solver(
    solver: NewtonSolver,
    problem: NonlinearMaterialProblem,
    unknown: fem.Function,
    qmap: QuadratureMap,
    t0: float,
    checkpoint_times: NDArray[float_],
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt: fem.Constant,
    material: MFrontMaterial,
    update_variables: Dict[fem.Function, fem.Function] = {},
    callback: Callable[[float],None] = none_callback,
) -> bool:
    t = t0
    current_dt = dt_init
    for next_t in checkpoint_times:
        current_dt = next_t - t
        while t < next_t:
            if current_dt > 0.99 * (next_t - t):
                current_dt = next_t - t  # To avoid arbitrarily small time steps
            print(f"Current time: {t:.2f} hours, current dt: {current_dt*3600:.0f} seconds")
            dt.value = current_dt
            material.dt = current_dt
            backup_fluxes = {
                name: flux.vector.copy() for (name, flux) in qmap.fluxes.items()
            }
            backup_isv = {
                name: flux.vector.copy()
                for (name, flux) in qmap.internal_state_variables.items()
            }
            backup_unknown = unknown.vector.copy()
            try:
                converged, it = problem.solve(solver, print_solution=True)
                t += current_dt
                for function_n, function in update_variables.items():
                    function.vector.copy(function_n.vector)
                current_dt = min(dt_max, 1.1 * current_dt)
            except RuntimeError:
                backup_unknown.copy(unknown.vector)
                for name, flux in qmap.fluxes.items():
                    backup_fluxes[name].copy(flux.vector)
                for name, flux in qmap.internal_state_variables.items():
                    backup_isv[name].copy(flux.vector)
                current_dt = current_dt / 2
                if current_dt < dt_min:
                    return False
        callback(t)
    return True
