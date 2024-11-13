from dolfinx import fem, io, mesh
from typing import List, Tuple, Dict
from dolfinx_materials.material.mfront import MFrontMaterial
from dolfinx_materials.quadrature_map import QuadratureMap


def VTX_init(
    domain: mesh.Mesh,
    filename: str,
    material: MFrontMaterial,
    variable_names: List[str],
    interpolation: Tuple[str, int] = ("CG", 1),
) -> Tuple[io.utils.VTXWriter, Dict[str, fem.Function]]:
    """
    Initialises a file to save functions in the VTX format (ADIOS2)
    Inputs:
        domain: dolfinx.mesh.Mesh
        filename: str, name for the VTX file
        material: dolfinx_materials.material.mfront.MFrontMaterial
        variable_names: List[str], list of variable names to save
        interpolation: Tuple[str, int], the interpolation type and degree of the VTX file
    Outputs:
        vtx: dolfinx.io.utils.VTXWriter, the VTX file handle
        functions: List[dolfinx.fem.Function], the list of functions to save
    """
    functions = {}
    external_state_variables = dict(zip(material.external_state_variable_names,material.external_state_variable_sizes))
    for name in variable_names:
        if name == "Displacement":
            size = domain.geometry.dim
        elif name in external_state_variables:
            size = external_state_variables[name]
        elif name in material.gradients:
            size = material.gradients[name]
        elif name in material.internal_state_variables:
            size = material.internal_state_variables[name]
        else:
            raise NameError(f"No variable named {name}.")
        function = fem.Function(
            fem.functionspace(
                domain, interpolation if size == 1 else (*interpolation, (size,))
            ),
            name=name,
        )
        functions.update({name: function})
    vtx_file = io.VTXWriter(domain.comm, f"{filename}.bp", list(functions.values()))
    return vtx_file, functions


def VTX_write(
    time: float,
    vtx_file: io.VTXWriter,
    functions: Dict[str, fem.Function],
    external_variable: fem.Function,
    external_state_variable_names: List[str],
    qmap: QuadratureMap,
    interpolation: Tuple[str, int] = ("CG", 1),
) -> None:
    for function_name, function in functions.items():
        if function_name in external_state_variable_names:
            function.interpolate(
                external_variable.sub(
                    external_state_variable_names.index(function_name)
                ).collapse()
            )
        else:
            function.interpolate(qmap.project_on(function_name, interpolation))
    vtx_file.write(time)
