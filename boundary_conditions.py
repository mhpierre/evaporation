from typing import Union
from dolfinx import fem
import ufl


def p_sat(T: Union[float, fem.function.Function]) -> ufl.algebra.Product:
    """
    Inputs:
        T: temperature in K
    Outputs:
        p_sat: Saturated water vapour pressure in Pa
    """
    return 610.5 * ufl.exp(17.269 * (T - 273.15) / (257.3 + T - 273.15))


def relative_humidity(pl, T):
    return ufl.exp(ufl.min_value(pl, 0.0) * 18.01528e-3 / (1000.0 * 8.31 * (T)))


def evap_fluid(pl, T, T_ext, Rh_ext):
    """
    Inputs:
        pl: internal liquid pressure (Pa)
        T: internal temperature (K)
        T_ext: external temperature (K)
        Rh_ext: external relative humidity (-)
    Outputs:
        W: water evaporation flux (kg/m^2/h)
    """
    return (
        3.6e3
        * 8.7e-8
        * 0.253
        * (relative_humidity(pl, T) * p_sat(T) - Rh_ext * p_sat(T_ext))
    )


def evap_heat(pl, T, T_ext, Rh_ext):
    """
    Inputs:
        pl: internal liquid pressure (Pa)
        T: internal temperature (K)
        T_ext: external temperature (K)
        Rh_ext: external relative humidity (-)
    Outputs:
        H: Heat flux from evaporation (J/m^2/h)
    """
    return (44200 / 18.01528e-3) * evap_fluid(pl, T, T_ext, Rh_ext)


def heat_convection(T, T_ext):
    return 3.6e3 * 25 * (T - T_ext)
