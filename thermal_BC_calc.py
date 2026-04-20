#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ====================================================================================================================================================

"""
Thermophysical properties of mediums in CHAPSim2 are provided in classes.
Functions for generating property tables or temp. difference from Grahsof number are provided.
"""

from utils import (
    LiquidLithiumProperties,
    LiquidPbLiProperties,
    LiquidSodiumProperties,
    LiquidLeadProperties,
    LiquidBismuthProperties,
    LiquidLBEProperties,
    LiquidFLiBeProperties,
    get_fluid_properties,
)

def generate_property_table(T_min, T_max, T_ref, pressure=0.1, n_points=20,
                           save_tsv=False, filename='lithium_properties.tsv'):
    """
    Generate a property table for liquid lithium over a temperature range.
    
    Parameters:
    -----------
    T_min : float
        Minimum temperature in K
    T_max : float
        Maximum temperature in K
    T_ref : float
        Reference temperature for enthalpy and entropy calculations
    pressure : float
        Pressure in MPa (default 0.1 MPa = ~1 atm)
    n_points : int
        Number of temperature points
    save_tsv : bool
        Whether to save the table as CSV
    filename : str
        Output filename if save_tsv=True
    
    Returns:
    --------
    pandas.DataFrame
        Property table with requested columns
    """
    
    li = LiquidLithiumProperties()
    
    # Check validity
    if T_min < li.T_melt:
        print(f"Warning: T_min ({T_min} K) is below melting point ({li.T_melt} K)")
    if T_max > li.T_boil:
        print(f"Warning: T_max ({T_max} K) is above boiling point ({li.T_boil} K)")
    
    # Generate temperature array
    T_array = np.linspace(T_min, T_max, n_points)
    
    # Calculate properties in the requested order
    data = {
        'Temperature (K)': T_array,
        'Pressure (MPa)': [pressure] * n_points,
        'Density (mol/l)': [li.density_molar(T) for T in T_array],
        'Volume (l/mol)': [li.molar_volume(T) for T in T_array],
        'Internal Energy (kJ/mol)': [li.internal_energy(T, T_ref) for T in T_array],
        'Enthalpy (kJ/mol)': [li.enthalpy(T, T_ref) for T in T_array],
        'Entropy (J/mol*K)': [li.entropy(T, T_ref) for T in T_array],
        'Cv (J/mol*K)': [li.heat_capacity_v(T) for T in T_array],
        'Cp (J/mol*K)': [li.heat_capacity_p_molar(T) for T in T_array],
        'Sound Spd. (m/s)': [li.speed_of_sound(T) for T in T_array],
        'Joule-Thomson (K/MPa)': [li.joule_thomson(T) for T in T_array],
        'Viscosity (uPa*s)': [li.viscosity(T) for T in T_array],
        'Therm. Cond. (W/m*K)': [li.thermal_conductivity(T) for T in T_array],
        'Phase': [li.phase(T, pressure) for T in T_array]
    }
    
    df = pd.DataFrame(data)
    
    # Save if requested
    if save_tsv:
        df.to_csv(filename, sep='\t', index=False)
        print(f"Property table saved to {filename}")
    
    return df

def Grahsof_to_temp_diff(grahsof, beta, L, mu, rho):
    """
    Convert Grahsof number to temperature difference in K.
    
    Grahsof = g * beta * Delta_T * L^3 / nu^2
    Delta_T = Grahsof * nu^2 / (g * beta * L^3)
    
    Assumptions:
    - g = 9.81 m/s²
    
    Parameters:
    -----------
    Grahsof : float
        Grahsof number
    Returns:
    """
    g = 9.81
    nu = mu / rho
    delta_T = (grahsof * nu**2) / (g * beta * L**3)
    
    return delta_T

def get_heat_flux(delta_T, k, L_ref):
    """
    Calculate heat flux from temperature difference.

    Parameters:
    -----------
    delta_T : float
        Temperature difference in K
    k : float
        Thermal conductivity in W/(m·K)
    L_ref : float
        Reference length in m

    Returns:
    --------
    float
        Heat flux in W/m²
    """
    heat_flux = k * delta_T / L_ref
    return heat_flux

def calc_fourier_number(k, rho, cp, L, timestep, nondim_time=True):
    """
    Calculate Fourier number.

    For dimensional time:    Fo = alpha * t / L^2
    For non-dimensional time (U_ref=1): Fo = alpha * t* / L
        where t_actual = t* * L (since t* = t / L with U_ref = 1 m/s)

    Parameters:
    -----------
    k : float
        Thermal conductivity in W/(m·K)
    rho : float
        Density in kg/m³
    cp : float
        Specific heat capacity at constant pressure in J/(kg·K)
    L : float
        Characteristic length in m
    timestep : float
        Time (dimensional in s, or non-dimensional if nondim_time=True)
    nondim_time : bool
        If True, timestep is non-dimensional (t* = t/L with U_ref=1 m/s)
        If False, timestep is dimensional in seconds

    Returns:
    --------
    float
        Fourier number (dimensionless)
    """
    alpha = k / (rho * cp)
    if nondim_time:
        # t_actual = t* * L, so Fo = alpha * t* * L / L^2 = alpha * t* / L
        Fo = alpha * timestep / L
    else:
        # Dimensional time: Fo = alpha * t / L^2
        Fo = alpha * timestep / (L**2)
    return Fo

def get_prandtl(temp, fluid):
    """
    Calculate Prandtl number.

    Pr = (c_p * mu) / k

    Parameters:
    -----------
    temp : float
        Temperature in K
    fluid : object
        Fluid properties object (LiquidLithiumProperties or LiquidPbLiProperties)

    Returns:
    --------
    float
        Prandtl number (dimensionless)
    """
    if isinstance(fluid, LiquidLithiumProperties):
        mu = fluid.viscosity(temp) * 1e-6  # Convert from µPa·s to Pa·s
    else:
        mu = fluid.viscosity(temp)  # Already in Pa·s
    k = fluid.thermal_conductivity(temp)
    c_p = fluid.heat_capacity_p(temp)
    Pr = (c_p * mu) / k

    return Pr

def get_viscosity_Pa_s(fluid, T):
    """
    Get viscosity in Pa·s for any fluid type.
    """
    if isinstance(fluid, LiquidLithiumProperties):
        return fluid.viscosity(T) * 1e-6  # Convert from µPa·s to Pa·s
    else:
        return fluid.viscosity(T)  # Already in Pa·s

FLUID_NAMES = {
    LiquidLithiumProperties: "Liquid Lithium (Li)",
    LiquidSodiumProperties: "Liquid Sodium (Na)",
    LiquidLeadProperties: "Liquid Lead (Pb)",
    LiquidBismuthProperties: "Liquid Bismuth (Bi)",
    LiquidLBEProperties: "Liquid Lead-Bismuth Eutectic (LBE)",
    LiquidFLiBeProperties: "Liquid FLiBe (2LiF-BeF2)",
    LiquidPbLiProperties: "Liquid Lead-Lithium Eutectic (Pb-83Li-17)",
}


def interactive_calculation():
    """
    Interactive input for calculating temperature difference and heat flux
    from Grashof number, with material property summary.
    """
    print("\n" + "=" * 70)
    print("  Thermophysical Properties Calculator")
    print("=" * 70)

    # Get medium selection
    print("\nAvailable media:")
    print("  Li    - Liquid Lithium")
    print("  Na    - Liquid Sodium")
    print("  Pb    - Liquid Lead")
    print("  Bi    - Liquid Bismuth")
    print("  LBE   - Lead-Bismuth Eutectic")
    print("  FLiBe - Molten Salt (2LiF-BeF2)")
    print("  PbLi  - Lead-Lithium (Pb-17Li)")

    while True:
        medium_input = input("\nSelect medium (Li/Na/Pb/Bi/LBE/FLiBe/PbLi): ").strip()
        try:
            fluid = get_fluid_properties(medium_input)
            break
        except ValueError as e:
            print(f"Error: {e}")

    # Get Grashof number
    while True:
        try:
            grashof_input = input("Enter Grashof number (e.g., 5e7): ").strip()
            grashof = float(grashof_input)
            if grashof <= 0:
                print("Grashof number must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get reference temperature
    while True:
        try:
            T_ref = float(input(f"Enter reference temperature in K (valid range: {fluid.T_melt:.1f} - {fluid.T_boil:.1f}): ").strip())
            if T_ref < fluid.T_melt:
                print(f"Warning: Temperature below melting point ({fluid.T_melt} K)")
            if T_ref > fluid.T_boil:
                print(f"Warning: Temperature above boiling point ({fluid.T_boil} K)")
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get reference length
    while True:
        try:
            L_ref = float(input("Enter reference length in m: ").strip())
            if L_ref <= 0:
                print("Reference length must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get timestep for Fourier number calculation
    while True:
        try:
            timestep = float(input("Enter timestep (non-dimensional, t* = t/L with U_ref=1): ").strip())
            if timestep <= 0:
                print("Timestep must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Calculate properties
    mu = get_viscosity_Pa_s(fluid, T_ref)
    rho = fluid.density_mass(T_ref)
    k = fluid.thermal_conductivity(T_ref)
    cp = fluid.heat_capacity_p(T_ref)
    beta = fluid.coeff_vol_exp(T_ref)
    Pr = get_prandtl(T_ref, fluid)

    # Calculate temperature difference and heat flux
    delta_T = Grahsof_to_temp_diff(grashof, beta, L_ref, mu, rho)
    heat_flux = k * delta_T / L_ref
    T_hot = T_ref + delta_T

    # Calculate Fourier number (using non-dimensional time from CHAPSim2)
    Fo = calc_fourier_number(k, rho, cp, L_ref, timestep, nondim_time=True)
    alpha = k / (rho * cp)  # thermal diffusivity for display

    # Print results
    medium_name = FLUID_NAMES.get(type(fluid), "Unknown Fluid")

    print("\n" + "=" * 70)
    print(f"  Results for {medium_name}")
    print("=" * 70)

    print("\n--- Input Parameters ---")
    print(f"  Grashof number:            {grashof:.3e}")
    print(f"  Reference temperature:     {T_ref:.2f} K")
    print(f"  Reference length:          {L_ref:.4f} m")
    print(f"  Timestep (non-dim):        {timestep:.6e}")

    print("\n--- Material Properties at T_ref ---")
    print(f"  Density (rho):             {rho:.2f} kg/m³")
    print(f"  Dynamic viscosity (mu):    {mu:.6e} Pa·s")
    print(f"  Thermal conductivity (k):  {k:.4f} W/(m·K)")
    print(f"  Specific heat (Cp):        {cp:.2f} J/(kg·K)")
    print(f"  Thermal diffusivity (α):   {alpha:.6e} m²/s")
    print(f"  Vol. expansion (beta):     {beta:.6e} 1/K")
    print(f"  Prandtl number (Pr):       {Pr:.6f}")

    print("\n--- Calculated Results ---")
    print(f"  Temperature difference:    {delta_T:.6f} K")
    print(f"  Hot wall temperature:      {T_hot:.2f} K")
    print(f"  Heat flux (q''):           {heat_flux:.4f} W/m²")
    #print(f"  Fourier number (Fo):       {Fo:.6e}")

    print("\n" + "=" * 70 + "\n")

    return {
        'grashof': grashof,
        'T_ref': T_ref,
        'L_ref': L_ref,
        'timestep': timestep,
        'delta_T': delta_T,
        'heat_flux': heat_flux,
        'Fo': Fo,
        'alpha': alpha,
        'Pr': Pr,
        'rho': rho,
        'mu': mu,
        'k': k,
        'cp': cp,
        'beta': beta
    }

if __name__ == '__main__':
    interactive_calculation()
