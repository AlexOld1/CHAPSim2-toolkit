# CHAPSim2-toolkit
A python post-processing and toolkit program based on NumPy and Matplotlib for DNS solver CHAPSim2.

## Install:

Dependencies are given in requirements.txt.
pip: Navigate to base directory and run 'pip install .'
conda: Navigate to base directory and run 'conda env create -f environment.yml' to create a conda environment for the program then 'conda activate chapsim2-toolkit' to use the environment.

## Scripts:

**gui.py**: This launches a user interface for turbulence statistics, slice visualisation, monitoring points and mesh analysis, run 'python gui.py'. The Mesh Analysis tab is interactive: load an input_chapsim.ini (or start from the built-in template), adjust cell counts, stretching and flow parameters with sliders, and the resolution report, headline metrics and spacing plot update live. The adjusted settings can be written back out as an input file. This will likely not work on HPCs, use interactive script input instead (run each script individually).

**quick_turb_stats.py**: Single case post-processing script, outputs a figure for velocity/TKE/Temperature and a figure for Reynolds stresses. Ideal for use on HPCs with only numpy, matplotlib and tqdm dependencies. Recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**slice.py**: 2D visualisation of any output parameter with matplotlib plotting options. Also recommended to be run on a serial/ data analysis or interactive node as bandwidth is typically throttled on login nodes. Interactive input.

**turb_stats.py**: Main post-processing script to provide velocity, temperature, Reynolds stress profiles, Reynolds stress budget terms and heat transfer statistics such as turbulent Prantl number and Nusselt number from CHAPSim2 text file or xdmf output. Input parameters, cases for comparison, plotting options etc. are specified on config.py file for interactive input. Plots saved in turb_stats_plots/ and to file path.

**monitor_points.py**: Plotting for bulk and point monitors, includes functionality to crop diverged data. Run the script in the directory containing monitor point files or specify a path to files. Interactive Input.

**mesh_analysis.py**: Pre-processing mesh resolution analysis. Reads a case's input_chapsim.ini, rebuilds the wall-normal grid exactly as the solver does, and assesses it against DNS resolution requirements (dy+, dx+, dz+, grid stretching, MHD boundary layer, recommended minimum mesh and time step). Python port of the estimate_spacial_resolution/estimate_temporal_resolution routines in CHAPSim2's apx_prerun_mod, so a mesh can be checked before a job is submitted. Run 'python mesh_analysis.py path/to/input_chapsim.ini', or with no argument to be prompted. Optionally saves a mesh distribution plot.

**thermal_BC_calc.py**: Property functions for liquid metals in CHAPSim2, functionality to output NIST format data file, convert a given Grashof number to constant wall temperature difference or heat flux (channel flow), calculate Prandtl number. Interactive input.

## Reference Data:

Isothermal channel (MKM180), square duct (KTH) reference data is provided as well as isothermal and heated MHD reference data (NK). All reference data is openly accessible from published sources. Copyright for reference datasets remains with the original authors/publishers. See individual data files for citations.