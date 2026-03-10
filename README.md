# DustDevol

Python package used to run chemical evolution models, designed for easy implementation of new models. Provides a function, `evolve_2o_FC`, which takes in various galaxy model parameters, and returns a dictionary containing the masses, derivatives, and an interpolation function for the gas, stars, metals, and dust, among others. The code comes from a restructuring of the BEDE code, used in De Vis et. al 2021, with primarily back-end changes, although some newer models and considerations are added as well. 

## Installation
To install, clone this repository, `cd` into it, and run your favorite python package installer on the directory. For example, using pip,
```
pip install .
```

## Requirements

### Python Packages
- numpy
- scipy
- astropy

### Input Parameters
`evolve_2o_FC` requires several parameters. These can either be passed as positional arguments to the function, or, how we recommend, is creating a dictionary which is then unpacked, e.g. `evolve_2o_FC(**my_model)`. The parameters needed are as follows:
- time_start: time at which to start the simulation, measured in Gyr after the big bang.
- time_end: time at which to end the simulation, also measured in Gyr after the big bang.
- sfr_model: function which returns the star formation rate given the current state of the galaxy.
- imf: function which takes in a stellar mass and returns the value of the imf at that mass.
- inflow_model: function for the inflow rate
- outflow_model: function for the outflow rate
- recycling_model: function for the recycling rate
- grain_growth_model: function for dust grain growth
- destruction_model: function for dust destruction
- ejecta_model: function for ejecta from dying stars
- init_gas: array of gas masses at t=0
- init_star: array of star masses at t=0
- init_metal: array of metal masses at t=0
- init_dust: array of dust masses at t=0
- model_params: dictionary containing all the needed "extra" parameters fo each of the models
- absolute_tolerance: error tolerance for each component in solar masses
- relative_tolerance: error tolerance for each component in fraction of current mass

### Output Structure
`evolve_2o_FC` returns a dictionary with the following components;
- times: array of times the integrator visits
- gas_masses: gas masses at each time
- star_masses: star masses at each time
- metal_masses: metal masses at each time
- dust_masses: dust masses at each time
- dgas_masses: derivative of the gas mass at each time step
- dstar_masses: likewise with stars
- dmetal_masses: likewise with metals
- ddust_masses: likewise with dust
- gas_func: function which returns gas mass at a given time
- star_func: likewise with stars
- metal_func: likewise with metal
- dust_func: likewise with dust
- sfr: star formation rate at each time
- sfr_func: function which returns sfr at a given time
- cache: dictionary containing various miscellaneous outputs. Will always contain the key "attempted_steps", giving how many steps the integrator tried, including steps that failed due to too large error. All other contents are specific to the gas_masses model.
