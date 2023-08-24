# ANALYSEA
Broad set of routines to clean, process and visuaise metocean conditions:
- free surface,
- tides (using [utide](https://github.com/wesleybowman/UTide))
- waves

and statiscal analysis of extreme events.

⚠️ PROGRAM IN ACTIVE DEVELOPMENT. WHEN A STABLE API IS REACHED, THE PROGRAM WILL BE AVAILABLE VIA pypi
## Requirements
 - Time series
   - xarray
   - pandas
   - geopandas
   - pyarrow
   - fastparquet
   - ruptures
   - searvey
 - Tide analysis
   - utide
 - plotting
   - owslib
   - cartopy
   - shapely
   - matplotlib

## Install
to install directly the dependencies for running the package, you can run the following:

    mamba env create -f requirements/env.yaml

or git clone this repository, and run the following command

    pip install .

# Applications
