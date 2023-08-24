# analysea
broad set of routine to analyse metocean condtions (free surface, tides, waves) and extreme events.

## requirements
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

to install directly the dependencies for running the package, you can run the following:

    mamba env create -f requirements/env.yaml
