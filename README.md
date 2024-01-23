# A Control-Bounded Quadrature Leapfrog ADC
This repository contains the simulation code for the parper [A Control-Bounded Quadrature Leapfrog ADC][paper].

## Install

### Virtual Python Enviroment
To run the code it's reccomended to first create a fresh virutal python enviroment as
``` zsh
python3 -m venv venv
```
to activate the enviroment type
``` zsh
source venv/bin/activate
```
to deactivate type
``` zsh
deactivate
```

### Install dependencies
All package requirements are listed in the [requirements.txt](./requirements.txt) file.
To install them simply type
``` zsh
python -m pip install -r requirements.txt
```

#### Optional
In case you want to use the parallel execution mode of [simset] make sure to also install [GNU Parallel]. 

## Simulations
Next follows details on how to execute and resulting files for each simulation from the [paper][paper].
The simulations are presented in connection to the figures which they ultimately generated.

### Fig. 10: Transfer functions
The transfer functions where generated using the [./transfer_function.py](./transfer_function.py) script.
To execute it type
```zsh
python transfer_function.py
```
The script generates a csv file [./csv/tranfer_function/transfer_functions.csv](./csv/transfer_function/transfer_functions.csv) which was plotted using the 
[pgfplots] framework in the LaTex source file for the [paper]. In addition, the script generate the matplotlib plot [./figures/transfer_function/transfer_functions.png](./figures/transfer_function/transfer_function.png).

![transfer_functions](./figures/transfer_function/transfer_function.png)

### Fig. 11: Power Spectral Densities
The power spectral density related simulations can be found in the [./psd.py](./psd.py) script.
As the file contains several parameter sweeps we have used the [simset] package to orchestrate this. 
This slightly obscures the execution commands as we now type the following
```zsh
python psd.py simulate setup local
./bash_scripts/local_simulation.sh
python psd.py process
```
to execute the simulations.

Alternatively, for parallel execution, execute
```zsh
python psd.py simulate setup parallel
./bash_scripts/parallel_simulation.sh
python psd.py process
```
Note that this requires the [GNU parallel] to be installed, see [Optional install Section.](#optional-optional-install). Additionally, for parallel execution, take note of the 
```python
simset.concurrent_jobs = 48
```
setting in the [./psd.py](./psd.py) file.

#### Results
The script generates a number of csv files and png figures. 

The csv files contains the date used for plotting Fig. 11, using [pgfplots], in the [paper]. Specifically,
- [./csv/psd/psd_8_4](./csv/psd/psd_8_4.csv), 
- [./csv/psd/psd_6_4](./csv/psd/psd_6_4.csv),
- and [./csv/psd/psd_6_8](./csv/psd/psd_6_8.csv),

contains the PSD data whereas [./csv/psd/snr_8_4.csv](./csv/psd/snr_8_4.csv), [./csv/psd/snr_6_4.csv](./csv/psd/snr_6_4.csv), and [./csv/psd/snr_6_8.csv](./csv/psd/snr_6_8.csv) correspond to the evaluated SNRs.

The resulting figures can be found in [./figures/psd/](./figures/psd/) and contains PSD plots, SNR plots, state evolution plots, the final estimate in time-domain, and more. Some highligts follow below

##### PSDs
![N=6, OSR=8](./figures/psd/psd_6_8.png)
![N=6, OSR=4](./figures/psd/psd_6_4.png)
![N=8, OSR=4](./figures/psd/psd_6_8.png)

### Fig. 12: Excess Loop Delay

### Fig. 13: Component Variations

### Fig. 15 and Fig. 16: Opamp Implemenation

## Simset
[Simset][simset] is used as a convenience for orchestating the various parameter sweeps.
Some usefull commands when using [simset] are:

#### Print status, i.e., the number of successfull and unsuccessfull simulations for a given file
```zsh
python python_script_using_simset.py info
```
#### Clean directory from all simset simulation files
```zsh
simset clean
```
Note that [simset] will create utilitie files as part of its operations these files are confined in the [.data](./data), [./bash_scripts](./bash_scripts/), [./local](./local/), and [.parallel](./parallel/) directories.

[paper]: https://arxiv.org/abs/2211.06745
[simset]: https://github.com/hammal/simset
[GNU Parallel]: https://www.gnu.org/software/parallel/
[pgfplots]: https://ctan.org/pkg/pgfplots