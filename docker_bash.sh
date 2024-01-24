#!/bin/bash
alias python=python3
python -m pip install --upgrade pip
python -m pip uninstall -y cbadc
python -m pip install -r requirements.txt
python opamp.py simulate setup local
bash ./bash_scripts/local_simulation.sh
python opamp.py process