### To activate this environment, use

    conda activate ./conda-env
or

    conda activate /Users/sogasg/dev/cluda/tb2/tb2-local/models/ml-1/conda-env

### To deactivate an active environment, use

    conda deactivate

## The fix
    export PYTHONPATH=.

## Example of running on Raspberry Pi
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1 # to make python print to the log file continually
    nohup python3 strategies/third/live-runner.py &>> ./log/6-jun-2020.txt & # to write logs to file and make sure that the process keeps running when the terminal is exited

Then show live logs:
    tail -F ./log/6-jun-2020.txt 

## Installing packages

First try with conda install
then:
     ./conda-env/bin/pip install black 

### Backtest a strategy

### Build features

### Evaluate a model

### Test if target is a good target
1. Create target
2. trade target directly.
3. Then we know if its a good target