## Setup
Make sure you have Python (3.12.0 recommended) and pip installed

Run `pip install -r requirements.txt` 

Note: this includes several packages that may or may not actually be explicitly required as they were used during development in some way. They may take up some additional diskspace, but will not interfere with the code.

## Reproducing the results
To reproduce the results from the thesis it is required to download the datasets first. They can be found at `https://data.4tu.nl/`. The specific datasets are specified in the thesis itself. Take not of the filenames, and make sure they match with those in the code.

After downloading and placing the datasets, you can run any of the specific functions in `main.py` to produce the results. Take note that the `preliminary()` is intended to be used without heurstic. You can disable the heuristic in the `coset_base.py` file at line 285 by setting `h` to `0`. In the `preliminary()` function you will have to adjust the noise threshold and filenames yourself to match those in experiment 1. The `main()` function requires you to run it once with, and once without heuristic. After running this function, move the results over to another folder to ensure they are not overwritten when running the function again.

## Reproducing the artificial models
The artificial models for experiment 2 can be produced by running `python model_builder.py`.

## Reproducing the figures and tables
The functions in `stats.py` can reproduce the figures and tables in the thesis. Some figures may or may not require first running the `fix_sp_count()`, `fix_sp_count_split()`, or `add_metrics()` functions if you have reproduced the results yourself.
