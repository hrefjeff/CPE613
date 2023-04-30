# Reference Implementations for Convolution

## Files

1. `driver.py` - script with the main function
2. `convolve_funcs.py` - implementations of the different convolution methods
3. `utils.py` - utility functions such as reading/writing from files
4. `requirements.txt` - python packages required for download into virtualenv
5. `../test_data_gold` - location for data that we know is correct
6. `result_checker.py` - script that can be used to check accuracy of output

## How to run on fresh environment

1. Ensure Python3 & Pip is installed
2. Run cmd `python3 -m venv env` to create the virtual environment
3. Run cmd `python3 -m pip install -r requirements.txt`
4. Edit `driver.py`'s "Define info for run section" for desired dataset
5. Run cmd `./driver`
6. View results in `../test_data` folder

## How to test

1. Open the `result_checker.py`
2. Update the `test_file` & `gold_file` variables to which data sets to compare
3. Run `./result_checker.py`

## List of Convolution Methods

1. Discrete time domain
2. Discrete FFT
3. (WORK IN PROGRESS) Discrete overlap add
4. (WORK IN PROGRESS) Discrete overlap save
