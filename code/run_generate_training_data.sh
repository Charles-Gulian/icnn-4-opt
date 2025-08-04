#!/bin/bash

# Set test parameters
export CASE_NAME=case14
export INPUT_DIM=3
export NUM_SAMPLES=5000

# Call MATLAB and run the script
matlab -nodisplay -nosplash -r "try, generate_training_data; catch e, disp(getReport(e)), exit(1); end; exit(0);"
