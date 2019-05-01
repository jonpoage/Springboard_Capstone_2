# Springboard_Capstone_2
This repository contains my second Springboard capstone project.

Title: Chest X-Ray Image Classification Using Deep Learning

Procedure for obtaining the raw data:
1. Download the file ZhangLabData.zip from the website
    https://data.mendeley.com/datasets/rscbjbr9sj/3
2. Extract the directory chest_xray/ from the downloaded .zip file.  

The repository contains the directories listed below.

===============================================================================
code/
===============================================================================
This directory contains the Python scripts that were used to prepare the 
data and perform the analysis for this project.  

NOTE - The global variables in the script definitions.py are set for my 
machine.  The user should alter the global variables in this script before
running any of the other scripts.

Scripts:

definitions.py --------------- Global variables used by the other scripts
eda.py ----------------------- Exploratory Data Analysis (EDA) 
                               for Analysis part 1
load_processed_data.py ------- Load the processed data as well as 
                               feature and target arrays
models.py -------------------- Models for Analysis parts 2 and 3
process_raw_data.py ---------- Process the raw data

===============================================================================
data/
===============================================================================
This directory contains the raw and processed data used for this project.

NOTE - The data sets are too large to store on GitHub. 
The user should create the following subdirectories in their repository:

raw/ 
processed/

Next, the user should obtain the raw data using the procedure above and copy 
the directory chest_xray/ into the raw/ subdirectory.

Finally, the user should run the script process_raw_data.py (after modifying
the global variables in definitions.py) to process the raw data and fill the
processed/ subdirectory.

The processed/ subdirectory will then have the structure described below.

files: 

y_ohe_class_names.txt -------- Class labels for the target array columns

subdirectories:

test_data/ ------------------- test data set
    image_files/ ----------------- test set JPEG image files
    test_data.pickle ------------- Pandas DataFrame with processed data
                                   for the images in the test set
train_data/ ------------------ train data set
    image_files/ ----------------- train set JPEG image files
    train_data.pickle ------------ Pandas DataFrame with processed data
                                   for the images in the train set
validation_data/ ------------- validation data set
    image_files/ ----------------- validation set JPEG image files
    validation_data.pickle ------- Pandas DataFrame with processed data
                                   for the images in the validation set

===============================================================================
docs/
===============================================================================
This directory is currently empty.

Relevant repository documentation will be added to this directory as needed.

===============================================================================
models/
===============================================================================
Models for Analysis parts 2 and 3 are stored here as HDF5 files.

===============================================================================
notebooks/
===============================================================================
Jupyter notebooks were stored here.

NOTE - jupyter notebooks were used to perform a first pass of data processing
and analysis. The notebooks were saved to this directory while they were being
used. After the analysis in a particular notebook was completed, the notebook 
was refactored into one or more Python scripts, after which the notebook was 
deleted.

===============================================================================
references/
===============================================================================
This directory contains relevant reference materials.

===============================================================================
reports/
===============================================================================
This directory contains the reports for this project.

Files: 

Capstone Project 2 Proposal.pdf --- Project Proposal

Subdirectories:

figures/ -------------------------- Image files used in the report documents
