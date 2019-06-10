See below for a description of the scripts in the code/ directory.

NOTE: the user should modify the global variables in definitions.py before 
	  running the other scripts.

===============================================================================
Scripts
===============================================================================
+----------------+
| definitions.py |
+----------------+
Defines global variables used by the other scripts.

+---------------------+
| process_raw_data.py |
+---------------------+
Processes the raw data.  Creates processed train, validation, and test
subdirectories. Creates processed data files.

+------------------------+
| load_processed_data.py |
+------------------------+
Contains functions used to load the processed data.

+--------+
| eda.py |
+--------+
Analyzes general image properties. Creates image files with results.

+-----------+
| models.py |
+-----------+
Contains functions used to construct the classifier and convolutional bases. 

+----------------+
| fit_model_1.py |
| fit_model_2.py |
| fit_model_3.py |
+----------------+
Fits model X (X = 1, 2, 3). Saves the HDF5 model file and creates an image
file with a plot of the training history.  

+---------------------------+
| utils_model_fitting.py    |
| utils_training_history.py |
+---------------------------+
Contains functions used during the model fitting.

+--------------------+
| evaluate_models.py |
+--------------------+
Evaluates the models. Creates image files of the results.

+---------------------------+
| utils_model_evaluation.py |
+---------------------------+
Contains functions used to evaluate the models.

+-------------------------------------+
| binary_classification_evaluation.py |
+-------------------------------------+
Evaluates model performance for a binary classification task. Creates image
files of the results.
