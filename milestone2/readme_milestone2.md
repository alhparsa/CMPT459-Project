You can download the compressed data from:
https://github.com/alhparsa/CMPT459-Project/raw/main/data/joined.csv.gz

The folder data contains the module load_data which contains two classes, `Data` and `CleanedData`. For this milestone, we will be using the class `CleanedData` which does the data imputing and cleaning from our joined dataset. After downloading the `joined.csv.gz`, make sure to pass the correct path to `loc` parameter of the object intializer in the `milestone_2.ipynb` file.

Before running the code of `milestone_2.ipynb`, please make sure you have all the packages required for our program. You can simply install them by running:
``` pip install -r requirements.txt```

The pickled models for adaboost and knn are available in the `data` folder and the `models` folder contains the modules for the neural network model. Besides the data location, you wouldn't need to make any changes to the notebook file to get the program running.