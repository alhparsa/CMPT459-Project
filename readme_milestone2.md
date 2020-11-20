Before running the code of `milestone_2.ipynb`, please make sure you have all the packages required for our program. You can simply install them by running:
``` pip install -r requirements.txt```

The pickled models for adaboost and knn should be available in the `data` folder after running the second cell in the notebook (it will automatically download the necassary files) and the `models` folder contains the modules for the neural network model.

The `data` folder also contains the `load_data` module which has the `CleanData` class that firstly downloads the required files from the repository and then impute and cleans the `joined.csv.gz` dataset (the one from previous milestone). Make sure you are connected to the internet when running the object initialization cell in the notebook as it will download around 120mb of data.