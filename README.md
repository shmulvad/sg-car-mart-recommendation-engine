# KDDM Project

Our project for the module [CS5228 Knowledge Discovery and Data Mining][mod] at National University of Singapore.

* **Team**: All That Data
* **Group members**: Soeren Hougaard Mulvad, Kanav Sabharwal, Saurabh Jain, Sai Dhawal Phaye

### Getting Started

Make sure you have Python installed. The code should work on Python 3.8+. Then just run the following and you should be good to go!

```bash
# We set clone depth to 1 so prior data files are not downloaded
$ git clone --depth 1 https://github.com/shmulvad/kddm-project.git
$ cd kddm-project
$ pip install -r requirements.txt
```


## Project Structure

### Data

* `data/` is a folder containing the raw train/test data, the same data after doing preliminary cleaning and replacing missing values using similarity approach, an embedding matrix for the cleaned train data (for use in task 2) and an embedding matrix for just the titles for task 3.
* `scraped_data/` is a folder containing all the data that was scraped from the internet. We mainly scraped the fuel type.

### Library Code

* `constants.py` defines a number of constants, such as the threshold for the number of NaN entries at which we deem a row not valuable to keep in our training data, the paths to different data, columns that we have determined to drop entirely, and lists of which columns are purely string columns and which are purely numerical columns.
* `utils.py` just holds utility functions, such as safely checking if a value is NaN for different types, getting the squared difference of the 5th quantile and 95th quantile for a certain column, etc.
* `cleaner.py` contains the main cleaning function responsible for doing all preliminary cleaning. It also has functions that, when the DataFrame is cleaned, makes certain columns categorical or likewise.
* `sim_func.py` holds functions that compute the similarity between different types of elements, such as between two strings, between two sets, etc.
* `similarity.py` defines code for computing the row-wise similarity between (optionally) all element in two DataFrames. This is used both for cleaning by replacing NaN values with the most similar values and for task 2 for finding similar items. It uses the functions defined in `sim_func.py`.
* `item_filters.py` defines code that is used for task 2 for filtering items based on whether certain criteria are met.
* `predictor.py` defines code that is used to train and infer machine learning models for filling missing values in the dataset.


### Generator Code

Our generator code is included for reproducibility. It generates the data we have and is set up as Python scripts that can be executed directly by running `python [FILE.py]` (given the environment is set up correctly). They may take a long time to run, but in the end, they will produce the data mentioned.

* `generate_sim_df.py` generates the similarity dataframes for the train and test data and new CSV files where the missing numerical values have been replaced.
* `generate_sentence_embeddings.py` generates the sentence embedding matrices that are used in respectively task 2 and task 3.
* `scraper.py` scrapes sgCarMart for car codes, sub codes and fuel type.

### Notebooks

All our notebooks can be found in the `notebooks/`-folder. They are as follows:

* `notebooks/eda.ipynb` contains the code that was used for doing exploratory data analysis and the code to generate the plots we include in our report.
* `notebooks/task1_approach1.ipynb` implements the baseline regressor over naive data with minimal pre-processing and gives a baseline RMSE score
* `notebooks/task1_approach2.ipynb` implements the LightGBM regressor over ml-filled and similarity filled data
* `notebooks/task1_approach3.ipynb` implements the LightGBM and CatBoost regressors over naive data cleaned with directed pre-processing
* `notebooks/task2_recommender.ipynb` implements and presents the recommender system. Note that a sentence transformer model is used which may take some time to download a model.
* `notebooks/task3_car_finder.ipynb` implements and presents the system we have that can search for car listings based on a textual search query.


## Tasks

### Task 1 - Regression of Car Resale Price

Task1 has been sub-divided into 3 notebooks corresponding to the 3 approaches that our report mentions. The notebooks are self-explanatory.

* `notebooks/task1_approach1.ipynb`: Approach 1 shows the working of a basic regressor setup on naive data cleaned dataset.
* `notebooks/task1_approach2.ipynb`: Approach 2 highlights the over-engineering situation that ml-filled/similarity-filled datasets suffer from. One can run the data-filling setup using code-cells. Alternatively, we provide loading of filled data to show the prime objective - the over-fit solution.
* `notebooks/task1_approach3.ipynb`: Approach 3 expands on using naive data with directed pre-processing and using LightGBM or CatBoost Regressor.


### Task 2 - Recommendation of Similar Rows

Our entry point for Task 2, recommendations for similar rows, can be found in `notebooks/task2_recommender.ipynb`. Important supplementary files for task 2 if one has an interest in diving deeper into the code are `similarity.py`, `sim_func.py`, `item_filters.py` and `generate_sentence_embeddings.py`.


### Task 3 - General

Task 3 is implemented in a two-fold manner.

First, there is a [Streamlit][streamlit] web app that can be found on the branch [`streamlit`][streamlitBranch] in the file [`app.py`][streamlitApp] and is deployed to <https://best-sg-car-deals.herokuapp.com>. It is deployed on a free tier at Heroku, so it may load for quite some time initially since the server has to start up.

Secondly, we have the car finder in `notebooks/task3_car_finder.ipynb` that searches for car titles where the features or other values match a textual description given as a query by the user.


[streamlit]: https://streamlit.io
[mod]: https://nusmods.com/modules/CS5228/knowledge-discovery-and-data-mining
[streamlitBranch]: https://github.com/shmulvad/kddm-project/tree/streamlit
[streamlitApp]: https://github.com/shmulvad/kddm-project/blob/streamlit/app.py
