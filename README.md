# KDDM Project

<!-- TODO: What is the 0_dev folder? -->

Our project for the module [CS5228 Knowledge Discovery and Data Mining][mod] at National University of Singapore.

* **Team**: All That Data
* **Group members**: Soeren Hougaard Mulvad, Kanav Sabharwal, Saurabh Jain, Sai Dhawal Phaye


### Getting Started

Make sure you have Python installed. Then just run `pip install -r requirements.txt` and you should be good to go!

### Project Structure

`constants.py` defines a number of constants, such as the threshold for the number of NaN entries at which we deem a row not valuable to keep in our training data, the paths to different data, columns that we have determined to drop entirely, and lists of which columns are purely string columns and which are purely numerical columns.

`utils.py` just holds utility functions, such as safely checking if a value is NaN for different types, getting the squared difference of the 5th quantile and 95th quantile for a certain column, etc.

`cleaner.py` contains the main cleaning function responsible for doing all preliminary cleaning. It also has functions that, when the DataFrame is cleaned, makes certain columns categorical or likewise.

`sim_func.py` holds functions that compute the similarity between different types of elements, such as between two strings, between two sets, etc. These are used in `similarity.py` that defines code for computing the row-wise similarity between (optionally) all element in two DataFrames. This is used both for cleaning by replacing NaN values with the most similar values and for task 2 for finding similar items.

The code glueing it all together is defined in the corresponding Jupyter Notebooks.

### Task 1

`task1/regressor.ipynb`.

TODO: Expand upon this.


### Task 2

Task 2 can be found in `task2/task2.ipynb`. It uses the similarity functions to find the most similar rows in the training dataset after preliminary cleaning, skipping all columns that are not present in the

Since a similarity is computed for the row in question and all rows in the training dataset, it can be somewhat slow to execute. In general, it takes about 2-3 seconds to compute the recommendations the first time for a given row.


### Task 3

Task 3 is implemented as a [Streamlit][streamlit] web app and can be found in `app.py`. It can be run by  running `streamlit run app.py` (assuming the user has already run `pip install -r requirements.txt`).

TODO: Expand upon this.


[streamlit]: https://streamlit.io
[mod]: https://nusmods.com/modules/CS5228/knowledge-discovery-and-data-mining
