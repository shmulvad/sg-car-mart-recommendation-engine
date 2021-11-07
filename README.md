# KDDM Project

Our project for the module [CS5228 Knowledge Discovery and Data Mining][mod] at National University of Singapore.

* **Team**: All That Data
* **Group members**: Soeren Hougaard Mulvad, Kanav Sabharwal, Saurabh Jain, Sai Dhawal Phaye


## TODO

- [ ] We have a weird folder called `0_dev`. I am not really sure what the two notebooks in there do. Looks like they might be related to cleaning before we unified the cleaning methods? If that's the case, it should be deleted.
- [ ] I cannot find the code for the regressors. Has this been committed?
- [ ] We have data files in `/data`, `/scraped_data`, `/task1` and `/task1/data`.
- [ ] What is the point of all the files in `/scraped_data/cols_to_drop`? They look very weird on my end.
- [ ] There should be some more explanation of the files in `/scraped_data`. For example, what is `train_3.tsv` and `test_3.tsv`?
- [ ] Are we using the contents of `predictor.py` in any notebooks? I cannot find it if that is the case.
- [ ] `dl.py` is extremely unreadable. There are many variables that are undefined, there is no documentation, etc. In general, it deviates a lot from the remaining codebase.
- [ ] Maybe we should move the Streamlit app to a separate repo. Often, the Heroku deployment may fail due to a lot of large files or installs that actually aren't needed for Streamlit. If so, then we can also remove files like `Procfile`, `setup.sh`, etc.
- [ ] We should probably include the code that was used for scraping


### Getting Started

Make sure you have Python installed. The code should work on Python 3.8+. Then just run the following and you should be good to go!

```bash
# We set clone depth to 1 so prior data files are not downloaded
$ git clone --depth 1 https://github.com/shmulvad/kddm-project.git
$ cd kddm-project
$ pip install -r requirements.txt
```

### Project Structure

`data/` is a folder containing the raw train/test data, the same data after doing preliminary cleaning and replacing missing values using similarity approach and an embedding matrix for the cleaned train data (for use in task 2).

`scraped_data/` is a folder containing all the data that was scraped from the internet. We mainly scraped the fuel type.

`constants.py` defines a number of constants, such as the threshold for the number of NaN entries at which we deem a row not valuable to keep in our training data, the paths to different data, columns that we have determined to drop entirely, and lists of which columns are purely string columns and which are purely numerical columns.

`utils.py` just holds utility functions, such as safely checking if a value is NaN for different types, getting the squared difference of the 5th quantile and 95th quantile for a certain column, etc.

`cleaner.py` contains the main cleaning function responsible for doing all preliminary cleaning. It also has functions that, when the DataFrame is cleaned, makes certain columns categorical or likewise.

`sim_func.py` holds functions that compute the similarity between different types of elements, such as between two strings, between two sets, etc. These are used in `similarity.py` that defines code for computing the row-wise similarity between (optionally) all element in two DataFrames. This is used both for cleaning by replacing NaN values with the most similar values and for task 2 for finding similar items.

`item_filters.py` defines code that is used for task 2 for filtering items based on whether certain criteria are met.

TODO: `dl.py`, `predictor.py`, `eda_pr.ipynb`, `ensemble_csvs.ipynb`,

### Task 1 - Regression of Car Resale Price

TODO: Expand upon this.


### Task 2 - Recommendation of Similar Rows

Our entry point for Task 2, recommendations for similar rows, can be found in `task2.ipynb`. Important supplementary files for task 2 if one has an interest in diving deeper into the code are `similarity.py`, `sim_func.py` and `item_filters.py`.


### Task 3 - General

Task 3 is implemented in a two-fold manner.

First, there is a [Streamlit][streamlit] web app that can be found in `app.py` and is deployed to <https://best-sg-car-deals.herokuapp.com>. It can be run locally by executing `streamlit run app.py` (assuming the user has already followed the installation guide and navigated to the project folder). Otherwise it can be accessed by following the deployment link. It is deployed on a free tier at Heroku, so it may load for quite some time initially since the server has to start up.

Secondly, TODO: Expand upon this.


[streamlit]: https://streamlit.io
[mod]: https://nusmods.com/modules/CS5228/knowledge-discovery-and-data-mining
