# KDDM Project - Streamlit App

This branch contains the code for our Streamlit app.

The [Streamlit][streamlit] web app that can be found in `app.py` and is deployed to <https://best-sg-car-deals.herokuapp.com>. It is deployed on a free tier at Heroku, so it may load for quite some time initially since the server has to start up.


### Running Locally

Make sure you have Python installed. The code should work on Python 3.8+. Then just run the following and you should be good to go!

```bash
# We set clone depth to 1 so prior data files are not downloaded
$ git clone --depth 1 https://github.com/shmulvad/kddm-project.git
$ cd kddm-project
$ git checkout streamlit
$ pip install -r requirements.txt
$ streamlit run app.py
```


[streamlit]: https://streamlit.io
[mod]: https://nusmods.com/modules/CS5228/knowledge-discovery-and-data-mining
