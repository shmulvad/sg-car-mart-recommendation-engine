import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import utils
from constants import TRAIN_PATH

st.set_page_config(
    page_title='The Best Car Deals in SG',
    page_icon='💰🚗',
    initial_sidebar_state='collapsed'
)


FAIR, GOOD, VERY_GOOD, EXCELLENT = 'Fair', 'Good', 'Very Good', 'Excellent'
DEAL_OPTIONS = [FAIR, GOOD, VERY_GOOD, EXCELLENT]
DEAL_PCT_CUTOFFS = 0.1, 0.2, 0.3, 0.6

COLS_TO_SHOW = ['title', 'make', 'model', 'type_of_vehicle', 'price', 'price_predicted']
BASE_URL = 'https://www.sgcarmart.com/used_cars/info.php?ID='


@st.cache(ttl=60)
def get_df():
    # Just for experimenting, in the end we will use the data from our best
    # predictor for the price_predicted
    df = pd.read_csv(TRAIN_PATH)
    mult = np.random.normal(0, 0.2, size=df.shape[0])
    df['price_predicted'] = df.price + df.price * mult
    diff = df.price_predicted - df.price
    df['pct_diff'] = diff / df.price
    for label, pct_cutoff in zip(DEAL_OPTIONS, DEAL_PCT_CUTOFFS):
        df[label] = df['pct_diff'] > pct_cutoff
    return df


def get_brands(df):
    return [brand.capitalize() for brand in df['make'].unique()
            if not utils.isnan(brand)]


def get_type_of_vehicle(df):
    return [vehicle.capitalize() for vehicle in df['type_of_vehicle'].unique()
            if not utils.isnan(vehicle)]


def lowercase_iter(iterable):
    return (item.lower() for item in iterable)


def title_with_link(row):
    return f'<a href="{BASE_URL}{row.listing_id}" target="_blank">{row.title}</a>'


def filter_df(df, brands, vehicles, deal_opt):
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered[deal_opt]]
    if brands:
        df_filtered = df_filtered[df_filtered['make'].isin(lowercase_iter(brands))]

    if vehicles:
        df_filtered = df_filtered[df_filtered['type_of_vehicle'].isin(lowercase_iter(vehicles))]

    return df_filtered\
        .sort_values(by=['pct_diff'], ascending=False)\
        .head(10)\
        .reset_index()


def get_df_html(df_filtered) -> str:
    df_show = df_filtered.copy()
    df_show['title'] = df_show.apply(title_with_link, axis=1)
    df_show.price = df_show.price.apply(round)
    df_show.price_predicted = df_show.price_predicted.apply(round)
    raw_html = df_show[COLS_TO_SHOW].to_html(escape=False, index=False)
    return f'<div style="overflow-x:auto;">{raw_html}</div>'


def get_plot(df_filtered):
    df = df_filtered.copy()
    df = df.rename(columns={'title': 'Car', 'price': 'Real Price', 'price_predicted': 'Predicted Price'})
    fig = px.bar(df,
                 x='Car',
                 y=['Real Price', 'Predicted Price'],
                 barmode='group',
                 height=500)
    fig.update_yaxes(title_text='Price')
    return fig


st.write('''
# Find Good Deals on Cars in the Singapore Market

With this application, you can find good deals on cars in the Singapore market.
''')

st.sidebar.header('How Does it Work?')
st.sidebar.markdown('''
We scrape sgcarmart.com for the latest car listings. Based on previous resale
prices, we have trained a predictor to predict the price of the cars. Now, if
a new car arrives that has a much lower price than what we predict, we can
tell you that it is a good deal.

Based on your criteria, the cars are shown in the tables below where the ones
with the biggest relative difference in real price and predicted price are shown
first.
''')

df = get_df()
deal_opt = st.select_slider('Show all deals that are at least...', DEAL_OPTIONS)
brands = st.multiselect('Choose the car brands you want shown:', get_brands(df))
vehicles = st.multiselect('Choose the type of vehicle you want shown:', get_type_of_vehicle(df))
df_filtered = filter_df(df, brands, vehicles, deal_opt)

if len(df_filtered) == 0:
    st.write('No cars found matching all your criteria.')
    st.stop()


st.write('''
Here you can check out what the best deals are in the market. For each car, we
are showing the current listing price on the left as well as what we predict the
car *should* actually resale for.
''')
st.plotly_chart(get_plot(df_filtered), use_container_width=True)

st.write('''
You can check out all the details of the listings above and follow
the link to purchase the car:
''')
st.write(get_df_html(df_filtered), unsafe_allow_html=True)
