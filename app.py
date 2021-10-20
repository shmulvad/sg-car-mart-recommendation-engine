import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

from constants import TRAIN_PATH
import utils


st.set_page_config(
    page_title='The Best Car Deals in SG',
    page_icon='💰🚗',
    initial_sidebar_state='collapsed'
)


FAIR, GOOD, VERY_GOOD, EXCELLENT = 'Fair', 'Good', 'Very Good', 'Excellent'
DEAL_OPTIONS = [FAIR, GOOD, VERY_GOOD, EXCELLENT]
DEAL_PCT_CUTOFFS = 0.1, 0.2, 0.3, 0.6

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


def prepare_df_for_showing(df_filtered):
    df_show = df_filtered.copy()
    df_show['title'] = df_show.apply(title_with_link, axis=1)
    df_show.price = df_show.price.apply(round)
    df_show.price_predicted = df_show.price_predicted.apply(round)
    return df_show[['title', 'make', 'model', 'type_of_vehicle', 'price', 'price_predicted']]


# def get_plot(df_filtered):
#     # TODO
#     df = pd.DataFrame(
#         columns=['']
#     )
#     fig = px.bar(df, x='Price', y=['Real Price', 'Predicted Price'], barmode='group')
#     return fig


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
''')

df = get_df()
deal_opt = st.select_slider('Show all deals that are at least...', DEAL_OPTIONS)
brands = st.multiselect('Choose the car brands you want shown:', get_brands(df))
vehicles = st.multiselect('Choose the type of vehicle you want shown:', get_type_of_vehicle(df))

st.write('''
Here are the best deals we found that match your criteria:
''')
df_filtered = filter_df(df, brands, vehicles, deal_opt)
df_show = prepare_df_for_showing(df_filtered)
st.write(df_show.to_html(escape=False), unsafe_allow_html=True)

# TODO
# st.write('See how good the deals are:')
# st.pyplot(get_plot(df_filtered))
