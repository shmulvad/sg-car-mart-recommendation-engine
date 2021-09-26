from pathlib import Path

GENERATE_SIM_DF = False

UNKNOWN_STR = 'UNKNOWN'

MAX_NUM_NANS = 8
NAN_PENALTY = 0.01

TRAIN_PATH = Path('data/train.csv')
TEST_PATH = Path('data/test.csv')
CSV_PREDS_OUT = Path('preds.csv')

MOST_SIMILIAR_TRAIN_PATH = Path('data/most_similar_train.pkl')
MOST_SIMILIAR_TEST_PATH = Path('data/most_similar_test.pkl')


COLS_TO_DROP = [
    'indicative_price',
    'eco_category',
    'opc_scheme'
]

CRITICAL_COLS = [
    'title',
    'model',
    'manufactured',
    'transmission',
    'curb_weight',
    'power',
    # 'fuel_type',
    'engine_cap',
    'no_of_owners',
    'depreciation',
    'coe',
    'road_tax',
    'dereg_value',
    'mileage',
    'omv',
    'arf'
]

STR_COLS = [
    'make',
    'title',
    'model',
    'description',
    'type_of_vehicle',
    'category',
    'fuel_type',
    'features',
    'accessories'
]

NUMBER_COLS = [
    'manufactured',
    'road_tax',
    'curb_weight',
    'power',
    'engine_cap',
    'no_of_owners',
    'depreciation',
    'coe',
    'dereg_value',
    'mileage',
    'omv',
    'arf'
]

TO_SKIP = {
    'listing_id',
    'description',
    'original_reg_date',  # Probably handle later
    'reg_date',  # Probably handle later
    'lifespan',  # Probably handle later
    'features',  # Maybe handle later
    'accessories',  # Maybe handle later
    'indicative_price'
}

TRANSMISSION_MAP = {
    'auto': 0,
    'manual': 1
}


FUEL_TYPE_MAP = {
    'electric': 0.0,
    'petrol-electric': 0.5,
    UNKNOWN_STR: 0.7,  # Arbitrarily chosen value
    'diesel': 1.0,
    'petrol': 1.0
}

VEHICLE_CATEGORIES = [
    {'sports car'},
    {'luxury sedan', 'suv'},
    {'others', 'mpv', 'stationwagon', 'mid-sized sedan'}
]
