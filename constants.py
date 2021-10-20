from pathlib import Path

GENERATE_SIM_DF = False

UNKNOWN_STR = "UNKNOWN"

MAX_NUM_NANS = 8
NAN_PENALTY = 0.01

MAX_CAR_AGE = 50

DATA_PATH = Path("data")
SCRAPED_PATH = Path("scraped_data")

TRAIN_PATH = DATA_PATH / "train.csv"
TEST_PATH = DATA_PATH / "test.csv"
CSV_PREDS_OUT = Path("preds.csv")

MOST_SIMILIAR_TRAIN_PATH = DATA_PATH / "most_similar_train.pkl"
MOST_SIMILIAR_TEST_PATH = DATA_PATH / "most_similar_test.pkl"

FUEL_TYPE_PATH = SCRAPED_PATH / "listing_id_to_fuel_type.json"
MAKE_DICT_PATH = SCRAPED_PATH / "make_dict.pkl"
MAKE_MODEL_BIN_PATH = SCRAPED_PATH / "make_model_bins.pkl"
MAKE_MODEL_DICT_MEAN_PATH = SCRAPED_PATH / "make_dict_mean_norm.pkl"
TEXT_PROCESS_TRAIN_PATH = SCRAPED_PATH / "train_3.tsv"
TEXT_PROCESS_TEST_PATH = SCRAPED_PATH / "test_3.tsv"
TEXT_SORTED_FEATURES_JSON_PATH = SCRAPED_PATH / "text_features.json"
USED_CARS_SIMPLIFIED_PATH = DATA_PATH / "sg-used-cars-final-simplified.csv"


COLS_TO_DROP = [
    "indicative_price",
    "eco_category",
]

CRITICAL_COLS = [
    "title",
    "model",
    "manufactured",
    "transmission",
    "curb_weight",
    "power",
    # 'fuel_type',
    "engine_cap",
    "no_of_owners",
    "depreciation",
    "coe",
    "road_tax",
    "dereg_value",
    "mileage",
    "omv",
    "arf",
]

STR_COLS = [
    "title",
    "model",
    "description",
    "type_of_vehicle",
    #     'fuel_type', #     Since Fuel Type is being handled by function we remove it
    "features",
    "accessories",
]

NUMBER_COLS = [
    "manufactured",
    "road_tax",
    "curb_weight",
    "power",
    "engine_cap",
    "no_of_owners",
    "depreciation",
    "coe",
    "dereg_value",
    "mileage",
    "omv",
    "arf",
]

NOMINAL_TO_REMOVE = [
    "listing_id",
    "title",
    "make",
    "model",
    "description",
    "features",
    "accessories",
    "opc_scheme",
]

TO_SKIP = {
    "listing_id",
    "description",
    "original_reg_date",  # Probably handle later
    "reg_date",  # Probably handle later
    "lifespan",  # Probably handle later
    "features",  # Maybe handle later
    "accessories",  # Maybe handle later
    "indicative_price",
}

TRANSMISSION_MAP = {"auto": 0, "manual": 1}

VEHICLE_CATEGORIES = [
    {"sports car"},
    {"luxury sedan", "suv"},
    {"others", "mpv", "stationwagon", "mid-sized sedan"},
]

MAKE_MODEL_BINS = [20000,30000,40000,50000,55000,60000,65000,70000,75000,
          80000,85000,90000,95000,100000,110000,120000,130000,
          140000,150000,175000,200000,250000,300000,400000,600000,
          900000,1200000,1600000]

MAKE_MODEL_PRICE_MIN = 8700
MAKE_MODEL_PRICE_MAX = 1813887.5

