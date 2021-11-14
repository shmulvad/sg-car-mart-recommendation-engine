import pandas as pd

import cleaner
import constants as const
import similarity as sim


def main():
    # Read data and run cleaner
    df_train_raw = pd.read_csv(const.TRAIN_PATH)
    df_test_raw = pd.read_csv(const.TEST_PATH)
    df_train = cleaner.clean_preliminary(df_train_raw)
    df_test = cleaner.clean_preliminary(df_test_raw, is_test=True)

    # TRAIN (NB: Takes ~4 hours to run)
    # Generate similarity matrix, pickle it, replace values and write to csv
    sim_df = sim.compute_similarities(df_train, df_train)
    sim_df.to_pickle(const.MOST_SIMILIAR_TRAIN_PATH)
    replaced_train = sim.replace_nan_with_most_similar(
        main_df=df_train,
        sim_df=sim_df,
        verbose=True
    )
    replaced_train.to_csv(const.SIM_REPLACED_TRAIN)

    # TEST (NB: Takes ~2.5 hours to run)
    # Generate similarity matrix, pickle it, replace values and write to csv
    sim_df_test = sim.compute_similarities(df_test, df_test)
    sim_df_test.to_pickle(const.MOST_SIMILIAR_TEST_PATH)
    replaced_test = sim.replace_nan_with_most_similar(
        main_df=df_test,
        train_df=df_train,
        sim_df=sim_df_test,
        verbose=True
    )
    replaced_test.to_csv(const.SIM_REPLACED_TEST)
    print('Done')


if __name__ == '__main__':
    main()
