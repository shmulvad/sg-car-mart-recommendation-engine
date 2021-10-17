from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from utils import rmse


def get_predictor(full_df, col):
    df = full_df.copy()
    retain_columns = [col for col in df.columns if sum(df[col].isna()) < 2784]
    df = df[retain_columns]
    df.dropna(inplace=True)

    X = df[col]
    Y = df.drop([col], axis=1)

    X_train, X_test, y_train, y_test \
        = train_test_split(X, Y, random_state=0, train_size=0.2)
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print("The Mean squared error for the predictor is: ", rmse(y_test, y_pred))

    # TODO: To be completed according to nature of cleaned data
    return reg
