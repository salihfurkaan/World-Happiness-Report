from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import GradientBoostingRegressor

def linear(data, random_state=42):
    x = data.drop(columns="happiness_score")
    y = data["happiness_score"]
    model = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=random_state)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(f"MSE for linear regression model: {mse(y_test, pred)}")



def knn(data, n_neighbors=5, leaf_size=30, metric='minkowski', n_jobs=-1):

    x = data.drop(columns="happiness_score")
    y = data["happiness_score"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42)

    model = KNeighborsRegressor(n_neighbors=n_neighbors, leaf_size=leaf_size, metric=metric, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("MSE for KNN : ", mse(y_test,pred))


def gradient_boosting(data, lr=0.1, max_depth=3, estimators=100, random_state=42):
    x = data.drop(columns="happiness_score")
    y = data["happiness_score"]

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=random_state)

    model = GradientBoostingRegressor(learning_rate=lr, n_estimators=estimators,max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("MSE for Gradient Boosting Regressor: ", mse(y_test, pred))