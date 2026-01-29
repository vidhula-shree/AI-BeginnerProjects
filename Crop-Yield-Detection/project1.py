import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
df = pd.read_csv('yield_df.csv')
df.dropna(inplace=True)
df_encoded = pd.get_dummies(df)
X = df_encoded.drop(columns=['hg/ha_yield']) # Features
y = df_encoded['hg/ha_yield'] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
