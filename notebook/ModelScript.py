import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns',200)
df = pd.read_csv("train.csv")
df.shape
df.head()

df.info()
df.duplicated().sum()
df.describe()
a=df.isnull().sum()/df.shape[0]*100#.to_string().split('\n')
a.to_string().split('\n')
df.info()

df.Alley = df.Alley.fillna("NoAlley")
df.shape

for i in df.select_dtypes(include="object").columns:
        print('='*30)
        k=(df[i].value_counts(normalize =True))*100
        
        a=df.groupby(i)["SalePrice"].mean()
        summean = df["SalePrice"].mean()
        
        rep = pd.concat([k,a,a-summean],axis=1)
        rep.columns=["Percentage","mean","ratio"]

        print(rep.sort_values(by="mean"))
        print('='*30)
#new
    
dropList = ['Street','Alley','Utilities','ExterCond','BsmtCond','BsmtFinSF2', 
            'Heating','CentralAir', 'Functional','GarageQual','GarageCond',
            'PavedDrive','MiscFeature'
            ]

nominal_groupings = {
    "MSZoning": {"RM": "RMRH", "RH": "RMRH", "C (all)": "Other"},
    "LotShape": {"IR2": "IR", "IR3": "IR"},
    "LandContour": {"Low": "HLOW", "HLS": "HLOW"},
    "LotConfig": {"Inside": "InsideF2", "FR2": "InsideF2", "FR3": "CulDSac"},
    "LandSlope": {"Mod": "ModSev", "Sev": "ModSev"},
    "BldgType": {"2fmCon": "DuplexTwnhs", "Duplex": "DuplexTwnhs", "Twnhs": "DuplexTwnhs"},
    "RoofStyle": {"Gambrel": "Gable", "Mansard": "Gable", "Flat": "HipShed", "Shed": "HipShed"},
    "Condition1": {"Artery": "Feedr", "RRAe": "Feedr", "RRAn": "RR", "RRNe": "RR", "RRNn": "RR", "PosN": "Pos", "PosA": "Pos"}
}
ordinal_mappings = {
    "ExterQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1},
    "BsmtQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1},
    "BsmtFinType1": {"GLQ": 3, "ALQ": 2, "BLQ": 2, "Rec": 1, "LwQ": 1, "Unf": 0},
    "HeatingQC": {"Ex": 3, "Gd": 2, "TA": 2, "Fa": 1, "Po": 1},
    "KitchenQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1},
    "FireplaceQu": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 1},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1},
    "PoolQC": {"Ex": 3, "Gd": 2, "Fa": 1},
    "Fence": {"GdPrv": 3, "MnPrv": 2, "GdWo": 2, "MnWw": 1}
}

def bin_neighborhood(x):
    low = ['MeadowV', 'IDOTRR', 'BrDale', 'BrkSide', 'Edwards', 'OldTown', 'Sawyer', 'Blueste', 'SWISU', 'NPkVill', 'NAmes']
    mid = ['Mitchel', 'SawyerW', 'NWAmes', 'Gilbert', 'Blmngtn', 'CollgCr']
    high = ['Crawfor', 'ClearCr', 'Somerst', 'Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']
    if x in low: return 1
    if x in mid: return 2
    if x in high: return 3
    return 0
df.drop(columns=dropList, inplace=True, errors='ignore')
df.replace(nominal_groupings, inplace=True)
df['Neighborhood'] = df['Neighborhood'].apply(bin_neighborhood)

for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(0)    

df = pd.get_dummies(df, drop_first=True)


df["LotFrontage"].isna().sum()
df["GarageYrBlt"].isna().sum()
df["LotFrontage"] =df["LotFrontage"].fillna(df["LotFrontage"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())
sns.histplot(df,x="GarageYrBlt")
sns.histplot(df,x="LotFrontage")
to_drop = ['GarageArea', '1stFlrSF', 'TotRmsAbvGrd']
df.drop(columns=to_drop, inplace=True, errors='ignore')


# Calculate correlation with the target only
price_corr = df.corr()['SalePrice'].sort_values(ascending=False)

# Show the top 15 most influential features
print("Top 15 Features affecting Price:")
print(price_corr.head(15))

# Show the bottom features (Negative correlation)
print("\nTop 5 Features that decrease Price:")
print(price_corr.tail(5))# Only select features with a correlation > 0.5 with SalePrice
top_features = price_corr[abs(price_corr) > 0.5].index
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, cmap='RdYlGn', fmt=".2f")
plt.title("High Correlation Features (>0.5)")
plt.show()
df = df[df["SalePrice"]<=500000]
df.shape

X = df.drop("SalePrice",axis=1)
y = df['SalePrice']
y.shape

# 1. Shuffle the data randomly
df = df.sample(frac=1).reset_index(drop=True)

# 2. Define the split point (80% for training)
split = int(0.8 * len(df))

# 3. Slice the data
X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split:]
y_test = y.iloc[split:]
null_rows = df[df.isnull().any(axis=1)]
print(null_rows)
X["MasVnrArea"] = X["MasVnrArea"].fillna(0)

X = X.drop(X.index[[271, 120]])
y = y.drop(y.index[[271, 120]])

print(X.shape)
print(y.shape)
fmodel = LinearRegression()
plt.boxplot(df.SalePrice)
losses = dict()
scores = dict()
num_rows = len(X)
fold_size = num_rows // 5

for i in range(5):
    start = i * fold_size
    end = (i + 1) * fold_size
    
    x_test = X.iloc[start:end]
    y_test = y.iloc[start:end]
    
    x_train = pd.concat([X.iloc[:start], X.iloc[end:]])
    y_train = pd.concat([y.iloc[:start], y.iloc[end:]])

    fmodel.fit(x_train,y_train)
    pre = fmodel.predict(x_test)
    
    loss = metrics.root_mean_squared_error(y_test, pre)
    score = metrics.r2_score(y_test, pre)

    losses[i+1] = loss
    scores[i+1] = score
    
    print(f"Round {i+1}: Training on {len(x_train)} rows \nloss: {losses[i+1]}\nScore: {score},\nTesting on {len(x_test)} rows.\n ============================")
fmodel.fit(X,y)
pre = fmodel.predict(X)

print(losses)
X.isna().sum().to_dict()
X.shape
y.shape

#error analysis taking round 1 data 
testx = X.iloc[:290]
testy= y.iloc[:290]

trainx = X.iloc[290:]
trainy=y.iloc[290:]
fmodel.fit(trainx,trainy)

pre = fmodel.predict(testx)

errors = abs(testy-pre)

core = ['GrLivArea', 'OverallQual', 'YearBuilt', 'Neighborhood','error','actual','predict']

analyse = testx.copy()
analyse["error"] = errors
analyse["actual"] = testy
analyse["predict"] = pre
a=(analyse.sort_values(by="error",ascending=False).head(10))
print(a[core])


model = LinearRegression()

final_df = X.copy()
final_df['SalePrice'] = y
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(0.8 * len(final_df))
train_final = final_df.iloc[:split_idx]
test_final = final_df.iloc[split_idx:]

X_train_final = train_final.drop('SalePrice', axis=1)
y_train_final = train_final['SalePrice']
X_test_final = test_final.drop('SalePrice', axis=1)
y_test_final = test_final['SalePrice']

final_model = LinearRegression()
final_model.fit(X_train_final, y_train_final)

final_preds = final_model.predict(X_test_final)

final_rmse = metrics.root_mean_squared_error(y_test_final, final_preds)
final_r2 = metrics.r2_score(y_test_final, final_preds)

print("\n" + "="*40)
print("FINAL MODEL PERFORMANCE")
print(f"Final RMSE: ${final_rmse:,.2f}")
print(f"Final R2 Score: {final_r2:.4f}")
print("="*40)

# 5. Visual Check (Optional)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_final, final_preds, alpha=0.5)
plt.plot([y_test_final.min(), y_test_final.max()], [y_test_final.min(), y_test_final.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Final Model: Actual vs Predicted")
plt.show()