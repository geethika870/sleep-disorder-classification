import pandas as pd
import numpy as np, random, pickle, os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 1️⃣ Load dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
df.drop("Person ID", axis=1, inplace=True)

# Parse BP
df[["Systolic_BP","Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df.drop("Blood Pressure",axis=1,inplace=True)

# Encode categoricals
label_encoders={}
for col in df.select_dtypes(include="object").columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    label_encoders[col]=le

X=df.drop("Sleep Disorder",axis=1)
y=df["Sleep Disorder"]

# 2️⃣ SMOTETomek Balancing
smt=SMOTETomek(random_state=SEED)
X_res,y_res=smt.fit_resample(X,y)

# 3️⃣ Train-Test Split
X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.2,stratify=y_res,random_state=SEED)

# 4️⃣ Scaling
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)

# 5️⃣ Define 5 models (LightGBM tuned for high accuracy)
models={
  "SVM": SVC(C=1,kernel="rbf",probability=True,random_state=SEED),
  "RandomForest": RandomForestClassifier(n_estimators=400,max_depth=18,random_state=SEED),
  "LightGBM": LGBMClassifier(
      n_estimators=1200, learning_rate=0.02, max_depth=25,
      num_leaves=80, subsample=0.85, colsample_bytree=0.85,
      random_state=SEED
  ),
  "XGBoost": XGBClassifier(eval_metric="mlogloss",use_label_encoder=False,n_estimators=600,learning_rate=0.03,random_state=SEED),
  "ANN": MLPClassifier(hidden_layer_sizes=(256,128,64),max_iter=700,early_stopping=True,random_state=SEED)
}

# 6️⃣ Train + Test
results={}
for name,m in models.items():
    m.fit(X_train_scaled,y_train)
    results[name]=accuracy_score(y_test,m.predict(X_test_scaled))*100

# 7️⃣ Compare results
acc_df=pd.DataFrame(list(results.items()),columns=["Model","Accuracy"])
acc_df["Accuracy"]=acc_df["Accuracy"].round(4)
print(acc_df)

# 8️⃣ Save best model
best=acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
with open("best_model.pkl","wb") as f:
    pickle.dump((models[best],sc,label_encoders,list(X.columns)),f)

print(f"✅ Best model saved: {best} with {acc_df['Accuracy'].max()}% accuracy")
