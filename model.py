import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("credit_risk_dataset.csv")
df.rename(columns={'cb_person_default_on_file':'defaulter',
                   'cb_person_cred_hist_length':'credit_hist_length (years)'},
                  inplace=True)
dups =df.duplicated()
df[dups]
df.query("person_age==23 & person_income==42000 & person_home_ownership=='RENT' & loan_int_rate==9.99")
df.drop_duplicates(inplace=True)
df['loan_int_rate'].mode()[0]
x = df['loan_int_rate'].mode()[0]
df['loan_int_rate'].fillna(x,inplace=True)

X = df[['person_age','person_income','loan_amnt','loan_int_rate','loan_status','credit_hist_length (years)']]
y = df["defaulter"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)


# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
