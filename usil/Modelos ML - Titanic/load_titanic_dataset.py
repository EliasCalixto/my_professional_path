# Import Sk-learn
from sklearn.utils import Bunch

# Import Other Libraries
import pandas as pd
import numpy as np

def load_titanic(url):
    # 1. Load data and drop columns
    titanic = pd.read_csv(url)
    titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # 2. Convert column names to lowercase
    titanic.columns = titanic.columns.str.lower()
    
    # 3. Handle missing values (using assignment instead of inplace)
    titanic['age'] = titanic['age'].fillna(titanic['age'].median())
    titanic['fare'] = titanic['fare'].fillna(titanic['fare'].median())
    titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
    
    # 4. One-hot encode categorical features
    titanic = pd.get_dummies(
        titanic,
        columns=['pclass', 'sex', 'embarked'],
        drop_first=True
    )
    
    # 5. Define features and target
    feature_names = [
        'age', 'sibsp', 'parch', 'fare',
        'pclass_2', 'pclass_3', 
        'sex_male',
        'embarked_Q', 'embarked_S'
    ]
    target = 'survived'
    
    # 6. Create Bunch object
    return Bunch(
        data=titanic[feature_names].values,
        target=titanic[target].values.astype(int),
        feature_names=feature_names,
        target_names=np.array(['died', 'survived']),
        DESCR="Titanic Survival Dataset"
    )

def main():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    load_titanic(url)

if __name__ == "__main__":
    main()