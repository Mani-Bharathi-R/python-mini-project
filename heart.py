# Import Necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
print("\n===========================\n")
data = pd.read_csv('heart.csv')
print('shape of the data: ',data.shape)

print("\n===========================\n")
print('displays whether the person has heart disease or not based on the colestral level: ',data[data['chol']>280]['target'].value_counts())
print("\n===========================\n")
print('\n  displays all the data present in the data set:\n',data.head())
print("\n===========================\n")
data.info()

data.describe()
print("\n===========================\n")
print('\n displays all the data present are null set or not:\n',data.isnull().sum())
print("\n===========================\n")
print('displays the total count of persons haveing heart disease and not haveing heart disease:\n',data['target'].value_counts())
print("\n===========================\n")
data.corr()['target'].sort_values(ascending = False)

plt.figure(figsize = (20,15))
plt.subplot(3,2,1)
plt.title('patients with & without heart disease')
sns.countplot(x = 'target', data = data)
plt.subplot(3,2,2)
plt.title('Gender VS target')
sns.countplot(x = 'sex', data = data, hue = 'target')
plt.xlabel('0 - Female &   1-Male')
plt.subplot(3,2,3)
plt.title('Type of chest Pain')
sns.countplot(x = 'cp', data = data, hue = 'target')
plt.xlabel('0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic')
plt.subplot(3,2,4)
plt.title('Exercise induced Angina')
sns.countplot(x = 'exang', data = data, hue = 'target')
plt.xlabel('0 - No &   1-yes')
plt.subplot(3,2,5)
plt.title('No of major vessels')
sns.countplot(x = 'ca', data = data, hue='target')
plt.subplot(3,2,6)
plt.title('Results of blood flow observed')
sns.countplot(x = 'thal', data = data, hue = 'target')

print("Splitting the dataset into traiing and test datasets and then training the model\n")
x_train, x_test, y_train, y_test = train_test_split(data.drop(['target'], axis = 1), data['target'], test_size = 0.2, random_state=5)

lg = LogisticRegression().fit(x_train, y_train)

res = lg.predict(x_test)

print("\n===========================\n")
print('\nThe result which we calculated:\n',res)
print("\n===========================\n")
print('\ndisplays the precission,recall,f1-score,support :\n',classification_report(y_test, res))
print("\n===========================\n")
cnf=confusion_matrix(y_test,res)
print('Confusion Matrix:\n ',cnf)
print("\n===========================\n")