'''
Created on Feb 14, 2021

@author: maria
'''
#import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

#set pandas to show all columns
pd.set_option("expand_frame_repr", False)

fraud = pd.read_csv('/Users/maria/Document/python/tuxsa/csv/creditcard.csv')

#DATA PREPROCESSING

#make a copy of dataframe
origin_fraud = fraud.copy()

#see all information of dataframe
info = fraud.info()
print(info)

#check for null value, result none
null = fraud.isnull().sum().sum()
print('Null value in dataset:', null)

#count non-fraud and fraud cases
cases = len(fraud)
nonfraud_count = len(fraud[fraud.Class == 0])
fraud_count = len(fraud[fraud.Class == 1])
fraud_percentage = round(fraud_count/nonfraud_count*100, 2)

print('\nCASE COUNT')
print('Total number of cases are {}'.format(cases))
print('Number of Non-fraud cases are {}'.format(nonfraud_count))
print('Number of fraud cases are {}'.format(fraud_count))
print('Percentage of fraud cases is {}'.format(fraud_percentage))

#DATA VISUALISATION
x_axis = ['Non-Fraud', 'fraud']
x_axis = pd.Series(x_axis)
y_axis = [nonfraud_count, fraud_count]
df = pd.DataFrame({'Fraud or Not': x_axis, 'Case Number': y_axis})
print(df)

ax = df.plot(kind='bar', 
             x='Fraud or Not', y='Case Number', fontsize = 16)
ax.set_title("Percentage of Fraud Found", fontsize = 16)
ax.legend(fontsize = 16)
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
for p in ax.patches:
    ax.annotate("{:.02%}".format(p.get_height()/284807),
                xy=(p.get_x()+0.02, p.get_y()+0.02))
plt.xlabel('Fraud Or Not')
plt.ylabel('Percentage of Case Number')
plt.show()

#see a target column, result target not balance
sns.countplot(fraud['Class'])
plt.title('Target column before balancing')
plt.show()

#histogram of all columns
plt.figure()
for col in fraud.columns[:-1]:
    plt.title(f'Histogram of column {col}')
    fraud[col].hist(bins=20)
    plt.show()

#DATA PROCESSING

#drop time column
fraud.drop('Time', axis=1, inplace=True)
print(fraud)

#scale amount and replace the whole column with the new scaled value
scaler = StandardScaler()
fraud['NormalizedAmount'] = scaler.fit_transform(fraud['Amount'].values.reshape(-1, 1))
#see the new dataframe after scale amount value
print(fraud)

#detect outlier using boxplot
#see boxplot of every columns using for loop
#ignore the boxplot for Class
for col in fraud.columns[:-1]:
    plt.title(f'Boxplot of column {col}')
    plt.boxplot(fraud[col])
    plt.show()

#select outlier from every column
#drop column 'Class'
outlier = fraud.columns.drop('Class')

#create function to remove outlier
def remove_outlier(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    clean = data[~((data < (Q1-1.5*IQR)) | (data > (Q3+1.5*IQR))).any(axis=1)]
    
    return clean

#remove outliers using function, replace with NaN value
fraud[outlier] = remove_outlier(fraud[outlier])
print('\nReplace outliers with NaN value:\n',fraud)

#remove all rows with NaN value
fraud.dropna(inplace=True)
print('\nRemove row with NaN value:\n',fraud)

#summary data
print('Shape of data with outliers: ', origin_fraud.shape)
print('Shape of data without outliers: ', fraud.shape)
print('Number of removed outliers: ', origin_fraud.shape[0] - fraud.shape[0])

#setup target column
X = fraud.drop(['Class'], axis = 1)
y = fraud['Class']

#balance target column using smote
smote = SMOTE(random_state=0)
X, y = smote.fit_resample(X, y)

#plot target column again
plt.figure()
sns.countplot(y)
plt.title('Target Column after balancing')
plt.show()

#SPLITTING DATA INTO TRANING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#BUILD MODEL AND TRAIN THE MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

#MAKE PREDICTION ON THE TEST SET RESULTS
y_prediction = model.predict(X_test)
print(y_prediction)

#CALCULATE THE ACCURACY USING CLASSIFICATION_REPORT SKLEARN
report = classification_report(y_test,y_prediction)
print(report)

#see f1-score
f1 = f1_score(y_test, y_prediction)
print(f1)

#CHECK CONFUSION MATRIX
confusion_matrix = confusion_matrix(y_test, y_prediction)
print(confusion_matrix)

print("AUC score is: ", roc_auc_score(y_test, y_prediction))

#TEST ACCURACY WITH ROC CURVE
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()