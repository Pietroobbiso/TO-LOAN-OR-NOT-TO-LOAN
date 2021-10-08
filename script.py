import pandas as pd
import numpy as np
import sys
import datetime
import seaborn as sns
from tqdm import tqdm
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.utils import resample




def get_feature_balance_to_montlhypayment_percentage(loan,test):

    df_cur_transactions = df_all_transactions_test if test else df_all_transactions
    
    df_loan_transactions = df_cur_transactions.loc[df_cur_transactions['account_id'] == loan['account_id']] #get all transactions for the account of the loan

    year = []
    month = []

    for index, transaction in df_loan_transactions.iterrows(): #gets year and month for each transaction
        trans_date = datetime.datetime.strptime(str(transaction['date']), "%y%m%d")
        year.append(trans_date.year)
        month.append(trans_date.month)

    df_loan_transactions['year'] = year
    df_loan_transactions['month'] = month

    df_mean_balance_bymonth = df_loan_transactions.groupby(['year','month'])['balance'].mean().reset_index(name='balance')
    df_mean_balance_allmonth = df_mean_balance_bymonth['balance'].mean()

    return  df_mean_balance_allmonth / loan['payments']


def get_client_district_from_account(account_id):
    
    df_disposition = df_dispositions.loc[(df_dispositions['account_id'] == account_id) & (df_dispositions['type'] == 'OWNER')] #get the disposition of the owner of the account
    df_client = df_clients.loc[df_clients['client_id'] == df_disposition.iloc[0]['client_id']] #get the info of the owner of the account
    return df_districts.loc[df_districts['code '] == df_client.iloc[0]['district_id']].iloc[0] #get the district info of the owner of the account

    
def get_feature_average_no_crimes_per_100_habitants(loan):
    
    district = get_client_district_from_account(loan['account_id'])
    
    no_crimes_95 = district['no. of commited crimes \'95 ']
    no_crimes_96 = district['no. of commited crimes \'96 ']
    
    no_crimes_95 = no_crimes_96 if no_crimes_95 == '?' else no_crimes_95
    no_crimes_96 = no_crimes_95 if no_crimes_96 == '?' else no_crimes_96
    
    return  ((int(no_crimes_95)+int(no_crimes_96))/2)/int(district['no. of inhabitants'])*100

def get_feature_average_unemployment_rate(loan):
    
    district = get_client_district_from_account(loan['account_id'])
    
    unemploymant_rate_95 = district['unemploymant rate \'95 ']
    unemploymant_rate_96 = district['unemploymant rate \'96 ']
    
    unemploymant_rate_95 = unemploymant_rate_96 if unemploymant_rate_95 == '?' else unemploymant_rate_95
    unemploymant_rate_96 = unemploymant_rate_95 if unemploymant_rate_96 == '?' else unemploymant_rate_96
    
    return  (float(unemploymant_rate_95)+float(unemploymant_rate_96))/2

def get_feature_proportion_avgsalary_monthlypayments(loan):
    
    district = get_client_district_from_account(loan['account_id'])
    
    return  int(district['average salary '])/int(loan['payments'])

def get_feature_account_credit_Card_type(loan,test):
    
    df_cur_credit_cards = df_credit_cards_test if test else df_credit_cards 
    
    df_loan_disposition = df_dispositions.loc[(df_dispositions['account_id'] == loan['account_id'])& (df_dispositions['type'] == 'OWNER')]
    df_credit_card_disposition = df_cur_credit_cards.loc[df_cur_credit_cards['disp_id'] == df_loan_disposition.iloc[0]['disp_id']]
    if (len(df_credit_card_disposition.index) == 1):
        return df_credit_card_disposition.iloc[0]["type"]
    else:
        return "no credit card"
    
    
def get_feature_sex(loan):
    
    df_loan_disposition = df_dispositions.loc[df_dispositions['account_id'] == loan['account_id']]
    df_client_disposition = df_clients.loc[df_clients['client_id'] == df_loan_disposition.iloc[0]['client_id']]
        
    trans_date = list(str(df_client_disposition.iloc[0]['birth_number']))
    
    month = int(trans_date[2] + trans_date[3])
    #print(month)
      
    if (month > 12):
        return 'F'
    else:
        return 'M'
        
        
def get_feature_age(loan):
    
    df_loan_disposition = df_dispositions.loc[df_dispositions['account_id'] == loan['account_id']]
    df_client_disposition = df_clients.loc[df_clients['client_id'] == df_loan_disposition.iloc[0]['client_id']]
        
    
    trans_date = list(str(df_client_disposition.iloc[0]['birth_number']))
    
    year = int(trans_date[0] + trans_date[1])
    age = 97 - year
       
    return age
 
    

df_train = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\loan_train.csv',sep=';')
df_test = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\loan_test.csv',sep=';')

df_dispositions = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\disp.csv',sep=';')
df_clients = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\client.csv',sep=';')
df_districts = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\district.csv',sep=';')
df_all_transactions = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\trans_train.csv',sep=';')
df_all_transactions_test = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\trans_test.csv',sep=';')
df_credit_cards = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\card_train.csv', sep=';', header=0)
df_credit_cards_test = pd.read_csv(r'C:\Users\39327\Desktop\ARTIFICIAL INTELLIGENCE\YEAR 2\SEMESTER 1 (PORTO)\KE & ML\card_test.csv', sep=';', header=0)

'''    
df_train = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/loan_train.csv', sep=';', header=0)
df_test = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/loan_test.csv', sep=';', header=0)

df_dispositions = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/disp.csv', sep=';', header=0)
df_clients = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/client.csv', sep=';', header=0)
df_districts = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/district.csv', sep=';', header=0)
df_all_transactions = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/trans_train.csv', sep=';', header=0)
df_all_transactions_test = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/trans_test.csv', sep=';', header=0)
df_credit_cards = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/card_train.csv', sep=';', header=0)
df_credit_cards_test = pd.read_csv(filepath_or_buffer='../input/to-loan-or-not-to-loan-that-is-the-question-7/public data/card_test.csv', sep=';', header=0)
'''

df_train_processed = pd.DataFrame(columns=['amount', 'duration', 'payments', 'balance_monthlypayment_percentage', 'average_no_crimes_per_100_habitants', 'average_unemployment_rate', 'proportion_avgsalary_monthlypayments','account_credit_Card_type','sex','age', 'status'])
df_test_processed = pd.DataFrame(columns=['amount', 'duration', 'payments', 'balance_monthlypayment_percentage', 'average_no_crimes_per_100_habitants', 'average_unemployment_rate', 'proportion_avgsalary_monthlypayments','account_credit_Card_type','sex','age', 'loan_id'])   
    


    
for index_loan, loan in tqdm(df_train.iterrows()):
    df_train_processed.loc[index_loan] = [loan['amount'], loan['duration'], loan['payments'], get_feature_balance_to_montlhypayment_percentage(loan,False), get_feature_average_no_crimes_per_100_habitants(loan), get_feature_average_unemployment_rate(loan), get_feature_proportion_avgsalary_monthlypayments(loan),get_feature_account_credit_Card_type(loan,False),get_feature_sex(loan),get_feature_age(loan), loan['status']]
    
#print(df_train_processed)


for index_loan, loan in tqdm(df_test.iterrows()):
    df_test_processed.loc[index_loan] = [loan['amount'], loan['duration'], loan['payments'], get_feature_balance_to_montlhypayment_percentage(loan,True), get_feature_average_no_crimes_per_100_habitants(loan), get_feature_average_unemployment_rate(loan), get_feature_proportion_avgsalary_monthlypayments(loan),get_feature_account_credit_Card_type(loan,True),get_feature_sex(loan),get_feature_age(loan), loan['loan_id']]
    
#print(df_test_processed)



df_data = pd.get_dummies(df_train_processed.drop(columns=['status']), columns=['account_credit_Card_type','sex'])
df_target = df_train_processed[['status']]
df_target = df_target.astype(int)
df_test_target = pd.get_dummies(df_test_processed.drop(columns=['loan_id']), columns=['account_credit_Card_type','sex'])
df_test_id = df_test_processed[['loan_id']]

#UP-SAMPLING STEP

df_merged = pd.concat([df_data,df_target],axis=1)

# Separate majority and minority classes
df_merged_majority = df_merged[df_merged.status==1]
df_merged_minority = df_merged[df_merged.status==-1]

df_minority_upsampled = resample(df_merged_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=282,    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_merged_majority, df_minority_upsampled])

df_upsampled.status.value_counts()


df_upsample_data = df_upsampled.drop('status',axis=1)

df_upsample_target = df_upsampled[['status']]
df_upsample_target = df_upsample_target.astype(int)


#DOWN-SAMPLING STEP
df_merged_majority = df_merged[df_merged.status==1]
df_merged_minority = df_merged[df_merged.status==-1]


df_majority_downsampled = resample(df_merged_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=46,     # to match minority class
                                 random_state=123) # reproducible results

df_downsampled = pd.concat([df_majority_downsampled, df_merged_minority])



df_downsampled.status.value_counts()


df_downsampled_data = df_downsampled.drop('status',axis=1)

df_downsampled_target = df_downsampled[['status']]
df_downsampled_target = df_downsampled_target.astype(int)



# COMBINING STEP


df_majority_downsampled_balance = resample(df_merged_majority, 
                                 replace=False,     # sample with replacement
                                 n_samples=164,    # to match minority class
                                 random_state=123) # reproducible results


df_minority_upsampled_balance = resample(df_merged_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=164,    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
df_balance = pd.concat([df_majority_downsampled_balance, df_minority_upsampled_balance])

df_balance.status.value_counts()

df_data_balance = df_balance.drop('status',axis=1)

df_target_balance = df_balance[['status']]
df_target_balance = df_target_balance.astype(int)
'''






#########################################
df_target['status'].value_counts()

df_target= df_target.astype(int)
df_test_id=df_test_id.astype(int)






'''
#Model1 : Decision Tree Classifier

model = tree.DecisionTreeClassifier(criterion="entropy")
scores = cross_val_score(model, df_data, df_target, scoring='roc_auc')
avg_scores = np.mean(scores)
print(avg_scores)


#Model2 : K-Nearest Neighbour

#ORIGINAL MODEL
model1 = KNeighborsClassifier(n_neighbors=5,p=2, weights='distance')
scores1 = cross_val_score(model1, df_data, df_target, cv=10, scoring='roc_auc')
avg_scores1 = np.mean(scores1)
print(avg_scores1) #0.55 - 0.60 with cv=10 & weights = 'distance'

model1.fit(df_data, df_target)
y_predicted_train = model1.predict(df_test_target)

#UP-SAMPLED MODEL

model2 = KNeighborsClassifier(n_neighbors=7)
scores2 = cross_val_score(model2, df_upsample_data, df_upsample_target, cv = 5, scoring='roc_auc')
avg_scores2 = np.mean(scores2) #0.81 - 0.90 with best paramters
print(avg_scores2)

model2.fit(df_upsample_data, df_upsample_target)
y_upsampled_predicted_train = model2.predict_proba(df_test_target)[:,0]
print(y_upsampled_predicted_train)

#hyperparameters tuning site

k_range = range(1, 31)
# empty list to store scores
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, df_upsample_data, df_upsample_target, cv=5, scoring='roc_auc')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())


print(k_scores)
plt.plot(k_range, k_scores, 'b', label='Train AUC')
plt.xlabel('Value of K for KNN')
plt.ylabel('roc-auc accuracy')



#DOWNSAMPLED MODEL

model3 = KNeighborsClassifier(n_neighbors=5)
scores3 = cross_val_score(model3, df_downsampled_data, df_downsampled_target, scoring='roc_auc')
avg_scores3 = np.mean(scores3) #0.51 - 0.61 - 0.77 
print(avg_scores3)

model3.fit(df_dwonsampled_data, df_downsampled_target)
y_downsampled_predicted_train = model3.predict(df_test_target)


#BALANCE MODEL

model4 = KNeighborsClassifier(n_neighbors=5)
scores4 = cross_val_score(model4, df_data_balance, df_target_balance, scoring='roc_auc')
avg_scores4 = np.mean(scores4)
print(avg_scores4) #0.70 - 0.86

model4.fit(df_data_balance, df_target_balance)
y_balance_predicted_train = model4.predict_proba(df_test_target)[:,0]
print(model4.predict(df_test_target))
print(y_balance_predicted_train)



#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
#Fit the model
best_model = clf.fit(df_upsample_data, df_upsample_target)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

#Best parameters:
#Best leaf_size: 1
#Best p: 2
#Best n_neighbors: 7



#Model3: Neural Networks

encoded_df_target = df_target["status"]. replace(-1, 0)
'''
# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(16, input_dim=14, activation='relu'))
    #model.add(Dense(16, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
'''
def create_model():
    model = Sequential()
    model.add(Dense(4,activation = 'relu',input_dim = 14))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


'''
# create model
model = Sequential()
model.add(Dense(16, input_dim=14, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(df_data,encoded_df_target,epochs=75, batch_size=10, verbose=0)

y_predicted_train = model.predict_classes(df_test_target)
print(y_predicted_train)
'''

'''
# evaluate model with standardized dataset
estimator = KerasClassifier(create_model, epochs=50, batch_size=20, verbose=0)
estimator.fit(df_data, encoded_df_target)
y_predicted_train = estimator.predict(df_test_target)

kfold = StratifiedKFold(n_splits=5, shuffle=False)
results = cross_val_score(estimator, df_data, encoded_df_target, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

#Model 4: Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, df_data, df_target, cv=5, scoring='roc_auc').mean())


#Model 5.1: Standard Bagging Classifier

#from sklearn.svm import SVC
model_bag = BaggingClassifier()

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_bag, df_data, df_target, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores)) #0.651


#Model 5.2: Bagging With Random Undersampling

# define model
from imblearn.ensemble import BalancedBaggingClassifier
model_bal_bag = BalancedBaggingClassifier()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_bal_bag, df_data, df_target, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores)) #0.651


# MODEL 6 RANDOM FOREST

#6.1 Standard Random Forest
from sklearn.ensemble import RandomForestClassifier
model_st_rf = RandomForestClassifier(n_estimators=10)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_st_rf, df_data, df_target, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores)) #

# 6.2 Random Forest With Bootstrap Class Weighting 
from sklearn.ensemble import RandomForestClassifier
model_rf_boot_class = RandomForestClassifier(n_estimators=10,class_weight='balanced_subsample')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_rf_boot_class, df_data, df_target, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores)) #


#6.2 Easy Ensemble
# define model

from imblearn.ensemble import EasyEnsembleClassifier
model_easy_ens = EasyEnsembleClassifier(n_estimators=10)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_easy_ens, df_data, df_target, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % np.mean(scores)) #



#PLOTS





df_train_processed['account_credit_Card_type'] = df_train_processed['account_credit_Card_type'].replace({'classic':'credit card', 'junior':'credit card','gold':'credit card'})
print (df_train_processed)





from matplotlib.pyplot import figure


df_train_processed['account_credit_Card_type'].value_counts().plot(kind='pie', autopct='%1.0f%%',textprops={'fontsize': 20},title='Customers with credit card vs without credit card',label='')

df_train_processed['status'].value_counts().plot(kind='pie', autopct='%1.0f%%',textprops={'fontsize': 20},title='Customers who paid vs not paid',label='')

#y_predicted_train = create_baseline().fit(df_data,encoded_df_target)

# calculate correlation matrix
plt.figure(figsize = (10,10))
corr = df_train_processed.corr()# plot the heatmap
#sns.set(font_scale=2)
#ax1 = figure.add_axes([0.4,0.2,0.5,0.6])
x_axis_labels = ['bal_monpay_per','av_n°_cri_per_100_inh','av_unem_rate','prop_av_sal_monpay_','status'] # labels for x-axis
y_axis_labels = ['bal_monpay_per','av_n°_cri_per_100_inh','av_unem_rate','prop_av_sal_monpay_','status']
mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
mask[np.diag_indices_from(mask)] = False
res = sns.heatmap(corr, xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=mask, vmin=-1, vmax = 1, annot=True, annot_kws={"size": 18}, cmap=sns.diverging_palette(220, 20, as_cmap=True),cbar_kws={"shrink": .5})
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 20)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 20)


ax = sns.swarmplot(x="sex", y="age", hue="status", data=df_train_processed)
fig = plt.figure()
fig.patch.set_facecolor('xkcd:mint green')
g = sns.catplot(x="sex", y="age",
                hue="status",
                data=df_train_processed, kind="swarm",
                height=8, aspect=.8)

#g.ax.set_facecolor('xkcd:gray')
g.ax.patch.set_facecolor('black')
plt.savefig('aa', transparent = True, bbox_inches = 'tight', pad_inches = -0.25)






original_stdout = sys.stdout  # Save a reference to the original standard output


with open('result.csv', 'w') as f:
    sys.stdout = f

    print('Id,Predicted')
    print()
    for i in range(0, len(df_test_id['loan_id'])):
        print(df_test_id['loan_id'][i], ',', y_balance_predicted_train[i])
        print()

    sys.stdout = original_stdout
