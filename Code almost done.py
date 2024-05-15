#!/usr/bin/env python
# coding: utf-8

# # Unveiling Risk Patterns: Investigating Risk in Online Peer-to-Peer Lending
# 
# Managing risk is pretty much the name of the game when it comes to handling loans. When you throw P2P lending platforms like Lending Club into the mix, the game gets even more interesting. They're using technology to take things to a whole new level, boosting their performance and making things more profitable.
# 
# In this Project I will be using their loan data from 2007 to 2018 to understand how and where they are doing better and where not. I will try to get insights from the available data to see if i can get something out of it and make their prediction model more better.
# 

# ## Importing libraries and Packages

# In[1]:





# In[2]:





# ## Importing the Data

# In[3]:


# loading the data set

data= pd.read_csv('M:\\pc\\downloads\\accepted_2007_to_2018Q4.csv')


# ## Understanding the Data 

# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.info


# # Univariate Analysis
# ###  1) Discarding features characterized by low variance

# In[8]:


# knowing the variables which have no variance



no_variance_cols = []
for col in data.columns:
    if data[col].nunique() <= 1:
        no_variance_cols.append(col)
        
print("Columns with no variance or constant values:")
print(no_variance_cols)



# In[9]:


# droping the above columns with no variance or constant values
data = data.drop(columns=no_variance_cols)


# ### 2) Eliminating features with substantial proportion of missing values.

# In[10]:


# lets see the null values 
data.isnull().sum()


# In[11]:


# Visualizing missing values
msno.bar(data,color='red')
plt.show()


# The above plot illustrates the proportion of missing values in each column.

# In[12]:


# now droping columns which has null values more than 30 %
def drop_null_columns(data, null_threshold=30):
    # Calculate the percentage of null values in each column
    null_percentages = (data.isnull().sum() / len(data)) * 100
    
    # Get the names of columns with null percentage greater than the threshold
    null_columns = null_percentages[null_percentages > null_threshold].index
    
    # Drop the null columns from the dataset
    data.drop(null_columns, axis=1, inplace=True)

    
drop_null_columns(data, 30)


# In[13]:


data.isnull().sum()


# In[14]:


# Again Visualizing missing values
msno.bar(data,color='blue')
plt.show()


# In[15]:


data.shape


# ### 3) Exploration of Target variable

# In[16]:


# Lets see the data type of the remaining columns
print(data.dtypes)


# In[17]:


# Calculate value counts and sort
data.dtypes.value_counts().sort_values().plot(kind='barh', color='skyblue')


# Set title and labels
plt.title('Number of columns by Data Types',fontsize=16)
plt.xlabel('Number of columns',fontsize=12)
plt.ylabel('Data type',fontsize=12)

# Show the plot
plt.show()


# #### Lets see our target variable first if we can detect something from it. The ultimate aim is to reduce the number of independent variables

# In[18]:


# checking our response variable in detail if we can get any insight from it

# Calculate proportions
loan_status_counts = data['loan_status'].value_counts(normalize=True)

# Generate a bar plot
plt.figure(figsize=(10,6))
sns.barplot(x=loan_status_counts.index, y=loan_status_counts.values, palette="viridis")

# Add title and labels
plt.title('Proportions of unique values in loan_status', fontsize=16)
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[19]:


# checking our response variable in detail if we can get any insight from it

data['loan_status'].value_counts(dropna=False)


# #### Current: This means the person who borrowed the money is making all the payments on time. They are not behind on any payments.
# 
# #### In Grace Period: This is a small window (15 days in this case) after the payment due date. If the borrower has not paid on the due date, they can still make the payment within these 15 days without getting penalized.
# 
# #### Late (16-30): If the borrower hasn't made the payment for 16 to 30 days after it was due, the loan is considered to be in this state.
# 
# #### Late (31-120): If the borrower hasn't made the payment for 31 to 120 days after it was due, the loan is considered to be in this state.
# 
# #### Fully Paid: The borrower has paid back all the money they borrowed. They can do this either by following the regular payment plan over 3 to 5 years, or by paying back all the money earlier than planned.
# 
# #### Default: This term is used when the borrower hasn't made payments for a very long time. It suggests the borrower is in serious trouble, possibly unable to repay the loan.
# 
# #### Charged Off: This is when the lender decides that they probably won't get their money back from the borrower. In other words, the lender has given up hope of being repaid. The loan is basically considered a loss by the lender.

# #### As we are going to make our analysis on loans that are either fully paid or Charged off, we will replace 'Current', 'In Grace Period', and 'Does not meet the credit policy. Status:Fully Paid' with 'Fully Paid' also replace 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off', and 'Default' with 'Charged Off.

# In[20]:


def merge_categories(status):
    if pd.isna(status):  # if the status is NaN, just return it as is
        return status
    elif status in ['Current', 'In Grace Period', 'Does not meet the credit policy. Status:Fully Paid']:
        return 'Fully Paid'
    elif status in ['Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off', 'Default']:
        return 'Charged Off'
    else:
        return status  # if it's not in any of the specified categories, return as is

data['loan_status'] = data['loan_status'].apply(merge_categories)



# In[21]:


data['loan_status'].value_counts(dropna=False)


# In[22]:


# checking our response variable in detail if we can get any insight from it

# Calculate proportions
loan_status_counts = data['loan_status'].value_counts(normalize=True)

# Generate a bar plot
plt.figure(figsize=(10,6))
sns.barplot(x=loan_status_counts.index, y=loan_status_counts.values, palette="viridis")

# Add title and labels
plt.title('Proportions of unique values in loan_status', fontsize=16)
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Proportion', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# In[23]:


data['loan_status'].value_counts(normalize= True, dropna=False)


# In[24]:


data.shape


# ### 4) Feature extraction

# In[25]:


data.columns


# In[26]:


# see unique values of each columns
for col in data.columns:
    unique_values = data[col].unique()
    print(f"Unique values of '{col}': {unique_values}")


# In[27]:


# Lendig club have also provided us with the description of their each column

col_discription = pd.read_excel('M:\\pc\\downloads\\LCDataDictionary_2018.xlsx',sheet_name=1)
col_discription


#  This is a crucial step to extract the important features, the col_discription file is very helpful in deciding our independent variables.
# 
#     

# #### I will try to do this in a strategic way, my strategy will go like this

# ### a) Removing features, based on finanical perspective
# Using what I know about finance and the details in the file, I'm going to narrow down our list of features. I'll do this by dropping the features that I don't think will help us predict loan defaults. Since we're building a model to predict defaults, we'll focus on the features that investors would know from the start

# In[28]:


# droping feaures based on the dictionary provided, 

short_list = data.drop(['id','issue_d','url','zip_code','out_prncp','out_prncp_inv','total_rec_late_fee','recoveries','collection_recovery_fee',
'total_rev_hi_lim','mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl','num_accts_ever_120_pd','num_actv_bc_tl',
'num_bc_tl','num_il_tl','num_op_rev_tl','total_pymnt_inv','total_rec_prncp','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m',
 'pct_tl_nvr_dlq','percent_bc_gt_75','delinq_2yrs','funded_amnt','funded_amnt_inv','inq_last_6mths','total_pymnt','total_rec_int', 'last_pymnt_d','last_pymnt_amnt', 'last_credit_pull_d','last_fico_range_high', 'last_fico_range_low',
'collections_12_mths_ex_med','acc_now_delinq','tot_coll_amt', 'tot_cur_bal', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt',
'mths_since_recent_bc', 'num_actv_rev_tl','num_bc_sats', 'tax_liens','total_il_high_credit_limit', 'hardship_flag',
'debt_settlement_flag','pymnt_plan'] ,axis=1, inplace=True)


# In[29]:


data.shape


# In[30]:


data.dtypes


# ### b) Checking Multicollinearity
# 
# Investigating the numeric features more, to see if they are influencing each other 

# In[31]:


# get the numerical columns
numerical_cols = data.select_dtypes(include=['float64'])

print(numerical_cols)


# In[32]:


# Correlation heatmap for numerical features

plt.figure(figsize=(25, 25))
correlation = numerical_cols.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()


# In[33]:


# Pairwise correlation
pairs = correlation.unstack()

# Select pairs whose correlation coefficient > 0.8 and < 1 (as correlation of variable with itself is 1)
#also the abs(pair) is taking both positive and negaitve values
strong_pairs = pairs[(abs(pairs) > 0.8) & (abs(pairs) < 1)]

print(strong_pairs)




# we can see that there are multilpe features which needs to be drop because of high collinearity, but before droping lets see its relation with loan_status.

# In[34]:


# Looping through all the highly correlated features

highly_correlated_features = [
    'loan_amnt','installment','fico_range_low', 'fico_range_high', 
    'bc_open_to_buy', 'total_bc_limit']

fig, axs = plt.subplots(2, 3, figsize=(15, 15))
for i, feature in enumerate(highly_correlated_features):
    row = i // 3
    pos = i % 3
    sns.boxplot(x='loan_status', y=feature, data=data, ax=axs[row][pos])
    axs[row][pos].set_title('Loan status vs ' + feature)

plt.tight_layout()
plt.show()


# In[35]:


sns.set_palette('pastel')
fig, axs = plt.subplots(2, 3, figsize=(15, 15))
for i, feature in enumerate(highly_correlated_features):
    row = i // 3
    pos = i % 3
    sns.violinplot(x='loan_status', y=feature, data=data, ax=axs[row][pos])
    axs[row][pos].set_title('Loan status vs ' + feature)

plt.tight_layout()
plt.show()


# In[36]:


data.columns


# In[37]:


# Based on the above plots I am droping those highly_correlated_features which will have no impact on the response variable
# other variables could play a crucial role in prediction, so i am not droping it 

data.drop(['fico_range_high'], axis=1, inplace=True) 


# In[38]:


data.dtypes


# In[39]:


# loan_amnt

data.groupby('loan_status')['loan_amnt'].describe().round(2)


# Loan amounts range from 500 to 40,000 with a median of $12,000, Charged-off loans tend to have higher loan amounts

# In[40]:


#int_rate

data.groupby('loan_status')['int_rate'].describe()


# Deafult people have higher int_rate 15.7%

# In[41]:


#installment

data.groupby('loan_status')['installment'].describe()


# Charged-off loans tend to have higher installments.

# In[42]:


#annual_inc

data.groupby('loan_status')['annual_inc'].describe()
#Because of the large range of incomes, we should take a log transform of the annual income variable.


# In[ ]:





# In[43]:


# Apply log transformation to the 'annual_inc' column
data['annual_inc'] = np.log1p(data['annual_inc'])


# In[44]:


data.groupby('loan_status')['annual_inc'].describe()


# Full paid loans have higher annual income 

# In[45]:


#dti(debt-to-income ratios)

data.groupby('loan_status')['dti'].describe()


# The variable seems to have alot of outliers, but still we can conclude that charged off loans have higher dti

# In[46]:


#open credit lines

data.groupby('loan_status')['open_acc'].describe()


# Clearly the Full paid loans have more open credit lines

# In[47]:


#Derogatory public records

data.groupby('loan_status')['pub_rec'].describe()


# The numbwer of derogatory public records has higher ratio in default loans

# In[48]:


#Total credit revolving balance

data['revol_bal'].describe()


# In[49]:


# Apply log transformation to the 'revol_bal' column
data['revol_bal'] = np.log1p(data['revol_bal'])

data.groupby('loan_status')['revol_bal'].describe()


# There is no significan difference 

# In[50]:


# Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.


data['revol_util'].describe()


# In[51]:


# Apply log transformation to the 'revol_util' column
data['revol_util'] = np.log1p(data['revol_util'])

data.groupby('loan_status')['revol_util'].describe()


# Its higher for the fully paid loans

# In[52]:


# total_acc

data.groupby('loan_status')['total_acc'].describe()


# No difference here

# In[ ]:





# In[53]:


data.groupby('loan_status')['bc_open_to_buy'].describe()


# In[54]:


# Apply log transformation to the 'bc_open_to_buy' column
data['bc_open_to_buy'] = np.log1p(data['bc_open_to_buy'])

data.groupby('loan_status')['bc_open_to_buy'].describe()


# In[55]:


#mortgage accounts

data.groupby('loan_status')['mort_acc'].describe()


# Mortgage accounts have more mortgage accounts

# In[56]:


#months since recent inquiry
data.groupby('loan_status')['mths_since_recent_inq'].describe()


# Fully paid loans leading here also

# In[57]:


#pub_rec_bankruptcies

data.groupby('loan_status')['pub_rec_bankruptcies'].describe()


# although there are alot of outliers but still its evident that charged off loans have more bankrupticies

# In[58]:


#tot_hi_cred_lim
data.groupby('loan_status')['tot_hi_cred_lim'].describe()


# In[59]:


# Apply log transformation to the 'bc_open_to_buy' column
data['tot_hi_cred_lim'] = np.log1p(data['tot_hi_cred_lim'])

data.groupby('loan_status')['tot_hi_cred_lim'].describe()


# No major difference 

# In[60]:


# total_bal_ex_mort

data.groupby('loan_status')['total_bal_ex_mort'].describe()


# In[61]:


# Apply log transformation to the 'total_bal_ex_mort' column
data['total_bal_ex_mort'] = np.log1p(data['total_bal_ex_mort'])
data.groupby('loan_status')['total_bal_ex_mort'].describe()


# No difference

# In[62]:


#total_bc_limit

data.groupby('loan_status')['total_bc_limit'].describe()



# In[63]:


# Apply log transformation to the 'total_bc_limit' column
data['total_bc_limit'] = np.log1p(data['total_bc_limit'])

data.groupby('loan_status')['total_bc_limit'].describe()


# Again no big difference

# In[ ]:





# In[64]:


data.shape


# ### c) Exploring Categorical variables

# In[65]:


data.select_dtypes(include=['object'])


# In[66]:


data.dtypes


# In[67]:


#term

data['term'].value_counts(dropna=False)
# as values are in 36 and 60, lets convert it to integers


# In[68]:


# Custom function to convert the values to integers, skipping NaN
def convert_to_int(s):
    try:
        return int(s.split()[0])
    except (AttributeError, ValueError):
        return np.nan

# Apply the custom function to the 'term' column
data['term'] = data['term'].apply(convert_to_int)





# In[69]:


data['term'].value_counts(dropna=False)


# In[70]:


plt.figure(figsize=(10,6))
sns.countplot(x='term', hue='loan_status', data=data, palette='coolwarm')
plt.title('Term vs Loan Status')
plt.show()


# The above plot suggests that people tends to be not default if the term is less

# In[71]:


#grade and sub grade

data['grade'].unique()


# In[72]:


data['sub_grade'].unique()


# The grade is implied by the subgrade, so let's drop the grade column.

# In[73]:


data.drop('grade', axis=1, inplace=True)


# In[74]:


#emp_title

data['emp_title'].describe()


# These are alot of employment title and i dnt think, it is of any use to use so we will drop it

# In[75]:


data.drop('emp_title', axis=1, inplace=True)


# In[76]:


#emp_length

data['emp_length'].value_counts(dropna=False).sort_index()


# In[77]:


data['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
data['emp_length'].replace('< 1 year', '0 years', inplace=True)


# In[78]:


# Define a function to handle conversion
def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])

# Apply the function to the 'emp_length' column
data['emp_length'] = data['emp_length'].apply(emp_length_to_int)



# In[79]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=data, hue='loan_status', palette='viridis')
plt.title('Employment Length vs Loan Status')
plt.xlabel('Employment Length (years)')
plt.ylabel('Count')
plt.show()


# Loan status does not appear to vary much with employment length on average, except for a small drop in charge-offs for borrowers with over 10 years of employment.

# In[80]:


#home_ownership

data['home_ownership'].value_counts(dropna=False)


# In[81]:


#replacing any and none with OTHER
data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)


# In[82]:


home_ownership_counts = data.groupby(['home_ownership', 'loan_status']).size().unstack()

fig, ax = plt.subplots(1, home_ownership_counts.shape[0], figsize=(20, 10))

for i, (home_ownership, counts) in enumerate(home_ownership_counts.iterrows()):
    ax[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax[i].set_title(f'Loan Status for Home Ownership: {home_ownership}')

plt.tight_layout()
plt.show()


#  Renters and homeowners have a higher probability of charge-off.

# In[83]:


#purpose

data['purpose'].value_counts()


# In[84]:


plt.figure(figsize=(15, 8))
sns.countplot(y='purpose', data=data, hue='loan_status')
plt.title('Loan Purpose Distribution by Loan Status')
plt.ylabel('Purpose')
plt.xlabel('Count')
plt.show()


# deb_consolidation has the highest purpose of the loan
# 

# In[85]:


#title
data['title'].value_counts().head(10)
# the purpose variable appears to already contain this information. So we drop the title variable.


# In[86]:


data.drop('title', axis=1, inplace=True)


# In[87]:


#state address

plt.figure(figsize=(15,8))
sns.countplot(x='addr_state', hue='loan_status', data=data, palette='bright')
plt.title('Loan Status Counts for Each State')
plt.xticks(rotation=90)
plt.show()


# Most of the loan apllicants are from CA

# In[88]:


#earliest_cr_line(The month the borrower's earliest reported credit line was opened)

data['earliest_cr_line']



# In[89]:


# extracting year from the column
data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line']).dt.year


# In[90]:


data['earliest_cr_line']


# In[91]:


#verification_status(Indicates if income was verified by [Lending Club], not verified, or if the income source was verified.)

# Create a cross-tabulation (contingency table) between the two variables
cross_tab = pd.crosstab(data['verification_status'], data['loan_status'])

print(cross_tab)


# In[92]:


#initial_list_status ( it tells how a loan was first available for investment, f,w)

# if the value is "W", it means the loan was initially listed in the whole market,
#If the value is "F", it means the loan was initially listed in the fractional market, 


data['initial_list_status'].value_counts()


# In[93]:


# Create a cross-tabulation (contingency table) between the two variables
cross_tab2 = pd.crosstab(data['initial_list_status'], data['loan_status'])


# In[94]:


print(cross_tab)


# In[95]:


#application_type
data['application_type'].value_counts()


# In[96]:


# Create a count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='loan_status', hue='application_type', data=data, palette='Set2')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.title('Relationship between Application Type and Loan Status')
plt.legend(title='Application Type', loc='upper right')
plt.show()


# Hence we can analyze from the above plot, that most of the application were individual

# In[97]:


#disbursement_method (The method by which the borrower receives their loan. Possible values are: CASH, DIRECT_PAY)

custom_colors = ['#3498db', '#e74c3c']

# Create a cross-tabulation (contingency table) between the two variables
cross_tab = pd.crosstab(data['loan_status'], data['disbursement_method'])

# Plot the stacked bar plot with customized colors
plt.figure(figsize=(8, 6))
cross_tab.plot(kind='bar', stacked=True, color=custom_colors, edgecolor='black')
plt.xlabel('Loan Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Relationship between disbursement_method and Loan Status', fontsize=14)
plt.legend(title='disbursement_method', loc='upper left', fontsize=10)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Amlost 90% loans have been issued via cash.

# In[98]:


# From above analysis we can further reduced our features by droping those which we think will have no influence on our model


columns_to_drop = ['addr_state', 'disbursement_method', 'initial_list_status','revol_bal','total_acc',
                   'bc_open_to_buy','total_bal_ex_mort','total_bc_limit','emp_length']
data.drop(columns=columns_to_drop, inplace=True)  


# In[99]:


data.shape


# ### Imputation of null values

# In[100]:


data.isnull().sum()


# In[101]:


data.dtypes


# In[102]:


data['loan_status'] = data.loan_status.map({'Fully Paid':1, 'Charged Off':0})


# In[103]:


data['loan_status'].value_counts(dropna=False)


# In[104]:


data= data.dropna(subset=['loan_status'])


# #### I am using median as my imputation method to fill the null values in the respected features

# In[105]:


from sklearn.impute import SimpleImputer

num_cols = data.select_dtypes(include=['float64']).columns

imputer = SimpleImputer(strategy='median')
data[num_cols] = imputer.fit_transform(data[num_cols])


# In[106]:


data.isnull().sum()


# #### For the transformation of categorical variables to feed it to the models. I will be using one-hot encoding

# In[107]:


data = pd.get_dummies(data, drop_first=True)  # one-hot encoding


# In[108]:


data.head()


# In[109]:


from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, fbeta_score, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight




# In[ ]:





# In[110]:


X= data.drop('loan_status', axis=1)


# In[111]:


y = data['loan_status']


# In[112]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)



# #### I am experimenting with custom class weights based on the class distribution. This way, I will give more importance to the minority class

# In[113]:


# Calculate custom class weights based on the class distribution
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}


# #### I will be applying Random Under sampling method to make my majority class matched the minority, so that our results are better

# In[114]:


# Apply Random Undersampling to the majority class in the training set only
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)





# In[115]:


# Apply Random Undersampling to the majority class in the training set only
#smote = SMOTE(random_state=42)
#X_train_rus, y_train_rus = smote.fit_resample(X_train, y_train)


# In[116]:


# Normalize the data before feeding to the model
#scaler = StandardScaler()
#X_train_rus_normalized = scaler.fit_transform(X_train_rus)
#X_test_normalized = scaler.transform(X_test)


# In[117]:


# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the data using the MinMaxScaler
X_train_scaled = scaler.fit_transform(X_train_rus)
X_test_scaled = scaler.transform(X_test)


# ### RandomForestClassifier

# In[118]:


# Random Forest Classifier with custom class weights
rf = RandomForestClassifier(class_weight=class_weights_dict, random_state=42)


# In[119]:


# Fit the Random Forest model on the undersampled training data
rf.fit(X_train_scaled, y_train_rus)


# In[120]:


# Predict on the test set using Random Forest
y_pred_rf = rf.predict(X_test_scaled)


# In[121]:


# Print classification report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


# In[122]:


# Calculate F2-score for Random Forest
f2_score_rf = fbeta_score(y_test, y_pred_rf, beta=2)
print(f"Random Forest F2-score: {f2_score_rf}")


# In[123]:


# Check the accuracy score for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")


# In[124]:


# Calculate and print AUC-ROC and AUC-PR for Random Forest
y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_roc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest AUC-ROC: {auc_roc_rf}")


# In[125]:


auc_pr_rf = average_precision_score(y_test,  y_pred_rf)
print(f"Random Forest AUC-PR: {auc_pr_rf}")


# ### Gradient Boosting Classifier
# 

# In[126]:


from sklearn.utils.class_weight import compute_sample_weight


# In[127]:


# Calculate custom class weights based on the class distribution
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_rus)

# Manually assign custom class weights during training using sample_weight
sample_weights = y_train_rus.map({0: class_weights[0], 1: class_weights[1]})


# In[128]:


# Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train_rus, sample_weight=sample_weights)
y_pred_gb = gb.predict(X_test_scaled)










# In[ ]:





# In[129]:


# Print classification report for Gradient Boosting
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))


# In[130]:


# Calculate F2-score for Random Forest
f2_score_rf = fbeta_score(y_test, y_pred_rf, beta=2)
print(f"Gradient Boosting F2-score: {f2_score_rf}")


# In[131]:


# Check the accuracy score for Gradient Boosting
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Gradient Boosting Accuracy: {accuracy_rf}")



# In[132]:


# Calculate and print AUC-ROC and AUC-PR for Gradient Boosting
y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_roc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Gradient Boosting AUC-ROC: {auc_roc_rf}")


# In[133]:


auc_pr_rf = average_precision_score(y_test,  y_pred_rf)
print(f"Gradient Boosting: {auc_pr_rf}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### XGBoost Classifier

# In[134]:


# XGBoost Classifier
xgb = XGBClassifier(random_state=42, use_label_encoder=False)
xgb.fit(X_train_scaled, y_train_rus,sample_weight=sample_weights)
y_pred_xgb = xgb.predict(X_test_scaled)






# In[135]:


# Print classification report for XGBoost
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# In[136]:


# Check the accuracy score for XGBoost
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"XGBoost: {accuracy_rf}")


# In[137]:


# Calculate and print AUC-ROC and AUC-PR for XGBoost
y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_roc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"XGBoost: {auc_roc_rf}")


# In[138]:


auc_pr_rf = average_precision_score(y_test,  y_pred_rf)
print(f"XGBoost: {auc_pr_rf}")


# In[140]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Function to plot classification report as a heatmap
def plot_classification_report_heatmap(classification_report_str):
    report_lines = classification_report_str.split('\n')
    report_data = []
    row_labels = []
    for line in report_lines[2:-5]:
        row = line.split()
        row_labels.append(row[0])
        report_data.append([float(x) for x in row[1:]])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.heatmap(report_data, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=['precision', 'recall', 'f1-score'], yticklabels=row_labels)
    ax.set_xlabel("Metrics")
    ax.set_title("Classification Report")
    plt.show()

# Get classification report for each model
classification_report_rf = classification_report(y_test, y_pred_rf, target_names=['0', '1'])
classification_report_gb = classification_report(y_test, y_pred_gb, target_names=['0', '1'])
classification_report_xgb = classification_report(y_test, y_pred_xgb, target_names=['0', '1'])

# Plot classification report as a heatmap for each model
plot_classification_report_heatmap(classification_report_rf)
plot_classification_report_heatmap(classification_report_gb)
plot_classification_report_heatmap(classification_report_xgb)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




