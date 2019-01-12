
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pd


# In[336]:


train = pd.read_csv('train_values.csv')
test = pd.read_csv('test_values.csv')


# In[337]:


train.head(10)


# In[334]:


train.head(10)


# In[338]:


train['thal'] = train['thal'].map(
    {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2}).astype(int)


# In[339]:


test['thal'] = test['thal'].map(
    {'normal': 0, 'fixed_defect': 1, 'reversible_defect': 2}).astype(int)


# In[340]:


train_labels = pd.read_csv('train_labels.csv')


# In[341]:


train_labels.head(10)


# In[362]:


train.head(10)


# In[363]:


str(train.shape)


# In[364]:


str(test.shape)


# In[365]:


train_labels.shape


# In[366]:


train.count().min() == train.shape[0] and test.count().min() == test.shape[0]


# In[114]:


train.dtypes


# In[367]:


dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
print(dtype_df.groupby("Column Type").aggregate('count').reset_index()    )


# In[368]:


train.max()


# In[369]:


train.min() 


# In[117]:


test['slope_of_peak_exercise_st_segment'] = test['slope_of_peak_exercise_st_segment'].astype(int)


# In[370]:


test['slope_of_peak_exercise_st_segment'].dtypes


# In[119]:


import matplotlib.pyplot as plt


# In[120]:


plt.scatter(train['sex'], train_labels['heart_disease_present'])


# In[349]:


train_labels.shape


# In[350]:


import numpy as np


# In[351]:


X_sex = train['sex'].values


# In[352]:


X_chol.max()


# In[353]:


type(X_sex)


# In[354]:


Y_train = train_labels['heart_disease_present'].values


# In[355]:


Y_train


# In[356]:


plt.scatter(X_chol, Y_train)


# In[40]:


X_age = train['age'].values


# In[41]:


Y_train


# In[357]:


color = ['red' if yt==1 else 'green' for yt in Y_train] 


# In[358]:


plt.scatter( X_age,X_sex,X_chol, color=color)
plt.xlabel('X_age')
plt.ylabel('X_sex')
plt.title('Dataset')
plt.show()


# In[359]:


train.head(10)


# In[371]:


del train['patient_id']


# In[372]:


del test['patient_id']


# In[373]:


train['resting_blood_pressure'].min()


# In[48]:


def min_max( tr):
    mini = tr.min()
    maxi = tr.max()
    for i in range(0,len(tr)):
        tr[i] = (tr[i]-mini)/(maxi-mini)
    return tr    


# In[374]:


x_bp = train['resting_blood_pressure'].values


# In[375]:


train.head()


# In[51]:


x_bp.astype(float)


# In[139]:


x_bp = min_max(x_bp)


# In[140]:


x_bp


# In[376]:


test.shape


# In[377]:


train.shape


# In[393]:


train.head()


# In[379]:


# res = train.copy()
for i in train['max_heart_rate_achieved']:
    max_val = train['max_heart_rate_achieved'].max()
    min_val = train['max_heart_rate_achieved'].min()
    train['max_heart_rate_achieved'] = (train['max_heart_rate_achieved']-min_val)/(max_val-min_val)


# In[381]:


# res = train.copy()
for i in train['age']:
    max_val = train['age'].max()
    min_val = train['age'].min()
    train['age'] = (train['age']-min_val)/(max_val-min_val)


# In[383]:


# res = train.copy()
for i in train['resting_blood_pressure']:
    max_val = train['resting_blood_pressure'].max()
    min_val = train['resting_blood_pressure'].min()
    train['resting_blood_pressure'] = (train['resting_blood_pressure']-min_val)/(max_val-min_val)


# In[385]:


# res = train.copy()
for i in train['serum_cholesterol_mg_per_dl']:
    max_val = train['serum_cholesterol_mg_per_dl'].max()
    min_val = train['serum_cholesterol_mg_per_dl'].min()
    train['serum_cholesterol_mg_per_dl'] = (train['serum_cholesterol_mg_per_dl']-min_val)/(max_val-min_val)


# In[388]:


# res = train.copy()
for i in test['max_heart_rate_achieved']:
    max_val = test['max_heart_rate_achieved'].max()
    min_val = test['max_heart_rate_achieved'].min()
    test['max_heart_rate_achieved'] = (test['max_heart_rate_achieved']-min_val)/(max_val-min_val)


# In[389]:


# res = train.copy()
for i in test['age']:
    max_val = test['age'].max()
    min_val = test['age'].min()
    test['age'] = (test['age']-min_val)/(max_val-min_val)


# In[390]:


# res = train.copy()
for i in test['resting_blood_pressure']:
    max_val = test['resting_blood_pressure'].max()
    min_val = test['resting_blood_pressure'].min()
    test['resting_blood_pressure'] = (test['resting_blood_pressure']-min_val)/(max_val-min_val)


# In[391]:


# res = train.copy()
for i in test['serum_cholesterol_mg_per_dl']:
    max_val = test['serum_cholesterol_mg_per_dl'].max()
    min_val = test['serum_cholesterol_mg_per_dl'].min()
    test['serum_cholesterol_mg_per_dl'] = (test['serum_cholesterol_mg_per_dl']-min_val)/(max_val-min_val)


# In[394]:


train.head()


# In[395]:


train[train['resting_ekg_results']==2].shape


# In[396]:


def sigmoid(z):
    return 1.0/(1+np.exp(-z))


# In[397]:


lmao = sigmoid(2)


# In[398]:


lmao


# In[513]:


X = train.values


# In[517]:


Y = train_labels.values


# In[518]:


Y.shape


# In[401]:


X_test = test.values


# In[519]:


X_test.shape


# In[403]:


Y_train.shape


# In[520]:


del train_labels['patient_id']


# In[521]:


train_labels.shape

Y_train = train_labels.values
# In[406]:


Y_train = train_labels.values


# In[407]:


Y_train.shape


# In[408]:


X_train.dtype


# In[409]:


type(X_train)


# In[411]:


X_train.T


# In[412]:


X_train_cv = X_train[150:180]


# In[413]:


X_train_cv.shape


# In[414]:


X_train = X_train[:150]


# In[415]:


Y_train_cv = Y_train[150:180] 


# In[416]:


Y_train = Y_train[:150]


# In[417]:


Y_train.shape[0]


# In[418]:


X_train.shape


# In[419]:


Y_train.shape


# In[420]:


X_test.shape


# In[421]:


X_train_cv.shape


# In[422]:


Y_train_cv.shape


# In[423]:


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


# In[424]:


theta_init = np.random.rand(X_train.shape[1])


# In[425]:


theta_init


# In[429]:


type(theta)


# In[427]:


def cost(X, Y, theta):
    h = np.dot(X.T, theta)
    H = float(sigmoid(h))
    return (-Y*np.log(H)-(1-Y)*np.log(1-H))


# In[428]:


h = np.dot(X_train[0].T, theta)
H = sigmoid(h)
h
1/(1+np.exp(-H))


# In[430]:


def compute_cost(X, Y, theta):
    total_cost = 0
    M = float(X.shape[0])
    for i in range(0, X.shape[0]):
        total_cost += cost(X[i], Y[i,0],theta)
    return total_cost/M


# In[431]:


def gradient_descent(X, Y, start_theta, alpha, num_iter):
    theta = start_theta
    cost_graph = []
    theta_prog = []
    for i in range(num_iter):
        cost_graph.append(compute_cost(X, Y, theta))
        theta = step_gradient(theta, alpha, X, Y)
        theta_prog.append(theta)
    return (theta, theta_prog, cost_graph) 


# In[432]:


def step_gradient(theta_curr, alpha, X, Y):
    theta_grad = np.zeros(X.shape[1])
    M = float(X.shape[0])
    for i in range(0, X.shape[0]):
        x = X[i]
        y = Y[i,0]
        for j in range(x.shape[0]):
            theta_grad[j] += -(1/M)*(y-sigmoid(np.dot(x, theta_curr)))*x[j]
    theta_up = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        theta_up[i] = theta_curr[i] - alpha*theta_grad[i]
    return theta_up    


# In[489]:


alpha = 0.04
num_iters = 10000
theta_init = np.zeros(13)


# In[490]:


theta, theta_prog, cost_graph = gradient_descent(X_train, Y_train, theta_init, alpha, num_iters)


# In[491]:


plt.plot(cost_graph)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.title('Cost per iteration')
plt.show()


# In[492]:


sigmoid(np.dot(X_train_cv[24], theta))


# In[493]:


Y_train_cv[24]


# In[494]:


theta


# In[495]:


def predict_prob(X, theta):
    return sigmoid(np.dot(X, theta))


# In[496]:


def predict(X, theta, threshold=0.5):
    if predict_prob(X, theta) >= threshold:
        return 1
    else:
        return 0


# In[497]:


X_train_cv.shape[0]


# In[498]:


Y_train_cv.shape


# In[499]:


predict(X_train_cv[0], theta)


# In[500]:


Y_train_cv[0,0]


# In[501]:


for i in range(X_train_cv.shape[0]):
    predict_cv.append(predict(X_train_cv[i], theta))


# In[502]:


predict_cv[4]


# In[503]:


for i in range(X_train_cv.shape[0]):
    predict_prob_cv.append(predict_prob(X_train_cv[i], theta))


# In[504]:


performance_matrix_cv = np.array([[0, 0], [0, 0]])


# In[505]:


performance_matrix_cv


# In[506]:


for i in range(X_train_cv.shape[0]):
    performance_matrix_cv[1-predict_cv[i], 1-Y_train_cv[i,0]] += 1


# In[507]:


performance_matrix_cv


# In[508]:


train.shape


# In[488]:


train.head()


# In[509]:


test.head()


# In[510]:


X_train.shape


# In[512]:


X = np.concatenate(X_train, X_train_cv)


# In[522]:


theta_full, theta_prog_full, cost_graph_full = gradient_descent(X, Y, theta_init, alpha, num_iters)


# In[523]:


plt.plot(cost_graph_full)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.title('Cost per iteration')
plt.show()


# In[524]:


predict_full = []
for i in range(X.shape[0]):
    predict_full.append(predict(X[i], theta_full))


# In[525]:


predict_prob_full = []
for i in range(X.shape[0]):
    predict_prob_full.append(predict_prob(X[i], theta_full))


# In[526]:


performance_matrix_full = np.array([[0, 0], [0, 0]])


# In[527]:


for i in range(X.shape[0]):
    performance_matrix_full[1-predict_full[i], 1-Y[i,0]] += 1


# In[529]:


performance_matrix_full


# In[530]:


predict_test = []
for i in range(X_test.shape[0]):
    predict_test.append(predict(X_test[i], theta_full))


# In[531]:


predict_prob_test = []
for i in range(X_test.shape[0]):
    predict_prob_test.append(predict_prob(X_test[i], theta_full))


# In[535]:


predict_prob_test


# In[536]:


test_data = pd.read_csv('test_values.csv')


# In[537]:


test_data.shape


# In[539]:


pred = np.asarray(predict_prob_test)


# In[541]:


pred.shape


# In[542]:


patient_id_test = test_data['patient_id']


# In[544]:


patient_id_test.shape


# In[546]:


submission = pd.DataFrame({
    "patient_id": test_data["patient_id"],
    "heart_disease_present": pred
})


# In[547]:


submission.to_csv('sol.csv', index = False)

