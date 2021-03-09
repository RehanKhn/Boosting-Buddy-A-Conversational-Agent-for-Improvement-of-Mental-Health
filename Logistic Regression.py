#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[1]:


import numpy as np
from sklearn.datasets import fetch_openml

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt


# In[2]:


mnist = fetch_openml('mnist_784')
mnist


# In[3]:


X,Y = mnist["data"], mnist["target"]
X.shape


# In[4]:


Y.shape


# In[5]:


Y = Y.astype(np.int8)
total = 0
for i in range(10):
    print ("digit", i, "appear", np.count_nonzero(Y == i), "times")


# In[6]:


def plot_digit(some_digit):
    
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.axis("off")
    plt.show()
    
plot_digit(X[36003])


# In[7]:


Y[36003]


# In[8]:


plot_digit(X[8000])
print(Y[8000])


# In[9]:


X_96 = X[np.any([Y == 9 , Y == 6], axis = 0)]
Y_96 = Y[np.any([Y == 9 ,Y == 6], axis = 0)]


# In[10]:


plot_digit(X_96[8000])
print(Y_96[8000])
plot_digit(X_96[9355])
print(Y_96[9355])
plot_digit(X_96[877])
print(Y_96[877])
plot_digit(X_96[144])
print(Y_96[144])


# In[11]:


print(X_96.shape)
print(Y_96.shape)


# In[12]:


print("number of 6:", np.count_nonzero(Y_96 == 6))
print("number of 9:", np.count_nonzero(Y_96 == 9))


# In[42]:


shuffle_index = np.random.permutation(X_96.shape[0])
X_96_shuffled, Y_96_shuffled = X_96[shuffle_index], Y_96[shuffle_index]

train_proportion = 0.8
train_test_cut = int(len(X_96)*train_proportion)

X_train, X_test, y_train, y_test =     X_96_shuffled[:train_test_cut],     X_96_shuffled[train_test_cut:],     Y_96_shuffled[:train_test_cut],     Y_96_shuffled[train_test_cut:]
    
print("Shape of X_train is", X_train.shape)
print("Shape of X_test is", X_test.shape)
print("Shape of y_train is", y_train.shape)
print("Shape of y_test is", y_test.shape)


# In[43]:


np.count_nonzero(Y_96 == 6) / np.count_nonzero(Y_96 == 9)


# In[44]:


print(np.count_nonzero(y_train == 6) / np.count_nonzero(y_train == 9))
print(np.count_nonzero(y_test == 6) / np.count_nonzero(y_test == 9))


# In[45]:


X_train_normalised = X_train/255.0
X_test_normalised = X_test/255.0


# In[46]:


X_train_tr = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1,y_train.shape[0])
X_test_tr = X_test_normalised.transpose()
y_test_tr = y_test.reshape(1,y_test.shape[0])

print(X_train_tr.shape)
print(y_train_tr.shape)
print(X_test_tr.shape)
print(y_test_tr.shape)

dim_train = X_train_tr.shape[1]
dim_test = X_test_tr.shape[1]

print("The training dataset has dimensions equal to", dim_train)
print("The test set has dimensions equal to", dim_test)


# In[47]:


y_train_tr = y_train_tr.astype(np.int8)
y_test_tr = y_test_tr.astype(np.int8)
row, col = y_train_tr.shape
row1, col1 = y_test_tr.shape
for i in range(col):
    if (y_train_tr[0,i] == 6):
        y_train_shifted[0,i] = y_train_tr[0,i] - 6
    elif (y_train_tr[0,i] == 9):
        y_train_shifted[0,i] = y_train_tr[0,i] - 8
    else:
        print("Error")

for i in range(col1):
    if (y_test_tr[0,i] == 6):
        y_test_shifted[0,i] = y_test_tr[0,i] - 6
    elif (y_test_tr[0,i] == 9):
        y_test_shifted[0,i] = y_test_tr[0,i] - 8
    else:
        print("Error")

#y_train_shifted = y_train_tr - 1
#y_test_shifted = y_test_tr - 1


# In[48]:


plot_digit(X_train_tr[:,1005])
print(y_train_shifted[:,1005])
plot_digit(X_train_tr[:,1432])
print(y_train_shifted[:,1432])
plot_digit(X_train_tr[:,456])
print(y_train_shifted[:,456])
plot_digit(X_train_tr[:,567])
print(y_train_shifted[:,567])

Xtrain = X_train_tr
ytrain = y_train_shifted
Xtest = X_test_tr
ytest = y_test_shifted


# In[23]:


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s


# In[24]:


def initialize(dim):
    w = np.zeros((dim,1))
    b = 0
    
    assert (w.shape == (dim,1))
    assert (isinstance(b, float) or isinstance(b,int))
    
    return w,b


# In[25]:


def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    z = np.dot(w.T,X)+b
    A = sigmoid(z)
    cost = -1.0/m*np.sum(Y*np.log(A)+(1.0-Y)*np.log(1.0-A))
    
    dw = 1.0/m*np.dot(X, (A-Y).T)
    db = 1.0/m*np.sum(A-Y)
    
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    
    grads = {"dw": dw, 
             "db":db}
    
    return grads, cost


# In[26]:


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i % 100 == 0:
            costs.append(cost)
            
        if print_cost and i % 100 == 0:
            print ("Cost (iteration %i) = %f" %(i, cost))
            
    grads = {"dw": dw, "db": db}
    params = {"w": w, "b": b}
        
    return params, grads, costs


# In[30]:


def predict (w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid (np.dot(w.T, X)+b)
    
    for i in range(A.shape[1]):
        if (A[:,i] > 0.5): 
            Y_prediction[:, i] = '1'
        elif (A[:,i] <= 0.5):
            Y_prediction[:, i] = '0'
            
    assert (Y_prediction.shape == (1,m))
    
    return Y_prediction


# In[49]:


def model (X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.5, print_cost = False):
    
    w, b = initialize(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict (w, b, X_test)
    Y_prediction_train = predict (w, b, X_train)
    
    train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train-Y_train)*100.0)
    test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test-Y_test)*100.0)
    
    d = {"costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    print ("Accuarcy Test: ",  test_accuracy)
    print ("Accuracy Train: ", train_accuracy)
    
    return d
    


# In[57]:


#Xtrain = Xtrain.astype(np.int8)
#ytrain = ytrain.astype(np.int8) 
#Xtest = Xtest.astype(np.int8)
#ytest = ytest.astype(np.int8)
d = model (Xtrain, 
           ytrain, 
           Xtest, 
           ytest, 
           num_iterations = 4000, 
           learning_rate = 0.05, 
           print_cost = True)


# In[51]:


ypred = d["Y_prediction_test"]
ypred_ravel = ypred.ravel()
ytest_ravel = ytest.ravel()
ypred_ravel_tr = ypred_ravel.reshape(1,ypred_ravel.shape[0])
dim_pred = ypred_ravel_tr.shape[1]
ytest_ravel_tr = ytest_ravel.reshape(1,ytest_ravel.shape[0])
dim_test = ytest_ravel_tr.shape[1]

def confusion(ytest_ravel, ypred_ravel):
    TP, TN, FP, FN = 0,0,0,0
    for i in range(dim_pred):
        if((ytest_ravel[i]==0) & (ypred_ravel[i] == 0)):
            TP = TP+1
        elif((ytest_ravel[i]==1) & (ypred_ravel[i] == 1)):
            TN = TN+1
        elif((ytest_ravel[i]==0) & (ypred_ravel[i] == 1)):
            FP = FP+1
        elif((ytest_ravel[i]==1) & (ypred_ravel[i] == 0)):
            FN = FN+1
        else:
            Print("Error")
    Array= ([[TP, FP], [FN , TN]])
    print(Array)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    return print(Accuracy,  Recall, Precision)
    
c=confusion(ytest_ravel, ypred_ravel)

from sklearn.metrics import confusion_matrix

confusion_matrix (ytest_ravel, ypred_ravel)


# In[52]:


plt.plot(d["costs"])
plt.xlim([1,40])
plt.ylim([0,0.12])
plt.title("Cost Function",fontsize = 15)
plt.xlabel("Number of iterations x 100", fontsize = 14)
plt.ylabel("$J(w,b)$", fontsize = 17)
plt.show()


# In[53]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()


# In[54]:


XX = Xtrain.T
YY = ytrain.T.ravel()

logistic.fit(XX,YY)


# In[55]:


logistic.score(XX,YY)


# In[56]:


sum(logistic.predict(XX) == YY) / len(XX)


# In[ ]:




