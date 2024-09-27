import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as metrics 
import matplotlib.pyplot as plt
import os


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

#path = os. getcwd()
#os.chdir('/Users/marin/TEACHING/2324/M2-GLM-HAX912X/TP')

# Braking distance as a function of vehicle speed (speed in km/h and distance in meters) 

data = np.loadtxt("freinage.txt")
print("\n Dataset:\n")
print(data)

X_data = data[:,0].reshape(len(data),1)  
Y_data = data[:,1].reshape(len(data),1)  
print("\n Number of observations: %d" %len(X_data))

scaler = StandardScaler().fit(X_data)
X_data = scaler.transform(X_data)

plt.figure(figsize=(10,6))
plt.xlabel("X: Speed")
plt.ylabel("Y: Braking distance")
plt.grid()
plt.xlim(-2.5, 1.5)
plt.ylim(0, 100)
plt.scatter(X_data, Y_data)

lr = lm.LinearRegression()
lr.fit(X_data, Y_data)  
print(lr.intercept_)  
print(lr.coef_)

X = np.linspace(-2.5,2.5,num=30).reshape(30,1)
Y_pred_lr = lr.predict(X)  

plt.figure(figsize=(10,6))
plt.plot(X_data, Y_data,'o')
plt.plot(X, Y_pred_lr, '-g')
plt.xlim(-2.5, 2.5)
plt.ylim(-10, 100)
plt.xlabel("X: Speed")
plt.ylabel("Y: Braking distance")
plt.grid()
plt.title('Dataset and linear regression')
plt.legend(["Dataset","Linear regression"])

print(metrics.mean_squared_error(Y_data,lr.predict(X_data)))

poly2 = PolynomialFeatures(degree=2,include_bias=False)
X_data2 = poly2.fit_transform(X_data)
scaler = StandardScaler().fit(X_data2)
X_data2 = scaler.transform(X_data2)
X2 = poly2.fit_transform(X)
X2 = scaler.transform(X2)

poly6 = PolynomialFeatures(degree=6,include_bias=False)
X_data6 = poly6.fit_transform(X_data)
scaler = StandardScaler().fit(X_data6)
X_data6 = scaler.transform(X_data6)
X6 = poly6.fit_transform(X)
X6 = scaler.transform(X6)

lrp2 = lm.LinearRegression()
lrp2.fit(X_data2,Y_data)
print("\n Polynomial regression of order 2")
print(lrp2.intercept_)
print(lrp2.coef_)

lrp6 = lm.LinearRegression()
lrp6.fit(X_data6,Y_data)
print("\n Polynomial regression of order 6")
print(lrp6.intercept_)
print(lrp6.coef_)


Y_pred_lrp2=lrp2.predict(X2)  
Y_pred_lrp6=lrp6.predict(X6)

plt.figure(figsize=(10,6))
plt.plot(X_data, Y_data,'o')
plt.plot(X, Y_pred_lr, '-g')
plt.plot(X, Y_pred_lrp2, '-b')
plt.plot(X, Y_pred_lrp6, '-c')
plt.xlim(-2.5, 2.5)
plt.ylim(-10, 100)
plt.xlabel("X: Speed")
plt.ylabel("Y: Braking distance")
plt.grid()
plt.title('Regression')
plt.legend(["Dataset","Linear regression","Model of order 2","Model of order 6"])

print("\n Linear reression: MSE = %.3f" %metrics.mean_squared_error(Y_data,lr.predict(X_data)))
print("\n Polynomial regression of order 2, MSE = %.3f" %metrics.mean_squared_error(Y_data,lrp2.predict(X_data2)))
print("\n Polynomial regression of order 6, MSE = %.3f" %metrics.mean_squared_error(Y_data,lrp6.predict(X_data6)))

ridgealpha0 = lm.Ridge(alpha=0)
ridgealpha0.fit(X_data6,Y_data)
print("\n Ridge regression alpha=0")
print(ridgealpha0.intercept_)
print(ridgealpha0.coef_)
Y_pred_ridgealpha0 = ridgealpha0.predict(X6)

ridgealpha01 = lm.Ridge(alpha=0.1)
ridgealpha01.fit(X_data6,Y_data)
print("\n Ridge regression alpha=0.1")
print(ridgealpha01.intercept_)
print(ridgealpha01.coef_)
Y_pred_ridgealpha01 = ridgealpha01.predict(X6)

ridgealpha1 = lm.Ridge(alpha=1)
ridgealpha1.fit(X_data6,Y_data)
print("\n Ridge regression alpha=1")
print(ridgealpha1.intercept_)
print(ridgealpha1.coef_)
Y_pred_ridgealpha1 = ridgealpha1.predict(X6)

ridgealpha10 = lm.Ridge(alpha=10)
ridgealpha10.fit(X_data6,Y_data)
print("\n Ridge regression alpha=10")
print(ridgealpha10.intercept_)
print(ridgealpha10.coef_)
Y_pred_ridgealpha10 = ridgealpha10.predict(X6)

ridgealpha100 = lm.Ridge(alpha=100)
ridgealpha100.fit(X_data6,Y_data)
print("\n Ridge regression alpha=100")
print(ridgealpha100.intercept_)
print(ridgealpha100.coef_)
Y_pred_ridgealpha100 = ridgealpha100.predict(X6)

plt.figure(figsize=(10,6))
plt.plot(X_data, Y_data,'o')
plt.plot(X, Y_pred_ridgealpha0, '-g')
plt.plot(X, Y_pred_ridgealpha01, '-b')
plt.plot(X, Y_pred_ridgealpha1, '-c')
plt.plot(X, Y_pred_ridgealpha10, '-r')
plt.plot(X, Y_pred_ridgealpha100, '-k')
plt.xlim(-2.5, 2.5)
plt.ylim(-10, 80)
plt.xlabel("X: Speed")
plt.ylabel("Y: Braking distance")
plt.grid()
plt.title('Ridge regression, d=6')
plt.legend(["Dataset","alpha=0","alpha=0.1","alpha=1","alpha=10","alpha=100"])

ridge1 = lm.RidgeCV(alphas=np.logspace(-5, 5, 20), cv=5)
ridge1.fit(X_data,Y_data)
print("Ridge regression, order 1")
print(ridge1.intercept_)
print(ridge1.coef_)
print("Optimal alpha: %.5f" %ridge1.alpha_)

ridge2 = lm.RidgeCV(alphas=np.logspace(-5, 5, 20), cv=5)
ridge2.fit(X_data2,Y_data)
print("\n Ridge regression, order 2")
print(ridge2.intercept_)
print(ridge2.coef_)
print("Optimal alpha: %.5f" %ridge2.alpha_)

ridge6 = lm.RidgeCV(alphas=np.logspace(-5, 5, 20), cv=5)
ridge6.fit(X_data6,Y_data)
print("\n Ridge regression, order 6")
print(ridge6.intercept_)
print(ridge6.coef_)
print("Optimal alpha: %.5f" %ridge6.alpha_)

Y_pred_lrr1=ridge1.predict(X)
Y_pred_lrr2=ridge2.predict(X2)
Y_pred_lrr6=ridge6.predict(X6)

plt.figure(figsize=(10,6))
plt.plot(X_data, Y_data,'o')
plt.plot(X, Y_pred_lrr1, '-g')
plt.plot(X, Y_pred_lrr2, '-b')
plt.plot(X, Y_pred_lrr6, '-c')
plt.xlim(-2.5, 2.5)
plt.ylim(-10, 80)
plt.xlabel("X: Speed")
plt.ylabel("Y: Braking distance")
plt.grid()
plt.title('ridge regression')
plt.legend(["Dataset","Model of order 1","Model of order 2","Model of order 6"])

print("\n Ridge regression, order 1, MSE = %.2f" %metrics.mean_squared_error(Y_data,ridge1.predict(X_data)))
print("\n Ridge regression, order 2, MSE = %.2f" %metrics.mean_squared_error(Y_data,ridge2.predict(X_data2)))
print("\n Ridge regression, order 6, MSE = %.2f" %metrics.mean_squared_error(Y_data,ridge6.predict(X_data6)))

lasso1 = lm.LassoCV()
lasso1.fit(X_data,np.ravel(Y_data))  
print("Lasso regression")
print(lasso1.intercept_)
print(lasso1.coef_)
print("Optimal alpha: %.5f" %lasso1.alpha_)

lasso2 = lm.LassoCV()
lasso2.fit(X_data2,np.ravel(Y_data))  
print("\n Lasso regression, polynomial of 2")
print(lasso2.intercept_)
print(lasso2.coef_)
print("Optimal alpha: %.5f" %lasso2.alpha_)

lasso6 = lm.LassoCV(max_iter=10000)   
lasso6.fit(X_data6,np.ravel(Y_data))
print("\n lasso regression, polynome degré 6")
print(lasso6.intercept_)
print(lasso6.coef_)
print("Optimal alpha: %.5f" %lasso6.alpha_)

Y_pred_lasso1=lasso1.predict(X)
Y_pred_lasso2=lasso2.predict(X2)
Y_pred_lasso6=lasso6.predict(X6)

plt.figure(figsize=(10,6))
plt.plot(X_data, Y_data,'o')
plt.plot(X, Y_pred_lr, '-g')
plt.plot(X, Y_pred_lasso2, '-b')
plt.plot(X, Y_pred_lasso6, '-c')
plt.xlim(-2.5, 2.5)
plt.ylim(-10, 80)
plt.xlabel("X: Speed")
plt.ylabel("Y: Braking distance")
plt.grid()
plt.title('Lasso')
plt.legend(["Dataset","Linear regression","Lasso order 2","Lasso order 6"])

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

Valid_croisee = RepeatedKFold(n_splits=5, n_repeats=100)
scores_lr = cross_val_score(lr, X_data, Y_data, cv=Valid_croisee)
print("lr - Accuracy: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))

alphas = np.logspace(-4, -0.5, 30)
parameters = {'alpha':alphas}
lasso6 = GridSearchCV(lm.Lasso(),parameters,cv=Valid_croisee,n_jobs=-1)
lasso6.fit(X_data6,np.ravel(Y_data))
print(lasso6.best_params_)
print(lasso6.best_estimator_.coef_)