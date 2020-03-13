import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
print(predictors)

numero_arboles_bootstrap = 100

x_train, x_test, y_train, y_test = train_test_split(data[predictors], data['Target'], train_size=0.5)

size_train = x_train.shape[0]
size_test = x_test.shape[0]
caracteristicas = x_train.shape[1]
max_depth = np.arange(1,10)


F1_train = np.zeros([numero_arboles_bootstrap, max_depth.shape[0]])
F1_test = np.zeros([numero_arboles_bootstrap, max_depth.shape[0]])

AFI = np.zeros([numero_arboles_bootstrap, max_depth.shape[0], caracteristicas])

for j in range(max_depth.shape[0]):
    for i in range(numero_arboles_bootstrap):
        sub_conjunto = np.random.randint(0, size_train,size_train )
        sub_conjunto_x_train = x_train.loc[x_train.index.values[sub_conjunto]]
        sub_conjunto_y_train = y_train.loc[x_train.index.values[sub_conjunto]]
        clf = DecisionTreeClassifier(max_depth = max_depth[j])
        clf = clf.fit(sub_conjunto_x_train, sub_conjunto_y_train)
        
        F1_train[i,j] = f1_score(y_train, clf.predict(x_train))
        F1_test[i,j] = f1_score(y_test, clf.predict(x_test))
        
        AFI[i,j,:] = clf.feature_importances_
        

plt.figure(figsize = (10,5))
plt.errorbar(max_depth, np.mean(F1_train, axis = 0), yerr = 
             np.std(F1_train, axis = 0), marker='^', label = 'test 50%')       
plt.errorbar(max_depth, np.mean(F1_test, axis = 0), yerr = 
             np.std(F1_test, axis = 0), marker='o', label = 'train 50%')      
plt.legend()   
plt.ylabel('Average F1 score')
plt.xlabel('max depth')
plt.savefig('F1_training_test.png')

AFI_final = np.mean(AFI, axis = 0)

plt.figure(figsize = (10,5))
for i in range(caracteristicas):
    plt.plot(max_depth, AFI_final[:, i], label = predictors[i])
    
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=5)
plt.xlabel('max_depth')
plt.ylabel('Average feature importance')
plt.savefig('features.png')