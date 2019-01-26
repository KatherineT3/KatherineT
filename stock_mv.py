
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# =============================================================================
# data_split
# =============================================================================
def data_split(X, y, test_size=0.2):                
    test_rows = int(test_size * X.shape[0]+1)
    X_train = X.iloc[0:-test_rows,:]
    X_test = X.iloc[-test_rows:,:]
    y_train = y.iloc[0:-test_rows]
    y_test = y.iloc[-test_rows:]     
                                                               
    return X_train, X_test, y_train, y_test  
# =============================================================================
# model_validation
# =============================================================================
def model_validation(clf, X, y, **kwargs):    
    score = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()

    X_train, X_test, y_train, y_test = data_split(X, y)    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)    
    accuracy = metrics.accuracy_score(y_test, y_pred)  

    if(len(np.unique(y_pred)) == 1):
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    roc_auc = metrics.roc_auc_score(y_test, y_pred)

    out = {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1,
        'roc_auc' : roc_auc,
        'score' : score,
        'clf' : clf
    }
    return out  
# =============================================================================
# ranking
# =============================================================================
def ranking(clf, model='Dtree', feature_cols=None):
    ranks = []
    if(model == 'Dtree'):
        ranks = sorted(zip(clf.feature_importances_, feature_cols), reverse=True)
    else:
        raise AssertionError('Invalid ranking model: {m}'.format(m=model))
    return ranks 
# =============================================================================
# best_tree_depth
# =============================================================================
def best_tree_depth(X, y, **kwargs):
    if('kwargs' in kwargs.keys()): kwargs = kwargs['kwargs']
    
    max_depth_range = range(2,21)
    accuracy = []
    out_list = []
   
    if('random_state' in kwargs.keys()): random_state = kwargs['random_state']
    else: random_state = 1
        
    for depth in max_depth_range:                    
        clf = DecisionTreeClassifier(max_depth=depth, random_state=random_state)         
        vout = model_validation(clf, X, y) 
        out_list.append(vout)
        accuracy.append(vout['accuracy'])                   

    if('plot' in kwargs.keys() and kwargs['plot'] == 'yes'):
        label = kwargs['label']
        plt.plot(max_depth_range, accuracy, label=label)
        plt.legend(loc='best')
        plt.xlabel('max_depth')
        plt.ylabel('accuracy')
        
    depths = sorted(zip(accuracy, max_depth_range, out_list),reverse=True)
    best = depths[0][2]   
    best['params'] = {'max_depth':depths[0][1]}
      
    return best
# =============================================================================
# recursive_feature_evaluation
# - feature_cols is a list of feature vars
# - class_col is a target var
# =============================================================================
def recursive_feature_evaluation(df, model='Dtree', feature_cols=None, class_col=None, **kwargs):       
    out_list = []
    f_len = len(feature_cols)    
    feature_cols = feature_cols[:f_len]
   
    while f_len >= 2:
        feature_cols = feature_cols[0:f_len]
        X = df[feature_cols]          
        y = df[class_col].astype(int)             
        out = best_tree_depth(X, y)        
        out['feature_size'] = f_len
        out['feature_cols'] = feature_cols        
        out_list.append(out)

        clf = out['clf']

        ranks = ranking(clf, model=model, feature_cols=feature_cols)
        feature_cols = [x[1] for x in ranks] 
  
        f_len -= 1
   
    cols = list(out_list[0].keys())
    mdf = pd.DataFrame(out_list, columns=cols)
    
    return mdf
# =============================================================================
# best_RF_params
# =============================================================================
def best_RF_params(X, y, **kwargs):
    if('kwargs' in kwargs.keys()): kwargs = kwargs['kwargs']
    
    max_depth_range = range(2,21)
    accuracy = []
    out_list = []  
   
    if('random_state' in kwargs.keys()): random_state = kwargs['random_state']
    else: random_state = 1
    if('n_estimators' in kwargs.keys()): n_estimators = kwargs['n_estimators']
    else: n_estimators = 50
            
    for depth in max_depth_range:                             
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth,random_state=random_state)
        vout = model_validation(clf, X, y) 
        out_list.append(vout)        
        accuracy.append(vout['accuracy'])            

    depths = sorted(zip(accuracy, max_depth_range, out_list),reverse=True)   
    best = depths[0][2]
    best['params'] = {'depth':depths[0][1],'n_estimators':n_estimators}
    
    return best   
# =============================================================================
# best_k
# =============================================================================
def best_k(X, y, **kwargs): 
    if('kwargs' in kwargs.keys()): kwargs = kwargs['kwargs']
    
    accuracy = []
    out_list = []
 
    k_range = range(2,51)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)               
        pipe = make_pipeline(StandardScaler(), knn)        
        vout =  model_validation(pipe, X, y)       
        out_list.append(vout)
        accuracy.append(vout['accuracy'])          

    if('plot' in kwargs.keys() and kwargs['plot'] == 'yes'):
        label = kwargs['label']
        plt.plot(k_range, accuracy, label=label)
        plt.legend(loc='best')
        plt.xlabel('K')
        plt.ylabel('accuracy')

    depths = sorted(zip(accuracy, k_range, out_list),reverse=True)   
    best = depths[0][2]   
    best['params'] = {'k_value': depths[0][1]}
     
    return best
# =============================================================================
# logReg
# =============================================================================
def logReg(X, y, **kwargs):    
    if('kwargs' in kwargs.keys()): kwargs = kwargs['kwargs']
        
    logreg = LogisticRegression(solver='lbfgs')    
    pipe = make_pipeline(StandardScaler(), logreg)
    out =  model_validation(pipe, X, y) 
    out['params'] = {'params':None}     

    return out
# =============================================================================
# model_eval
# =============================================================================
def model_eval(df, model='Dtree', feature_sets=None, **kwargs): 
    
    if('random_state' in kwargs.keys()): random_state=kwargs['random_state']
    else: random_state = 1
        
    if(model == 'Dtree' and 'max_depth' in kwargs.keys()):
        f_cols = kwargs['feature_cols']
        y_col = kwargs['y_col']
        X = df[f_cols]
        y = df[y_col]
        
        depth = kwargs['max_depth']                
        clf = DecisionTreeClassifier(max_depth=depth, random_state=random_state)         
        out = model_validation(clf, X, y) 
        return out
    elif(model == 'knn' and 'n_neighbors' in kwargs.keys()):
        f_cols = kwargs['feature_cols']
        y_col = kwargs['y_col']
        X = df[f_cols]
        y = df[y_col]
        
        k = kwargs['n_neighbors']        
        knn = KNeighborsClassifier(n_neighbors=k)               
        pipe = make_pipeline(StandardScaler(), knn)        
        out =  model_validation(pipe, X, y)
        return out
    else:
        out_list = []
        for index, row in feature_sets.iterrows():
            class_key, f_id, feature_cols = row['label'], row['feature_id'], row['feature_cols'] 
            
            y = df[class_key].astype(int)                             
            X = df[feature_cols]                              
            out = best_params[model](X, y, kwargs=kwargs)

            item = {'label' : class_key,'model' : model, 'feature_id':f_id}            
            item.update(out)                 
            item['feature_size'] = len(feature_cols)
            item['feature_cols'] = feature_cols                                  
            out_list.append(item)
        
        cols = list(out_list[0].keys()) 
        mdf = pd.DataFrame(out_list, columns=cols) 

        return mdf

# =============================================================================
# tree_plot
# =============================================================================
from sklearn.tree import export_graphviz
import graphviz

def tree_plot(tree_clf, tree_name='tree_stock', feature_cols=None):
    dot_data = export_graphviz(tree_clf, out_file=None, 
                           feature_names=feature_cols,                           
                           filled=True, 
                           rounded=True,
                           special_characters=True)    
  
    # write pdf graph
    graph = graphviz.Source(dot_data)
    graph.render("stock_tree") 

# =============================================================================
# 
# =============================================================================
best_params = {}
best_params['Dtree'] = best_tree_depth
best_params['RF'] = best_RF_params
best_params['knn'] = best_k
best_params['logReg'] = logReg
