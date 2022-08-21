''' This file contains all functions used to extract features, load data, define models, plot examples, plot confusion matrices'''

import os
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
from imageio import imread
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.metrics import matthews_corrcoef

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

import tensorflow
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


#%% 
def plot_confusion_matrix(y_train,y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_train, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(15,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]), 
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
  

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
#%%
def plot_examples(path, index):
  """This functions plots a single-cell image along with its segmentation"""

  img = imread(path +'/' + str(index)+ '.bmp') # read image
  img_segCyt = imread(path +'/' + str(index)+ '_segCyt.bmp') # read cytoplasm segmentation image
  img_segNuc = imread(path +'/' + str(index)+ '_segNuc.bmp') # read nucleus segmentation image

  fig = plt.figure(figsize=(12, 12)) # size of the figure
  grid = AxesGrid(fig, 111,
                  nrows_ncols = (1, 3),
                  axes_pad = 0.5) # code to create subplots
  grid[0].imshow(img)
  grid[0].axis('off')
  grid[0].set_title('Original image')
  grid[1].imshow(img_segCyt,cmap='gray')
  grid[1].axis('off')
  grid[1].set_title("Segmentation Cytoplasma")
  grid[2].imshow(img_segNuc,cmap='gray')
  grid[2].axis('off')
  grid[2].set_title("Segmentation Nucleus")
  plt.show()

#%%
def Area(im):
    ''' This function takes as input a segmentation image of a specific area and returns the Nucleus area'''
    c = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] == 255 :
                c += 1
    return c/(im.shape[0]*im.shape[1])
#%%
def NC_ratio(im_seg_nuc,im_seg_cyt):
    ''' This function takes as input an image and returns the size of nucleus relative to cell size'''
    return Area(im_seg_nuc)/(Area(im_seg_nuc)+Area(im_seg_cyt))
#%%
def Brightness(im, im_seg):
    ''' This function takes as input an image and its segmentation and returns the brightness of a specific area'''
    red_mu = im[im_seg==255][:,0].mean()
    green_mu = im[im_seg==255][:,1].mean()
    blue_mu = im[im_seg==255][:,2].mean()
    return 0.999*red_mu+0.587*green_mu+0.114*blue_mu
#%%
def Longest_diameter(im): 
    ''' The diameter of the circle circumscribing the segmented area'''
    x = [B[i][0] for i in range(B.shape[0])]
    y = [B[i][1] for i in range(B.shape[0])]
    diam1 = max(x)-min(x)
    diam2 = max(y)-min(y)
    diam = max(diam1,diam2)
    return diam
#%%
def Shortest_diameter(im): 
    ''' The diameter of the inscribed circle of the segmented area'''
    B = np.argwhere(im==im.max())
    x = [B[i][0] for i in range(B.shape[0])]
    y = [B[i][1] for i in range(B.shape[0])]
    diam1 = max(x)-min(x)
    diam2 = max(y)-min(y)
    diam = min(diam1,diam2)
    return diam
#%%
def Elongation(im):
    return Shortest_diameter(im)/Longest_diameter(im)
#%%
def Roundness(im):
    circle = np.pi/4 * Longest_diameter(im)**2
    return Area(im)/circle
#%%
def Perimeter_nuc(im, im_nuc):
  countours, _ = cv2.findContours(cv2.convertScaleAbs(im_nuc), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  try:
    x=len(countours[0])
  except:
    x=0
  return x
#%%
def Perimeter_cyt(im, im_cyt):
  im_cyt = scipy.ndimage.morphology.binary_fill_holes(im_cyt)
  im_cyt = im_cyt.astype(int)
  countours, _ = cv2.findContours(cv2.convertScaleAbs(im_cyt), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  return len(countours[0])
#%%
def N_pos(im_seg_nuc,im_seg_cyt):
    B = np.argwhere(im_seg_nuc==im_seg_nuc.max())
    x = [B[i][0] for i in range(B.shape[0])]
    y = [B[i][1] for i in range(B.shape[0])]
    xn = (min(x)+max(x))/2
    yn = (min(y)+max(y))/2
    
    B = np.argwhere(im_seg_cyt==im_seg_cyt.max())
    x = [B[i][0] for i in range(B.shape[0])]
    y = [B[i][1] for i in range(B.shape[0])]
    xc = (min(x)+max(x))/2
    yc = (min(y)+max(y))/2
    
    return 2*np.sqrt((xn-xc)**2 + (yn-yc)**2)/Longest_diameter(im_seg_cyt)
#%%
def Maxima_Minima(im,im_nuc,im_cyt):
  count_max1 = 0
  count_min1 = 0
  count_max2 = 0
  count_min2 = 0
  im1 = im.copy()
  im2 = im.copy()
  im1[im_nuc!=im_nuc.max()]=0
  im2[im_cyt!=im_cyt.max()]=0
  for x in range(1,im.shape[0]-1):
    for y in range(1,im.shape[1]-1):
      window1 = im1[x-1:x+2,y-1:y+2]
      window2 = im2[x-1:x+2,y-1:y+2]
      count_max1 += window1[window1==np.max(window1)].shape[0]
      count_min1 += window1[window1==np.min(window1)].shape[0]
      count_max2 += window2[window2==np.max(window2)].shape[0]
      count_min2 += window2[window2==np.min(window2)].shape[0]
  return count_max1, count_min1, count_max2, count_min2
#%%
def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name, 
            files in os.walk(root) for f in files if f.endswith(file_type)]
#%%
def extract_features(path_train_images,df_train):

    train_all_images = filelist(path_train_images, 'bmp')
    Ids_train = []
    for c in train_all_images :
        id_ = c.split('/')[-1].split('.bmp')[0]
        if id_.isnumeric() :
            Ids_train.append(id_)
        
    df_train['Id'] = Ids_train
    
    N_area = []
    C_area = []
    Nlong = []
    Clong = []
    Nshort = []
    Cshort = []
    Nround = []
    Cround = []
    Nperim = []
    Cperim = []
    Npos = []
    Nelong = []
    Celong = []
    Nmax = []
    Nmin = []
    Cmax = []
    Cmin =[]
    
    for id_ in Ids_train :
        im = np.array(imread(path_train_images+'/'+str(id_)+'.bmp'))
        im_nuc=np.array(imread(path_train_images+'/'+str(id_)+'_segNuc.bmp'))  
        im_cyt = np.array(imread(path_train_images+'/'+str(id_)+'_segCyt.bmp'))

        N_area.append(Area(im_nuc))
        C_area.append(Area(im_cyt))
        Nlong.append(Longest_diameter(im_nuc))
        Clong.append(Longest_diameter(im_cyt))
        Nshort.append(Shortest_diameter(im_nuc))
        Cshort.append(Shortest_diameter(im_cyt))
        Nround.append(Roundness(im_nuc))
        Cround.append(Roundness(im_cyt))
        Nperim.append(Perimeter_nuc(im, im_nuc))
        Cperim.append(Perimeter_cyt(im, im_cyt))
        Npos.append(N_pos(im_nuc,im_cyt))
        Nelong.append(Elongation(im_nuc))
        Celong.append(Elongation(im_cyt))
        nmax, nmin, cmax, cmin = Maxima_Minima(im,im_nuc,im_cyt)
        Nmax.append(nmax)
        Nmin.append(nmin)
        Cmax.append(cmax)
        Cmin.append(cmin)

    df_train['Narea'] = N_area
    df_train['Carea'] = C_area
    df_train['NCarea'] = [a/b for (a,b) in zip(N_area,C_area)]
    df_train['Nlong'] = Nlong
    df_train['Clong'] = Clong
    df_train['Nshort'] = Nshort
    df_train['Cshort'] = Cshort
    df_train['Nround'] = Nround
    df_train['Cround'] = Cround
    df_train['Nperim'] = Nperim
    df_train['Cperim'] = Cperim
    df_train['Npos'] = Npos
    df_train['Nelong'] = Nelong
    df_train['Celong'] = Celong
    df_train['Nmax'] = Nmax
    df_train['Nmin'] = Nmin
    df_train['Cmax'] = Cmax
    df_train['Cmin'] = Cmin
    
    return df_train

#%%
def linear_classifier(X_train, y_train,X_test,y_test,multiclass=False):
  # Define linear classifier
  linear_clf = LinearRegression()

  if multiclass == False : 
    # Fit linear classifier
    linear_clf.fit(X_train, y_train)
    # Define class names
    class_names = ['Normal','Abnormal']
    y_pred_train = linear_clf.predict(X_train)
    y_pred_train = np.rint(y_pred_train)
    y_pred_train_ = y_pred_train.reshape(y_pred_train.shape[0])
    y_pred_test = linear_clf.predict(X_test)
    y_pred_test = np.rint(y_pred_test)
    y_pred_test_ = y_pred_test.reshape(y_pred_test.shape[0])
    # Evaluate the linear classifier
    print('Training score: {}.'.format(linear_clf.score(X_train,y_train)))
    print('Validation score: {}.\n'.format(linear_clf.score(X_test,y_test)))
    
  elif multiclass == True :
    # Categorize target variable 
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    # Fit linear classifier
    linear_clf.fit(X_train, y_train1)
    # Define class names
    class_names = ['0','1','2','3','4','5','6','7','8']
    # Make predictions 
    y_pred_train = linear_clf.predict(X_train)
    y_pred_train_ = np.argmax(y_pred_train,axis=1)
    y_pred_test = linear_clf.predict(X_test)
    y_pred_test_ = np.argmax(y_pred_test,axis=1)
    # Evaluate the linear classifier
    print('Training score: {}.'.format(linear_clf.score(X_train,y_train1)))
    print('Validation score: {}.\n'.format(linear_clf.score(X_test,y_test1)))

  # Compute confusion matrix
  plot_confusion_matrix(y_test, y_pred_test_, classes =class_names ,normalize=True, cmap=plt.cm.Blues);

#%%
def misclassified_samples_(model,train_data,train_labels,class_names,ID,path_images) : 
  y_pred = model.predict(train_data)
  miss_ids = []
  miss_indices = []
  for i in range(len(y_pred)):
    if y_pred[i]!=train_labels[i]:
      miss_indices.append(i)
      miss_ids.append(ID[i])
    if len(miss_ids)==3:
      break

  print('\nID {}: {} cell predicted as {}'.format(miss_ids[0], class_names[train_labels[miss_indices[0]]], class_names[y_pred[miss_indices[0]]]))
  plot_examples(path_images, miss_ids[0])
  print('\nID {}: {} cell predicted as {}'.format(miss_ids[1], class_names[train_labels[miss_indices[1]]], class_names[y_pred[miss_indices[1]]]))
  plot_examples(path_images, miss_ids[1])
  print('\nID {}: {} cell predicted as {}'.format(miss_ids[2], class_names[train_labels[miss_indices[2]]], class_names[y_pred[miss_indices[2]]]))
  plot_examples(path_images, miss_ids[2])

#%%
def NareaCarea_distribution(df_train,train_labels):
  fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
  sns.boxplot(ax=axes[0], x='ABNORMAL',y='Narea',data=df_train.merge(pd.DataFrame(train_labels ,index=df_train.index , columns=['ABNORMAL']),on=df_train.index), showfliers=False)
  axes[0].set_title('Range values of Narea grouped by classes')
  sns.boxplot(ax=axes[1], x='ABNORMAL',y='Carea',data=df_train.merge(pd.DataFrame(train_labels ,index=df_train.index , columns=['ABNORMAL']),on=df_train.index), showfliers=False)
  axes[1].set_title('Range values of Carea grouped by classes')
  sns.boxplot(ax=axes[2], x='ABNORMAL',y='NCarea',data=df_train.merge(pd.DataFrame(train_labels ,index=df_train.index , columns=['ABNORMAL']),on=df_train.index), showfliers=False)
  axes[2].set_title('Range values of NCarea grouped by classes');

#%%
def load_data(all_images, shape,col='ABNORMAL', labels=None, test = False):
  y = []
  ids =[]
  images = []

  for c in tqdm(all_images) :

    # Retrieve images
    if c.split('/')[-1].split('.bmp')[0].isnumeric() :
      images.append(resize(imread(c), shape).astype('float32'))

    if test == False : 
      # Match labels to training images
      id = c.split('/')[-1].split('.bmp')[0]
      if id.isnumeric() :
        y.append(labels[labels['ID']==int(id)][col].iloc[0])
    
    # Retrieve ids
    id = c.split('/')[-1].split('.bmp')[0]
    if id.isnumeric() :
      ids.append(int(id))
  
  y = np.array(y)
  ids = np.array(ids)
  images = np.array(images)

  return images, y, ids

#%%
def plot_accuracy_loss_curves(history) :
  plt.figure(figsize=(10,5))
  # Plot training & validation accuracy values
  plt.subplot(1,2,1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')

  # Plot training & validation loss values
  plt.subplot(1,2,2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()


#%%
def SVM_classifier(X_train, y_train,X_test,y_test,multiclass=False):

  if multiclass == False :
    # Replace target variable values : {0,1} -> {-1,1} because SVM uses the sign function.
    y_train_svm=[y if y==1 else -1 for y in y_train]
    y_test_svm=[y if y==1 else -1 for y in y_test]
    # Fitting linear SVM
    Lsvm = SVC(kernel='rbf', C=1, random_state=0)
    # Apply cross validation
    Lsvm_cv = cross_validate(Lsvm, X_train, y_train, cv=5,
                            return_train_score=True, return_estimator=True)
    # Define class names
    class_names = ['Normal','Abnormal']

  elif multiclass == True:
    # Fitting linear SVM
    Lsvm = SVC(kernel='linear', C=1, decision_function_shape='ovo', random_state=0)
    # Apply cross validation
    Lsvm_cv = cross_validate(Lsvm, X_train, y_train, cv=5,
                            return_train_score=True, return_estimator=True)
    class_names = ['0','1','2','3','4','5','6','7','8']

  # Display scores
  print(" Average and std TRAIN CV accuracy : {0} +- {1}".format(
      Lsvm_cv['train_score'].mean(), Lsvm_cv['train_score'].std()))
  print(" Average and std TEST CV accuracy : {0} +- {1}".format(
      Lsvm_cv['test_score'].mean(), Lsvm_cv['test_score'].std()))

  # Look for the best estimator
  index_best = np.argmax(Lsvm_cv['test_score'])
  estimator_best = Lsvm_cv['estimator'][index_best]

  # Predict 
  y_pred = estimator_best.predict(X_test)
  # Compute confusion matrix
  
  cnf_matrix = confusion_matrix(y_test, y_pred)
  # Plot confusion matrix
  plot_confusion_matrix(y_test, y_pred, classes = class_names ,normalize=True, cmap=plt.cm.Blues);
  return estimator_best
#%%
def Decision_Tree_Classifier(X_train, y_train,X_test,y_test,multiclass=False):

  if multiclass == False :
    class_names = ['Normal','Abnormal']
  elif multiclass == True :
    class_names = ['0','1','2','3','4','5','6','7','8']

  # Define classifier 
  Tree = DecisionTreeClassifier(random_state=0)

  # Define parameters for cross validation
  p_grid_tree = {'min_samples_split': [2,3,4,5,6,7,8,9,10,11], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11], 
                'max_leaf_nodes': [10,11,12,13,14,15,16]} 

  # Compute cross validation
  grid_tree = GridSearchCV(estimator=Tree, param_grid=p_grid_tree, scoring='accuracy', cv=5)

  # Train the classifier
  grid_tree.fit(X_train, y_train)

  # Retrieve best parameters
  best_params = grid_tree.best_params_
  print('Best parameters : ',best_params)

  # Compute best model
  best_model_tree = DecisionTreeClassifier(min_samples_split = best_params['min_samples_split'], 
  min_samples_leaf = best_params['min_samples_leaf'], max_leaf_nodes = best_params['max_leaf_nodes'])

  # Train best model
  best_model_tree.fit(X_train,y_train)

  # predict
  y_pred_train = best_model_tree.predict(X_train)
  y_pred_test = best_model_tree.predict(X_test)

  # Print train and test scores
  print("Decision Tree train accuracy :", best_model_tree.score(X_train,y_train))
  print("Decision Tree test accuracy :", best_model_tree.score(X_test,y_test))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, y_pred_test)

  # Plot confusion matrix
  print(), plot_confusion_matrix(y_test, y_pred_test, classes = class_names ,normalize=True, cmap=plt.cm.Blues);


#%%
def Random_Forest_Classifier(X_train, y_train,X_test,y_test,multiclass=False):
  
  if multiclass == False :
    class_names = ['Normal','Abnormal']
  elif multiclass == True :
    class_names = ['0','1','2','3','4','5','6','7','8']

  # Define classifier 
  RF = RandomForestClassifier(random_state=0)

  # Define parameters for cross validation
  p_grid_RF = {'n_estimators': [10, 15, 20, 25, 30], 'min_samples_leaf': [
      2, 3, 4, 5, 6], 'max_features': ['sqrt', 'log2'],'max_leaf_nodes': [10,11,12,13,14,15,16,17,18,19]}

  # Compute cross validation
  grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF,
                        scoring='accuracy', cv=5)

  # Train the classifier
  grid_RF.fit(X_train, y_train)

  # Retrieve best parameters
  best_params = grid_RF.best_params_
  print("Best params: {}".format(best_params))

  # Compute best model
  best_model_RF = RandomForestClassifier(n_estimators=best_params['n_estimators'], min_samples_leaf=best_params['min_samples_leaf'],
                                        max_features=best_params['max_features'], max_leaf_nodes = best_params['max_leaf_nodes'], random_state=0)

  # Fit best model
  best_model_RF.fit(X_train,y_train)

  # predict
  y_pred_train = best_model_RF.predict(X_train)
  y_pred_test = best_model_RF.predict(X_test)

  # Print train and test scores
  print("Decision Tree train accuracy :", best_model_RF.score(X_train,y_train))
  print("Decision Tree test accuracy :", best_model_RF.score(X_test,y_test))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, y_pred_test)

  # Plot confusion matrix
  plot_confusion_matrix(y_test, y_pred_test, classes =class_names ,normalize=True, cmap=plt.cm.Blues);

  return best_model_RF

#%%
def plot_features_importance(best_model_RF, df_train_pca):
  # Compute features importances
  importances = best_model_RF.feature_importances_

  # Retrieve features indices
  indices = np.argsort(importances)[::-1]
  n_features = len(df_train_pca.columns)

  # Plot the feature importances of the forest
  plt.figure(figsize=(15,5))
  plt.title("Feature importance")
  plt.bar(range(n_features), importances[indices[0:n_features]], color="r", align="center")
  plt.xticks(range(n_features), df_train_pca.columns[indices[0:n_features]])
  plt.xlim([-1, n_features])
  plt.show()

#%%
def Gradient_Boosting_Classifier(X_train, y_train,X_test,y_test):

  # Define parameters for cross validation
  param_grid = {'n_estimators': [200, 300, 400],
                'learning_rate': [0.1,0.15, 0.05],
                'max_depth': [4, 5, 6],
                'min_samples_leaf': [2,3,4],
                'max_features': [1,2,3]}

  # Define classifier
  estimator = GradientBoostingClassifier(random_state=0)

  # Apply cross validation
  classifier = GridSearchCV(estimator=estimator, cv=5, param_grid=param_grid)
  classifier.fit(X_train, y_train)

  # Retrieve best parameters
  best_param = classifier.best_estimator_

  # Define best model
  best_model = GradientBoostingClassifier(learning_rate=best_param.learning_rate, max_depth=best_param.max_depth,
                                          max_features=best_param.max_features,
                                          min_samples_leaf=best_param.min_samples_leaf,
                                          n_estimators=best_param.n_estimators,
                                          random_state=0)

  # Fit best model
  best_model.fit(X_train, y_train)

  # Make predictions
  y_pred_train = best_model.predict(X_train)
  y_pred_test = best_model.predict(X_test)

  # Evaluate model
  print("Gradient Boosting train accuracy:", best_model.score(X_train, y_train))
  print("Gradient Boosting test accuracy :", best_model.score(X_test,y_test))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, y_pred_test)

  # Plot confusion matrix
  plot_confusion_matrix(y_test, y_pred_test, classes =['Normal','Abnormal'] ,normalize=True, cmap=plt.cm.Blues);

#%%
def create_model_binary(shape,dropout_rate=0.0, lrate=0.00006, finetuned_layers = 10):

  prior = tensorflow.keras.applications.DenseNet169(include_top=False, 
                                                    weights='imagenet',
                                                    input_shape=shape)
  model = tensorflow.keras.Sequential()
  model.add(prior)

  model.add(Flatten()) 
  model.add(Dense(192, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(128, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(64, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(32, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(16, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(1, activation='sigmoid'))

  # Fine-tune selected layers
  c = 0
  for layer in model.layers[0].layers:  
    if (c >= finetuned_layers):
      layer.trainable = False
      c += 1
    else:
      layer.trainable = True

  # Compile model
  model.compile(optimizer=Adam(lr=lrate),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
  
  return model

#%%
def create_model_multiclass(shape,dropout_rate=0.0, lrate=0.00006, finetuned_layers = 10):

  prior = tensorflow.keras.applications.DenseNet169(include_top=False, 
                                                    weights='imagenet',
                                                    input_shape=shape)
  model = tensorflow.keras.Sequential()
  model.add(prior)

  model.add(Flatten()) 
  model.add(Dense(192, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(128, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(64, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(32, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(16, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(dropout_rate))
  model.add(Dense(9, activation='softmax'))
  
  # Fine-tune selected layers
  c = 0
  for layer in model.layers[0].layers:  
    if (c >= finetuned_layers):
      layer.trainable = False
      c += 1
    else:
      layer.trainable = True

  # Compile model
  model.compile(optimizer=Adam(lr=lrate),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
  
  return model

#%%
def plot_samples_cell_categories(Working_dir,labels):
  plt.figure(figsize=(30,5))
  unique_groups = []
  for i in range(labels.shape[0]):
    if not(labels['GROUP'].iloc[i] in unique_groups):
      unique_groups.append(labels['GROUP'].iloc[i])
      plt.subplot(1,9,len(unique_groups))
      plt.imshow(imread(Working_dir + 'Train'+'/'+str(labels['ID'].iloc[i])+'.bmp'))
      plt.title('Class '+str(len(unique_groups)-1))
      if len(unique_groups)==9:
        break
  plt.show()

#%%
def Gradient_Boosting_Classifier_multiclass(X_train, y_train,X_test,y_test):

  param_grid = {'n_estimators': [300, 500],
                'learning_rate': [0.1, 0.05],
                'max_depth': [4, 5, 6],
                'min_samples_leaf': [3, 5],
                'max_features': [1,2,3]}

  estimator = GradientBoostingClassifier(random_state=0)

  classifier = GridSearchCV(estimator=estimator, cv=5, param_grid=param_grid)

  classifier.fit(X_train, y_train)

  best_param = classifier.best_estimator_

  best_model = GradientBoostingClassifier(learning_rate=best_param.learning_rate, max_depth=best_param.max_depth,
                                          max_features=best_param.max_features,
                                          min_samples_leaf=best_param.min_samples_leaf,
                                          n_estimators=best_param.n_estimators,
                                          random_state=0)

  best_model.fit(X_train, y_train)

  y_pred_train = best_model.predict(X_train)
  y_pred_test = best_model.predict(X_test)

  print("Gradient Boosting train accuracy:", best_model.score(X_train, y_train))
  print("Gradient Boosting test accuracy :", best_model.score(X_test,y_test))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(y_test, y_pred_test)

  # Plot confusion matrix
  plot_confusion_matrix(y_test, y_pred_test, classes =['0','1','2','3','4','5','6','7','8'] ,normalize=True, cmap=plt.cm.Blues);

#%%
