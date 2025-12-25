import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

sys.path.append(r"C:\Users\MurrayLab\sensory-dependent-gait")

from processing import data_loader, utils_math
from processing.data_config import Config
from figures.fig_config import Config as FigConfig

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import shuffle

"""
INJECTION SIDE CLASSIFICATION BASED ON LIMB PHASE DATA
"""

def classify_injection_side(df, dataToClassify, target_col = 'injection', crossvalidation=False):
        # resample the majority class
        class_shapes = (df[df[target_col] == 0].shape[0],
                        df[df[target_col] == 1].shape[0])
        if class_shapes[0] != class_shapes[1]:
            majority_class = np.argmax(class_shapes)
            minority_class = np.argmin(class_shapes)
            df_majority = df[df[target_col] == majority_class]
            df_minority = df[df[target_col] == minority_class]
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=df_minority.shape[0],
                random_state=42   
                ) 
            
            # recombine the dataset and reshuffle
            df_balanced = pd.concat((df_minority, df_majority_downsampled))
            df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df_balanced = df
        
        # separate X and Y
        if type(dataToClassify) != list:
            raise TypeError("`dataToPlot` must be a list!")
        X = df_balanced[dataToClassify] if len(dataToClassify)>1 else df_balanced[dataToClassify].values.reshape(-1,1)
        y = df_balanced[target_col]
        
        # convert to sin/cos
        X = np.hstack((np.cos(X), np.sin(X)))
        
        # split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
                                    X,
                                    y,
                                    test_size=0.25,
                                    random_state=42
                                        )
        
        # scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # training a classifier
        classifier = SVC(probability=True, random_state=42)
        print(f"Classifying injection side based on {dataToClassify}...")
        print(f"Number of datapoints: {df_balanced.shape[0]}")
                           
        classifier.fit(X_train, y_train)        # takes about 10 minutes to run
    
        # predict
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        # predict for ROC-AUC
        y_probs = classifier.predict_proba(X_test)[:,1]
        fpr, tpr, threshold = roc_curve(y_test, y_probs)
        
        cv_scores=[]
        if crossvalidation:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(classifier, X, y, cv=kf, scoring='accuracy')
            print(f"cv score: {np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
            
            N_permutations = 100
            permuted_scores = []
            for i in range(N_permutations):
                if i%10==0:
                    print(f"Permutation {i}...")
                y_train_permuted = np.random.permutation(y_train)
                classifier.fit(X_train, y_train_permuted)
                y_pred_permuted = classifier.predict(X_test)
                permuted_scores.append(accuracy_score(y_test, y_pred_permuted))
            permuted_scores = np.array(permuted_scores)
            p_val = np.mean(permuted_scores >= accuracy)
            print(f"Permutation p-value: {p_val:.4f}")
            
                  
        return fpr, tpr, accuracy, cv_scores
    
def prepare_data(df, target_col, dataToClassify, mice, mouse_col = 'mouseID', 
                 left_inj_mice = Config.injection_config['left_inj_imp'],
                 right_inj_mice = Config.injection_config['right_inj_imp']):
    mice_left = np.intersect1d(left_inj_mice, mice)
    mice_right = np.intersect1d(right_inj_mice, mice)

    # add an injection side column
    left_mask = [m in mice_left for m in df[mouse_col]]
    right_mask = [m in mice_right for m in df[mouse_col]]
    df.loc[left_mask, target_col] = 0
    df.loc[right_mask,  target_col] = 1
    
    if type(dataToClassify) != list:
        raise TypeError("`dataToPlot` must be a list!")
        
    # convert data to radians
    df[dataToClassify] = df[dataToClassify]*2*np.pi
    df[dataToClassify] = df[dataToClassify].where(df[dataToClassify]>=0, df[dataToClassify]+2*np.pi)
    
    # subset data
    cols_to_select = [mouse_col] + dataToClassify + [target_col]
    df_sub = df[cols_to_select]
    df_sub = df_sub.dropna()
    
    # balance the data wrt mice
    mouse_counts = df_sub[mouse_col].value_counts()
    samples_per_mouse = mouse_counts.min()
    # keep all data from the most underrepresented mouse
    df_sub_resampled = df_sub[df_sub[mouse_col] == mouse_counts.idxmin()].reset_index(drop=True)
    for m in np.setdiff1d(df_sub[mouse_col].unique(), [mouse_counts.idxmin()]):
        df_mouse = df_sub[df_sub[mouse_col] == m].reset_index(drop=True)
        sample = df_mouse.sample(n=samples_per_mouse, random_state=42)
        df_sub_resampled = pd.concat([df_sub_resampled, sample], ignore_index=True)
    
    # shuffle rows so that data from different mice are mixed
    df_sub_resampled = df_sub_resampled.drop([mouse_col], axis=1)
    df_sub_final = df_sub_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
           
    return df_sub_final