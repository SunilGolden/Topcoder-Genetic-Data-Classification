import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
import argparse


def train(train_data_path='./train.csv'):
    '''
    This function will fit a Histogram-based Gradient Boosting Classification 
    Tree on the provided data and save that model in the disk.

    Args:
        train_data_path (str): path of training data
    '''
    # Read the training data
    train_data = pd.read_csv(train_data_path, sep=' ', header=None)
    
    # Get the features and labels
    train_X = train_data.iloc[:,:-1]
    train_y = train_data.iloc[:,-1]

    # Define a Classifier 
    # [Histogram-based Gradient Boosting Classification Tree]
    clf = HistGradientBoostingClassifier(random_state=9)
    
    # Fit the Classifier
    clf.fit(train_X, train_y)

    # Save the model to disk
    pickle.dump(clf, open('trained_model.sav', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
        'It will fit a Histogram-based Gradient Boosting Classification\
        Tree on the provided data and save that model in the disk.'\
        )
    parser.add_argument('--train_data_path', type=str, default='./train.csv')

    args = parser.parse_args()

    train(train_data_path=args.train_data_path)