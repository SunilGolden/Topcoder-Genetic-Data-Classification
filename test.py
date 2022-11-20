import numpy as np
import pandas as pd
import pickle
import argparse


def test(test_data_path='./test_x.csv', 
         trained_model_path='./trained_model.sav'):
    '''
    This function predicts the class probability based on Histogram-based 
    Gradient Boosting Classification Tree  for the features in the test data.
    It then saves the predicted probalility by concatinating with the test 
    data as a csv file.

    Args:
        test_data_path (str): path of test data
        trained_model_path (str): path of trained model  
    '''
    # Load the saved model
    trained_model = pickle.load(open(trained_model_path, 'rb'))
    
    # Read the test data
    test_data = pd.read_csv(test_data_path, header=None) 
    
    # Predict the class probabilities
    class_probability = trained_model.predict_proba(test_data)

    # Convert the array to DataFrame
    class_probability = pd.DataFrame(np.array(class_probability))
    
    # Concat the class probabilities with the test data
    solution = pd.concat([test_data, class_probability], axis="columns")
    
    # Save the solution as csv file
    solution.to_csv('solution.csv', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
    	'It predicts the class probability based on Histogram-based \
    	Gradient Boosting Classification Tree  for the features in the test data.\
    	It then saves the predicted probalility by concatinating with the test \
    	data as a csv file.'\
    )
    parser.add_argument('--test_data_path', type=str, default='./test_x.csv')
    parser.add_argument('--trained_model_path', type=str, default='./trained_model.sav')

    args = parser.parse_args()

    test(test_data_path=args.test_data_path, trained_model_path=args.trained_model_path)