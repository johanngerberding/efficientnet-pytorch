import numpy as np 
import pickle 
from os import listdir 
from os.path import isfile, join 
import argparse 
import os 


def unpickle_cifar10_data(directory):
    # Initialize the variables
    train = dict()
    test = dict()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    # Iterate through all files that we want, train and test
    # Train is separated into batches
    for filename in listdir(directory):
        if isfile(join(directory, filename)):
            
            # train data
            if 'data_batch' in filename:
                print('Handing file: {}'.format(filename))
                
                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')

                if 'data' not in train:
                    train['data'] = data[b'data']
                    train['labels'] = np.array(data[b'labels'])
                else:
                    train['data'] = np.concatenate((train['data'], data[b'data']))
                    train['labels'] = np.concatenate((train['labels'], data[b'labels']))
            # test data
            elif 'test_batch' in filename:
                print('Handing file: {}'.format(filename))
                
                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')
                
                test['data'] = data[b'data']
                test['labels'] = data[b'labels']
    
    # Manipulate the data to the propper format
    for image in train['data']:
        train_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
    train_y = [label for label in train['labels']]
    
    for image in test['data']:
        test_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
    test_y = [label for label in test['labels']]
    
    # Transform the data to np array format
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    return (train_x, train_y), (test_x, test_y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', 
                        help="path to cifar10 root directory", 
                        type=str, 
                        required=True)
    parser.add_argument('--out', 
                        help="path to cifar10 output dir", 
                        type=str, 
                        required=True)
    args = parser.parse_args()

    (train_x, train_y), (test_x, test_y) = unpickle_cifar10_data(args.path)
    
    os.makedirs(join(args.out, 'train'))
    os.makedirs(join(args.out, 'test'))
    
    with open(join(args.out, 'train') + '/train_imgs.pkl', 'wb') as f:
        pickle.dump(train_x, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(args.out, 'test') + '/test_imgs.pkl', 'wb') as f:
        pickle.dump(test_x, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(args.out, 'train') + '/train_labels.pkl', 'wb') as f:
        pickle.dump(train_y, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(join(args.out, 'test') + '/test_labels.pkl', 'wb') as f:
        pickle.dump(test_y, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()