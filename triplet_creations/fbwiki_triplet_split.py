# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:51:42 2024

@author: Eduin Hernandez

Summary: Splits the triplet into a Train, Test, and Validation set.
"""
from utils.process_triplets import split_triplets

if __name__ == '__main__':
    'Input File'
    triplet_file_path = './data/modified_triplet.txt'
    
    'Output File'
    train_file_path = './data/train.txt'
    test_file_path = './data/test.txt'
    valid_file_path = './data/valid.txt'

    split_triplets(triplet_file_path, train_file_path, test_file_path, valid_file_path)
    
    print("Triplets have been split into train, test, and valid files.")
