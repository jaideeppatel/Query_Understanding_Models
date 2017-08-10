from bs4 import BeautifulSoup
from collections import defaultdict
import nltk
import itertools
import re
import string
import time
import sys
import os
import io
import codecs
import glob
import MITIE_NER_package

# -------- Read the aruguments/instructions file ----------
def get_arguments(f_name):
    with open('instructions_files/'+f_name, mode='r') as f:
        command_list = f.read().split(',')
    ins = [c.strip() for c in command_list]
    return ins

# ---------- NER processing ------------------------
if __name__ == '__main__':
    ins_file_name = 'mitie_train_instructions.txt'
    instructions = get_arguments(ins_file_name)

    # --------------- Data Preprocessing ------------------------
    # Here we convert the LN case law documents into sentence entity format ....
    # Instructions for preprocessing train data ...
    # Format: filename (any file in the lncase_data folder), directory name (any directory under lncase_data dir),
    # file format (xml, csv, txt, dat)...
    file_type = instructions[2].strip()
    files = []

    if len(instructions[0])>0:
        f_name = instructions[0].strip().split('.')[0]
        files.append('lncase_data/'+f_name + '.' + file_type)
    else:
        dir_name = instructions[1].strip()
        for file in glob.glob(dir_name+"/*."+file_type):
            print(file)
            files.append(file)

    print("Total number of files in Training data:", len(files))

    # Calling the preporcessing function from the MITIE package which will take all the input file names and
    # convert + merge each file into a sentence entity training data
    train_file_name = MITIE_NER_package.preprocess_train_data(files)

    # ---------------------- Train NER Model ------------------
    ner_model = MITIE_NER_package.MITIE_train(train_file_name)

    # OR when you want to use an external pre porcessed file use below code example...
    ner_model = MITIE_NER_package.MITIE_train('mitie_train_data/MITIE_sent_ent_data.txt')

    # The command below saves the trained ner model as a .dat file to the directed repository ...
    ner_model.save_to_disk("ner_models/trained_ner_model.dat")

    # --------------------- Testing NER Model ----------------------------
    # This is a two step process ....
    # 1. Preprocess the test files if the test data is one or more case law documents ...
    # 2. Read the pretrained NER model and perform predictions as per instructions in the mitie_test_instructions file

    # Instructions for preprocessing test data ...
    # Format: filename (any file in the ner_test_files folder), directory name (any directory under ner_test_files dir),
    # file format (xml, csv, txt, dat)...
    file_type = instructions[2].strip()
    files = []
    if len(instructions[0].strip())>0:
        f_name = instructions[0].strip().split('.')[0]
        files.append('test_files/'+f_name + '.' + file_type)
    else:
        dir_name = instructions[1].strip()
        for file in glob.glob('test_files/'+dir_name+"/*."+file_type):
            files.append(file)

    print("Total number of files in Test data:", len(files),files)
    test_files = MITIE_NER_package.preprocess_test_data(files)

    # Once the preporcessed files are available ....
    # the below command will return the ner predictions for the one/more test files with the metrics ...
    test_entities = MITIE_NER_package.MITIE_test(instructions)

