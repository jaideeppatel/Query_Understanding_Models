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
import pandas as pd
import csv
import glob
from mitie import *

# This will return the keys for performing grouped sorting for the entity list ...
def get_key(item):
    return len(item[0].split())

# To clean the query text, primarily ro ignore the LN boolean and string connectors ...
def clean_query(q):
    word_list = [p for p in re.split("( |\(|\))", q) if p.strip()]
    wl = [re.sub("[()\";,:&]+",'',p.strip()) for p in word_list if len(p)>1]
    return wl

# Preprocess all the LN case law documents to generate the training data for the MITIE model ...
def preprocess_train_data(files):
    # Standard file name convention to be used to save the training data ...
    sent_ent_data_file_name = 'mitie_train_data/MITIE_sent_ent_data.txt'
    ent_data_file_name = 'mitie_train_data/MITIE_entities_identified.txt'
    ent_file = codecs.open(ent_data_file_name, 'w+', encoding='utf-8')
    train_data = codecs.open(sent_ent_data_file_name, 'w+', encoding='utf-8')

    # A dictionary based entity map highly important when needed to group different entity types in one ...
    entity_map = {'attorney': 'attorney', 'judge': 'judge', 'judicialBranch': 'judicialBranch',
                  'legislativeBranch': 'legislativeBranch', 'geography': 'geography',
                  'citation': 'citation', 'lawFirm': 'lawFirm',
                  'organization': 'organization'}
    entity_dict = defaultdict(int)

    overall_time = time.time()
    train_data.write('-----\n')
    FL = 0
    total_entities = 0
    for f in files:
        FL += 1
        with codecs.open(f, 'r', encoding='utf-8') as case:
            data_file = case.readlines()
            CS = 0
            for line in data_file:
                CS += 1
                case_time = time.time()
                xml_file = line
                soup = BeautifulSoup(xml_file, 'xml')
                doc = soup.find('doc')
                p_tags = doc.findAll('p')

                for tag in p_tags:
                    para = tag.text
                    if tag.find('entity') or (len(tag.findAll('cite')) > 0):
                        ent_list = []
                        if len(tag.findAll('cite')) > 0:
                            cites = list(tag.findAll('cite'))
                            cite_list = [c.text for c in cites]

                            for c in range(len(cite_list)):
                                para = para.replace(cite_list[c], 'citexxx' + str(c))

                            for cts in cite_list:
                                ent_list.append([cts, entity_map['citation']])

                        if tag.find('entity'):
                            entity_tag = tag.findAll('entity')
                            for ent in entity_tag:
                                entity_tag = tag.findAll('entity')
                                entity_name, entity_type = ent.text, ent.attrs['role'].split(':')[-1]
                                ent_list.append([entity_name, entity_map[entity_type]])

                        # Group entities based on entity text and entity type ...
                        ent_list_grouped = list((ent_list for ent_list, _ in itertools.groupby(ent_list)))

                        # Sort entities based on the len of entity text ...
                        # This is to avoid multiple entity predictions on the same word ...
                        ent_list_sorted = sorted(ent_list_grouped, key=get_key, reverse=True)
                        for e in ent_list_sorted:
                            ent_file.write(e[0] + ":" + e[1] + "\n")
                            total_entities += 1

                        sentences = nltk.sent_tokenize(para)

                        for s in sentences:
                            for c in range(len(cite_list)):
                                s = s.replace('citexxx' + str(c), cite_list[c])
                            sent = nltk.word_tokenize(s)
                            s_bool = [0] * len(sent)

                            # Use the sent_flag to avoid printing the same sentence multiple times ...
                            sent_flag = False
                            flag = False
                            for e in ent_list_sorted:
                                ent_name = e[0].split()
                                ent_type = e[1]
                                i = 0
                                while (i < len(sent)):
                                    j = i
                                    res_len = 0
                                    for wd in ent_name:
                                        if j < len(sent):
                                            if wd.lower() == sent[j].lower():
                                                j += 1
                                                res_len += 1
                                    if res_len == len(ent_name):
                                        flag = True
                                        if not any(s_bool[v] == 1 for v in range(j - res_len, j)):
                                            for v in range(j - res_len, j):
                                                s_bool[v] = 1
                                            if sent_flag == False:
                                                train_data.write(' '.join(sent) + "\n")
                                                sent_flag = True
                                            entity_dict[ent_type] += 1
                                            train_data.write(
                                                str(j - res_len) + '-' + str(j) + "-" + e[0] + "-" + e[1] + "\n")
                                    i += 1
                            if flag != False:
                                train_data.write('-----\n')
                print('Case number:', CS)
                print("Case processing time", "--- %s seconds ---" % (time.time() - case_time))
        print('File number:', FL)

    ent_file.close()
    train_data.close()

    print("Entities in train data:",entity_dict.keys())
    print('Entities counts are:', entity_dict)
    print("Total entities:", total_entities)
    print("Overall preprocessing time", "--- %s seconds ---" % (time.time() - overall_time))

    return sent_ent_data_file_name

# Call the MITIE trainer, add training data to the NER trainer instance and train the ner model ...
def MITIE_train(train_data_file):
    fname = train_data_file

    # Call the MITIE word feature extractor ....
    trainer = ner_trainer("MITIE-master/MITIE-models/english/total_word_feature_extractor.dat")
    total_sentences = 0
    total_entities = 0
    data_list = []
    ln = 0
    with codecs.open(fname,encoding='utf-8') as fp:
        for line in fp:
            # print("Line:",line)
            ln += 1
            if line.strip() == '-----':
                if len(data_list) > 1:
                    # Add a sentence as a new instance to the ner trainer ...
                    sample = ner_training_instance(data_list[0])
                    total_sentences += 1
                    for ent in data_list[1:]:
                        item = ent.split('-')
                        a = int(item[0])
                        b = int(item[1])
                        type = str(item[-1]).strip()
                        sample.add_entity(range(a,b), type)
                        # print(range(a,b), type)
                        total_entities += 1
                    trainer.add(sample)
                    data_list = []
                else:
                    data_list = []
            if line.strip() != '-----':
                data_list.append(line)

    print("Total number of sentences in training data:",total_sentences)
    print("Total number of entities in training data:",total_entities)
    print('Training began ---------------------')
    train_start = time.time()
    trainer.num_threads = 4
    ner = trainer.train()
    # ner.save_to_disk("ner_models/trained_ner_model.dat")
    print("Training time taken for 4", "--- %s seconds ---" % (time.time() - train_start))

    return ner


# To preprocess one or more test files and identify total entities in each of the test files ....
# Here we convert the test case law documents to individual test files
# Note: Here we save each case file as a separate test file as we are keen to understand the performance
# of the entity model on different case law file ...
def preprocess_test_data(files):
    entity_map = {'attorney': 'attorney', 'judge': 'judge', 'judicialBranch': 'judicialBranch',
                  'legislativeBranch': 'legislativeBranch','geography': 'geography',
                  'citation': 'citation', 'lawFirm': 'lawFirm',
                  'organization': 'organization'}
    overall_time = time.time()
    FL = 0
    for f in files:
        f_name = f.split('/')[-1].split('.')[0]
        sent_ent_data_file_name = 'test_files/MITIE_sent_ent_data_'+f_name+'.txt'
        train_data = codecs.open(sent_ent_data_file_name, 'w+', encoding='utf-8')
        train_data.write('-----\n')
        entity_dict = defaultdict(int)
        total_entities = 0
        total_sent = 0

        FL += 1
        with codecs.open(f, 'r', encoding='utf-8') as case:
            data_file = case.readlines()
            CS = 0
            for line in data_file:
                CS += 1
                case_time = time.time()
                xml_file = line
                soup = BeautifulSoup(xml_file, 'xml')
                doc = soup.find('doc')
                p_tags = doc.findAll('p')

                for tag in p_tags:
                    para = tag.text
                    if tag.find('entity') or (len(tag.findAll('cite')) > 0):
                        ent_list = []
                        if len(tag.findAll('cite')) > 0:
                            cites = list(tag.findAll('cite'))
                            cite_list = [c.text for c in cites]

                            for c in range(len(cite_list)):
                                para = para.replace(cite_list[c], 'citexxx' + str(c))

                            for cts in cite_list:
                                ent_list.append([cts, entity_map['citation']])

                        if tag.find('entity'):
                            entity_tag = tag.findAll('entity')
                            for ent in entity_tag:
                                entity_tag = tag.findAll('entity')
                                entity_name, entity_type = ent.text, ent.attrs['role'].split(':')[-1]
                                ent_list.append([entity_name, entity_map[entity_type]])
                        # group entities
                        ent_list_grouped = list((ent_list for ent_list, _ in itertools.groupby(ent_list)))
                        # sort entities
                        ent_list_sorted = sorted(ent_list_grouped, key=get_key, reverse=True)

                        sentences = nltk.sent_tokenize(para)
                        for s in sentences:
                            for c in range(len(cite_list)):
                                s = s.replace('citexxx' + str(c), cite_list[c])
                            sent = nltk.word_tokenize(s)
                            s_bool = [0] * len(sent)

                            sent_flag = False
                            flag = False
                            for e in ent_list_sorted:
                                ent_name = e[0].split()
                                ent_type = e[1]
                                i = 0
                                while (i < len(sent)):
                                    j = i
                                    res_len = 0
                                    for wd in ent_name:
                                        if j < len(sent):
                                            if wd.lower() == sent[j].lower():
                                                j += 1
                                                res_len += 1
                                    if res_len == len(ent_name):
                                        flag = True
                                        if not any(s_bool[v] == 1 for v in range(j - res_len, j)):
                                            for v in range(j - res_len, j):
                                                s_bool[v] = 1
                                            if sent_flag == False:
                                                train_data.write(s + "\n")
                                                total_sent += 1
                                                sent_flag = True
                                            entity_dict[ent_type] += 1
                                            train_data.write(
                                                str(j - res_len) + '-' + str(j) + "-" + e[0] + "-" + e[1] + "\n")
                                    i += 1
                            if flag != False:
                                train_data.write('-----\n')
        print('File number:', FL)
        train_data.close()
        print('Total Sentences with Entities:',total_sent)
        print("Total entities:", total_entities)
        print('Total entities in sentences', entity_dict)

    print("Overall preprocessing time", "--- %s seconds ---" % (time.time() - overall_time))

    return True

# Returns the NER predictions for the every test record ....
def get_entities(test_records,exp_val,format,model):
    print('MITIE Extracting Entities ...')
    print("loading NER model...")
    ner = named_entity_extractor(model)
    print("\nTags output by this NER model:", ner.get_possible_ner_tags())

    # This is to compare the prediction results to expected results, only if the test file contains NER labels ...
    if exp_val.upper() == "T":
        test_file = test_records
        test_file['entity_output'] = ''
        query_list = list(test_file['Query'])

        if format.upper() == 'QUERY':
            query_list = [clean_query(q) for q in query_list]
        else:
            query_list = [nltk.word_tokenize(q) for q in query_list]
        results_file = pd.DataFrame()
        cols = list(test_file.columns)
        for c in cols:
            results_file[c] = ''
        r=0
        for q in query_list:
            try:
                tokens = q
                # print("Tokenized Input",tokens)
                entities = ner.extract_entities(tokens)
                # print("\nEntities found:", entities)
                # print("\nNumber of entities detected:", len(entities))

                ent_list = []
                for e in entities:
                    range = e[0]
                    tag = e[1]
                    score = e[2]
                    score_text = "{:0.3f}".format(score)
                    entity_text = " ".join(tokens[i] for i in range)
                    ent_list.append(entity_text + " : " + tag + " : " + score_text)
                    # print("   Score: " + score_text + ": " + tag + ": " + entity_text)
                cur_row = test_file.iloc[r].set_value('entity_output', ent_list)
                results_file = results_file.append(cur_row, ignore_index=True)
                r+=1
            except (UnicodeDecodeError, UnicodeEncodeError, UnicodeError) as e:
                ent_list = ['Unicode error']
                cur_row = test_file.iloc[r].set_value('entity_output', ent_list)
                results_file = results_file.append(cur_row, ignore_index=True)
                r += 1

    # Here we only perform the predictions without calculating any performance matrics ....
    else:
        results_file = pd.DataFrame()
        query_list = test_records

        r = 0
        for q in query_list:
            try:
                tokens = q
                # print("Tokenized input:", tokens)
                entities = ner.extract_entities(tokens)
                # print("\nEntities found:", entities)
                # print("\nNumber of entities detected:", len(entities))

                ent_list = []
                for e in entities:
                    range = e[0]
                    tag = e[1]
                    score = e[2]
                    score_text = "{:0.3f}".format(score)
                    entity_text = " ".join(tokens[i] for i in range)
                    ent_list.append(entity_text + " : " + tag + " : " + score_text)
                    # print("   Score: " + score_text + ": " + tag + ": " + entity_text)

                cur_row = pd.Series([q,ent_list])
                results_file = results_file.append(cur_row, ignore_index=True)

            except (UnicodeDecodeError, UnicodeEncodeError, UnicodeError) as e:
                ent_list = ['Unicode error']
                cur_row = pd.Series([q, ent_list])
                results_file = results_file.append(cur_row, ignore_index=True)
        results_file.columns = ['Query','entity_output']
    return results_file

# Specifically designed to process the test files in sentence entity format ...
def get_entities_mitie(test_records, model):
    ner = model
    ner_results = pd.DataFrame(columns=['sent', 'exp_output', 'pred_output'])
    ner_results['exp_cnt'] = ''
    ner_results['out_cnt'] = ''
    ner_results['result'] = ''

    print('Loading test file')
    item_list = []
    ent_list = []
    entity_dict = defaultdict(list)
    entity_dict_out = defaultdict(list)
    all_entity_dict = defaultdict(int)
    all_entity_dict_out = defaultdict(int)
    sent = ''
    xo = 0
    with codecs.open(test_records, mode='r',encoding='utf-8') as f:
        for line in f:
            try:
                xo += 1
                if line != '-----\n':
                    item_list.append(line.strip())
                else:
                    if len(item_list) > 0:
                        sent = item_list[0]
                        ent_list = [[i.split('-')[0], i.split('-')[1], i.split('-')[2], i.split('-')[-1]] for i in item_list[1:]]
                        ent_count = len(item_list) - 1
                        for e in ent_list:
                            entity_dict[e[-1]].append([int(e[0]), int(e[1])])
                            all_entity_dict[e[-1]] += 1

                        tokens = nltk.word_tokenize(sent)
                        # print("Tokenized input:", tokens)

                        entities = ner.extract_entities(tokens)
                        # print("\nEntities found:", entities)
                        # print("\nNumber of entities detected:", len(entities))

                        ent_out_list = []
                        ent_count_out = 0
                        for e in entities:
                            range = e[0]
                            print('range is:',range)
                            tag = e[1]
                            score = e[2]
                            score_text = "{:0.3f}".format(score)
                            entity_text = " ".join(tokens[i] for i in range)
                            # print("   Score: " + score_text + ": " + tag + ": " + entity_text)
                            entity_dict_out[tag].append([range.start,range.stop])
                            all_entity_dict_out[tag] += 1
                            ent_count_out +=1
                            ent_out_list.append([range, entity_text, tag])

                        print('exp',entity_dict)
                        print('pred',entity_dict_out)
                        correct = 0
                        for key in entity_dict.keys():
                            if key in entity_dict_out.keys():
                                first_set = set(map(tuple, entity_dict[key]))
                                secnd_set = set(map(tuple, entity_dict_out[key]))
                                diff = first_set - secnd_set
                                if diff.__len__() == 0:
                                    correct += first_set.__len__()
                                elif diff.__len__() >0 :
                                    correct += first_set.__len__() - diff.__len__()


                        ner_results = ner_results.append(
                            pd.Series([sent, ent_list, ent_out_list,ent_count,ent_count_out,correct],
                                index=['sent','exp_output','pred_output','exp_cnt','out_cnt','result']), ignore_index=True)

                    entity_dict = defaultdict(list)
                    entity_dict_out = defaultdict(list)
                    item_list = []
                    ent_list = []
                    sent = ''
            except(UnicodeDecodeError, UnicodeEncodeError, UnicodeError) as e:
                ner_results = ner_results.append(
                    pd.Series(['Unicode error', 'Unicode error', 'Unicode error','Unicode error','Unicode error','Unicode error'],
                              index=['sent','exp_output','pred_output','exp_cnt','out_cnt','result']), ignore_index=True)
                item_list = []
                ent_list = []
                entity_dict = defaultdict(int)
                entity_dict_out = defaultdict(int)
                sent = ''
    print('Total number of Actual Entities:',all_entity_dict)
    print('Total number of Entities identified:', all_entity_dict_out)
    print('Actual: ',sum(all_entity_dict.values()),'Predicted: ',sum(all_entity_dict_out.values()))
    return ner_results


# Performs all the processing to generate predictions for the test files ...
# Returns a csv file with NER predictions and performance metrics ...
def MITIE_test(instructions):
    test_format = instructions[0]
    file_type = instructions[1]
    file_name = instructions[2]
    dir_name = instructions[3]
    exp_value = instructions[4].strip()
    model_name = instructions[5]
    print('MITIE Testing Started ...')
    print(test_format,file_type,file_name,dir_name,exp_value,model_name)

    if dir_name.strip() == '':

        if test_format.upper() == 'DOCUMENT':

            with codecs.open('test_files/' + file_name, encoding='utf-8') as tst:
                data_file = tst.read()
            file_sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(data_file)]
            test_result_file = get_entities(file_sentences, exp_value, test_format, model_name)

            test_result_file.to_csv('test_results/' + file_name.split('.')[0] + '_entity_results.csv', sep=',',
                                    quotechar='"', quoting=csv.QUOTE_ALL)

        if test_format.upper() == 'SENT':

            if file_type.upper() != 'CSV':
                with codecs.open('test_files/' + file_name, encoding='utf-8') as tst:
                    file_sentences = [nltk.word_tokenize(line.strip()) for line in tst.readlines()]
                test_result_file = get_entities(file_sentences, exp_value, test_format, model_name)
                test_result_file.to_csv('test_results/' + file_name.split('.')[0] + '_entity_results.csv', sep=',',
                                        quotechar='"', quoting=csv.QUOTE_ALL)

            elif file_type.upper() == 'CSV':
                records = pd.read_csv('test_files/' + file_name, encoding='utf-8')
                test_result_file = get_entities(records, exp_value, test_format, model_name)

                test_result_file.to_csv('test_results/' + file_name.split('.')[0] + '_entity_results.csv', sep=',',
                                        quotechar='"', quoting=csv.QUOTE_ALL)

        if test_format.upper() == 'QUERY':
            print('file is Query')
            if file_type.upper() != 'CSV':
                with codecs.open('test_files/' + file_name, encoding='utf-8') as tst:
                    file_sentences = [clean_query(line.strip()) for line in tst.readlines()]
                test_result_file = get_entities(file_sentences, exp_value, test_format, model_name)
                test_result_file.to_csv('test_results/' + file_name.split('.')[0] + '_entity_results.csv', sep=',',
                                        quotechar='"', quoting=csv.QUOTE_ALL)

            elif file_type.upper() == 'CSV':
                records = pd.read_csv('test_files/' + file_name, encoding='utf-8')
                test_result_file = get_entities(records, exp_value, test_format, model_name)
                test_result_file.to_csv('test_results/' + file_name.split('.')[0] + '_entity_results.csv', sep=',',
                                        quotechar='"', quoting=csv.QUOTE_ALL)

        if test_format.upper() == 'MITIE':
            test_result_file = get_entities_mitie('test_files/' + file_name, model_name)

            test_result_file.to_csv('test_results/' + file_name.split('.')[0] + '_entity_results.csv', sep=',',
                                    quotechar='"', quoting=csv.QUOTE_ALL)

    if dir_name.strip() != '':
        print("dir name is not null")
        print("loading NER model...")
        ner = named_entity_extractor(model_name)
        print("\nTags output by this NER model:", ner.get_possible_ner_tags())

        test_files = []
        for file in glob.glob('test_files/'+dir_name + "/*." + file_type):
            test_files.append(file)
        print("Total number of files in Training data:", len(test_files),test_files)

        for file_name in test_files:
            f_name = file_name.split('/')[-1].split('.')[0]

            if test_format.upper() == 'DOCUMENT':

                with codecs.open(file_name,encoding='utf-8') as tst:
                    data_file = tst.read()
                file_sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(data_file)]
                test_result_file = get_entities(file_sentences, exp_value, test_format, ner)

                test_result_file.to_csv('test_results/' + f_name + '_entity_results.csv', sep=',',
                                        quotechar='"', quoting=csv.QUOTE_ALL)

            if test_format.upper() == 'SENT':

                if file_type.upper() != 'CSV':
                    with codecs.open(file_name,encoding='utf-8') as tst:
                        file_sentences = [nltk.word_tokenize(line.strip()) for line in tst.readlines()]
                    test_result_file = get_entities(file_sentences, exp_value, test_format, ner)

                    test_result_file.to_csv('test_results/' + f_name + '_entity_results.csv', sep=',',
                                            quotechar='"', quoting=csv.QUOTE_ALL)

                elif file_type.upper() == 'CSV':
                    records = pd.read_csv(file_name,encoding='utf-8')
                    test_result_file = get_entities(records, exp_value, test_format, ner)

                    test_result_file.to_csv('test_results/' + f_name + '_entity_results.csv', sep=',',
                                            quotechar='"', quoting=csv.QUOTE_ALL)

            if test_format.upper() == 'QUERY':

                if file_type.upper() != 'CSV':
                    with codecs.open(file_name,encoding='utf-8') as tst:
                        file_sentences = [clean_query(line.strip()) for line in tst.readlines()]
                    test_result_file = get_entities(file_sentences, exp_value, test_format, ner)
                    test_result_file.to_csv('test_results/' + f_name + '_entity_results.csv', sep=',',
                                            quotechar='"', quoting=csv.QUOTE_ALL)

                elif file_type.upper() == 'CSV':
                    records = pd.read_csv(file_name,encoding='utf-8')
                    test_result_file = get_entities(records, exp_value, test_format, ner)

                    test_result_file.to_csv('test_results/' + f_name + '_entity_results.csv', sep=',',
                                            quotechar='"', quoting=csv.QUOTE_ALL)

            if test_format.upper() == 'MITIE':
                test_result_file = get_entities_mitie(file_name, ner)

                test_result_file.to_csv('test_results/' + f_name + '_entity_results.csv', sep=',',
                                        quotechar='"', quoting=csv.QUOTE_ALL)

    return True

