import  sys
import os
import re
import time
import pandas as pd
import csv
import hunspell

# This function reads the test file and returns the query list...
def read_queries_text(filename,f_type):
    fname = filename +'.'+f_type
    query_file = open('test_files/' + fname, 'r')
    query_list = []
    for line in query_file:
        query_list.append(line.replace('\n', ''))

    return query_list

# To clean the query text, primarily ro ignore the LN boolean and string connectors ...
def clean_query(q):
    word_list = [p for p in re.split("( |\(|\))", q) if p.strip()]
    wl = [re.sub("[()\";,:&]+",'',p.strip()) for p in word_list if len(p)>1]
    return wl

# Function to get hunspell suggestion for list of tokens from the test query ....
def get_suggestion(tk,hobj,n):
    try:
        pattern = re.compile("[!/?]+| |^atleast[a-zA-Z0-9]]*")
        chk_list = [pattern.search(t) for t in tk]
        suggest_list = []
        proc_que = []
        incorrect_spells = 0
        for i in range(len(chk_list)):
            if chk_list[i] == None:
                spell_list = hobj.spell(tk[i])
                if spell_list == False:
                    incorrect_spells+=1
                    pred = hobj.suggest(tk[i])
                    proc_que.append('"'+ tk[i] + '"')
                    if len(pred) >= n:
                        suggest_list.append([p.lower() for p in pred[0:n]])
                    elif len(pred) > 0 and len(pred)<n:
                        suggest_list.append([p.lower() for p in pred])
                    else:
                        suggest_list.append(tk[i])
                else:
                    proc_que.append(tk[i])
                    suggest_list.append(tk[i])
            else:
                proc_que.append(tk[i])
                suggest_list.append(tk[i])
        processed_query = ' '.join(proc_que)
    except (UnicodeDecodeError,UnicodeEncodeError) as e:
        suggest_list = ['Unicode error']
        incorrect_spells = 0
        processed_query = ''

    return [suggest_list, incorrect_spells, processed_query]

# Function to process the test file and generate the results csv file ...
def process_results_span(query_list,instructions,exp_res,hobj):
    hunspell_suggestions = pd.DataFrame()
    hunspell_suggestions['Query'] = ''
    hunspell_suggestions['processed query'] = ''
    hunspell_suggestions['number of words identified as misspelled'] = ''

    if instructions[3] == 'Y':
        hunspell_suggestions['Expected Results'] = ''
        hunspell_suggestions['correct query'] = ''

    colname = 'Hunspell suggestions'
    total_misspelled = 0
    total_identified = 0
    start_time = time.time()
    if instructions[3].upper() == 'Y':
        hunspell_suggestions['number of actual misspelled words'] = ''

        for j in range(len(query_list)):
            print(j)
            total_bad_words = 0
            q_new = clean_query(query_list[j])
            q_exp = exp_res[j].replace('<span>', '').replace('</span>', '')
            q_clean = clean_query(q_exp)

            hunspell_suggestions.set_value(j, 'Query', query_list[j].replace('\r', ''))
            cnt = exp_res[j].count("<span>")
            total_bad_words +=cnt
            total_misspelled +=cnt
            hunspell_suggestions.set_value(j, 'number of actual misspelled words', total_bad_words)
            exp_res_list = re.findall(r"<span>(.*?)</span>", exp_res[j])
            exp_res_text = ' '.join(exp_res_list)
            hunspell_suggestions.set_value(j, 'Expected Results', exp_res_text)
            hunspell_suggestions.set_value(j, 'correct query',q_clean)

            suggestions = get_suggestion(q_new, hobj, int(instructions[2]))
            hunspell_suggestions.set_value(j, colname, suggestions[0])
            hunspell_suggestions.set_value(j, 'number of words identified as misspelled', suggestions[1])
            total_identified += suggestions[1]
            hunspell_suggestions.set_value(j, 'processed query', suggestions[2])
    else:
        for j in range(len(query_list)):
            print(j)
            q_new = clean_query(query_list[j])
            q_exp = exp_res[j].replace('<span>', '').replace('</span>', '')
            q_clean = clean_query(q_exp)
            hunspell_suggestions.set_value(j, 'Query', query_list[j].replace('\r', ''))

            suggestions = get_suggestion(q_new, hobj, int(instructions[2]))
            hunspell_suggestions.set_value(j, colname, suggestions[0])
            hunspell_suggestions.set_value(j, 'number of words identified as misspelled', suggestions[1])
            hunspell_suggestions.set_value(j, 'processed query', suggestions[2])
    # print('Total number of actual bad words in '+str(len(query_list))+" queries: ",str(total_bad_words))
    print("Total time taken to process "+ str(len(query_list))+' queries:',time.time() - start_time)
    print("Total number of actual misspelled words: ",total_misspelled)
    print("Total number of words identified as misspelled: ", total_identified)
    return hunspell_suggestions

# Retuns the test file metrics like the accuracy score and False positives ...
def get_scores(results_df,inst):

    sug_col = 'Hunspell suggestions'
    colname = 'Hunspell result check'
    results_df[colname] = ''

    rows = results_df.shape[0]
    total_rows = rows
    correct_pred = 0
    incorrect_pred = 0
    for i in range(rows):
        exp_result = results_df.iloc[i]['Expected Results'].replace('#', '').split(' ')
        huns_result = str(results_df.iloc[i][sug_col]).replace('[', '').replace(']', '')
        chk_flag = True
        for w in range(len(exp_result)):
            st = "'"+exp_result[w].lower()+"'"
            if st not in huns_result:
                chk_flag = False
        if chk_flag == False:
            results_df.set_value(i, colname, 'Incorrect')
            incorrect_pred += 1
        else:
            results_df.set_value(i, colname, 'Correct')
            correct_pred += 1

    accuracy = (correct_pred / total_rows)
    print("Total number of Queries:", total_rows)
    print("Total number of correct predictions:",correct_pred)
    print("Total number of incorrect predictions:", incorrect_pred)
    print('Overall Accuracy:', accuracy)
    return results_df
