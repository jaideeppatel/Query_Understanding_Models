import  sys
import os
import re
import timeit
import pandas as pd
import csv
import hunspell
import hunspell_package


# Get user input format, name, dic, model...
# # Instructions for preprocessing train data ...
# Format: filename, all the dictionary names separated by a <space> character,
# number of top predictions to be retrieved, expected_results_bool,
# expected results in span or non-span formats ....
with open('hunspell_instruction.txt',mode='r') as f:
      command_list = f.read().split(',')

instructions = [c.strip().split() for c in command_list]

f_name = instructions[0][0].split('.')[0]
file_type = instructions[0][0].split('.')[1]
expr = instructions[3][0]
top = instructions[2][0]
span_ins = instructions[4][0]

# Adding the default Hunspell idictionary and rule files to the Hunspell instance ....
hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
for dic in instructions[1]:
      hobj.add_dic('/usr/share/hunspell/'+dic)

# Add all the needed dictionaries to the Hunspell instance ...
inst = [f_name, file_type, top, expr]
if len(instructions[1])>0:
    op_fname = f_name + '_Hunspell_multiple_dicts_top' + top + '_results.csv'
else:
    op_fname = f_name + '_Hunspell_top' + top + '_results.csv'

# Processing .txt test files ....
if (file_type=='TXT' or file_type=='DAT'):
      query_list = hunspell_package.read_queries_text(f_name,file_type)
      exp_results = [''] * len(query_list)

# Processing .csv test files ....
elif file_type.upper()=='CSV':
      input_file = pd.read_csv('test_files/' + f_name +'.' + file_type, sep=",")
      query_list = list(input_file['Query'])
      if expr.upper() == 'Y':
            exp_results = list(input_file['Expected Results'])
      else:
            exp_results = [''] * len(query_list)

      # Begin the test file processing to get Hunspell suggestions ....
      if expr.upper() == 'Y':
            if span_ins.upper() == 'SPAN':
                  results_df = hunspell_package.process_results_span(query_list,inst,exp_results,hobj)
                  results_df = hunspell_package.get_scores(results_df, instructions)

                  results_df.to_csv('test_files/' + op_fname, sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
                  print('Hunspell results saved in the file', op_fname)

            elif span_ins.upper() == 'NON-SPAN':
                  # Yet to add the non-span processing function to the Hunspell package ....
                  results_df = hunspell_package.process_results(query_list, inst, exp_results,hobj)
                  results_df = hunspell_package.get_scores(results_df, instructions)
                  results_df.to_csv('test_files/' + op_fname, sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
                  print('Hunspell results saved in the file', op_fname)

