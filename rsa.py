from math import exp, log, inf
import pandas as pd
from numpy import nan
import argparse
import sys
import os
from datetime import datetime
import glob

parser = argparse.ArgumentParser()
parser.add_argument('data_files', nargs = '?', default = 'rsa-comb-data.csv',
	help = 'Argument to specify where to find the CSVs containing the possible objects and utterances to model. Default is a file named "rsa-comb-data.csv" in the current directory.')
parser.add_argument('--output_file', '-o', nargs = '?', default = '',
	help = 'Argument to specify a file to output predictions to. Default does not output predictions, but prints them to the console.')
parser.add_argument('--alpha', '-a', type = float, default = 1.,
	help = 'Argument to specify an alpha for the speaker to adopt.')
parser.add_argument('--append', '-ap', default = False, action = 'store_true',
	help = 'If the output file already exists, append results to it without asking.')
parser.add_argument('--succinct', '-s', default = False, action = 'store_true',
	help = 'Argument to not print each stage of the model to the console.')
parser.add_argument('--nonincremental', '-ni', default = False, action = 'store_true',
	help = 'Argument to predict values at the utterance level instead of incrementally.')
parser.add_argument('--predict_single', '-ps', default = False, action = 'store_true',
	help = 'Argument to make predictions using the incremental model for partial utterances instead of whole utterances. Note that this will really only be interpretable if you are looking at the first word; otherwise, the prediction won\'t be properly incremental.')
parser.add_argument('--compare', '-c', default = False, action = 'store_true',
	help = 'Set this to run incremental, partial utterance, and utterance-level predicitions so you can compare the results. Note that comparisons for single-word utterances must be specified in the CSV.')

args = parser.parse_args()

verbose = not args.succinct

# If we are not verbose and we have no output, warn the user
if not verbose and not args.output_file:
	print('Warning: predictions will not be printed to the console or saved to file.')

# If we have an output specification, check if the file already exists and ask the user to specify another if so
if args.output_file:
	if not args.output_file.endswith('.txt'):
		args.output_file += '.txt'

	while os.path.isfile(args.output_file) and not args.append:
		new_file = input(f'Warning: file {args.output_file} already exists. Please enter a new location, or press enter to append new results to existing file: ')
		if not new_file:
			break

		args.output_file = new_file
		if args.output_file != '' and not args.output_file.endswith('.txt'):
			args.output_file += '.txt'

# predict_single has no use if we are nonincremental (but we will override this if we have compare set)
if args.nonincremental and args.predict_single and not args.compare:
	print('Warning: predict_single has no use when predicting non-incrementally.')
	args.predict_single = False

# Parse the data files into a list
args.data_files = args.data_files.split(':')
args.data_files = [file for sublist in [[file for file in glob.glob(data_file) if file.endswith('.csv')] for data_file in args.data_files] for file in sublist]

# If an utterance is true of an object, return 1 else 0. Used when predicting non-incrementally 
def denotation(utterance, obj):
	try:
		# Get the lexical semantics associated with the utterance (i.e., the states of affairs it describes)
		lexical_semantics = possible_utterances[utterance]['lexical semantics']

		# Get the actual characteristics of the objects
		obj_characteristics = objects[obj]['characteristics']
	except KeyError:
		# If the utterance or object does not exist, return 0 (false)
		return 0

	# Check if the lexical semantics are a subject of the object's characteristics. If so, the utterance is true,
	# if not, it is false. Convert this to 1, 0 by using int()
	return int(lexical_semantics.items() <= obj_characteristics.items())

# Return a value corresponding to the literal probability of an incomplete string. Used when predicting incrementally
def truthiness(c, obj):
	# Get the lexical semantics of the current item, as well as any previous items and combine them to give us the current lexical semantics

	# First, get all the possible continuations associated with the current item
	continuations = [utterance for string, utterance in possible_utterances.items() if string.startswith(c)]

	# If there are no possible continuations, this is not a licit utterance, and we return 0
	if not continuations:
		return 0

	# Get the number of continuations of the utterance compatible with the current obj and with any object
	cur_compatible = 0
	total_compatible = 0

	# For each possible continuation of the utterance
	for continuation in continuations:
		# Start a dictionary to hold its lexical semantics
		lex_sem = dict()
		current_c = ''
		# For each lexical entry in the sequence of words
		for lex_entry in continuation['sequence']:
			# Get the word identifier so we can grab its lexical semantics
			word = list(lex_entry.keys())[0]

			# Add the current word to the partial utterance. 
			# The lstrip is neccesary so that if we're on the first word, we don't add a preceding space.
			current_c = f'{current_c} {word}'.lstrip()
			
			#current_c = ' '.join([item for sublist in [[current_c, word] if current_c else [word]] for item in sublist])

			# If the word has lexical semantics (i.e., if it's not the STOP token)
			if 'lexical semantics' in lex_entry[word].keys():
				# Add them to the list (we are assuming a conjuctive lexical semantics)
				#lex_sem.update({attribute : value for attribute, value in lex_entry[word]['lexical semantics'].items()})
				lex_sem.update(lex_entry[word]['lexical semantics'])

				# Check if the lexical semantics of the current partial utterance is compatible with the current object
				if not lex_sem.items() <= objects[obj]['characteristics'].items():
					# If we have checked all the continuations and none are left
					if continuations[len(continuations) - 1] == continuation:
						# If there are any compatible continuations
						if total_compatible:
							# Return the ratio of continuations true in the current world to the number of true continuations in any world
							return cur_compatible/total_compatible

						# Otherwise, return a probability evenly distributed over the remaining possible continuations at the current string
						all_possible_continuations = [utterance for string, utterance in possible_utterances.items() if string.startswith(current_c)]

						# If we only have one possible continuation, that is the current c, and we want to return 0
						return 1/len(all_possible_continuations) if len(all_possible_continuations) != 1 else 0
				
			# If we have a full utterance, return its truth value
			if current_c.endswith('STOP') and c == current_c:
				return int(lex_sem.items() <= objects[obj]['characteristics'].items())

		# Now that we have the complete lexical semantics of the continuation, check if it is compatible with the world state
		if lex_sem.items() <= objects[obj]['characteristics'].items():
			# If so, add one to that total
			cur_compatible += 1

		# Now check whether that continuation is possible with any other world states
		for o in objects:
			if lex_sem.items() <= objects[o]['characteristics'].items():
				total_compatible += 1

	# If there are any worlds compatible with a continuation of the current utterance, return the ratio of the number of continuations compatible with the current world to the number of total possible continuations		
	return cur_compatible/total_compatible

# Get the literal probability of each object given each utterance
def literal_listener(cs, objects, *, verbose = False, output = False):
	# Get a matrix of truth values for each utterance-object pair. If non-incremental, we want denotations; if incremental, we want truthiness values
	if args.nonincremental:
		truth_message = '\nTruth value matrix:'
		L_0_message = '\nL₀ p(w|u):'
		truth_value_matrix = {c : {obj : denotation(c, obj) for obj in objects} for c in cs}
	else:
		truth_message = '\nTruthiness value matrix:'
		L_0_message = '\nL₀ p(w|c, word):'
		truth_value_matrix = {c : {obj : truthiness(c, obj) for obj in objects} for c in cs}

	# Print/save the truth value matrix
	if verbose:
		print(truth_message)
		print(pd.DataFrame.from_dict(truth_value_matrix, orient = 'index').round(2))

	if args.output_file and output:
		with open(args.output_file, 'a', encoding = 'utf-8') as file:
			file.write(f'{truth_message}\n')
			file.write(pd.DataFrame.from_dict(truth_value_matrix, orient = 'index').round(2).to_string() + '\n')

	# Normalize the values by dividing each truth value times its probability by the sum of the total number of objects/worlds that utterance is true of times the probability of each world. this will ensure that the rows add to one, giving us a probability distribution for the literal listener
	L_0 = {c : {obj : (truth_value_matrix[c][obj]*objects[obj]['p'])/sum([truth_value_matrix[c][obj]*objects[obj]['p'] for obj in objects]) for obj in objects} for c in cs}

	if verbose:
		print(L_0_message)
		print(pd.DataFrame.from_dict(L_0, orient = 'index').round(2))

	if args.output_file and output:
		with open(args.output_file, 'a', encoding = 'utf-8') as file:
			file.write(f'{L_0_message}\n')
			file.write(pd.DataFrame.from_dict(L_0, orient = 'index').round(2).to_string() + '\n')

	return L_0

# For now, assume all (partial) utterances have 0 cost
def cost(c):
	return 0

# Get the utility of each utterance given each object and an alpha
def pragmatic_speaker(cs, objects, alpha = args.alpha, *, verbose = False, output = False):
	# Get the literal listener model
	L_0 = literal_listener(cs, objects, verbose = verbose, output = output)
	S_1_message = '\nS₁ p(u|w):'

	# If we are predicting incrementally, the probability of a sequence is the joint probability of its words multiplied together
	if not args.nonincremental:

		# Set the correct message
		S_1_message = ('\nS₁ p(word|c, w):')

		# Get the probability associated with each partial utterance
		# First, split each partial utterance into a list of words
		cs = {c : {'sequence' : c.split(' '), 'p' : 0} for c in cs}

		# For each c in the list of cs
		for c in cs:
			# Probability of starting an utterance in 1
			prob = 1
			# We get the probability of the sequence by multiplying the probabilities of the words
			for word in cs[c]['sequence']:
				prob *= word_list[word]['p']

			cs[c]['p'] = prob

	# Transform it by taking the log of each utterance's probability and subtracting its cost, then exponentiating that
	S_1 = {obj : {c : exp(alpha*log(L_0[c][obj]) - cost(c)) if L_0[c][obj] != 0 else exp(-inf) for c in cs} for obj in objects}

	# Now normalize by dividing each value by the total probability mass for that world. This gives us the probability that S will use that utterance given that world
	S_1 = {obj : {c : (S_1[obj][c]*cs[c]['p'])/sum([S_1[obj][c]*cs[c]['p'] for c in cs]) if sum([S_1[obj][c]*cs[c]['p'] > 0]) else 0. for c in cs} for obj in objects}

	if verbose:
		print(S_1_message)
		print(pd.DataFrame.from_dict(S_1, orient = 'index').round(2))

	if args.output_file and output:
		with open(args.output_file, 'a', encoding = 'utf-8') as file:
			file.write(f'{S_1_message}\n')
			file.write(pd.DataFrame.from_dict(S_1, orient = 'index').round(2).to_string() + '\n')

	return S_1

# Get the probability of each world given each utterance
def pragmatic_listener(cs, objects, *, verbose = False, output = False):
	# Get the pragmatic speaker model
	S_1 = pragmatic_speaker(cs, objects, verbose = verbose, output = output)

	# Flip it from p(u|w) to p(w|u)
	L_1 = {c : {obj : S_1[obj][c] for obj in objects} for c in cs}

	# Normalize each utterance's probability by dividing for the total probability mass for that world
	L_1 = {c : {obj : (L_1[c][obj]*objects[obj]['p'])/sum([L_1[c][obj]*objects[obj]['p'] for obj in objects]) for obj in objects} for c in cs}

	if args.nonincremental:
		L_1_message = '\nL₁ p(w|u):'
	else:
		L_1_message = '\nL₁ p(w|c, word):'

	if verbose:
		print(L_1_message)
		print(pd.DataFrame.from_dict(L_1, orient = 'index').round(2))

	if args.output_file and output:
		with open(args.output_file, 'a', encoding = 'utf-8') as file:
			file.write(f'{L_1_message}\n')
			file.write(pd.DataFrame.from_dict(L_1, orient = 'index').round(2).to_string() + '\n')

	return L_1

# Run the model for each step
def predict_incremental(utterances, objects, *, verbose = False):
	# Get and display the truthiness matrix
	truth_value_matrix = {utterance : {obj : truthiness(utterance + ' STOP', obj) for obj in objects} for utterance in utterances}

	# Convert utterances to a list of words for each
	utterances = [utterance.split(' ') for utterance in utterances]

	# Convert the utterances to a list of incrementally built strings = cs
	for utterance in utterances:
		# Start at the second word of the utterance
		for i in range(1, len(utterance)):
			# Set the c at that point in the utterance to it + the previous string
			utterance[i] = ' '.join([utterance[i - 1], utterance[i]])

	# For each c in each utterance, we want to compare to the other cs of the same length. To do this without raising an IndexError for the shortest list, we need to fill it out with dummy values (and then just not pass those dummy values to the evaluation functions)
	max_len = max([len(sequence) for sequence in utterances])
	for utterance in utterances:
		utterance.extend([0] * (max_len - len(utterance)))

	# Now that we've padded out each list, we can get the sets of partial utterances we want to compare
	# This puts all of the first words in a comparison set, all of the second words, and so on
	comparison_sets = [list(set(comparison_set)) for comparison_set in list(zip(*utterances))]

	# Remove the padded 0s since we are not actually comparing to them
	for comparison_set in comparison_sets:
		if 0 in comparison_set:
			comparison_set.remove(0)

	# For each comparison set, run the predictions and store them (also store the truthiness value matrix for the utterances)
	utterance_predictions = list()
	for i, comparison_set in enumerate(comparison_sets):
		order = [string for string in pd.Series([' '.join(word.split(' ')[:i+1]) for word in list(truth_value_matrix.keys())]).drop_duplicates().tolist() if string in comparison_set]
		T_V = {c : {obj : truthiness(c, obj) for obj in objects} for c in comparison_set}
		L_0 = literal_listener(comparison_set, objects, verbose = False)
		S_1 = pragmatic_speaker(comparison_set, objects, verbose = False)
		L_1 = pragmatic_listener(comparison_set, objects, verbose = False)
		utterance_predictions.append({'print_order' : order, 'T_V' : T_V, 'L_0' : L_0, 'S_1' : S_1, 'L_1' : L_1})

	# Print out the first models, since otherwise we wouldn't
	if verbose:
		for model in utterance_predictions[0]:
			order = utterance_predictions[0]['print_order']
			if model == 'T_V':
				print('\nWord 1: truthiness value matrix:')
				utterance_predictions[0][model] = {c : utterance_predictions[0][model][c] for c in order}
			elif model == 'L_0':
				print('\nWord 1: L₀ p(w|c = [], word):')
				utterance_predictions[0][model] = {c : utterance_predictions[0][model][c] for c in order}
			elif model == 'S_1':
				print('\nWord 1: S₁ p(word|c = [], w):')
				utterance_predictions[0][model] = {obj : {c : utterance_predictions[0][model][obj][c] for c in order} for obj in S_1}
			elif model == 'L_1':
				print('\nWord 1: L₁ p(w|c, word):')
				utterance_predictions[0][model] = {c : utterance_predictions[0][model][c] for c in order}

			if model != 'print_order':
				print(pd.DataFrame.from_dict(utterance_predictions[0][model], orient = 'index').round(2))

	if args.output_file:
		with open(args.output_file, 'a', encoding = 'utf-8') as file:
			for model in utterance_predictions[0]:
				order = utterance_predictions[0]['print_order']
				if model == 'T_V':
					file.write('\nWord 1: truthiness value matrix:\n')
				elif model == 'L_0':
					file.write('\nWord 1: L₀ p(w|c = [], word):\n')
				elif model == 'S_1':
					file.write('\nWord 1: S₁ p(word|c = [], w):\n')
				elif model == 'L_1':
					file.write('\nWord 1: L₁ p(w|c, word):\n')

				if model != 'print_order':
					file.write(pd.DataFrame.from_dict(utterance_predictions[0][model], orient = 'index').round(2).to_string() + '\n')

	# For each prediction after the first, iterate through the individual predictions and update them by multiplying by the probability of the appropriate previous prediction
	for i, predictions in enumerate(utterance_predictions[1:], start = 1):

		order = predictions['print_order']

		if verbose:
			T_V = predictions['T_V']
			if i == len(utterance_predictions[1:]):
				print(f'\nWord {i+1}: (partial) truthiness value matrix:')
				T_V = {c : T_V[c] for c in order}
				print(pd.DataFrame.from_dict(T_V, orient = 'index').round(2))
				order = list(truth_value_matrix.keys())
				print(f'\nWord {i+1}: (full) truthiness value matrix:')
				print(pd.DataFrame.from_dict(truth_value_matrix, orient = 'index').round(2))
			else:
				print(f'\nWord {i+1}: truthiness value matrix:')
				T_V = {c : T_V[c] for c in order}
				print(pd.DataFrame.from_dict(T_V, orient = 'index').round(2))

		if args.output_file:
			order = predictions['print_order']
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				if i == len(utterance_predictions[1:]):
					file.write(f'\nWord {i+1}: (partial) truthiness value matrix:\n')
					T_V = {c : T_V[c] for c in order}
					file.write(pd.DataFrame.from_dict(T_V, orient = 'index').round(2).to_string() + '\n')
					order = list(truth_value_matrix.keys())
					file.write(f'\nWord {i+1}: (full) truthiness value matrix:\n')
					file.write(pd.DataFrame.from_dict(truth_value_matrix, orient = 'index').round(2).to_string() + '\n')
				else:
					file.write(f'\nWord {i+1}: truthiness value matrix:\n')
					T_V = {c : T_V[c] for c in order}
					file.write(pd.DataFrame.from_dict(T_V, orient = 'index').round(2).to_string() + '\n')

		# Separate out the previous predictions
		prev_predictions = utterance_predictions[i - 1]
		prev_L_0 = prev_predictions['L_0']
		prev_S_1 = prev_predictions['S_1']
		prev_L_1 = prev_predictions['L_1']

		# Get the current predictions
		L_0 = predictions['L_0']
		S_1 = predictions['S_1']
		L_1 = predictions['L_1']

		# Update L_0
		# For each partial utterance in L_0
		for c in L_0:
			# Remove the last word from c so that we can use that to find the c leading into it
			prev_c = prev_L_0[c.rsplit(' ', 1)[0]]
			
			# For each object's compatibility with that partial utterance
			for obj, prob in L_0[c].items():
				prob *= prev_c[obj]

				L_0[c][obj] = prob

			# Normalize the results
			L_0[c] = {obj : L_0[c][obj]/sum([L_0[c][obj] for obj in objects]) for obj in objects}

		# Add the previous full utterances
		L_0 = {**{c : obj for c, obj in prev_L_0.items() if [c] + [0 for i in range(0, max_len - len([c]))] in utterances}, **L_0}

		# Normalize the results
		L_0 = {c: {obj : L_0[c][obj]/sum([L_0[c][obj] for obj in objects]) for obj in objects} for c in L_0}

		if verbose:
			print(f'\nWord {i+1}: L₀ p(w|c, word):')
			# Have to do this becase there's some weird issue where sometimes it prints out the right order and sometimes it doesn't that I can't track down
			L_0 = {c : L_0[c] for c in order} 
			print(pd.DataFrame.from_dict(L_0, orient = 'index').round(2))
		
		if args.output_file:
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				file.write(f'\nWord {i+1}: L₀ p(w|c, word):\n')
				L_0 = {c : L_0[c] for c in order}
				file.write(pd.DataFrame.from_dict(L_0, orient = 'index').round(2).to_string() + '\n')
	
		# Update S_1
		for obj in S_1:
			# Remove the last word from c so that we can use that to find the c leading into it
			prev_obj = prev_S_1[obj]

			# For each object's compatibility with that partial utterance
			for c, prob in S_1[obj].items():

				# Get the probability associated with the previous word of that utterance
				prev_prob = prev_obj[c.rsplit(' ', 1)[0]]

				# Multiply it by the current probability and update it
				prob *= prev_prob
				S_1[obj][c] = prob

		# Get the previous full utterances
		prev_S_1_utt = {obj : {c : prob for c, prob in cs.items() if [c] + [0 for i in range(0, max_len - len([c]))] in utterances} for obj, cs in prev_S_1.items()}

		for obj in S_1:
			# Get the utterances associated with that world
			prev_S_1_cs = prev_S_1_utt[obj]
			# For each utterance
			for c in prev_S_1_cs:
				# Add to the current world in the S_1 model for the next continuations
				#order = [c] + list(S_1[obj].keys())
				S_1[obj][c] = prev_S_1_cs[c]
				S_1[obj] = {c : S_1[obj][c] for c in order if c in S_1[obj]}

		# Normalize the results
		for obj in S_1:
			total_prob = sum([prob for prob in S_1[obj].values()])
			S_1[obj] = {c : prob/total_prob if total_prob != 0 else 0 for c, prob in S_1[obj].items()}

		if verbose:
			print(f'\nWord {i+1}: S₁ p(word|c, w):')
			# Another hack
			S_1 = {obj : {c : S_1[obj][c] for c in order} for obj in S_1}
			print(pd.DataFrame.from_dict(S_1, orient = 'index').round(2))

		if args.output_file:
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				file.write(f'\nWord {i+1}: S₁ p(word|c, w):\n')
				S_1 = {obj : {c : S_1[obj][c] for c in order} for obj in S_1}
				file.write(pd.DataFrame.from_dict(S_1, orient = 'index').round(2).to_string() + '\n')

		# Update 
		# For each partial utterance in L_1
		for c in L_1:
			# Remove the last word from c so that we can use that to find the c leading into it
			prev_c = prev_L_1[c.rsplit(' ', 1)[0]]
			
			# For each object's compatibility with that partial utterance
			for obj, prob in L_1[c].items():
				prob *= prev_c[obj]

				L_1[c][obj] = prob

			# Normalize the results
			L_1[c] = {obj : L_1[c][obj]/sum([L_1[c][obj] for obj in objects]) for obj in objects}

		# Add the previous full utterances
		L_1 = {**{c : obj for c, obj in prev_L_1.items() if [c] + [0 for i in range(0, max_len - len([c]))] in utterances}, **L_1}

		# Normalize the results
		L_1[c] = {obj : L_1[c][obj]/sum([L_1[c][obj] for obj in objects]) for obj in objects}

		if verbose:
			print(f'\nWord {i+1}: L₁ p(w|c, word):')
			# And again
			L_1 = {c : L_1[c] for c in order if c in L_1}
			print(pd.DataFrame.from_dict(L_1, orient = 'index').round(2))

		if args.output_file:
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				file.write(f'\nWord {i+1}:  L₁ p(w|c, word):\n')
				L_1 = {c : L_1[c] for c in order if c in L_1}
				file.write(pd.DataFrame.from_dict(L_1, orient = 'index').round(2).to_string() + '\n')

	# Add in the truth-value matrix to return
	utterance_predictions.insert(0, truth_value_matrix)

	return utterance_predictions

# For each data file, read it in and get the predictions
for data_file in args.data_files:

	# Read in the definition of worlds and utterances
	context = pd.read_csv(data_file)

	# Separate out the utterances, words and the worlds/objects
	objects = context[context.type == 'object'].copy().loc[:, ~context.columns.isin(['type'])].pivot(index = 'label', columns = 'attribute', values = 'value')
	if objects.empty:
		print('Error: no objects specified. Exiting.')
		sys.exit(1)

	words = context[context.type == 'word'].copy().loc[:, ~context.columns.isin(['type'])].pivot(index = 'label', columns = 'attribute', values = 'value')
	if words.empty:
		print('Error: no words specified. Words must be specified even for utterance-level predictions only. Exiting.')
		sys.exit(1)

	utterances = context[context.type == 'utterance'].copy().loc[:, ~context.columns.isin(['type'])].pivot(index = 'label', columns = 'attribute', values = 'value')
	if utterances.empty and not args.print_single:
		print('Error: no complete utterances found when not predicting partial utterances. Exiting.')
		sys.exit(1)

	utterances = utterances.loc[:, utterances.columns.notnull()]

	# Set any undefined ps to 1
	if not 'p' in objects.columns:
		objects['p'] = nan

	objects.p = [p if not pd.isnull(p) else 1. for p in objects.p]

	if not 'p' in words.columns:
		possible_utterances['p'] = nan

	words.p = [p if not pd.isnull(p) else 1. for p in words.p]

	if not 'p' in utterances.columns:
		utterances['p'] = nan

	utterances.p = [p if not pd.isnull(p) else 1. for p in utterances.p]

	if not 'sequence' in utterances.columns:
		utterances['sequence'] = utterances.index

	utterances.sequence = [sequence if not pd.isnull(sequence) else label for label, sequence in tuple(zip(utterances.index, utterances.sequence))]

	# Convert objects to dict
	p_objects = objects['p'].copy().T.to_dict()
	p_objects = {obj : {'p' : float(p)} for obj, p in p_objects.items()}
	characteristics_objects = objects.loc[:, ~objects.columns.isin(['p'])].copy()
	characteristics_objects = characteristics_objects.set_index(characteristics_objects.index).T.to_dict()
	characteristics_objects = {obj : {'characteristics' : attribute} for obj, attribute in characteristics_objects.items()}
	objects = {obj : {'characteristics' : characteristics_objects[obj]['characteristics'], 'p' : p_objects[obj]['p']} for obj in characteristics_objects}

	# Convert words to dict
	p_words = words['p'].copy().T.to_dict()
	p_words = {word : {'p' : float(p)} for word, p in p_words.items()}
	word_list = words.loc[:, ~words.columns.isin(['p'])].copy()
	word_list = word_list.set_index(word_list.index).T.to_dict()
	word_list = {word : {attribute : value for attribute, value in word_list[word].items() if not pd.isnull(value)} for word in word_list}
	word_list = {word : {'lexical semantics' : attribute} for word, attribute in word_list.items()}
	word_list = {word : {'lexical semantics' : word_list[word]['lexical semantics'], 'p' : p_words[word]['p']} for word in word_list}

	# Add the stop token to our list
	word_list.update({'STOP' : {'p' : 1.}})

	# Convert utterances to a dict
	possible_utterances = utterances.loc[:, ~utterances.columns.isin(['p'])].copy()
	possible_utterances = possible_utterances.set_index(possible_utterances.index).T.to_dict()
	possible_utterances = {f'{label} STOP' : {sequence : utterance.split(' ') + ['STOP'] for sequence, utterance in possible_utterances[label].items()} for label in possible_utterances}

	for utterance in possible_utterances:
		possible_utterances[utterance]['sequence'] = [{word : word_list[word]} for word in possible_utterances[utterance]['sequence']]

	if args.predict_single or args.compare:
		cs = context[context.type == 'c'].label.tolist()

	# If we are predicting non-incrementally, convert the utterances to the appropriate format
	if args.nonincremental or args.compare:
		p_utterances = utterances['p'].copy().T.to_dict()
		p_utterances = {utterance : {'p' : float(p)} for utterance, p in p_utterances.items()}
		
		# Remove the STOP token from the end of the label since we no longer need it to signal the end of the utterance
		interim_utterances = {utterance.rstrip(' STOP') : sequence for utterance, sequence in possible_utterances.items()}
		# Set up an accumulator to hold the reformatted utterances
		nonincremental_possible_utterances = dict()
		for utterance in interim_utterances:
			# Remove the STOP token from the end of the sequence

			interim_utterances[utterance]['sequence'] = [word for word in interim_utterances[utterance]['sequence'] if not 'STOP' in word.keys()]
			# Set up an accumulator to hold the semantics of the current utterance
			semantics = dict()
			# For each word, get its lexical semantics and update the semantics
			for lex_entry in interim_utterances[utterance]['sequence']:
				word = list(lex_entry.keys())[0]
				semantics.update(word_list[word]['lexical semantics'])

			# Now that we have the semantics of the whole utterance, add it to the accumulator
			nonincremental_possible_utterances.update({utterance : {'lexical semantics' : semantics, 'p' : p_utterances[utterance]['p']}})

	# Set a variable to output the data
	output = True if args.output_file else False

	# If we are comparing, run the separate models
	if args.compare:
		args.nonincremental = False
		args.predict_single = False
		if args.output_file:
			# Write the header to the output file
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				# If we're adding to an existing file, add a separator first
				if not file.tell() == 0:
					file.write('\n========================================\n========================================\n\n')
				file.write(f'Predictions for {data_file} run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
				file.write(f'\nArguments specified: {str(args)[10:-1]}\n')

				file.write('\nObjects/worlds:\n')
				for obj in objects:
					file.write(f'{obj}: {objects[obj]}\n')

				file.write('\nWords:\n')
				for word in word_list:
					if word != 'STOP':
						file.write(f'{word}: {word_list[word]}\n')

				file.write('\nPossible utterances:\n')
				for utterance in possible_utterances:
					file.write(f'{utterance.rstrip(" STOP")}: ')
					for i, word in enumerate(possible_utterances[utterance]['sequence']):
						if not list(word.keys())[0] == 'STOP':
							if i > 0:
								file.write(''.rjust(len(utterance.rstrip(' STOP') + ': ')))
							file.write(f'Word {i+1}: {possible_utterances[utterance]["sequence"][i]}\n')
					file.write('\n')
			
				file.write('\nPredictions:')
		predict_incremental([utterance.rstrip(' STOP') for utterance in possible_utterances], objects, verbose = verbose)
		args.predict_single = True
		if args.output_file:
			# Write the header to the output file
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				# If we're adding to an existing file, add a separator first
				if not file.tell() == 0:
					file.write('\n========================================\n========================================\n\n')
				file.write(f'Predictions for {data_file} run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
				file.write(f'\nArguments specified: {str(args)[10:-1]}\n')

				file.write('\nObjects/worlds:\n')
				for obj in objects:
					file.write(f'{obj}: {objects[obj]}\n')

				file.write('\nWords:\n')
				for word in word_list:
					if word != 'STOP':
						file.write(f'{word}: {word_list[word]}\n')

				file.write('\nPossible utterances:\n')
				for utterance in possible_utterances:
					file.write(f'{utterance.rstrip(" STOP")}: ')
					for i, word in enumerate(possible_utterances[utterance]['sequence']):
						if not list(word.keys())[0] == 'STOP':
							if i > 0:
								file.write(''.rjust(len(utterance.rstrip(' STOP') + ': ')))
							file.write(f'Word {i+1}: {possible_utterances[utterance]["sequence"][i]}\n')
					file.write('\n')
				
				if not cs:
					file.write('\nUnable to predict single since no partial utterances were specified.')
					if args.compare:
						file.write('\n')
					print('Warning: cannot predict single since no partial utterances were specified.')

				file.write(f'Partial utterances: {cs}\n\nPredictions for partial utterances:')
		if cs: pragmatic_listener(cs, objects, verbose = verbose, output = output)
		args.predict_single = False
		args.nonincremental = True
		# For the nonincremental predictions, set the possible utterances to the nonincremental versions
		incremental_possible_utterances, possible_utterances = possible_utterances, nonincremental_possible_utterances
		if args.output_file:
			# Write the header to the output file
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				# If we're adding to an existing file, add a separator first
				if not file.tell() == 0:
					file.write('\n========================================\n========================================\n\n')
				file.write(f'Predictions for {data_file} run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
				file.write(f'\nArguments specified: {str(args)[10:-1]}\n')

				file.write('\nObjects/worlds:\n')
				for obj in objects:
					file.write(f'{obj}: {objects[obj]}\n')

				file.write('\nWords:\n')
				for word in word_list:
					if word != 'STOP':
						file.write(f'{word}: {word_list[word]}\n')

				file.write('\nPossible utterances:\n')
				for utterance in possible_utterances:
					file.write(f'{utterance.rstrip(" STOP")}: {possible_utterances[utterance]}\n')

				file.write('\nPredictions:')	
		pragmatic_listener(possible_utterances, objects, verbose = verbose, output = output)
		# Restore them in case someone is running this interactively and wants them back
		possible_utterances = incremental_possible_utterances
	elif args.nonincremental:
		# For the nonincremental predictions, set the possible utterances to the nonincremental versions
		incremental_possible_utterances, possible_utterances = possible_utterances, nonincremental_possible_utterances
		if args.output_file:
			# Write the header to the output file
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				# If we're adding to an existing file, add a separator first
				if not file.tell() == 0:
					file.write('\n========================================\n========================================\n\n')
				file.write(f'Predictions for {data_file} run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
				file.write(f'\nArguments specified: {str(args)[10:-1]}\n')

				file.write('\nObjects/worlds:\n')
				for obj in objects:
					file.write(f'{obj}: {objects[obj]}\n')

				file.write('\nWords:\n')
				for word in word_list:
					if word != 'STOP':
						file.write(f'{word}: {word_list[word]}\n')

				file.write('\nPossible utterances:\n')
				for utterance in possible_utterances:
					file.write(f'{utterance.rstrip(" STOP")}: {possible_utterances[utterance]}\n')

				file.write('\nPredictions:')	
		pragmatic_listener(possible_utterances, objects, verbose = verbose, output = output)
		# Restore them in case someone is running this interactively and wants them back
		possible_utterances = incremental_possible_utterances
	elif args.predict_single:
		if args.output_file:
			# Write the header to the output file
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				# If we're adding to an existing file, add a separator first
				if not file.tell() == 0:
					file.write('\n========================================\n========================================\n\n')
				file.write(f'Predictions for {data_file} run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
				file.write(f'\nArguments specified: {str(args)[10:-1]}\n')

				file.write('\nObjects/worlds:\n')
				for obj in objects:
					file.write(f'{obj}: {objects[obj]}\n')

				file.write('\nWords:\n')
				for word in word_list:
					if word != 'STOP':
						file.write(f'{word}: {word_list[word]}\n')

				file.write('\nPossible utterances:\n')
				for utterance in possible_utterances:
					file.write(f'{utterance.rstrip(" STOP")}: ')
					for i, word in enumerate(possible_utterances[utterance]['sequence']):
						if not list(word.keys())[0] == 'STOP':
							if i > 0:
								file.write(''.rjust(len(utterance.rstrip(' STOP') + ': ')))
							file.write(f'Word {i+1}: {possible_utterances[utterance]["sequence"][i]}\n')
					file.write('\n')
				
				if not cs:
					file.write('\nUnable to predict single since no partial utterances were specified.')
					if args.compare:
						file.write('\n')
					print('Warning: cannot predict single since no partial utterances were specified.')

				file.write(f'Partial utterances: {cs}\n\nPredictions for partial utterances:')
		if cs: pragmatic_listener(cs, objects, verbose = verbose, output = output)
	else:
		if args.output_file:
			# Write the header to the output file
			with open(args.output_file, 'a', encoding = 'utf-8') as file:
				# If we're adding to an existing file, add a separator first
				if not file.tell() == 0:
					file.write('\n========================================\n========================================\n\n')
				file.write(f'Predictions for {data_file} run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
				file.write(f'\nArguments specified: {str(args)[10:-1]}\n')

				file.write('\nObjects/worlds:\n')
				for obj in objects:
					file.write(f'{obj}: {objects[obj]}\n')

				file.write('\nWords:\n')
				for word in word_list:
					if word != 'STOP':
						file.write(f'{word}: {word_list[word]}\n')

				file.write('\nPossible utterances:\n')
				for utterance in possible_utterances:
					file.write(f'{utterance.rstrip(" STOP")}: ')
					for i, word in enumerate(possible_utterances[utterance]['sequence']):
						if not list(word.keys())[0] == 'STOP':
							if i > 0:
								file.write(''.rjust(len(utterance.rstrip(' STOP') + ': ')))
							file.write(f'Word {i+1}: {possible_utterances[utterance]["sequence"][i]}\n')
					file.write('\n')
			
				file.write('\nPredictions:')
		predict_incremental([utterance.rstrip(' STOP') for utterance in possible_utterances], objects, verbose = verbose)