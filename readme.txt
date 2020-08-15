This Python script implements an utterance level and an incremental Rational Speech Acts model of pragmatic inference, both as described in Cohn-Gordon, Goodman, and Potts (2019).

Usage:

python rsa-combined.py [data_files] [--output_file -o OUTPUT_FILE] [--alpha (-a) ALPHA] [--append (-ap)] [--succinct (-s)] [--nonincremental (-ni)] [--predict_single (-ps)] [--compare (-c)]

Arguments:

data_files: The path(s) to CSV(s) containing a specification of the contextual situation, including worlds/objects, words, and (partial) utterances. See below for details on how this CSV should be formatted. The default path is a file in the current directory named 'rsa-comb-data.csv'. Additional files can be manually specified by separating each with ':'. You can also specify multiple files using Unix style wildcards.

--output_file (-o): The name of a txt file to output results to (you may leave out the '.txt' extension if you wish). If omitted, results will not be saved. If a txt file with the specified name already exists, you will be asked to either input a new file name, or confirm you want to append the results to the existing file.

--alpha (-a): Set the alpha value for the pragmatic speaker. Default is 1 (most rational).

--append (-ap): If set, output will automatically be append to a specified output file if it already exists without warning. Useful if you want to run the script several times in a row with different settings and output the results to the same file.

--succinct (-s): Don't print output to the console.

--nonincremental (-ni): Set this to make utterance-level predictions instead of incremental predictions.

--predict_single (-ps): Set this to make predictions using the incremental model for partial utterances instead of whole utterances. Note that this will really only be interpretable if you are looking at the first word; otherwise, the prediction won't be properly incremental. These must be specified separately in the CSV.

--compare (-c): Set this to run incremental, partial utterance, and utterance-level predictions so you can compare the results. Note that partial utterances must be specified separately in the CSV.

Input structure:

The input file(s) should be in CSV format, with four columns named:

label,type,attribute,value

label: a label for the specified model object. This is what groups multiple rows together, allowing you to specify multiple properties for a model object.

type: what kind of model object the label refers to. This can be one of four values: object, word, utterance, or c (i.e., a partial utterance, following Cohn-Gordon, Goodman, & Potts (2019)).

An object corresponds to a world state. It defines a set of attributes and values that the particular world state we are interested in has. It is called 'object' following the examples in Cohn-Gordon, Goodman, & Potts (2019), where the world-states correspond to which object the speaker has in mind from a list of objects. But you could specify events, etc., too, as the format for expressing things is quite flexible. An object has characteristics, which are the properties that are used to determine when an utterance is true of an object, and are specified the same way as a word's lexical semantics. An object also has a 'p', which corresponds to the prior probability of that world/that the speaker has that object in mind.

A word is a single incremental unit for predictive purposes. Note that this means you could specify a multi-word phrase as a "word"; that just means it will count as one incremental unit when predicting step by step. Due to the way utterances are split, you should use dashes between words when specifying multi-word sequences as a single predictive unit, and not spaces, which will cause each item separated by spaces to be treated as a separate word. A word has an associated lexical semantics and a prior probability associated with its use.

An utterance is a sequence of words. The sequence is determined by splitting on spaces. If you want to use a multi-word sequence as a single word, it is recommended that you use a dash to join them. An utterance can have a prior probability of its own, separate from the product of the probabilities of the words that comprise it. Utterances do not have their own semantics; their semantics is determined by the word that comprise them. For now, this means that only conjuctive semantics are supported; function application cannot be modeled currently.

A 'c' is a partial utterance. It is not treated as a sequence of words, but as a single unit. Note that this means that if you specify a multi-word sequence as a c, it will not be treated as an incremental sequence of words. These can model predictions for single partial utterances using the predictive model, but not predictively. This basically means that the results are only interpretable if the cs correspond to the first word(s) of an utterance, and show the predictions associated with that word.

attribute: a label for a category. This can be anything (with one exception), but it should share the same usage when used to specify the lexical semantics of a word and the characteristics of an object. Note that an object should have all of its (relevant) attributes and their values specified to result in accurate predictions. The one reserved attribute is 'p', which is used to specify the prior probability of its associated model object. Any prior probabilities not set are set to 1.

value: the value of the attribute. This can also be anything, and it should also share the same usage when used to specify the lexical semantics of a word and the characteristics of an object. The exception is 'p', which should be a number corresponding to the prior probability of the model object it is associated with.

See the included CSV file for an example of a model specification that corresponds to the one in Cohn-Gorden, Goodman, and Potts (2019).

Output:

There are two types of output: if --succinct is not set, output will be printed to the console. If an output file is set, output will be saved in the txt file specified, with information about the model specified in the CSV and the arguments passed to the script, for reproducibility.