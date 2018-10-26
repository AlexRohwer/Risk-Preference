import numpy as np
import csv
import matplotlib.pyplot as plt
from random import randint
from functools import reduce
import operator

# Fields from the input file, in same order as they appear in the file
# (ignoring respondent ID). Do not modify or change.
fields=['result','smoke','drink','gamble','skydive','speed','cheat','steak','cook','gender','age','income','education','location']
# Corresponding field values from the input file. Do not modify or change.
values=[('lottery a','lottery b'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),('no','yes'),
        ('rare','medium rare','medium','medium well','well'),('male','female'),
        ('18-29','30-44','45-60','> 60'),
        ('$0 - $24,999','$25,000 - $49,999','$50,000 - $99,999','$100,000 - $149,999','$150,000+'),
        ('less than high school degree','high school degree','some college or associate degree','bachelor degree','graduate degree'),
        ('east north central','east south central','middle atlantic','mountain','new england','pacific',
         'south atlantic','west north central','west south central')]

######################################################################
# This function reads a csv file and creates a new list of each row in the file, excluding the 0th element.
# Then, the function creates a new list of dictionaries with 'fields' as the keys and 'values' as the values,
# using a nested dictionary comprehension in a list comprehension that excludes rows that didn't answer the first
# field. Finally, the function iterates over listy to remove any elements of each dictionary in listy that do not
# provide a value.

def readData(filename='steak-risk-survey.csv', fields=fields, values=values):
    L = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            L.append(row[1:])
    listy = [ {fields[i]:L[resp][i].lower() for i in range(len(fields))} for resp in range(2, len(L)) if L[resp][0].lower() in values[0] ]
    for dictionary in listy:
        for element in list(dictionary.keys()):
            if dictionary[element] == '':
                del dictionary[element]
    return(listy)
        
######################################################################
# This function takes data selected from readData() and sorts it into a double bar graph. The primary components of this
# function are the two variables assigned to a nested list comprehension, which counts how many respondents that prefered either
# lottery a or lottery b and selected a certain value to a particular field. Then, it compares that number to the total population and
# and returns a tuple of each percentage for each lottery respondents. Lastly, using MatPlotLib, it plots this data into a
# double bar graph
def showPlot(D, field, values):
    n_groups = len(values)
    lotterya = tuple([ [ D[i][field] for i in range(len(D)) if D[i][fields[0]] in values[0][0] if field in D[i] ]
                       .count(value)/len(D) for value in values ])
    lotteryb = tuple([ [ D[i][field] for i in range(len(D)) if D[i][fields[0]] in values[0][1] if field in D[i] ]
                       .count(value)/len(D) for value in values ])

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = .35
    opacity = .8

    rects1 = plt.bar(index, lotterya, bar_width, alpha=opacity, color='b', label='Lottery A')
    rects2 = plt.bar(index + bar_width, lotteryb, bar_width, alpha=opacity, color='r', label='Lottery B')

    plt.xlabel('Values')
    plt.ylabel('Percentage of population')
    plt.title('Lottery preference by ' + '"' + field + '"')
    plt.xticks(index + .5*bar_width, tuple([value for value in values]))
    plt.legend()

    plt.tight_layout()
    plt.show()

######################################################################
# This function creates a dictionary with each field as a key, with values of dictionaries with each lottery type
# as a key, with values of dictionaries with each possible response of a field as a key, with values of the percentage
# of respondents who answered that response
def train(D, fields=fields, values=values):
    S = {}
    S[fields[0]] = {}
    for lottery in values[0]:
        S[fields[0]][lottery] = {}
        S[fields[0]][lottery] = [ D[i][fields[0]] == lottery for i in range(len(D)) ].count(True) / len(D)
    for field in fields[1:]:
        S[field] = {}
        for lottery in values[0]:
            S[field][lottery] = {}
            for value in values[fields.index(field)]:
                L = sum([[ D[i][field] for i in range(len(D)) if D[i][fields[0]] in lottery if field in D[i]].count(num) for num in values[fields.index(field)]])
                S[field][lottery][value] = ([ D[i][field] for i in range(len(D)) if D[i][fields[0]] in lottery if field in D[i] ].count(value))/L
                
    return(S)

######################################################################
# This function finds the probabiity that someone would prefer lottery a or lottery b using Naive Bay's Rule
# Given certain values in certain fields, and given the probability that someone chooses lottery a or b, this function
# compares the two numbers, and whichever probability for a certain lottery is higher, the predicted lottery
# is returned
def predict(example, P, fields=fields, values=values):
    lotteryA_result = reduce(operator.mul, [ P[field][values[0][0]][(example[field])] for field in fields[1:] if field in example ]
                             + [P[fields[0]][values[0][0]], 1] )
    lotteryB_result = reduce(operator.mul, [ P[field][values[0][1]][(example[field])] for field in fields[1:] if field in example ]
                             + [P[fields[0]][values[0][1]], 1] )
    if lotteryA_result > lotteryB_result:
        return('lottery a')
    else:
        return('lottery b')

######################################################################
# Predict by guessing. You're going to be about half right!
def guess(example, fields=fields, values=values):
    return(values[0][randint(0,1)]==example['result'])

######################################################################
# This function tests the predict() function's reliability to predict someone's lottery preference by
# comparing predict()'s outputs for each person in D, and compares them with the actual lottery
# preference of each person as a percentage
def test(D, P, fields=fields, values=values):
    count = 0
    for i in range(len(D)):
        if predict(D[i], P) == D[i][fields[0]]:
            count = count + 1
    return(count/len(D))
    
    

######################################################################
# Fisher-Yates-Knuth fair shuffle, modified to only shuffle the last k
# elements. S[-k:] will then be the test set and S[:-k] will be the
# training set.
def shuffle(D, k):
    # Work backwards, randomly selecting an element from the head of
    # the list and swapping it into the tail location. We don't care
    # about the ordering of the training examples (first len(D)-N),
    # just the random selection of the test examples (last N).
    i = len(D)-1
    while i >= len(D)-k:
        j = randint(0, i)
        D[i], D[j] = D[j], D[i]
        i = i-1
    return(D)

# Evaluate.
def evaluate(filename='steak-risk-survey.csv', fields=fields, values=values, trials=100):
    # Read in the data.
    D = readData(filename, fields, values)
    # Establish size of test set (10% of total examples available).
    N = len(D)//10
    result = 0
    random = 0
    for i in range(trials):
        # Shuffle to randomly select N test examples.
        D = shuffle(D, N)
        # Train the system on first 90% of the examples.
        P = train(D[:-N], fields=fields, values=values)
        # Test on last 10% of examples, chosen at random by shuffle().
        result += test(D[-N:], P, fields=fields, values=values)
        # How well would you do guessing at random?
        random += sum([ len([ True for x in D[-N:] if guess(x)])/N ])
    # Return average accuracy.
    print('NaiveBayes={}, random guessing={}'.format(result/trials, random/trials))


