
import os.path
import math
from itertools import chain
from collections import defaultdict
import csv

# Initialization
positiveTrainingData = "hotelPosT-train.txt"
positiveWordsText = "positive-words.txt"
testReviewData = "hotel-review-testset.txt"

negativeTrainingData = "hotelNegT-train.txt"
negativeWordsText = "negative-words.txt"

pronounList = ["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"]

# Reads and cleans the input positive/negative review
def readAndCleanFile(file):
    try:
        with open(file) as openedFile:
            cleanReviewList = []
            for line in openedFile:
                ID, reviewWords = line[0:7], line.lower().split('\t')[1].replace('\n','').split(' ')
                individualReviewList = [ID,reviewWords]
                cleanReviewList.append(individualReviewList)
        return cleanReviewList

    except:
        print("Error reading OR cleaning file")

# Creates a dictionary with positive/negative words from a positive/negative word text
def positiveNegativeWords(positiveNegativeWordsFile):
    with open(positiveNegativeWordsFile) as openedFile:
        words = []
        positiveNegativeWords = openedFile.readlines()
        for word in positiveNegativeWords:
            words.append(word.strip('\n'))
    return words

# Creates a dictionary with ID and count of positive/negative works in a given positive/negative reviews
# NOTE: It counts only unique positive/negative words
def positiveNegativeWordCounter(positiveNegativeWords, positiveNegativeReviewWords):
    featureDictionary = {}
    for i in range(len(positiveNegativeReviewWords)):
        count = 0
        words=[]
        for word in positiveNegativeReviewWords[i][1]:
            word = word.replace(".","")
            word = word.replace(",","")
            word = word.replace("!","")
            words.append(word)
        count = len(set(positiveNegativeWords) & set(words))
        featureDictionary[positiveNegativeReviewWords[i][0]] = int(count)
    return featureDictionary

def noPresence(reviewWords):
    noDictionary = {}
    noSet = {"no"}
    for i in range(len(reviewWords)):
        count = 0
        words=[]
        for word in reviewWords[i][1]:
            word = word.replace(".","")
            word = word.replace(",","")
            word = word.replace("!","")
            words.append(word)
        count = len(noSet & set(words)) > 0
        noDictionary[reviewWords[i][0]] = int(count)
    return noDictionary

# Creates a dictionary with ID and count of pronoun in a given reviews
# NOTE: It counts only unique pronouns
def pronounCount(pronounList, reviewWords):
    pronounDictionary = {}
    for i in range(len(reviewWords)):
        count = 0
        words=[]
        for word in reviewWords[i][1]:
            word = word.replace(".","")
            word = word.replace(",","")
            words.append(word)
        count = len(set(pronounList) & set(words))
        pronounDictionary[reviewWords[i][0]] = count
    return pronounDictionary

# Creates a dictionary with ID and presence of an exclamatory('!') mark in a given reviews
def excPresence(reviewWords):
    excDictionary = {}
    for i in range(len(reviewWords)):
        isPresent = 0
        for word in reviewWords[i][1]:
            if("!" in word):
                isPresent = 1
                break
        excDictionary[reviewWords[i][0]] = isPresent
    return excDictionary

# Creates a dictionary with ID and total words in a given reviews
def totalWordsCounter(reviewWords):
    featureDictionary = {}
    for i in range(len(reviewWords)):
        wordCount= len(reviewWords[i][1])
        logWordCount = round(math.log(wordCount),3)
        featureDictionary[reviewWords[i][0]] = logWordCount
    return featureDictionary

# Creates a dictionary with ID and positive class for positive reviews
def markPositiveClass(positiveReviewWords):
    featureDictionary = {}
    for i in range(len(positiveReviewWords)):
        featureDictionary[positiveReviewWords[i][0]] = 1
    return featureDictionary

# Creates a dictionary with ID and negative class for negative reviews
def markNegativeClass(reviewWords):
    featureDictionary = {}
    for i in range(len(reviewWords)):
        featureDictionary[reviewWords[i][0]] = 0
    return featureDictionary

# Main
if __name__ == '__main__':
    # Read and clean positive reviews
    trainPositiveReviews = readAndCleanFile(positiveTrainingData)

    # Read and clean negative reviews
    trainNegativeReviews = readAndCleanFile(negativeTrainingData)

    # Read positive words
    positiveWords = positiveNegativeWords(positiveWordsText)

    # Read negative words
    negativeWords = positiveNegativeWords(negativeWordsText)


    # |***********************************TRAIN***********************************|
    # |-------------------------Start For Positive Review-------------------------|
    # Feature 1 : count(positive words)
    positiveReviewCountPositiveReviews = positiveNegativeWordCounter(positiveWords, trainPositiveReviews)

    # Feature 2 : count(negative words)
    negativeReviewCountPositiveReviews = positiveNegativeWordCounter(negativeWords, trainPositiveReviews)

    # Feature 3 : Presence of "no"
    positiveNoDictionary = noPresence(trainPositiveReviews)

    # Feature 4 : count(Pronouns)
    positivePronounDictionary = pronounCount(pronounList, trainPositiveReviews)

    # Feature 5 : Presence of "!"
    positiveExcDictionary = excPresence(trainPositiveReviews)

    # Feature 6 : log(word count of doc)
    positiveTotalWordCountDictionary = totalWordsCounter(trainPositiveReviews)

    # Feature 7 : class (Positive : Negative :: 1 : 0)
    positiveClassDictionary = markPositiveClass(trainPositiveReviews)
    # |-------------------------End For Positive Review-------------------------|


    # |-------------------------Start For Negative Review-------------------------|
    # Feature 1 : count(positive words)
    positiveReviewCountNegativeReviews = positiveNegativeWordCounter(positiveWords, trainNegativeReviews)

    # Feature 2 : count(negative words)
    negativeReviewCountNegativeReviews = positiveNegativeWordCounter(negativeWords, trainNegativeReviews)

    # Feature 3 : Presence of "no"
    negativeNoDictionary = noPresence(trainNegativeReviews)

    # Feature 4 : count(Pronouns)
    negativePronounDictionary = pronounCount(pronounList, trainNegativeReviews)

    # Feature 5 : Presence of "!"
    negativeExcDictionary = excPresence(trainNegativeReviews)

    # Feature 6 : log(word count of doc)
    negativeTotalWordCountDictionary = totalWordsCounter(trainNegativeReviews)

    # Feature 7 : class (Positive : Negative :: 1 : 0)
    negativeClassDictionary = markNegativeClass(trainNegativeReviews)
    # |-------------------------End For Negative Review-------------------------|


    # |-------------------------Start Merging Dictionaries-------------------------|
    finalFeatureDictionary = defaultdict(list)

    for key, value in chain(positiveReviewCountPositiveReviews.items(), negativeReviewCountPositiveReviews.items(),
                            positiveNoDictionary.items(), positivePronounDictionary.items(), positiveExcDictionary.items(),
                           positiveTotalWordCountDictionary.items(), positiveClassDictionary.items(),
                           positiveReviewCountNegativeReviews.items(), negativeReviewCountNegativeReviews.items(),
                            negativeNoDictionary.items(), negativePronounDictionary.items(), negativeExcDictionary.items(),
                           negativeTotalWordCountDictionary.items(), negativeClassDictionary.items()):
        finalFeatureDictionary[key].append(value)
    # |-------------------------End Merging Dictionaries-------------------------|


    # |-------------------------Start Writing Dictionary to CSV-------------------------|
    try:
        with open('processedTrainHotelReviews.csv', 'w') as csv_file:
            writeString = ""
            for key, value in finalFeatureDictionary.items():
                writeString += str(key) + "," + ','.join(str(x) for x in value) +'\n'
            csv_file.write(writeString)
        print("SUCCESS: Feature Vector CSV(processedTrainHotelReviews.csv) for training successfully generated!")
    except:
        print("Error generating Feature Vector CSV for training :/")
    # |-------------------------End Writing Dictionary to CSV-------------------------|



    # |***********************************TEST***********************************|
    # Read and clean test reviews
    testReviews = readAndCleanFile(testReviewData)

    # |-------------------------Start For Test Review-------------------------|
    # Feature 1 : count(positive words)
    positiveReviewCountTestReviews = positiveNegativeWordCounter(positiveWords, testReviews)

    # Feature 2 : count(negative words)
    negativeReviewCountTestReviews = positiveNegativeWordCounter(negativeWords, testReviews)

    # Feature 3 : Presence of "no"
    testNoDictionary = noPresence(testReviews)

    # Feature 4 : count(Pronouns)
    testPronounDictionary = pronounCount(pronounList, testReviews)

    # Feature 5 : Presence of "!"
    testExcDictionary = excPresence(testReviews)

    # Feature 6 : log(word count of doc)
    testTotalWordCountDictionary = totalWordsCounter(testReviews)

    # |-------------------------End For Test Review-------------------------|


    # |-------------------------Start Merging Dictionaries-------------------------|
    finalTestDictionary = defaultdict(list)

    for key, value in chain(positiveReviewCountTestReviews.items(), negativeReviewCountTestReviews.items(),
                            testNoDictionary.items(), testPronounDictionary.items(), testExcDictionary.items(),
                           testTotalWordCountDictionary.items()):
        finalTestDictionary[key].append(value)
    # |-------------------------End Merging Dictionaries-------------------------|


    # |-------------------------Start Writing Dictionary to CSV-------------------------|
    try:
        with open('processedTestHotelReviews.csv', 'w') as csv_file:
            writeString = ""
            for key, value in finalTestDictionary.items():
                writeString += str(key) + "," + ','.join(str(x) for x in value) +'\n'
            csv_file.write(writeString)
        print("SUCCESS: Feature Vector CSV(processedTestHotelReviews.csv) for testing successfully generated!")
    except:
        print("Error generating Feature Vector CSV for testing :/")
    # |-------------------------End Writing Dictionary to CSV-------------------------|
