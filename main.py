from __future__ import division
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")

import sys
sys.path.append('./')
sys.path.append('/home/catle/Tools/horovod/build/lib.linux-x86_64-3.7/')

is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue

from collections import defaultdict
from termcolor import colored, cprint
import tensorflow as tf
# tf.compat.v1.enable_resource_variables() # 4 dynamic shape variable

import numpy as np
import threading
import random
import os
import time
import math
import h5py
import json
import pickle as pkl

from macnetwork.config import config, loadDatasetConfig, parseArgs
from macnetwork.preprocess import Preprocesser, bold, bcolored, writeline, writelist
# from macnetwork.model import MACnet
from model_gqa import MACnet
import horovod.tensorflow as hvd

import global_vars
from global_vars import ATmask, random_seed
global_vars.init()

############################################# loggers #############################################

# Writes log header to file
def logInit():
    with open(config.logFile(), "a+") as outFile:
        writeline(outFile, config.expName)
        headers = ["epoch", "trainAnswerAcc", "trainTaskAcc", "valAnswerAcc", "valTaskAcc",
                   "trainLoss", "valLoss", "trainAnswerLoss", "valAnswerLoss", "trainTaskLoss", "valTaskLoss",
                   "trainLogicLoss", "valLogicLoss"]
        if config.evalTrain:
            headers += ["evalTrainAnswerAcc", "evalTrainTaskAcc",
                        "evalTrainLoss",  "evalTrainAnswerLoss",
                        "evalTrainTaskLoss", "evalTrainLogicLoss"]
        if config.extra:
            if config.evalTrain:
                headers += ["thAnswerAcc", "thTaskAcc",
                            "thLoss", "thAnswerLoss",
                            "thTaskLoss", "thLogicLoss"]
            headers += ["vhAnswerAcc", "vhTaskAcc",
                        "vhLoss", "vhAnswerLoss",
                        "vhTaskLoss", "vhLogicLoss"]
        headers += ["time", "lr"]

        writelist(outFile, headers)
        # lr assumed to be last

'''
Writes log header of constraintWeights to file
'''
def logCWInit():
    with open(config.logFile()+'.cw', "a+") as outFile:
        writeline(outFile, config.expName)
        headers = ["epoch", ]
        # "not yes(p) and not no(p) <-> open_task(p)",
        if 'B' in config.tflRules:
            binary_headers = [
                "verifyGlobalTrue(p) -> yes(q)",
                "verifyAttrTrue(p) -> yes(q)",
                "allDiffTrue(p) -> yes(q)",
                "existTrue(p) -> yes(q)",
                "existNotTrue(p) -> yes(q)",
                "existAndTrue(p) -> yes(q)",
                "existAttrTrue(p) -> yes(q)",
                "existAttrNotTrue(p) -> yes(q)",
                "existAttrOrTrue(p) -> yes(q)",
                "existOrTrue(p) -> yes(q)",
                "existRelTrue(p) -> yes(q)",
                "allSameTrue(p) -> yes(q)",
                "twoDiffTrue(p) -> yes(q)",
                "twoSameTrue(p) -> yes(q)",
                "verifyAttrsTrue(p) -> yes(q)",
                "verifyAttrAndTrue(p) -> yes(q)"
                "verifyGlobalFalse(p) -> no(q)",
                "verifyAttrFalse(p) -> no(q)",
                "allDiffFalse(p) -> no(q)",
                "existFalse(p) -> no(q)",
                "existNotFalse(p) -> no(q)",
                "existAndFalse(p) -> no(q)",
                "existAttrFalse(p) -> no(q)",
                "existAttrNotFalse(p) -> no(q)",
                "existAttrOrFalse(p) -> no(q)",
                "existOrFalse(p) -> no(q)",
                "existRelFalse(p) -> no(q)",
                "allSameFalse(p) -> no(q)",
                "twoDiffFalse(p) -> no(q)",
                "twoSameFalse(p) -> no(q)",
                "verifyAttrsFalse(p) -> no(q)",
                "verifyAttrAndFalse(p) -> no(q)"
                "common(p) -> yes(q) or no(q)",
            ]

            # numCW1 += len(binary_headers)
            headers += binary_headers

        if 'G' in config.tflRules:
            global_headers = [
                "queryGlobal(p) -> global(a)",
                "chooseGlobal(p) -> global(a)",
            ]
            # numCW1 += len(global_headers)
            headers += global_headers

        if 'A' in config.tflRules:
            attr_headers = [
                "queryAttr(p) -> attr(a)",
                "queryAttrObj(p) -> attr(a)",
                "chooseAttr(p) -> attr(a)",
                "common(p) -> attr(a)"]
            # numCW1 += len(attr_headers)
            headers += attr_headers

        if 'O' in config.tflRules:
            obj_headers = [
                "queryObj(p) -> obj(a)",
                "chooseObj(p) -> obj(a)"]
            # numCW1 += len(obj_headers)
            headers += obj_headers

        if 'R' in config.tflRules:
            rel_headers = [
                "queryRel(p) -> rel(a)",
                "chooseRel(p) -> rel(a)"]
            # numCW1 += len(rel_headers)
            headers += rel_headers

        if 'E' in config.tflRules:
            entail_headers = [
            "queryObj(p) -> queryAttrObj(q)",
            "queryAttrObj(p) -> existAttrTrue(q)",
            "existAttrTrue(p) -> existAttrOrTrue(q)",
            "existAttrTrue(p) -> existNotTrue(q)",
            "existAttrTrue(p) -> existAttrNotTrue(q)",
            "existAttrOrTrue(p) -> existNotOrTrue(q)",
            "existNotOrTrue(p) -> existOrTrue(q)",
            "existOrTrue(p) -> existTrue(q)",
            "queryAttrObj(p) -> queryObj(q)",
            "queryNotObj(p) -> existNotTrue(q)",
            "existNotTrue(p) -> existTrue(q)",
            "existAttrNotTrue(p) -> existTrue(q)",
            "existAndTrue(p) -> existTrue(q)",
            "existRelTrue(p) -> existTrue(q)",
            "existRelTrue(p) <-> verifyRelTrue(q)",
            "verifyRelTrue(p) <-> queryRel(q)",
            "verifyRelTrue(p) <-> chooseRel(q)",
            "existOrFalse(p) -> existFalse(q)",
            "existFalse(p) -> existNotFalse(q)",
            "existFalse(p) -> existAttrNotFalse(q)",
            "existFalse(p) -> existRelFalse(q)",
            "existFalse(p) -> existAndFalse(q)",
            "existNotFalse(p) -> existAttrFalse(q)",
            "existAttrNotFalse(p) -> existAttrFalse(q)",
            "existNotOrFalse(p) -> existNotFalse(q)",
            "existNotOrFalse(p) -> existAttrNotFalse(q)",
            "existNotOrFalse(p) -> existAttrOrFalse(q)",
            "existAttrOrFalse(p) -> existAttrFalse(q)",
            "verifyAttrsTrue(p) -> verifyAttrTrue(q)",
            "verifyAttrAndTrue(p)-> verifyAttrTrue(q)",
            "verifyAttrTrue(p) -> queryAttr(q)",
            "queryAttr(p) -> verifyAttrFalse(q)",
            "queryAttr(p) -> chooseAttr(q)",
            "verifyAttrFalse(p) -> verifyAttrAndFalse(q)",
            "verifyAttrAndFalse(p) -> chooseAttr(q)",
            "chooseAttr(p) <-> chooseObj(q)",
            "verifyGlobalTrue(p) -> verifyGlobalFalse(q)",
            "verifyGlobalTrue(p) <-> queryGlobal(q)",
            "verifyGlobalFalse(p) -> chooseGlobal(q)",
            "compare(p) -> common(q)",
            "common(p) -> twoSameTrue(q)",
            "twoSameTrue(p) <-> twoDiffFalse(q)",
            "twoSameFalse(p) <-> twoDiffTrue(q)",
            "allSameTrue(p) <-> allDiffFalse(q)",
            "allSameFalse(p) <-> allDiffTrue(q)"]
            # numCW2 = len(entail_headers)
            headers += entail_headers

        headers += ["time", "lr"]
        # numCW = numCW1+numCW2
        writelist(outFile, headers)

# Writes log record to file
def logRecord(epoch, epochTime, lr, trainRes, evalRes, extraEvalRes):
    with open(config.logFile(), "a+") as outFile:
        record = [epoch, trainRes["answerAcc"], trainRes["taskAcc"],
                  evalRes["val"]["answerAcc"], evalRes["val"]["taskAcc"],
                  trainRes["loss"], evalRes["val"]["loss"],
                  trainRes["answerLoss"], evalRes["val"]["answerLoss"],
                  trainRes["taskLoss"], evalRes["val"]["taskLoss"],
                  trainRes["logicLoss"], evalRes["val"]["logicLoss"]]
        if config.evalTrain:
            record += [evalRes["train"]["answerAcc"],
                       evalRes["train"]["taskAcc"],
                       evalRes["train"]["loss"],
                       evalRes["train"]["answerLoss"],
                       evalRes["train"]["taskLoss"],
                       evalRes["train"]["logicLoss"]]
        if config.extra:
            if config.evalTrain:
                record += [extraEvalRes["train"]["answerAcc"],
                           extraEvalRes["train"]["taskAcc"],
                           extraEvalRes["train"]["loss"],
                           extraEvalRes["train"]["answerLoss"],
                           extraEvalRes["train"]["taskLoss"],
                           extraEvalRes["train"]["logicLoss"]]
            record += [extraEvalRes["val"]["answerAcc"],
                       extraEvalRes["val"]["taskAcc"],
                       extraEvalRes["val"]["loss"],
                       extraEvalRes["val"]["answerLoss"],
                       extraEvalRes["val"]["taskLoss"],
                       extraEvalRes["val"]["logicLoss"]]
        record += [epochTime, lr]

        writelist(outFile, record)

def logCWRecord(epoch, epochTime, lr, trainRes):
    with open(config.logFile()+'.cw', "a+") as outFile:
        record = [epoch]
        # from scipy.special import softmax
        # for weight in softmax(trainRes["constraintWeights"],1)[0,:]:
        for weight in trainRes["constraintWeights"][0,:]:
            record += [weight]
        record += [epochTime, lr]

        writelist(outFile, record)

# Gets last logged epoch and learning rate
def lastLoggedEpoch():
    with open(config.logFile(), "r") as inFile:
        lastLine = list(inFile)[-1+config.restoreEpoch].split(",")
    epoch = int(lastLine[0])
    lr = float(lastLine[-1])
    return epoch, lr

################################## printing, output and analysis ##################################

# Analysis by type
analysisQuestionLims = [(0,18),(19,float("inf"))]
analysisProgramLims = [(0,12),(13,float("inf"))]

toArity = lambda instance: instance["programSeq"][-1].split("_", 1)[0]
toType = lambda instance: "boolean" if (instance["answer"] in ["yes", "no"]) else "open"
toQType = lambda instance: instance["questionType"]
toAType = lambda instance: instance["answerType"]

def fieldLenIsInRange(field):
    return lambda instance, group: \
        (len(instance[field]) >= group[0] and
        len(instance[field]) <= group[1])

# Groups instances based on a key
def grouperKey(toKey):
    def grouper(instances):
        res = defaultdict(list)
        for instance in instances:
            res[toKey(instance)].append(instance)
        return res
    return grouper

# Groups instances according to their match to condition
def grouperCond(groups, isIn):
    def grouper(instances):
        res = {}
        for group in groups:
            res[group] = (instance for instance in instances if isIn(instance, group))
        return res
    return grouper

groupers = {
    "questionLength": grouperCond(analysisQuestionLims, fieldLenIsInRange("question")),
    "programLength": grouperCond(analysisProgramLims, fieldLenIsInRange("programSeq")),
    "arity": grouperKey(toArity),
    "type": grouperKey(toType),
    "qType": grouperKey(toQType),
    "aType": grouperKey(toAType)
}

# Computes average
def avg(instances, field):
    if len(instances) == 0:
        return 0.0
    return sum([(1 if instance["prediction"] == instance["answer"] else 0) for instance in instances]) / len(instances)

# Prints analysis of questions loss and accuracy by their group
def printAnalysis(res):
    if config.analysisType != "":
        print("Analysis by {type}".format(type = config.analysisType))
        groups = groupers[config.analysisType](res["preds"])
        for key in groups:
            instances = groups[key]
            # avgLoss = avg(instances, "loss") avgLoss
            avgAnswerAcc = avg(instances, "answerAcc")
            avgTaskAcc = avg(instances, "taskAcc")
            num = len(instances)
            print("Group {key}: Loss: {loss}, answerAcc: {aacc}, taskAcc: {tacc}, Num: {num}".format(
                key = key, loss = 0, aacc = avgAnswerAcc, tacc = avgTaskAcc, num = num))

# Print results for a tier
def printTierResults(tierName, res, color):
    if res is None:
        return

    print("{tierName} Loss: {loss}, {tierName} minLoss: {minLoss}, \
          answerAccuracy: {aacc}, maxAnswerAcc: {maxAnswerAcc}, \
          taskAccuracy: {tacc}, maxTaskAcc: {maxTaskAcc}".format(tierName = tierName,
        loss = bcolored("{:2.4f}".format(res["loss"]), color),
        minLoss = bcolored("{:2.4f}".format(res["minLoss"]), color),
        aacc = bcolored("{:2.4f}".format(res["answerAcc"]), color),
        tacc = bcolored("{:2.4f}".format(res["taskAcc"]), color),
        maxAnswerAcc = bcolored("{:2.4f}".format(res["maxAnswerAcc"]), color),
        maxTaskAcc = bcolored("{:2.4f}".format(res["maxTaskAcc"]), color)))

    printAnalysis(res)

# Prints dataset results (for several tiers)
def printDatasetResults(trainRes, evalRes, extraEvalRes):
    printTierResults("Training", trainRes, "magenta")
    printTierResults("Training EMA", evalRes["train"], "red")
    printTierResults("Validation", evalRes["val"], "cyan")
    if extraEvalRes != None:
        printTierResults("Extra Training EMA", extraEvalRes["train"], "red")
        printTierResults("Extra Validation", extraEvalRes["val"], "cyan")

# Writes predictions for several tiers
def writePreds(preprocessor, evalRes, extraEvalRes, suffix=""):
    preprocessor.writePreds(evalRes["train"], "train", suffix)
    preprocessor.writePreds(evalRes["val"], "val", suffix)
    preprocessor.writePreds(evalRes["test"], "test", suffix)
    preprocessor.writePreds(extraEvalRes["train"], "train", "H"+suffix)
    preprocessor.writePreds(extraEvalRes["val"], "val", "H"+suffix)
    preprocessor.writePreds(extraEvalRes["test"], "test", "H"+suffix)

def inp(msg):
    if sys.version_info[0] < 3:
        return raw_input(msg)
    else:
        return input(msg)

############################################# session #############################################
# Initializes TF session. Sets GPU memory configuration.
def setSession():
    sessionConfig = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
    if config.allowGrowth:
        sessionConfig.gpu_options.allow_growth = True
    if config.maxMemory < 1.0:
        sessionConfig.gpu_options.per_process_gpu_memory_fraction = config.maxMemory
    if config.hvdspeed:
        sessionConfig.gpu_options.visible_device_list = str(hvd.local_rank())
    return sessionConfig

############################################## savers #############################################
# Initializes savers (standard, optional exponential-moving-average and optional for subset of variables)
def setSavers(model):
    saver = tf.train.Saver(max_to_keep = config.weightsToKeep)

    subsetSaver = None
    if config.saveSubset:
        isRelevant = lambda var: any(s in var.name for s in config.varSubset)
        relevantVars = [var for var in tf.global_variables() if isRelevant(var)]
        subsetSaver = tf.train.Saver(relevantVars, max_to_keep = config.weightsToKeep, allow_empty = True)

    emaSaver = None
    if config.useEMA:
        emaSaver = tf.train.Saver(model.emaDict, max_to_keep = config.weightsToKeep)

    return {
        "saver": saver,
        "subsetSaver": subsetSaver,
        "emaSaver": emaSaver
    }

################################### restore / initialize weights ##################################
# Restores weights of specified / last epoch if on restore mod.
# Otherwise, initializes weights.
def loadWeights(sess, saver, init, bcast=None, feed=None):
    if config.restoreEpoch > 0 or config.restore:
        # restore last epoch only if restoreEpoch isn't set
        if config.restoreEpoch <= 0:
            # restore last logged epoch
            config.restoreEpoch, config.lr = lastLoggedEpoch()

        print(bcolored("Restoring epoch {} and lr {}".format(config.restoreEpoch, config.lr),"cyan"))
        print(bcolored("Restoring weights", "blue"))
        saver.restore(sess, config.weightsFile(config.restoreEpoch))
        epoch = config.restoreEpoch
    else:
        print(bcolored("Initializing weights", "blue"))
        if feed is None: sess.run(init)
        else: sess.run(init,feed_dict=feed)

        if (not config.hvdspeed) or (config.hvdspeed and hvd.rank() == 0):
            logInit()
            logCWInit()
        epoch = 0

    if config.hvdspeed: sess.run(bcast)
    return epoch

###################################### training / evaluation ######################################
# Chooses data to train on (main / extra) data.
def chooseTrainingData(data):
    trainingData = data["main"]["train"]
    alterData = None

    if config.extra:
        if config.trainExtra:
            if config.extraVal:
                trainingData = data["extra"]["val"]
            else:
                trainingData = data["extra"]["train"]
        if config.alterExtra:
            alterData = data["extra"]["train"]

    return trainingData, alterData

#### evaluation
# Runs evaluation on train / val / test datasets.
def runEvaluation(sess, model, data, dataOps, epoch, evalTrain = True, evalTest = False,
                  checkTrain = False, checkVal = False,
                  getPreds = False, getAtt = None, prevRes = None,
                  preprocessor=None, answerDict=None, keyId=None):
    if getAtt is None:
        getAtt = config.getAtt
    res = {"train": None, "val": None, "test": None}

    if data is not None:
        if evalTrain and config.evalTrain:
            res["train"] = runEpoch(sess, model, data["evalTrain"], dataOps, train = False, epoch = epoch, getPreds = getPreds, getAtt = getAtt,
                maxAnswerAcc = prevRes["train"]["maxAnswerAcc"] if prevRes else 0.0,
                maxTaskAcc = prevRes["train"]["maxTaskAcc"] if prevRes else 0.0,
                minLoss = prevRes["train"]["minLoss"] if prevRes else float("inf"),
                preprocessor=preprocessor, answerDict=answerDict, tier='evalTrain', keyId=keyId)


        res["val"] = runEpoch(sess, model, data["val"], dataOps, train = False, epoch = epoch, getPreds = getPreds, getAtt = getAtt,
            maxAnswerAcc = prevRes["val"]["maxAnswerAcc"] if prevRes else 0.0,
            maxTaskAcc = prevRes["val"]["maxTaskAcc"] if prevRes else 0.0,
            minLoss = prevRes["val"]["minLoss"] if prevRes else float("inf"),
            preprocessor=preprocessor, answerDict=answerDict, tier='val', keyId=keyId)


        if evalTest or config.test:
            res["test"] = runEpoch(sess, model, data["test"], dataOps, train = False, epoch = epoch, getPreds = getPreds, getAtt = getAtt,
                maxAnswerAcc = prevRes["test"]["maxAnswerAcc"] if prevRes else 0.0,
                maxTaskAcc = prevRes["test"]["maxTaskAcc"] if prevRes else 0.0,
                minLoss = prevRes["test"]["minLoss"] if prevRes else float("inf"),
                preprocessor=preprocessor, answerDict=answerDict, tier='test', keyId=keyId)

    return res

# Run model check on train / val datasets
def runLogicEval(sess, model, data, dataOps, epoch, evalTrain = True, evalTest = False,
                  checkTrain = False, checkVal = False,
                  getPreds = False, getAtt = None, prevRes = None,
                  preprocessor=None, answerDict=None, keyId=None):
    if getAtt is None:
        getAtt = config.getAtt
    res = {"train": None, "val": None, "test": None}

    if data is not None:
        if checkTrain and config.checkTrain:
            res["train"] = runEpoch(sess, model, data["evalTrain"], dataOps, train = False, epoch = epoch, getPreds = getPreds, getAtt = getAtt,
                maxAnswerAcc = prevRes["train"]["maxAnswerAcc"] if prevRes else 0.0,
                maxTaskAcc = prevRes["train"]["maxTaskAcc"] if prevRes else 0.0,
                minLoss = prevRes["train"]["minLoss"] if prevRes else float("inf"),
                preprocessor=preprocessor, answerDict=answerDict, tier='checkTrain', keyId=keyId)


        if checkVal and config.checkVal:
            res["val"] = runEpoch(sess, model, data["val"], dataOps, train = False, epoch = epoch, getPreds = getPreds, getAtt = getAtt,
                maxAnswerAcc = prevRes["val"]["maxAnswerAcc"] if prevRes else 0.0,
                maxTaskAcc = prevRes["val"]["maxTaskAcc"] if prevRes else 0.0,
                minLoss = prevRes["val"]["minLoss"] if prevRes else float("inf"),
                preprocessor=preprocessor, answerDict=answerDict, tier='checkVal', keyId=keyId)

    return res

## training conditions (comparing current epoch result to prior ones)
def improveEnough(curr, prior, lr):
    prevRes = prior["prev"]["res"]
    currRes = curr["res"]

    if prevRes is None:
        return True

    prevTrainLoss = prevRes["train"]["loss"]
    currTrainLoss = currRes["train"]["loss"]
    lossDiff = prevTrainLoss - currTrainLoss

    ## FOR CLEVR
    if config.dataset == "CLEVR":
        notImprove = ((lossDiff < 0.015 and prevTrainLoss < 0.5 and lr > 0.00002) or \
                      (lossDiff < 0.008 and prevTrainLoss < 0.15 and lr > 0.00001) or \
                      (lossDiff < 0.003 and prevTrainLoss < 0.10 and lr > 0.000005))
                      #(prevTrainLoss < 0.2 and config.lr > 0.000015)
    else:
        notImprove = (lossDiff < 0.02 and lr > 0.00005)
                      #(prevTrainLoss < 0.2 and config.lr > 0.000015)

    return not notImprove

def better(currRes, bestRes):
    return currRes["val"]["answerAcc"] > bestRes["val"]["answerAcc"]

############################################## data ###############################################
#### instances and batching
# Trims sequences based on their max length.
def trim2DVectors(vectors, vectorsLengths):
    maxLength = np.max(vectorsLengths)
    return vectors[:,:maxLength]

# Trims batch based on question length.
def trimData(data):
    data["questions"] = trim2DVectors(data["questions"], data["questionLengths"])
    return data

# Gets batch / bucket size.
def getLength(data):
    return len(data["indices"]) # len(data["instances"])

# Selects the data entries that match the indices.
def selectIndices(data, indices):
    def select(field, indices):
        if type(field) is np.ndarray:
            return field[indices]
        if type(field) is list:
            return [field[i] for i in indices]
        else:
            return field
    selected = {k : select(d, indices) for k,d in data.items()}
    return selected

# Batches data into a a list of batches of batchSize.
# Shuffles the data by default.
def getBatches(data, batchSize = None, shuffle = True):
    batches = []

    dataLen = getLength(data)
    if batchSize is None or batchSize > dataLen:
        batchSize = dataLen

    indices = np.arange(dataLen)
    if shuffle:
        np.random.shuffle(indices)

    for batchStart in range(0, dataLen, batchSize):
        batchIndices = indices[batchStart : batchStart + batchSize]
        # if len(batchIndices) == batchSize?
        if len(batchIndices) >= config.gpusNum:
            batch = selectIndices(data, batchIndices)
            batches.append(batch)
            # batchesIndices.append((data, batchIndices))

    return batches

# Organize data into a a list of families
# Shuffles the family data by default.
def preprocessFamily(questionData, tier, imageIndex, preprocessor, predicateDict):
    """
    Create a question family
    """
    noQuestion = len(questionData)
    instances = []
    for i, anItem in enumerate(questionData):
        imageId = anItem['data']['imageId']
        questionStr = anItem['data']['question']
        imageInfo = imageIndex[imageId]
        imageId = {"group": tier, "id": imageId,
                   "idx": imageInfo["index"]}  # int(imageId)

        taskLabel = predicateDict[anItem['task']] - 1845 # use config rather hardcode
        question = preprocessor.encodeQuestionStr(questionStr)
        instances.append({
            "questionStr": questionStr,
            "question": question,
            "answer": anItem['data']['answer'],
            "task": taskLabel,
            "imageId": imageId,
            "objectsNum": imageInfo['objectsNum'] if 'objectsNum' in imageInfo else 0,
            "tier": tier,
            "index": i
        })

    familyData = preprocessor.vectorizeData(instances)
    familyData['questionsNum'] = noQuestion
    # padding questions with zeros until size 28
    familyData['questions'] = np.pad(familyData['questions'],
               ((0,0),(0,28-familyData['questions'].shape[1])))
    return familyData

def getFamilies(imageIndex, tier, preprocessor, predicateDict, keyId=None, shuffle = True):
    families = []

    # replace the path
    if config.hvdspeed:
        if config.familyStyle=='unique':
            familiesDataPath = \
                '/data/catle/Projects/lyrics_tfnorm/data/{}_balanced_all_tasks_entailed_unique_raw_{}_sorted.pkl'\
                .format(tier,hvd.rank()) # rotate data partitions # _unique
                # .format(tier,(hvd.rank()+epoch-1) % hvd.size()) # rotate data partitions
        elif config.familyStyle=='ununique':
            familiesDataPath = \
                '/data/catle/Projects/lyrics_tfnorm/data/{}_balanced_all_tasks_entailed_raw_{}_sorted.pkl'\
                .format(tier,hvd.rank()) # rotate data partitions # _unique

    else:
        familiesDataPath = \
            '/data/catle/Projects/lyrics_tfnorm/data/{}_balanced_all_tasks_entailed_unique_raw_0_sorted.pkl'\
            .format(tier)
    with open(familiesDataPath, 'rb') as fh:
        questionData = pkl.load(fh)

    if keyId is None:
        # family batch training - option 1
        for keyId, questionFamilies in questionData['question_dict'].items():
            if int(keyId) >= config.minNoQuest and int(keyId) <= config.batchSize:
                tmp_list = []
                for questionFamily in questionFamilies:
                    familyData = preprocessFamily(
                        questionFamily, tier, imageIndex, preprocessor, predicateDict) # tier=train
                    tmp_list.append(familyData)

                if config.datapct < 100:
                # select 1% of data
                    datapct = float(config.datapct)/100.0
                    tmp_list = tmp_list[:int(np.ceil(len(tmp_list)*datapct))]

                random.shuffle(tmp_list); # ... randomly
                families += tmp_list
    else:
        # curriculum training
        tmp_list = []
        questionFamilies = questionData['question_dict'][keyId]
        for questionFamily in questionFamilies:
            familyData = preprocessFamily(
                questionFamily, tier, imageIndex, preprocessor, predicateDict) # tier=train
            tmp_list.append(familyData)

        if config.datapct < 100:
        # select 1% of data
            datapct = float(config.datapct)/100.0
            tmp_list = tmp_list[:int(np.ceil(len(tmp_list)*datapct))]

        random.shuffle(tmp_list);
        families += tmp_list

    return families


#### image batches
# Opens image files.
def openImageFiles(images):
    for group in images:
        images[group]["imagesFile"] = h5py.File(images[group]["imagesFilename"], "r")
        if config.dataset != "CLEVR":
            with open(images[group]["imgsInfoFilename"], "r") as file:
                images[group]["imgsInfo"] = json.load(file)

# Closes image files.
def closeImageFiles(images):
    for group in images:
        images[group]["imagesFile"].close()

# Loads an images from file for a given data batch.
def loadImageBatch(images, batch):
    imagesGroup = lambda imageId: images[imageId["group"]]
    toFile = lambda imageId: imagesGroup(imageId)["imagesFile"]
    toInfo = lambda imageId: imagesGroup(imageId)["imgsInfo"][str(imageId["id"])]

    if config.imageObjects:
        imageBatch = np.zeros((len(batch["imageIds"]), config.imageDims[0], config.imageDims[1]))
        for i, imageId in enumerate(batch["imageIds"]):
            numObjects = toInfo(imageId)["objectsNum"]

            imageBatch[i, 0:numObjects] = toFile(imageId)["features"][imageId["idx"], 0:numObjects]

    else:
        imageBatch = np.stack([toFile(imageId)["features"][imageId["idx"]]
            for imageId in batch["imageIds"]], axis = 0)

        config.imageDims = imageBatch.shape[1:]

    ret = {"images": imageBatch, "imageIds": batch["imageIds"]}

    return ret

# Loads images for several num batches in the batches list from start index.
def loadImageBatches(images, batches, start, num):
    batches = batches[start: start + num]
    return [loadImageBatch(images, batch) for batch in batches]

#### data alternation
# Alternates main training batches with extra data.
def alternateData(batches, alterData, dataLen):
    alterData = alterData["data"][0] # data isn't bucketed for altered data

    # computes number of repetitions
    needed = math.ceil(len(batches) / config.alterNum)
    print(bold("Extra batches needed: %d") % needed)
    perData = math.ceil(getLength(alterData) / config.batchSize)
    print(bold("Batches per extra data: %d") % perData)
    repetitions = math.ceil(needed / perData)
    print(bold("reps: %d") % repetitions)

    # make alternate batches
    alterBatches = []
    for _ in range(repetitions):
        repBatches = getBatches(alterData, batchSize = config.batchSize)
        random.shuffle(repBatches)
        alterBatches += repBatches
    print(bold("Batches num: %d") + len(alterBatches))

    # alternate data with extra data
    curr = len(batches) - 1
    for alterBatch in alterBatches:
        if curr < 0:
            break
        batches.insert(curr, alterBatch)
        dataLen += getLength(alterBatch)
        curr -= config.alterNum

    return batches, dataLen

############################################ threading ############################################

imagesQueue = queue.Queue(maxsize = 20) # config.tasksNum
inQueue = queue.Queue(maxsize = 1)
outQueue = queue.Queue(maxsize = 1)

def loaderRun(images, batches):
    batchNum = 0

    while batchNum < len(batches):
        nextItem = loadImageBatches(images, batches, batchNum, config.taskSize)
        assert len(nextItem) == min(config.taskSize, len(batches) - batchNum)
        batchNum += config.taskSize
        imagesQueue.put(nextItem)

########################################## stats tracking #########################################
# Computes exponential moving average.
def emaAvg(avg, value):
    if avg is None:
        return value
    emaRate = 0.98
    return avg * emaRate + value * (1 - emaRate)

# Initializes training statistics.
def initStats():
    return {
        "totalBatches": 0,
        "totalAnswerData": 0,
        "totalTaskData": 0,
        "totalLoss": 0.0,
        "totalAnswerLoss": 0.0,
        "totalTaskLoss": 0.0,
        "totalLogicLoss": 0.0,
        "totalAnswerCorrect": 0,
        "totalTaskCorrect": 0,
        "totalAnswerAcc": 0.0,
        "totalTaskAcc": 0.0,
        "totalConstraintWeights": np.zeros((1,global_vars.numCW)),
        "loss": 0.0,
        "answerLoss": 0.0,
        "taskLoss": 0.0,
        "logicLoss": 0.0,
        "constraintWeights": np.zeros((1,global_vars.numCW)),
        "answerAcc": 0.0,
        "taskAcc": 0.0,
        "emaLoss": None,
        "emaAnswerLoss": None,
        "emaTaskLoss": None,
        "emaLogicLoss": None,
        "emaAnswerAcc": None,
        "emaTaskAcc": None,
        "emaConstraintWeights": np.zeros((1,global_vars.numCW))
    }

# Updates statistics with training results of a batch
def updateStats(stats, res):
    stats["totalBatches"] += 1
    stats["totalAnswerData"] += res["batchSize"]
    if res["numQuestions"] is not None:
        stats["totalTaskData"] += res["numQuestions"]

    stats["totalLoss"] += res["loss"] / res["batchSize"]
    stats["totalAnswerLoss"] += res["answerLoss"] / res["batchSize"]

    if config.tflSS or config.tflFK:
        stats["totalTaskLoss"] += res["taskLoss"] / res["numQuestions"]
        stats["totalLogicLoss"] += res["logicLoss"] / res["numQuestions"]

    stats["totalAnswerCorrect"] += res["answerCorrectNum"]
    stats["totalAnswerAcc"] += res["answerAcc"]

    if config.tflSS or config.tflFK:
        stats["totalTaskCorrect"] += res["taskCorrectNum"]
        stats["totalTaskAcc"] += res["taskAcc"]
        if config.tflWeightMode=='sw':
            stats["totalConstraintWeights"] += res["constraintWeights"]
        else:
            stats["totalConstraintWeights"] = np.zeros((1,global_vars.numCW))
    else:
        stats["totalTaskCorrect"] = -1
        stats["totalTaskAcc"] = -1.0
        stats["totalConstraintWeights"] = np.zeros((1,global_vars.numCW))

    stats["loss"] = stats["totalLoss"] / stats["totalBatches"]
    stats["answerLoss"] = stats["totalAnswerLoss"] / stats["totalBatches"]

    if config.tflSS or config.tflFK:
        stats["taskLoss"] = stats["totalTaskLoss"] / stats["totalBatches"]
        stats["logicLoss"] = stats["totalLogicLoss"] / stats["totalBatches"]
        stats["constraintWeights"] = stats["totalConstraintWeights"] / stats["totalBatches"]
    else:
        stats["taskLoss"] = -1.0
        stats["logicLoss"] = -1.0
        stats["constraintWeights"] = np.zeros((1,global_vars.numCW))

    stats["answerAcc"] = stats["totalAnswerCorrect"] / stats["totalAnswerData"]

    if config.tflSS or config.tflFK:
        stats["taskAcc"] = stats["totalTaskCorrect"] / stats["totalTaskData"]

    stats["emaLoss"] = emaAvg(stats["emaLoss"], res["loss"])
    stats["emaAnswerLoss"] = emaAvg(stats["emaAnswerLoss"], res["answerLoss"])
    stats["emaAnswerAcc"] = emaAvg(stats["emaAnswerAcc"], res["answerAcc"])

    if config.tflSS or config.tflFK:
        stats["emaTaskLoss"] = emaAvg(stats["emaTaskLoss"], res["taskLoss"])
        stats["emaLogicLoss"] = emaAvg(stats["emaLogicLoss"], res["logicLoss"])
        stats["emaTaskAcc"] = emaAvg(stats["emaTaskAcc"], res["taskAcc"])
        if config.tflWeightMode=='sw':
            stats["emaConstraintWeights"] = emaAvg(stats["emaConstraintWeights"], res["constraintWeights"])
        else:
            stats["emaConstraintWeights"] = np.zeros((1,global_vars.numCW))
    else:
        stats["emaTaskLoss"] = -1.0
        stats["emaLogicLoss"] = -1.0
        stats["emaTaskAcc"] = -1.0
        stats["emaConstraintWeights"] = np.zeros((1,global_vars.numCW))

    return stats

# Translates training statistics into a string to print
def statsToStr(stats, res, epoch, batchNum, dataLen, startTime):
    formatStr = "eb {epoch},{batchNum} ({dataProcessed} / {dataLen:5d}), " + \
                "t = {time} ({loadTime:2.2f}+{trainTime:2.2f}), lr {lr}, g = {gradNorm}\t\t\t\n" + \
                "l = {loss}, al = {aloss}, tl = {tloss}, ll = {lloss}," + \
                "emL = {emaLoss}, emAL = {emaAnswerLoss}, emTL = {emaTaskLoss}, emLL = {emaLogicLoss}\t\t\t\n" + \
                "aa = {aacc}, ta = {tacc}, " + \
                "avAA = {avgAnswerAcc}, avTA = {avgTaskAcc}, " + \
                "emAA = {emaAnswerAcc}, emTA = {emaTaskAcc}; " + \
                "{expname}\n"

    s_epoch = bcolored("{:2d}".format(epoch),"green")
    s_batchNum = "{:3d}".format(batchNum)
    s_dataProcessed = bcolored("{:5d}".format(stats["totalBatches"]),"green")
    s_dataLen = dataLen
    s_time = bcolored("{:2.2f}".format(time.time() - startTime),"green")
    s_loadTime = res["readTime"]
    s_trainTime = res["trainTime"]
    s_lr = bold(bcolored(config.lr,"green"))
    s_gradNorm = bcolored("{:2.4}".format(res["gradNorm"]),"green")

    s_loss = bcolored("{:2.1f}".format(res["loss"]), "blue")
    s_answerLoss = bcolored("{:2.1f}".format(res["answerLoss"]), "blue")
    s_taskLoss = bcolored("{:2.1f}".format(res["taskLoss"]), "blue")
    s_logicLoss = bcolored("{:2.1f}".format(res["logicLoss"]), "blue")
    s_emaLoss = bcolored("{:2.1f}".format(stats["emaLoss"]), "blue")
    s_emaAnswerLoss = bcolored("{:2.1f}".format(stats["emaAnswerLoss"]), "blue")
    s_emaTaskLoss = bcolored("{:2.1f}".format(stats["emaTaskLoss"]), "blue")
    s_emaLogicLoss = bcolored("{:2.1f}".format(stats["emaLogicLoss"]), "blue")

    s_answerAcc = bcolored("{:2.4f}".format(res["answerAcc"]),"red")
    s_taskAcc = bcolored("{:2.4f}".format(res["taskAcc"]),"red")
    s_avgAnswerAcc = bcolored("{:2.4f}".format(stats["answerAcc"]),"red")
    s_avgTaskAcc = bcolored("{:2.4f}".format(stats["taskAcc"]),"red")
    s_emaAnswerAcc = bcolored("{:2.4f}".format(stats["emaAnswerAcc"]),"red")
    s_emaTaskAcc = bcolored("{:2.4f}".format(stats["emaTaskAcc"]),"red")

    s_expname = config.expName

    return formatStr.format(epoch = s_epoch, batchNum = s_batchNum, dataProcessed = s_dataProcessed,
                            dataLen = s_dataLen, time = s_time, loadTime = s_loadTime,
                            trainTime = s_trainTime, lr = s_lr,
                            loss = s_loss, aloss = s_answerLoss, tloss = s_taskLoss, lloss = s_logicLoss,
                            aacc = s_answerAcc, tacc = s_taskAcc,
                            avgAnswerAcc = s_avgAnswerAcc, avgTaskAcc = s_avgTaskAcc,
                            gradNorm = s_gradNorm, emaLoss = s_emaLoss,
                            emaAnswerLoss = s_emaAnswerLoss, emaTaskLoss = s_emaTaskLoss,
                            emaLogicLoss = s_emaLogicLoss, emaAnswerAcc = s_emaAnswerAcc,
                            emaTaskAcc = s_emaTaskAcc, expname = s_expname)

'''
Runs an epoch with model and session over the data.
1. Batches the data and optionally mix it with the extra alterData.
2. Start worker threads to load images in parallel to training.
3. Runs model for each batch, and gets results (e.g. loss,  accuracy).
4. Updates and prints statistics based on batch results.
5. Once in a while (every config.saveEvery), save weights.

Args:
    sess: TF session to run with.

    model: model to process data. Has runBatch method that process a given batch.
    (See model.py for further details).

    data: data to use for training/evaluation.

    epoch: epoch number.

    saver: TF saver to save weights

    calle: a method to call every number of iterations (config.calleEvery)

    alterData: extra data to mix with main data while training.

    getAtt: True to return model attentions.
'''
def runEpoch(sess, model, data, dataOps, train, epoch, saver = None, calle = None,
    alterData = None, getPreds = False, getAtt = False, maxAnswerAcc = 0.0, maxTaskAcc = 0.0,
    minLoss = float("Inf"),preprocessor=None, answerDict=None, tier='train', keyId=None):

    dataLen = sum(getLength(bucket) for bucket in data["data"])
    if dataLen == 0:
        return {
            "loss": 0,
            "answerLoss": 0,
            "taskLoss": 0,
            "logicLoss": 0,
            "answerAcc": 0,
            "taskAcc": 0,
            "maxAnswerAcc": 0,
            "maxTaskAcc": 0,
            "minLoss": 0,
            "answerPreds": [],
            "taskPreds": [],
            "constraintWeights": np.zeros((global_vars.numCW,))
            }

    # initialization
    startTime0 = time.time()

    stats = initStats()
    answerPreds = []
    taskPreds = []

    # open image files
    openImageFiles(data["images"])

    batches = []
    # tier == 'train' and
    if tier == 'train' and config.batchStyle == 'family': # (config.tflSS or config.tflFK):
        # make batches and randomize
        assert tier in ['train','val','test'], 'runEpoch on either train, val or test tier'
        with open(config.imgsInfoFile(tier), "r") as fh:
            imageIndex = json.load(fh)
        batches = getFamilies(imageIndex, tier, preprocessor, answerDict.sym2id, keyId, True)
        # batches.reverse() # reverse
        dataLen = len(batches)
        print('#{}'.format(dataLen))
    elif tier == 'train' and config.batchStyle == 'hybrid':
        # load families
        with open(config.imgsInfoFile(tier), "r") as fh:
            imageIndex = json.load(fh)
        # families = getFamilies(imageIndex, tier, preprocessor, answerDict.sym2id, keyId, True)
        families = getFamilies(imageIndex, tier, preprocessor, answerDict.sym2id, None, True)

        # load batches
        buckets = data["data"]
        dataLen = sum(getLength(bucket) for bucket in buckets)

        for bucket in buckets:
            batches += getBatches(bucket, batchSize = config.batchSize)

        partition_size = int(np.ceil(float(len(batches)) / float(hvd.size())))
        '''
        datapct = float(config.datapct)/100.0
        partition_size = int(np.ceil(float(len(batches))
                                     / float(hvd.size())*datapct))
        '''

        if (hvd.rank() + 1) == hvd.size():
            start_idx = int(len(batches) - partition_size)
            end_idx = int(len(batches))
        else:
            start_idx = int(hvd.rank()*partition_size)
            end_idx = int((hvd.rank()+1)*partition_size)
        print('[{},{}] - #{}'.format(start_idx,end_idx,end_idx-start_idx))
        batches = batches[start_idx:end_idx]
        dataLen = len(batches)

        # randomly select families from repo
        random_families = random.choices(families,k=dataLen)
        # replace random samples with family samples
        for i in range(0,dataLen): # for each batch
            questionsNum = random_families[i]['questionsNum']
            for k, v in random_families[i].items(): # inject family data
                if k in batches[i].keys(): # if key exist
                    batches[i][k][:questionsNum] = \
                                random_families[i][k] # at first few samples
                else:
                    '''
                    if k == 'tasks':
                        batches[i][k] = np.pad(random_families[i][k],
                                    ((0,config.batchSize-questionsNum)))
                    else:
                    '''
                    batches[i][k] = random_families[i][k]
    elif tier in ['checkTrain','checkVal']:
        # prepare the data for model checking on train/val data partition
        assert tier in ['checkTrain','checkVal'], 'modelCheck on either train or val tier'
        if tier == 'checkVal': tier = 'val'
        elif tier == 'checkTrain': tier = 'train'

        with open(config.imgsInfoFile(tier), "r") as fh:
            imageIndex = json.load(fh)
        batches = getFamilies(imageIndex, tier, preprocessor, answerDict.sym2id, keyId, True)
        # batches.reverse() # reverse
        dataLen = len(batches)
        print('#{}'.format(dataLen))
    else:
        ## prepare batches
        # if keyId is not None: config.batchSize=np.int32(keyId)
        buckets = data["data"]
        dataLen = sum(getLength(bucket) for bucket in buckets)

        for bucket in buckets:
            batches += getBatches(bucket, batchSize = config.batchSize)

        """
        start_idx = int(np.floor(hvd.rank()/hvd.size()*len(batches)))
        end_idx = int(np.floor((hvd.rank()+1)/hvd.size()*len(batches)))
        batches = batches[start_idx:end_idx]
        dataLen = len(batches)
        """
        partition_size = int(np.ceil(float(len(batches)) / float(hvd.size())))
        '''
        datapct = float(config.datapct)/100.0
        partition_size = int(np.ceil(float(len(batches))
                                     / float(hvd.size())*datapct))
        '''

        if (hvd.rank() + 1) == hvd.size():
            start_idx = int(len(batches) - partition_size)
            end_idx = int(len(batches))
        else:
            start_idx = int(hvd.rank()*partition_size)
            end_idx = int((hvd.rank()+1)*partition_size)
        print('[{},{}] - #{}'.format(start_idx,end_idx,end_idx-start_idx))
        batches = batches[start_idx:end_idx]
        dataLen = len(batches)
        '''
        if tier == 'train' and config.datapct < 100:
            datapct = float(config.datapct)/100.0
            batches = batches[0:int(np.ceil(dataLen*datapct))] # use 10%
        '''
        # random.shuffle(batches)
        # if train: batches = batches[0:408] # trick for equalize experiment

    # alternate with extra data
    if train and alterData is not None:
        batches, dataLen = alternateData(batches, alterData, dataLen)

    # for batchNum, batch in enumerate(batches):
    batchNum = 0
    batchesNum = len(batches)
    virtualBatchSize = 0
    vBatchSize = 0

    while batchNum < batchesNum:
        try:
            startTime = time.time()

            # prepare batch
            progress=float(batchNum+(epoch-1)*batchesNum)/float(batchesNum*config.epochs)
            # progress=float(batchNum)/float(batchesNum)

            batch = batches[batchNum]
            batch = trimData(batch)

            # load images batch
            imagesBatch = loadImageBatch(data["images"], batch)
            for i, imageId in enumerate(batch["imageIds"]):
                assert imageId == imagesBatch["imageIds"][i]

            # run batch
            if train and config.hvdoptim == 'adasum':
                virtualBatchSize += batch["questions"].shape[0]
                if virtualBatchSize < config.batchSize:
                    vBatchSize = 0
                    getUpdate = False
                else:
                    # print('Virtual batch size: ', virtualBatchSize)
                    vBatchSize = virtualBatchSize
                    virtualBatchSize = 0
                    getUpdate = True
            else:
                vBatchSize = 0
                getUpdate = True

            res = model.runBatch(sess, batch, imagesBatch, train,
                                 getUpdate=getUpdate, vBatchSize=vBatchSize,
                                 getPreds=getPreds, getAtt=getAtt, progress=progress)

            # update stats
            stats = updateStats(stats, res) # , batch
            answerPreds += res["answerPreds"]

            # update states and save checkpoint at rank 0 GPU only
            if config.hvdspeed and hvd.rank() != 0: continue

            sys.stdout.write(statsToStr(stats, res, epoch, batchNum, dataLen, startTime))
            if batchNum < batchesNum-1: sys.stdout.write("\033[F\033[F\033[F")
            sys.stdout.flush()
            sys.stderr.flush()

            # save weights
            if saver is not None:
                if batchNum > 0 and batchNum % config.saveEvery == 0:
                    print("\n\n")
                    print(bold("saving weights"))
                    saver.save(sess, config.weightsFile(epoch))

            # calle
            if calle is not None:
                if batchNum > 0 and batchNum % config.calleEvery == 0:
                    calle()

            batchNum += 1

        except tf.errors.OutOfRangeError:
            break

    sys.stdout.write("\r")
    sys.stdout.flush()

    closeImageFiles(data["images"])

    return {"loss": stats["loss"],
            "answerLoss": stats["answerLoss"],
            "taskLoss": stats["taskLoss"],
            "logicLoss": stats["logicLoss"],
            "answerAcc": stats["answerAcc"],
            "maxAnswerAcc": max(stats["answerAcc"], maxAnswerAcc),
            "taskAcc": stats["taskAcc"],
            "maxTaskAcc": max(stats["taskAcc"], maxTaskAcc),
            "minLoss": min(stats["loss"], minLoss),
            "answerPreds": answerPreds,
            "taskPreds": taskPreds,
            "constraintWeights": stats["constraintWeights"]
            }

'''
Trains/evaluates the model:
1. Set GPU configurations.
2. Preprocess data: reads from datasets, and convert into numpy arrays.
3. Builds the TF computational graph for the MAC model.
4. Starts a session and initialize / restores weights.
5. If config.train is True, trains the model for number of epochs:
    a. Trains the model on training data
    b. Evaluates the model on training / validation data, optionally with
       exponential-moving-average weights.
    c. Prints and logs statistics, and optionally saves model predictions.
    d. Optionally reduces learning rate if losses / accuracies don't improve,
       and applies early stopping.
6. If config.test is True, runs a final evaluation on the dataset and print
   final results!
'''
def main():
    # init horovod
    if config.hvdspeed: hvd.init()

    if (not config.hvdspeed) or (config.hvdspeed and hvd.rank() == 0):
        with open(config.configFile(), "a+") as outFile:
            json.dump(vars(config), outFile)

    # set gpus
    if config.gpus != "":
        config.gpusNum = len(config.gpus.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    tf.logging.set_verbosity(tf.logging.ERROR)

    # process data
    print(bold("Preprocess data..."))
    start = time.time()
    preprocessor = Preprocesser()
    data, embeddings, answerDict, questionDict = preprocessor.preprocessData()
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    nextElement = None
    dataOps = None

    # build model
    print(bold("Building model..."))
    start = time.time()

    model = MACnet(embeddings, answerDict, questionDict, nextElement)
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    # initializer
    '''
    if config.useDLM:
        import pickle as pkl
        fh = open('./sample_batch.pkl','rb')
        data, images = pkl.load(fh)
        feed = model.createFeedDict(data, images, True)
    '''

    init = tf.global_variables_initializer()
    if config.hvdspeed:
        bcast = hvd.broadcast_global_variables(0)
    else:
        bcast = tf.no_op()

    # savers
    savers = setSavers(model)
    saver, emaSaver = savers["saver"], savers["emaSaver"]

    # sessionConfig
    sessionConfig = setSession()

    with tf.Session(config = sessionConfig) as sess:

        # ensure no more ops are added after model is built
        sess.graph.finalize()

        # restore / initialize weights, initialize epoch variable
        epoch = loadWeights(sess, saver, init, bcast)

        trainRes, evalRes = None, None

        if config.train:
            start0 = time.time()

            bestEpoch = epoch
            bestRes = None
            prevRes = None

            # epoch in [restored + 1, epochs]
            for epoch in range(config.restoreEpoch + 1, config.epochs + 1):

                # balanced_set [6,25]
                # curriculum training
                # not unique
                '''
                keyStr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                          '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                          '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                          '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
                          '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61',
                          '62', '63', '67', '69', '71', '76', '80'] # 68 ind
                keyStr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                          '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
                          '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
                          '38'] # 38 ind with more than 10 samples per shaded
                '''
                # unique
                keyStr = ['1','2','3','4','5','6']

                keyStr = [ i for i in keyStr if np.int32(i) >= config.minNoQuest ]
                keyId = keyStr[int(np.floor(np.float((epoch-1)*len(keyStr))/np.float(config.epochs)))]

                # normal batch training
                # keyId = None

                print(bcolored("Training epoch {}...".format(epoch), "green"))
                start = time.time()

                # train
                # calle = lambda: model.runEpoch(), collectRuntimeStats, writer
                trainingData, alterData = chooseTrainingData(data)
                trainRes = runEpoch(sess, model, trainingData, dataOps, train = True, epoch = epoch,
                    saver = saver, alterData = alterData,
                    maxAnswerAcc = trainRes["maxAnswerAcc"] if trainRes else 0.0,
                    minLoss = trainRes["minLoss"] if trainRes else float("inf"),
                    preprocessor=preprocessor, answerDict=answerDict, tier="train", keyId=keyId)

                if (not config.hvdspeed) or (config.hvdspeed and hvd.rank() == 0):
                    # save weights
                    saver.save(sess, config.weightsFile(epoch))
                    if config.saveSubset:
                        subsetSaver.save(sess, config.subsetWeightsFile(epoch))

                # load EMA weights
                if config.useEMA:
                    print(bold("Restoring EMA weights"))
                    emaSaver.restore(sess, config.weightsFile(epoch))

                # evaluation
                getPreds = config.getPreds or (config.analysisType != "")

                print(bcolored("Evaluating epoch {}...".format(epoch), "green"))
                evalRes = runEvaluation(sess, model, data["main"],
                    dataOps, epoch, getPreds = getPreds, prevRes = evalRes,
                    preprocessor=preprocessor, answerDict=answerDict)
                extraEvalRes = runEvaluation(sess, model, data["extra"],
                    dataOps, epoch, evalTrain = not config.extraVal, getPreds = getPreds,
                    preprocessor=preprocessor, answerDict=answerDict)

                # restore standard weights
                if config.useEMA:
                    print(bold("Restoring standard weights"))
                    saver.restore(sess, config.weightsFile(epoch))

                if (not config.hvdspeed) or (config.hvdspeed and hvd.rank() == 0):
                    print("")

                    epochTime = time.time() - start
                    print("took {:.2f} seconds".format(epochTime))

                    # print results
                    printDatasetResults(trainRes, evalRes, extraEvalRes)

                    # stores predictions and optionally attention maps
                    if config.getPreds:
                        print(bcolored("Writing predictions...", "white"))
                        writePreds(preprocessor, evalRes, extraEvalRes)

                    logRecord(epoch, epochTime, config.lr, trainRes, evalRes, extraEvalRes)
                    logCWRecord(epoch, epochTime, config.lr, trainRes)

                    # update best result
                    # compute curr and prior
                    currRes = {"train": trainRes, "val": evalRes["val"]}
                    curr = {"res": currRes, "epoch": epoch}

                    if bestRes is None or better(currRes, bestRes):
                        bestRes = currRes
                        bestEpoch = epoch

                    prior = {"best": {"res": bestRes, "epoch": bestEpoch},
                            "prev": {"res": prevRes, "epoch": epoch - 1}}

                    # lr reducing
                    if config.lrReduce:
                        if not improveEnough(curr, prior, config.lr):
                            config.lr *= config.lrDecayRate
                            print(colored("Reducing LR to {}".format(config.lr), "red"))

                    # early stopping
                    if config.earlyStopping > 0:
                        if epoch - bestEpoch > config.earlyStopping:
                            break

                    # update previous result
                    prevRes = currRes

            # reduce epoch back to the last one we trained on
            epoch -= 1
            print("Training took {:.2f} seconds ({:} epochs)".format(time.time() - start0,
                epoch - config.restoreEpoch))

        if config.finalTest :
            print("Testing on epoch {}...".format(epoch))

            start = time.time()
            if epoch > 0:
                if config.useEMA:
                    emaSaver.restore(sess, config.weightsFile(epoch))
                else:
                    saver.restore(sess, config.weightsFile(epoch))

            evalRes = runEvaluation(sess, model, data["main"], dataOps, epoch,
                                    evalTest = False, getPreds = True,
                                    preprocessor=preprocessor, answerDict=answerDict)
            extraEvalRes = runEvaluation(sess, model, data["extra"], dataOps, epoch,
                evalTrain = not config.extraVal, evalTest = False, getPreds = True,
                preprocessor=preprocessor, answerDict=answerDict)

            if (not config.hvdspeed) or (config.hvdspeed and hvd.rank() == 0):
                print("took {:.2f} seconds".format(time.time() - start))
                printDatasetResults(trainRes, evalRes, extraEvalRes)

                print("Writing predictions...")
                writePreds(preprocessor, evalRes, extraEvalRes)

        if config.checkTrain or config.checkVal:
            print("Checking model on epoch {}...".format(epoch))

            start = time.time()
            if epoch > 0:
                if config.useEMA:
                    emaSaver.restore(sess, config.weightsFile(epoch))
                else:
                    saver.restore(sess, config.weightsFile(epoch))

            evalRes = runLogicEval(sess, model, data["main"], dataOps, epoch,
                                    checkTrain = config.checkTrain, checkVal = config.checkVal,
                                    preprocessor=preprocessor, answerDict=answerDict)
            extraEvalRes = runLogicEval(sess, model, data["extra"], dataOps, epoch,
                evalTrain = not config.extraVal, evalTest = False, getPreds = True,
                preprocessor=preprocessor, answerDict=answerDict)

        if config.interactive:
            if epoch > 0:
                if config.useEMA:
                    emaSaver.restore(sess, config.weightsFile(epoch))
                else:
                    saver.restore(sess, config.weightsFile(epoch))

            tier = config.interactiveTier
            images = data["main"][tier]["images"]

            imgsInfoFilename = config.imgsInfoFile(tier)
            with open(imgsInfoFilename, "r") as file:
                imageIndex = json.load(file)

            openImageFiles(images)

            resInter = {"preds": []}

            while True:

                text = inp("Enter <imageId>_<question>\n")
                if len(text) == 0:
                    break

                imageId, questionStr = text.split("_")

                imageInfo = imageIndex[imageId]

                imageId = {"group": tier, "id": imageId, "idx": imageInfo["idx"]} # int(imageId)
                question = preprocessor.encodeQuestionStr(questionStr)
                instance = {
                    "questionStr": questionStr,
                    "question": question,
                    "answer": "yes", # Dummy answer
                    "answerFreq": ["yes"], # Dummy answer
                    "imageId": imageId,
                    "tier": tier,
                    "index": 0
                }

                if config.imageObjects:
                    instance["objectsNum"] = imageInfo["objectsNum"]

                print(instance)

                datum = preprocessor.vectorizeData([instance])
                image = loadImageBatch(images, {"imageIds": [imageId]})
                res = model.runBatch(sess, datum, image, train = False, getPreds = True, getAtt = True)
                resInter["preds"].append(instance)

                print(instance["prediction"])

            if config.getPreds:
                print(bcolored("Writing predictions...", "white"))
                preprocessor.writePreds(resInter, "interactive".format())

            closeImageFiles(images)

        print(bcolored("Done!","white"))

        if config.hvdspeed:
            sess.run(hvd.shutdown()) # fail command to force stop

if __name__ == '__main__':
    parseArgs()
    loadDatasetConfig[config.dataset]()
    main()
