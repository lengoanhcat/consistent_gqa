import os
import sys
sys.path.append('/home/catle/Projects/lyrics_tfnorm/tf-logic-dev/')
sys.path.append('/home/catle/Tools/horovod/build/lib.linux-x86_64-3.7/')
import time
import math
import numpy as np
from scipy.special import comb
import tensorflow as tf
#import tensorflow.compat.v2 as tf
#tf.enable_v2_behavior()
# import tensorflow_probability as tfp

from macnetwork import ops
from macnetwork.config import config
from macnetwork.mac_cell import MACCell
import tfl
from tfl.functions import FromTFLogits
import horovod.tensorflow as hvd
import os
import contextlib, functools
import pickle as pkl

import global_vars
from global_vars import ATmask, random_seed, TSmask, ASmask

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
The MAC network model. It performs reasoning processes to answer a question over
knowledge base (the image) by decomposing it into attention-based computational steps,
each perform by a recurrent MAC cell.

The network has three main components.
Input unit: processes the network inputs: raw question strings and image into
distributional representations.

The MAC network: calls the MACcells (mac_cell.py) config.netLength number of times,
to perform the reasoning process over the question and image.

The output unit: a classifier that receives the question and final state of the MAC
network and uses them to compute log-likelihood over the possible one-word answers.
'''
class MACnet(object):

    '''Initialize the class.

    Args:
        embeddingsInit: initialization for word embeddings (random / glove).
        answerDict: answers dictionary (mapping between integer id and symbol).
    '''
    def __init__(self, embeddingsInit, answerDict, questionDict, nextElement = None):
        self.input = nextElement
        self.embeddingsInit = embeddingsInit
        self.answerDict = answerDict
        self.questionDict = questionDict
        self.build()

    '''
    Initializes placeholders.
        questionIndicesAll: integer ids of question words.
        [batchSize, questionLength]

        questionLengthsAll: length of each question.
        [batchSize]

        imagesPlaceholder: image features.
        [batchSize, channels, height, width]
        (converted internally to [batchSize, height, width, channels])

        answerIndicesAll: integer ids of answer words.
        [batchSize]

        lr: learning rate (tensor scalar)
        train: train / evaluation (tensor boolean)

        dropout values dictionary (tensor scalars)
    '''

    # change to H x W x C?

    def addPlaceholders(self):
        with tf.variable_scope("Placeholders"):
            ## data
            # questions
            self.questionIndicesAll = tf.placeholder(tf.int32, shape = (None, None))
            self.questionLengthsAll = tf.placeholder(tf.int32, shape = (None, ))

            # images
            # put image known dimension as last dim?
            if config.imageObjects:
                self.imagesAll = tf.placeholder(tf.float32, shape = (None, None, None))
                self.imagesObjectNumAll = tf.placeholder(tf.int32, shape = (None, ))
            else:
                self.imagesPlaceholder = tf.placeholder(tf.float32, shape = (None, None, None, None))
                self.imagesAll = tf.transpose(self.imagesPlaceholder, (0, 2, 3, 1))

            # answers
            self.answerIndicesAll = tf.placeholder(tf.int32, shape = (None, ))
            if config.tflSS or config.tflFK or config.batchStyle=='hybrid':
                self.numQuestions = tf.placeholder(tf.int32)
                # self.numQuestionsAll = tf.placeholder(tf.int32, shape = (None, ))
                self.taskIndicesAll = tf.placeholder(tf.int32, shape = (None, ))

            if config.useDLM:
                self.logitsMapAll = tf.placeholder(tf.float32, shape = (None, None))

            if config.dataset == "VQA":
                self.answerFreqListsAll = tf.placeholder(tf.int32, shape = (None, None))
                self.answerFreqNumsAll = tf.placeholder(tf.int32, shape = (None, ))

            if config.ansFormat == "mc":
                self.choicesIndicesAll = tf.placeholder(tf.int32, shape = (None, None))
                self.choicesNumsAll = tf.placeholder(tf.int32, shape = (None, ))
                # in general could consolidate that with mc and make it more general if i did choicesIndices all of them
                # in case of open ended

            ## optimization
            self.lr = tf.placeholder(tf.float32, shape = ())
            self.train = tf.placeholder(tf.bool, shape = ())
            self.batchSizeAll = tf.shape(self.questionIndicesAll)[0]

            ## dropouts
            # TODO: change dropouts to be 1 - current
            self.dropouts = {
                "encInput": tf.placeholder(tf.float32, shape = ()),
                "encState": tf.placeholder(tf.float32, shape = ()),
                "stem": tf.placeholder(tf.float32, shape = ()),
                "question": tf.placeholder(tf.float32, shape = ()),
                "read": tf.placeholder(tf.float32, shape = ()),
                "write": tf.placeholder(tf.float32, shape = ()),
                "memory": tf.placeholder(tf.float32, shape = ()),
                "output": tf.placeholder(tf.float32, shape = ()),
                "controlPre": tf.placeholder(tf.float32, shape = ()),
                "controlPost": tf.placeholder(tf.float32, shape = ()),
                "wordEmb": tf.placeholder(tf.float32, shape = ()),
                "word": tf.placeholder(tf.float32, shape = ()),
                "vocab": tf.placeholder(tf.float32, shape = ()),
                "object": tf.placeholder(tf.float32, shape = ()),
                "wordStandard": tf.placeholder(tf.float32, shape = ())
            }

            # batch norm params
            self.rmax = tf.placeholder(tf.float32, shape=())
            self.rmin = tf.placeholder(tf.float32, shape=())
            self.dmax = tf.placeholder(tf.float32, shape=())
            self.batchNorm = \
                {"decay": config.bnDecay,
                 "center": config.bnCenter,
                 "scale": config.bnScale,
                 "train": self.train,
                 "rmax": self.rmax,
                 "rmin": self.rmin,
                 "dmax": self.dmax}


            self.imageInDim = config.imageDims[-1]
            if not config.imageObjects:
                self.H, self.W, self.imageInDim = 7, 7, 2048# config.imageDims
                if config.dataset == "CLEVR":
                    self.H, self.W, self.imageInDim = 14, 14, 1024

    # Feeds data into placeholders. See addPlaceholders method for further details.
    def createFeedDict(self, data, images, train, progress, vBatchSize):
        # self.lr: config.lr*data["questions"].shape[0],
        # self.lr: config.lr
        # self.lr: config.lr * vBatchSize if config.hvdoptim == 'adasum' else config.lr*data["questions"].shape[0],
        feedDict = {
            self.questionIndicesAll: data["questions"],
            self.questionLengthsAll: data["questionLengths"],
            self.answerIndicesAll: data["answers"],
            self.dropouts["encInput"]: config.encInputDropout if train else 1.0,
            self.dropouts["encState"]: config.encStateDropout if train else 1.0,
            self.dropouts["stem"]: config.stemDropout if train else 1.0,
            self.dropouts["question"]: config.qDropout if train else 1.0, #_
            self.dropouts["memory"]: config.memoryDropout if train else 1.0,
            self.dropouts["read"]: config.readDropout if train else 1.0, #_
            self.dropouts["write"]: config.writeDropout if train else 1.0,
            self.dropouts["output"]: config.outputDropout if train else 1.0,
            self.dropouts["controlPre"]: config.controlPreDropout if train else 1.0,
            self.dropouts["controlPost"]: config.controlPostDropout if train else 1.0,
            self.dropouts["wordEmb"]: config.wordEmbDropout if train else 1.0,
            self.dropouts["word"]: config.wordDp if train else 1.0,
            self.dropouts["vocab"]: config.vocabDp if train else 1.0,
            self.dropouts["object"]: config.objectDp if train else 1.0,
            self.dropouts["wordStandard"]: config.wordStandardDp if train else 1.0,
            self.lr: config.lr * vBatchSize if config.hvdoptim == 'adasum' else config.lr*data["questions"].shape[0],
            self.train: train
        }

        if config.tflSS or config.tflFK or config.batchStyle=='hybrid':
            if "tasks" in data.keys():
                feedDict.update({
                    self.taskIndicesAll: data["tasks"],
                })
            else:
                feedDict.update({
                    self.taskIndicesAll: np.random.randint(
                        0,47,data["questions"].shape[0])
                })

            if "questionsNum" in data.keys():
                feedDict.update({
                    self.numQuestions: data["questionsNum"]
                })
            else:
                feedDict.update({
                    self.numQuestions: data["questions"].shape[0]
                })

        if progress is None:
            feedDict.update({
                self.rmax: 1.0,
                self.rmin: 1.0,
                self.dmax: 0.0
            })
        else:
            feedDict.update({
                self.rmax: 1.0 + 2.0*smoothstep(progress, 0.05, 0.3),
                self.rmin: 1.0/(1.0 + 2.0*smoothstep(progress, 0.05, 0.3)),
                self.dmax: 5.0*smoothstep(progress, 0.05,0.2)
            })

        if config.imageObjects:
            feedDict.update({
                self.imagesAll: images["images"],
                self.imagesObjectNumAll: data["objectsNums"],
            })
        else:
            feedDict.update({
                self.imagesPlaceholder: images["images"]
            })

        if config.dataset == "VQA":
            feedDict.update({
                self.answerFreqListsAll: data["answerFreqs"],
                self.answerFreqNumsAll: data["answerFreqNums"]
            })

        if config.ansFormat == "mc":
            feedDict.update({
                self.choicesIndicesAll: data["choices"],
                self.choicesNumsAll: data["choicesNums"]
            })

        return feedDict

    # Splits data to a specific GPU (tower) for parallelization
    def initTowerBatch(self, towerI, towersNum, dataSize):
        towerBatchSize = tf.floordiv(dataSize, towersNum)
        start = towerI * towerBatchSize
        end = (towerI + 1) * towerBatchSize if towerI < towersNum - 1 else dataSize

        self.questionIndices = self.questionIndicesAll[start:end]
        self.questionLengths = self.questionLengthsAll[start:end]

        self.images = self.imagesAll[start:end]

        self.imagesObjectNum = None
        if config.imageObjects:
            self.imagesObjectNum = self.imagesObjectNumAll[start:end]

        self.answerIndices = self.answerIndicesAll[start:end]
        if config.tflSS or config.tflFK or config.batchStyle=='hybrid':
            self.taskIndices = self.taskIndicesAll[start:end]
            # self.numQuestions = self.numQuestionsAll[start:end]

        self.answerFreqs = self.answerFreqNums = None
        if config.dataset == "VQA":
            self.answerFreqLists = self.answerFreqListsAll[start:end]
            self.answerFreqNums = self.answerFreqNumsAll[start:end]

        self.choicesIndices = self.choicesNums = None
        if config.ansFormat == "mc":
            self.choicesIndices = self.choicesIndicesAll[start:end]
            self.choicesNums = self.choicesNumsAll[start:end]

        self.batchSize = end - start

    '''
    The Image Input Unit (stem). Passes the image features through a CNN-network
    Optionally adds position encoding (doesn't in the default behavior).
    Flatten the image into Height * Width "Knowledge base" array.

    Args:
        images: image input. [batchSize, height, width, inDim]
        inDim: input image dimension
        outDim: image out dimension
        addLoc: if not None, adds positional encoding to the image

    Returns preprocessed images.
    [batchSize, height * width, outDim]
    '''
    def stem(self, images, inDim, outDim, addLoc = None):
        with tf.variable_scope("stem"):
            if config.stemNormalize:
                images = tf.nn.l2_normalize(images, dim = -1)

            if config.imageObjects: # VQA ??? or config.useBaseline:
                features, dim = images, inDim
                if config.stemLinear:
                    features = ops.linear(images, inDim, outDim, dropout = self.dropouts["stem"])
                    dim = outDim
                elif config.stemDeep:
                    dims = [inDim] + config.stemDims + [outDim]
                    features = ops.FCLayer(features, dims, dropout = self.dropouts["stem"])

                if config.stemAct != "NON":
                    features = ops.actF(config.stemAct)(features)

                return features, dim

            if addLoc is None:
                addLoc = config.locationAware

            if config.stemLinear:
                features = ops.linear(images, inDim, outDim)
            else:
                if config.stemNumLayers == 0:
                    outDim = inDim
                else:
                    dims = [inDim] + ([config.stemDim] * (config.stemNumLayers - 1)) + [outDim]

                    if addLoc:
                        images, inDim = ops.addLocation(images, inDim, config.locationDim,
                            h = self.H, w = self.W, locType = config.locationType)
                        dims[0] = inDim

                    features = ops.CNNLayer(images, dims,
                        batchNorm = self.batchNorm if config.stemBN else None,
                        dropout = self.dropouts["stem"],
                        kernelSizes = config.stemKernelSizes,
                        strides = config.stemStrideSizes)

                    if config.stemGridRnn:
                        features = ops.multigridRNNLayer(features, H, W, outDim)

            if config.baselineNew or (not config.useBaseline):
                features = tf.reshape(features, (self.batchSize, -1, outDim))

        return features, outDim

    # Embed question using parametrized word embeddings.
    # The embedding are initialized to the values supported to the class initialization
    def qEmbeddingsOp(self, qIndices, embInit):
        with tf.variable_scope("qEmbeddings"):
            embInit = tf.to_float(embInit)
            embeddingsVar = tf.get_variable("emb", initializer = embInit,
                dtype = tf.float32, trainable = (not config.wrdEmbQFixed))
            embeddings = tf.concat([tf.zeros((1, config.wrdQEmbDim)), embeddingsVar], axis = 0)
            questions = tf.nn.embedding_lookup(embeddings, qIndices)

        return questions, embeddings

    # Embed answer words
    def aEmbeddingsOp(self, aIndices, embInit):
        with tf.variable_scope("aEmbeddings"):
            if embInit is None:
                return None
            embInit = tf.to_float(embInit)
            embeddings = tf.get_variable("emb", initializer = embInit,
                dtype = tf.float32, trainable = (not config.wrdEmbAFixed))

            if config.ansFormat == "mc":
                answers = tf.nn.embedding_lookup(embeddings, aIndices)
            else:
                answers = embeddings
        return answers

    def vocabEmbeddings(self, embInit, name):
        with tf.variable_scope("vocabEmbeddings" + name):
            embInit = tf.to_float(embInit)
            embeddings = tf.get_variable("emb", initializer = embInit,
                dtype = tf.float32, trainable = (not config.semanticFixEmbs))
        return embeddings

    # Embed question and answer words with tied embeddings
    def qaEmbeddingsOp(self, qIndices, aIndices, embInit):
        questions, embeddings = self.qEmbeddingsOp(qIndices, embInit)
        answers = tf.nn.embedding_lookup(embeddings, aIndices)
        return questions, answers, embeddings

    '''
    Embed question (and optionally answer) using parametrized word embeddings.
    The embedding are initialized to the values supported to the class initialization
    '''
    def embeddingsOp(self, qIndices, aIndices, embInit):
        # nullWord = tf.tile(tf.expand_dims(nullWord, axis = 0), [self.batchSize, 1, 1])
        if config.ansEmbMod == "SHARED":
            if config.ansFormat == "oe":
            #if aIndices is None:
                aIndices = embInit["oeAnswers"]
            questions, answers, qaEmbeddings = self.qaEmbeddingsOp(qIndices, aIndices, embInit["qa"])
        else:
            questions, qEmbeddings = self.qEmbeddingsOp(qIndices, embInit["q"])
            answers = self.aEmbeddingsOp(aIndices, embInit["a"])

        if config.ansFormat == "oe" and config.ansEmbMod != "NON":
            answers = tf.tile(tf.expand_dims(answers, axis = 0), [self.batchSize, 1, 1])

        return questions, answers # , embeddings

    '''
    The Question Input Unit embeds the questions to randomly-initialized word vectors,
    and runs a recurrent bidirectional encoder (RNN/LSTM etc.) that gives back
    vector representations for each question (the RNN final hidden state), and
    representations for each of the question words (the RNN outputs for each word).

    The method uses bidirectional LSTM, by default.
    Optionally projects the outputs of the LSTM (with linear projection /
    optionally with some activation).

    Args:
        questions: question word embeddings
        [batchSize, questionLength, wordEmbDim]

        questionLengths: the question lengths.
        [batchSize]

        projWords: True to apply projection on RNN outputs.
        projQuestion: True to apply projection on final RNN state.
        projDim: projection dimension in case projection is applied.

    Returns:
        Contextual Words: RNN outputs for the words.
        [batchSize, questionLength, ctrlDim]

        Vectorized Question: Final hidden state representing the whole question.
        [batchSize, ctrlDim]
    '''
    def encoder(self, questions, questionLengths, projWords = False,
        projQuestion = False, projDim = None):

        with tf.variable_scope("encoder"):
            # variational dropout option
            varDp = None
            if config.encVariationalDropout:
                varDp = {"stateDp": self.dropouts["stateInput"],
                         "inputDp": self.dropouts["encInput"],
                         "inputSize": config.wrdQEmbDim}

            # rnns
            for i in range(config.encNumLayers):
                questionCntxWords, vecQuestions = ops.RNNLayer(questions, questionLengths,
                    config.encDim, bi = config.encBi, cellType = config.encType,
                    dropout = self.dropouts["encInput"], varDp = varDp, name = "rnn%d" % i)

            # dropout for the question vector
            vecQuestions = tf.nn.dropout(vecQuestions, self.dropouts["question"])

            # batchNorm vecQuetions and contextual words
            if config.questBN:
                qSyncBN = hvd.SyncBatchNormalization(
                    axis=-1,
                    momentum = self.batchNorm["decay"],
                    fused=False,
                    center = self.batchNorm["center"],
                    scale = self.batchNorm["scale"],
                    renorm_clipping={
                        'rmax':self.batchNorm['rmax'],
                        'rmin':self.batchNorm['rmin'],
                        'dmax':self.batchNorm['dmax']},
                    renorm=True)
                vSyncBN = hvd.SyncBatchNormalization(
                    axis=-1,
                    momentum = self.batchNorm["decay"],
                    fused=False,
                    center = self.batchNorm["center"],
                    scale = self.batchNorm["scale"],
                    renorm_clipping={
                        'rmax':self.batchNorm['rmax'],
                        'rmin':self.batchNorm['rmin'],
                        'dmax':self.batchNorm['dmax']},
                    renorm=True)

                qSyncBN.apply(questionCntxWords,
                    training = self.batchNorm["train"])
                vSyncBN.apply(vecQuestions,
                    training = self.batchNorm["train"])

            # projection of encoder outputs
            if projWords:
                questionCntxWords = ops.linear(questionCntxWords, config.encDim, projDim,
                    name = "projCW")
            if projQuestion:
                vecQuestions = ops.linear(vecQuestions, config.encDim, projDim,
                    act = config.encProjQAct, name = "projQ")

        return questionCntxWords, vecQuestions

    '''
    Stacked Attention Layer for baseline. Computes interaction between images
    and the previous memory, and casts it back to compute attention over the
    image, which in turn is summed up with the previous memory to result in the
    new one.

    Args:
        images: input image.
        [batchSize, H * W, inDim]

        memory: previous memory value
        [batchSize, inDim]

        inDim: inputs dimension
        hDim: hidden dimension to compute interactions between image and memory

    Returns the new memory value.
    '''
    def baselineAttLayer(self, images, memory, inDim, hDim, name = "", reuse = None):
        with tf.variable_scope("attLayer" + name, reuse = reuse):
            # projImages = ops.linear(images, inDim, hDim, name = "projImage")
            # projMemory = tf.expand_dims(ops.linear(memory, inDim, hDim, name = "projMemory"), axis = -2)
            # if config.saMultiplicative:
            #     interactions = projImages * projMemory
            # else:
            #     interactions = tf.tanh(projImages + projMemory)
            interactions, hDim = ops.mul(images, memory, inDim, proj = {"dim": hDim, "shared": False},
                interMod = config.baselineAttType)

            attention = ops.inter2att(interactions, hDim, mask = self.imagesObjectNum)
            summary = ops.att2Smry(attention, images)

            newMemory = memory + summary

        return newMemory


    '''
    Baseline approach:
    If baselineAtt is True, applies several layers (baselineAttNumLayers)
    of stacked attention to image and memory, when memory is initialized
    to the vector questions. See baselineAttLayer for further details.

    Otherwise, computes result output features based on image representation
    (baselineCNN), or question (baselineLSTM) or both.

    Args:
        vecQuestions: question vector representation
        [batchSize, questionDim]

        questionDim: dimension of question vectors

        images: (flattened) image representation
        [batchSize, imageDim]

        imageDim: dimension of image representations.

        hDim: hidden dimension to compute interactions between image and memory
        (for attention-based baseline).

    Returns final features to use in later classifier.
    [batchSize, outDim] (out dimension depends on baseline method)
    '''
    def baseline(self, vecQuestions, questionDim, images, imageDim, hDim):
        with tf.variable_scope("baseline"):
            if config.baselineAtt:
                memory = ops.linear(vecQuestions, questionDim, hDim, name = "qProj")
                images = ops.linear(images, imageDim, hDim, name = "iProj")

                for i in range(config.baselineAttNumLayers):
                    memory = self.baselineAttLayer(images, memory, hDim, hDim,
                        name = "baseline%d" % i)
                memDim = hDim
            else:
                if config.imageObjects:
                    cff = tf.get_variable("cff", shape = (imageDim, ), initializer = tf.random_normal_initializer())
                    interactions, hDim = ops.mul(images, cff, imageDim)
                    attention = ops.inter2att(interactions, hDim, mask = self.imagesObjectNum)
                    images = ops.att2Smry(attention, images)
                else:
                    images, imageDim = ops.linearizeFeatures(images, self.H, self.W,
                        imageDim, projDim = config.baselineProjDim)
                if config.baselineLSTM and config.baselineCNN:
                    memory = tf.concat([vecQuestions, images], axis = -1)
                    memDim = questionDim + imageDim
                elif config.baselineLSTM:
                    memory = vecQuestions
                    memDim = questionDim
                else: # config.baselineCNN
                    memory = images
                    memDim = imageDim

        return memory, memDim

    '''
    Runs the MAC recurrent network to perform the reasoning process.
    Initializes a MAC cell and runs netLength iterations.

    Currently it passes the question and knowledge base to the cell during
    its creating, such that it doesn't need to interact with it through
    inputs / outputs while running. The recurrent computation happens
    by working iteratively over the hidden (control, memory) states.

    Args:
        images: flattened image features. Used as the "Knowledge Base".
        (Received by default model behavior from the Image Input Units).
        [batchSize, H * W, memDim]

        vecQuestions: vector questions representations.
        (Received by default model behavior from the Question Input Units
        as the final RNN state).
        [batchSize, ctrlDim]

        questionWords: question word embeddings.
        [batchSize, questionLength, ctrlDim]

        questionCntxWords: question contextual words.
        (Received by default model behavior from the Question Input Units
        as the series of RNN output states).
        [batchSize, questionLength, ctrlDim]

        questionLengths: question lengths.
        [batchSize]

    Returns the final control state and memory state resulted from the network.
    ([batchSize, ctrlDim], [bathSize, memDim])
    '''
    def MACnetwork(self, images, vecQuestions, questionWords, questionCntxWords,
        questionLengths, name = "", reuse = None):

        with tf.variable_scope("MACnetwork" + name, reuse = reuse):

            self.macCell = MACCell(
                vecQuestions = vecQuestions,
                questionWords = questionWords,
                questionCntxWords = questionCntxWords,
                questionLengths = questionLengths,
                knowledgeBase = images,
                kbSize = self.imagesObjectNum,
                memoryDropout = self.dropouts["memory"],
                readDropout = self.dropouts["read"],
                writeDropout = self.dropouts["write"],
                controlDropoutPre = self.dropouts["controlPre"],
                controlDropoutPost = self.dropouts["controlPost"],
                wordDropout = self.dropouts["word"],
                vocabDropout = self.dropouts["vocab"],
                objectDropout = self.dropouts["object"],
                # qDropoutMAC = self.qDropoutMAC,
                batchSize = self.batchSize,
                train = self.train,
                reuse = reuse,
                rmax = self.rmax,
                rmin = self.rmin,
                dmax = self.dmax)

            state = self.macCell.zero_state(self.batchSize, tf.float32)

            none = tf.zeros((self.batchSize, 1), dtype = tf.float32)

            for i in range(config.netLength):
                self.macCell.iteration = i
                _, state = self.macCell(none, state)

            finalControl = state.control
            finalMemory = state.memory

        return finalControl, finalMemory

    '''
    Output Unit (step 1): chooses the inputs to the output classifier.

    By default the classifier input will be the the final memory state of the MAC network.
    If outQuestion is True, concatenate the question representation to that.
    If outImage is True, concatenate the image flattened representation.

    Args:
        memory: (final) memory state of the MAC network.
        [batchSize, memDim]

        vecQuestions: question vector representation.
        [batchSize, ctrlDim]

        images: image features.
        [batchSize, H, W, imageInDim]

        imageInDim: images dimension.

    Returns the resulted features and their dimension.
    '''
    def outputOp(self, memory, control, vecQuestions, images, imageInDim):
        with tf.variable_scope("outputUnit"):
            features = memory
            dim = config.memDim

            if config.outQuestion:
                q = vecQuestions
                eQ = ops.linear(q, config.ctrlDim, config.memDim, name = "outQuestion")
                features, dim = ops.concat(features, eQ, config.memDim, mul = config.outQuestionMul)

            # assumes imageObjects False
            if config.outImage:
                images, imagesDim = ops.linearizeFeatures(images, self.H, self.W, self.imageInDim,
                    outputDim = config.outImageDim)
                images = ops.linear(images, config.memDim, config.outImageDim, name = "outImage")
                features = tf.concat([features, images], axis = -1)
                dim += config.outImageDim

        return features, dim

    '''
    Output Unit (step 2): Computes the logits for the answers. Passes the features
    through fully-connected network to get the logits over the possible answers.
    Optionally uses answer word embeddings in computing the logits (by default, it doesn't).

    Args:
        features: features used to compute logits
        [batchSize, inDim]

        inDim: features dimension

        aEmbedding: supported word embeddings for answer words in case answerMod is not NON.
        Optionally computes logits by computing dot-product with answer embeddings.

    Returns: the computed logits.
    [batchSize, answerWordsNum]
    '''
    # in mc has to be ansMod not NON
    def classifier(self, features, inDim, choices = None, choicesNums = None, outType='ans'):
        with tf.variable_scope("classifier_"+outType):
            # outDim = config.answerWordsNum
            if outType == 'ans':
                outDim = ATmask[0]
                outClassifierDims = config.outAnsClassifierDims
            elif 'tsk' in outType:
                outDim = ATmask[1]-ATmask[0]
                outClassifierDims = config.outTskClassifierDims

            dims = [inDim] + outClassifierDims + [outDim]
            if config.answerMod != "NON":
                dims[-1] = config.wrdAEmbDim

            logits = ops.FCLayer(features, dims,
                batchNorm = self.batchNorm if config.outputBN else None,
                dropout = self.dropouts["output"])

            if config.answerMod != "NON":
                logits = ops.gatedAct(config.outAct, gate = config.outGate)(logits)
                logits = tf.nn.dropout(logits, self.dropouts["output"])
                concat = {"x": config.answerBias}
                interactions, interDim = ops.mul(choices, logits, dims[-1], interMod = config.answerMod, concat = concat)
                logits = ops.inter2logits(interactions, interDim, sumMod = config.answerSumMod)
                if config.ansFormat == "oe":
                    logits += ops.getBias((outDim, ), "ans")
                else:
                    logits = ops.expMask(logits, choicesNums)

        return logits

    def aggregateFreqs(self, answerFreqs, answerFreqNums):
        if answerFreqs is None:
            return None
        answerFreqs = tf.one_hot(answerFreqs, config.answerWordsNum) # , axis = -1
        mask = tf.sequence_mask(answerFreqNums, maxlen = config.AnswerFreqMaxNum)
        mask = tf.expand_dims(tf.to_float(mask), axis = -1)
        answerFreqs *= mask
        answerFreqs = tf.reduce_sum(answerFreqs, axis = 1)
        return answerFreqs

    def addLogicLossOp(self, logits, questions, answerIndices, taskIndices): #, tasks):
        ''' Add logic constrained loss operation '''
        print("Creating logic component graph for consistent GQA...")
        predicates_dict = self.answerDict.sym2id  # load from answer

        # Domains Definition
        softmax_logits = tf.concat(
            [tf.nn.softmax(logits[:,:ATmask[0]],axis=1),
             tf.nn.softmax(logits[:,ATmask[0]:ATmask[1]],axis=1)],axis=1) #38

        questions = tfl.Domain(
            "Question", size=self.numQuestions,
            data=tf.concat([questions, softmax_logits,
                            tf.cast(tf.expand_dims(answerIndices,axis=1),
                                    tf.float32),
                            tf.cast(tf.expand_dims(taskIndices,axis=1),
                                    tf.float32)
                            ],
                           axis=1)) # self.numQuestions

        # Function Definition
        class IsDiff(tfl.functions.AbstractFunction):
            def __call__(self, a, b):
                qA = a[:,0:config.encDim]; qB = b[:,0:config.encDim]
                dist = tf.sqrt(tf.reduce_sum(tf.square(qA - qB), axis=1))
                return tf.where(dist > .1 * tf.ones_like(dist),
                        tf.ones_like(dist),
                        tf.zeros_like(dist))

        class IsSamAns(tfl.functions.AbstractFunction):
            def __call__(self, a, b):
                # check whether task in the same semantic
                aA_logit = a[:,config.encDim:config.encDim+ATmask[0]]
                aB_logit = b[:,config.encDim:config.encDim+ATmask[0]]

                aA_label = tf.argmax(aA_logit, axis=1)
                aB_label = tf.argmax(aB_logit, axis=1)
                aA_prob = tf.reduce_max(aA_logit, axis=1)
                aB_prob= tf.reduce_max(aB_logit, axis=1)

                return tf.where(tf.equal(aA_label,aB_label),
                                aA_prob * aB_prob,
                                1. - aA_prob * aB_prob) # strong negation for TP


        class IsConTaskAns(tfl.functions.AbstractFunction):
            def __call__(self, a, b):
                # check whether task in the same semantic
                tA_logit = a[:,config.encDim+ATmask[0]:config.encDim+ATmask[1]]
                tB_logit = b[:,config.encDim+ATmask[0]:config.encDim+ATmask[1]]
                aA_logit = a[:,config.encDim:config.encDim+ATmask[0]]
                aB_logit = b[:,config.encDim:config.encDim+ATmask[0]]

                tA_indices = tf.argmax(tA_logit, axis=1)
                tB_indices = tf.argmax(tB_logit, axis=1)
                aA_indices = tf.argmax(aA_logit, axis=1)
                aB_indices = tf.argmax(aB_logit, axis=1)

                tA_prob = tf.reduce_max(tA_logit, axis=1)
                tB_prob = tf.reduce_max(tB_logit, axis=1)
                aA_prob = tf.reduce_max(aA_logit, axis=1)
                aB_prob = tf.reduce_max(aB_logit, axis=1)

                sTA_label = tf.reshape(tf.gather(TSmask, tA_indices),[-1,1,1])
                sTB_label = tf.reshape(tf.gather(TSmask, tB_indices),[-1,1,1])
                sAA_label = tf.gather(ASmask, aA_indices)
                sAB_label = tf.gather(ASmask, aB_indices)

                conTaskAnswerA = tf.map_fn(
                    lambda x: tf.shape(tf.sets.intersection(x[0],x[1]))[1]>0,
                    (sTA_label,sAA_label),dtype=tf.bool)

                conTaskAnswerB = tf.map_fn(
                    lambda x: tf.shape(tf.sets.intersection(x[0],x[1]))[1]>0,
                    (sTB_label,sAB_label),dtype=tf.bool)

                return tf.where(tf.logical_and(conTaskAnswerA, conTaskAnswerB),
                                    tA_prob * tB_prob * aA_prob * aB_prob,
                                    tf.zeros_like(tA_indices,dtype=tf.float32))

        class IsGoodAns(tfl.functions.AbstractFunction):
            def __call__(self, a, b):
                # check the truth degree of a label answer
                aA_logit = a[:,config.encDim:config.encDim+ATmask[0]]
                aB_logit = b[:,config.encDim:config.encDim+ATmask[0]]
                # AnsInd 2nd last column
                aA_indices = tf.cast(a[:,-2],tf.int32)
                aB_indices = tf.cast(b[:,-2],tf.int32)

                # gather truth degree w.r.t. answer label
                aA_prob = tf.map_fn(
                    lambda x: tf.gather(x[0],x[1]),
                    (aA_logit,aA_indices),dtype=tf.float32)
                aB_prob = tf.map_fn(
                    lambda x: tf.gather(x[0],x[1]),
                    (aB_logit,aB_indices),dtype=tf.float32)

                return tfl.World.tnorm([aA_prob,aB_prob])

        class IsTrueAns(tfl.functions.AbstractFunction):
            def __call__(self, a):
                # check whether the ans is correct or not
                aA_logit = a[:,config.encDim:config.encDim+ATmask[0]]
                aA_indices = tf.cast(a[:,-2],tf.int32)
                aA_prob = tf.map_fn(
                    lambda x: tf.gather(x[0],x[1]),
                    (aA_logit,aA_indices),dtype=tf.float32)
                return aA_prob

        class IsGoodTask(tfl.functions.AbstractFunction):
            def __call__(self, a, b):
                # check the truth degree of a label answer
                tA_logit = a[:,config.encDim+ATmask[0]:config.encDim+ATmask[1]]
                tB_logit = b[:,config.encDim+ATmask[0]:config.encDim+ATmask[1]]
                # TskInd last column
                tA_indices = tf.cast(a[:,-1],tf.int32)
                tB_indices = tf.cast(b[:,-1],tf.int32)

                # gather truth degree w.r.t. task label
                tA_prob = tf.map_fn(
                    lambda x: tf.gather(x[0],x[1]),
                    (tA_logit,tA_indices),dtype=tf.float32)
                tB_prob = tf.map_fn(
                    lambda x: tf.gather(x[0],x[1]),
                    (tB_logit,tB_indices),dtype=tf.float32)

                return tfl.World.tnorm([tA_prob,tB_prob])


        class IsSamSem(tfl.functions.AbstractFunction):
            def __call__(self, a, b):
                tA_logit = a[:,config.encDim+ATmask[0]:config.encDim+ATmask[1]]
                tB_logit = b[:,config.encDim+ATmask[0]:config.encDim+ATmask[1]]

                tA_indices = tf.argmax(tA_logit, axis=1)
                tB_indices = tf.argmax(tB_logit, axis=1)
                tA_prob = tf.reduce_max(tA_logit, axis=1)
                tB_prob = tf.reduce_max(tB_logit, axis=1)

                sA_label = tf.gather(TSmask, tA_indices)
                sB_label = tf.gather(TSmask, tB_indices)

                return tf.where(tf.equal(sA_label,sB_label),
                                tA_prob*tB_prob,
                                tf.zeros_like(sA_label,dtype=tf.float32))

        is_diff = IsDiff()
        tfl.Predicate("isDiff",domains=["Question","Question"],function=is_diff)
        is_good_ans = IsGoodAns()
        tfl.Predicate("isGoodAns",domains=["Question","Question"],function=is_good_ans)
        is_true_ans = IsTrueAns()
        tfl.Predicate("isTrueAns",domains=["Question"],function=is_true_ans)
        is_good_task = IsGoodTask()
        tfl.Predicate("isGoodTask",domains=["Question","Question"],function=is_good_task)
        is_sam_ans = IsSamAns()
        tfl.Predicate("isSamAns",domains=["Question","Question"],function=is_sam_ans)
        is_sam_sem = IsSamSem()
        tfl.Predicate("isSamSem",domains=["Question","Question"],function=is_sam_sem)
        is_con_task_ans = IsConTaskAns()
        tfl.Predicate("isConTaskAns",domains=["Question","Question"],function=is_con_task_ans)

        # Predicates Definition
        logits_func = FromTFLogits(softmax_logits)
        for k, v in predicates_dict.items():
            tfl.Predicate(
                k, domains=("Question",), function=tfl.functions.Slice(logits_func, v))

        pos_task_list = ['verifyGlobalTrue','verifyAttrTrue', 'allDiffTrue', 'existTrue', 'existNotTrue',
            'existAndTrue', 'existAttrTrue', 'existAttrNotTrue',
            'existAttrOrTrue', 'existOrTrue', 'existRelTrue',
            'allSameTrue', 'twoDiffTrue', 'twoSameTrue',
            'verifyAttrsTrue', 'verifyAttrAndTrue']

        pos_task_strQ = ' or '.join([task+'(q)' for task in pos_task_list])
        pos_task_strP = ' or '.join([task+'(p)' for task in pos_task_list])

        neg_task_list = ['verifyGlobalFalse','verifyAttrFalse', 'allDiffFalse', 'existFalse', 'existNotFalse',
            'existAndFalse', 'existAttrFalse', 'existAttrNotFalse',
            'existAttrOrFalse', 'existOrFalse', 'existRelFalse',
            'allSameFalse', 'twoDiffFalse', 'twoSameFalse',
            'verifyAttrsFalse', 'verifyAttrAndFalse']

        neg_task_strQ = ' or '.join([task+'(q)' for task in neg_task_list])
        neg_task_strP = ' or '.join([task+'(P)' for task in neg_task_list])

        bin_task_str = 'compare(q)'

        # unused rule
        '''
        open_task_str = 'queryGlobal(q)'
        open_task_str.join([
            ' or {}(q)'.format(task) for task in
                ['queryAttr', 'queryAttrObj', 'chooseAttr',
                'queryObj', 'chooseObj', 'queryRel',
                'chooseRel', 'chooseAttr', 'chooseGlobal',
                'common']])
        '''


        with open('./data/answerSemantic_dict.pkl','rb') as fh:
            sem_answer_dict = pkl.load(fh)

        # .replace(' ','_')
        global_ans_str = ' or '.join(
            [answer+'(q)' for answer in
             list(set(sem_answer_dict['global'])&set(predicates_dict.keys()))])

        attr_ans_str = ' or '.join(
            [answer+'(q)' for answer in
             list(set(sem_answer_dict['attr'])&set(predicates_dict.keys()))])

        cat_ans_str = ' or '.join(
            [answer+'(q)' for answer in
             list(set(sem_answer_dict['cat'])&set(predicates_dict.keys()))])

        rel_ans_str = ' or '.join(
            [answer+'(q)' for answer in
             list(set(sem_answer_dict['rel'])&set(predicates_dict.keys()))])

        constraints_str = [
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
            "allSameFalse(p) <-> allDiffTrue(q)"
        ]

        constraints_str_1 = [
            "(isTrueAns(p) and queryObj(p)) -> (queryAttrObj(q) and isTrueAns(q))",
            "(isTrueAns(p) and queryAttrObj(p)) -> (existAttrTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrTrue(p)) -> (existAttrOrTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrTrue(p)) -> (existNotTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrTrue(p)) -> (existAttrNotTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrOrTrue(p)) -> (existNotOrTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existNotOrTrue(p)) -> (existOrTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existOrTrue(p)) -> (existTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and queryAttrObj(p)) -> (queryObj(q) and isTrueAns(q))",
            "(isTrueAns(p) and queryNotObj(p)) -> (existNotTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existNotTrue(p)) -> (existTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrNotTrue(p)) -> (existTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAndTrue(p)) -> (existTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existRelTrue(p)) -> (existTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and existRelTrue(p)) <-> (verifyRelTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyRelTrue(p)) <-> (queryRel(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyRelTrue(p)) <-> (chooseRel(q) and isTrueAns(q))",

            "(isTrueAns(p) and existOrFalse(p)) -> (existFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and existFalse(p)) -> (existNotFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and existFalse(p)) -> (existAttrNotFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and existFalse(p)) -> (existRelFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and existFalse(p)) -> (existAndFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and existNotFalse(p)) -> (existAttrFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrNotFalse(p)) -> (existAttrFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and existNotOrFalse(p)) -> (existNotFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and existNotOrFalse(p)) -> (existAttrNotFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and existNotOrFalse(p)) -> (existAttrOrFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and existAttrOrFalse(p)) -> (existAttrFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and verifyAttrsTrue(p)) -> (verifyAttrTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyAttrAndTrue(p))-> (verifyAttrTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyAttrTrue(p)) -> (queryAttr(q) and isTrueAns(q))",
            "(isTrueAns(p) and queryAttr(p)) -> (verifyAttrFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and queryAttr(p)) -> (chooseAttr(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyAttrFalse(p)) -> (verifyAttrAndFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyAttrAndFalse(p)) -> (chooseAttr(q) and isTrueAns(q))",
            "(isTrueAns(p) and chooseAttr(p)) <-> (chooseObj(q) and isTrueAns(q))",

            "(isTrueAns(p) and verifyGlobalTrue(p)) -> (verifyGlobalFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyGlobalTrue(p)) <-> (queryGlobal(q) and isTrueAns(q))",
            "(isTrueAns(p) and verifyGlobalFalse(p)) -> (chooseGlobal(q) and isTrueAns(q))",

            "(isTrueAns(p) and compare(p)) -> (common(q) and isTrueAns(q))",
            "(isTrueAns(p) and common(p)) -> (twoSameTrue(q) and isTrueAns(q))",
            "(isTrueAns(p) and twoSameTrue(p)) <-> (twoDiffFalse(q) and isTrueAns(q))",

            "(isTrueAns(p) and twoSameFalse(p)) <-> (twoDiffTrue(q) and isTrueAns(q))",

            "(isTrueAns(p) and allSameTrue(p)) <-> (allDiffFalse(q) and isTrueAns(q))",
            "(isTrueAns(p) and allSameFalse(p)) <-> (allDiffTrue(q) and isTrueAns(q))"
        ]


        entailed_task_str = ' or '.join(
            ['(' +entail + ')' for entail in
             constraints_str])

        constraints = []
        numCW1 = 0
        numCW2 = 0

        # Is Entailed Answer
        if 'E' in config.tflRules:
            for c_str in constraints_str:
                constraints.append(
                    tfl.constraint(
                        "forall p: forall q: (isDiff(p,q) and ({})) ->  isConTaskAns(p,q)".format(c_str))\
                    /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += len(constraints_str)
            '''
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: isDiff(p,q) and ({}) -> isConTaskAns(p,q)".format(entailed_task_str))\
                /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += 1
            '''

        if 'E1' in config.tflRules:
            for c_str in constraints_str:
                constraints.append(
                    tfl.constraint(
                        "forall p: forall q: (isDiff(p,q) and ({})) ->  isGoodAns(p,q)".format(c_str))\
                    /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += len(constraints_str)

        if 'E2' in config.tflRules:
            for c_str in constraints_str:
                constraints.append(
                    tfl.constraint(
                        "forall p: forall q: (isDiff(p,q) and isGoodTask(p,q) and ({})) ->  isGoodAns(p,q)".format(c_str))\
                    /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += len(constraints_str)

        if 'E3' in config.tflRules:
            entailment_str = '({})'.format(constraints_str[0]) + ''.join([
                ' or ({})'.format(task) for task in constraints_str[1:]])
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: (isDiff(p,q) and isGoodTask(p,q) and ({})) ->  isGoodAns(p,q)".format(entailment_str))\
                /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += 1

        if 'E4' in config.tflRules:
            entailment_str = '({})'.format(constraints_str[0]) + ''.join([
                ' or ({})'.format(task) for task in constraints_str[1:]])
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: (isDiff(p,q) and ({})) ->  isGoodAns(p,q)".format(entailment_str))\
                /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += 1

        if 'E5' in config.tflRules:
            entailment_str = '({})'.format(constraints_str_1[0]) + ''.join([
                ' or ({})'.format(task) for task in constraints_str_1[1:]])
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: (isDiff(p,q) and ({}))".format(entailment_str))\
                /tf.cast(tf.square(self.batchSize),tf.float32)) # and isConTaskAns(p,q)
            numCW2 += 1


        # a possible rule
        # forall p: forall q: (isDiff(p,q) and isSamSem(p,q)) -> (isSamAns(p,q) and isGoodAns(p))
        # and isGoodAns(p,q)
        if 'C' in config.tflRules:
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: (isDiff(p,q) and isSamSem(p,q)) -> (isSamAns(p,q) and isConTaskAns(p,q))")
                /tf.cast(tf.square(self.batchSize),tf.float32))
            numCW2 += 1

        if 'C1' in config.tflRules:
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: (isDiff(p,q) and isSamSem(p,q)) -> (isGoodAns(p,q))")
                /tf.cast(tf.square(self.batchSize),tf.float32))
            numCW2 += 1

        if 'C2' in config.tflRules:
            constraints.append(
                tfl.constraint(
                    "forall p: forall q: (isDiff(p,q) and isGoodTask(p,q)) -> (isGoodAns(p,q))")
                /tf.cast(tf.square(self.batchSize),tf.float32))
            numCW2 += 1


        global_vars.numCW = numCW1+numCW2
        # print constraintWeights for debugging
        '''
        if hvd.rank() == 0:
            print_op = tf.print(
                'constr_vars: ',self.constraintWeights,'\n')
        else:
            print_op = tf.no_op()

        with tf.control_dependencies([print_op]):
        '''
        if config.tflWeightMode=='sw':
            # self.constraintWeights = tf.Variable(
            #    np.zeros([1,numCW],dtype=float), dtype=tf.float32, name="constraintWeights")
            # self.constraintWeights = ops.getWeight([1,numCW], name="constraintWeights")
            constraintWeights1 = tf.Variable(
                np.zeros([1,numCW1],dtype=float), dtype=tf.float32, name="constraintWeights1")
            constraintWeights2 = tf.Variable(
                np.zeros([1,numCW2],dtype=float), dtype=tf.float32, name="constraintWeights2")
            # potential bug here
            self.constraintWeights = tf.concat(
                [tf.nn.softmax(constraintWeights1,axis=1),
                    tf.nn.softmax(constraintWeights2,axis=1)]
                ,axis=1)

            # softmax
            constr_vars = tf.expand_dims(
                tf.stack(constraints,axis=0),axis=1) # TD: * constraint_variable parameters

            constr_loss = config.tflLossWeight*tf.squeeze(tf.matmul(
                    self.constraintWeights,constr_vars))
        elif config.tflWeightMode=='fw':
            # normal solution
            self.constraintWeights = tf.no_op()
            constr_vars = tf.stack(constraints,axis=0) # TD: * constraint_variable parameters
            constr_loss = config.tflLossWeight*tf.math.reduce_sum(constr_vars)

        return constr_loss


    # Computes mean cross entropy loss between logits and answers.
    def addAnswerLossOp(self, logits, answers, answerFreqs, answerFreqNums, tasks=None):
        # modify with masks for answers and tasks
        losses = list()
        if config.tflSS or config.tflFK:
            ATmask = [1845,1893]

            if config.lossType == "softmax": # or config.ansFormat == "mc":
                if len(logits) >= 1:
                    ansLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels = answers, logits = logits[0])
                    losses = [ansLoss]

                if len(logits) >= 2:
                    tskLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = tasks, logits = logits[1]) #48
                    losses += [tskLoss]
            else:
                raise NotImplementedError('loss type - {}'.format(config.lossType))

            self.answerLossList.append(tf.reduce_mean(ansLoss))
            self.taskLossList.append(tf.reduce_mean(tskLoss))
        else:
            if config.lossType == "softmax": # or config.ansFormat == "mc":
                losses = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = answers, logits = logits[0])

            elif config.lossType == "svm":
                answers = tf.one_hot(answers, config.answerWordsNum) # , axis = -1
                losses = ops.hingeLoss(labels = answers, logits = logits[0])

            elif config.lossType == "probSoftmax":
                answerFreqs = tf.to_float(answerFreqs)
                answerDist = answerFreqs / tf.expand_dims(tf.to_float(answerFreqNums), axis = -1)
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels = answerDist, logits = logits[0])
                if config.weightedSoftmax:
                    weights = tf.to_float(answerFreqNums) / float(config.AnswerFreqMaxNum)
                    losses *= weights
            elif config.lossType == "sigmoid":
                if config.dataset == "VQA":
                    answerFreqs = tf.to_float(answerFreqs)
                    answerDist = answerFreqs / float(config.AnswerFreqMaxNum)
                else:
                    answerDist = tf.one_hot(answers, config.answerWordsNum)
                if config.lossWeight == 1:
                    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = answerDist, logits = logits[0])
                else:
                    print("weighted sigmoid")
                    losses = tf.nn.weighted_cross_entropy_with_logits(targets = answerDist, logits = logits[0],
                        pos_weight = config.lossWeight)
                if config.ansWeighting or config.ansWeightingRoot:
                    losses *= self.answerDict.weights
                losses = tf.reduce_sum(losses, axis = -1)
            else:
                print("non-identified loss")

            self.answerLossList.append(losses)

        if config.tflSS or config.tflFK:
            if len(logits) == 2:
                loss = tf.reduce_mean(ansLoss) + tf.reduce_mean(tskLoss)
            else:
                loss = tf.reduce_mean(ansLoss)
        else:
            loss = tf.reduce_mean(losses)

        return loss, losses

    # Computes predictions (by finding maximal logit value, corresponding to highest probability)
    # and mean accuracy between predictions and answers.
    def addPredOp(self, logits, answers, outType='ans', predMask=None): # , answerFreqs
        with tf.variable_scope("pred_" + outType):
            if predMask != None:
                answerWordsNum = predMask[1] - predMask[0]
                predLogits = logits
            else:
                answerWordsNum = config.answerWordsNum
                predLogits = logits

            if config.ansFormat == "oe":# and config.ansAddUnk:
                mask = tf.to_float(tf.sequence_mask([2], answerWordsNum)) * (-1e30) # 1 or 2?
                predLogits += mask

            preds = tf.to_int32(tf.argmax(predLogits, axis = -1)) # tf.nn.softmax(

            if config.dataset == "VQA" and config.ansFormat == "oe":
                agreeing = tf.reduce_sum(tf.one_hot(preds, answerWordsNum) * self.answerFreqs, axis = -1)
                corrects = tf.minimum(agreeing * 0.3, 1.0) # /3 ?
            else:
                corrects = tf.to_float(tf.equal(preds, answers))

            correctNum = tf.reduce_sum(corrects)
            acc = tf.reduce_mean(corrects)
            if predMask[0] == 0:
                self.correctNumList.append(correctNum)
                self.answerAccList.append(acc)
            else:
                self.taskCorrectNumList.append(correctNum)
                self.taskAccList.append(acc)

        return preds, corrects, correctNum

    # Creates optimizer (adam)
    def addOptimizerOp(self):
        with tf.variable_scope("trainAddOptimizer"):
            self.globalStep = tf.Variable(0, dtype = tf.int32, trainable = False, name = "globalStep") # init to 0 every run?
            if config.hvdspeed:
                # optimizer = tf.train.AdamOptimizer(learning_rate = self.lr * hvd.size())
                if config.hvdoptim == 'average':
                    optimizer = tf.train.AdamOptimizer(learning_rate = self.lr * hvd.size()) #
                    self.passes_per_step = 1
                    optimizer = hvd.DistributedOptimizer(
                        optimizer, sparse_as_dense=True,
                        compression=hvd.Compression.fp16)
                elif config.hvdoptim == 'adasum':
                    optimizer = tf.train.AdamOptimizer(learning_rate = self.lr * hvd.local_size()) #
                    self.passes_per_step = tf.cast(
                        tf.ceil(float(config.batchSize)/tf.cast(self.batchSizeAll,tf.float32)),tf.int64)
                    # self.passes_per_step = tf.cast(6*(config.batchSize/21),tf.int64) # unique batch family style

                    optimizer = hvd.DistributedOptimizer(
                        optimizer, sparse_as_dense=True,
                        backward_passes_per_step=self.passes_per_step,
                        compression=hvd.Compression.fp16,
                        op=hvd.Adasum)
                # backward_passes_per_step=tf.cast(tf.ceil(32./tf.cast(self.batchSizeAll,tf.float32)),tf.int64),
                #,backward_passes_per_step=4,op=hvd.Adasum,backward_passes_per_step=2, compression=hvd.Compression.fp16,tf.cast(tf.round(32./self.batchSizeAll),tf.int64)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)

            if config.subsetOpt:
                self.subsetOptimizer = tf.train.AdamOptimizer(learning_rate = self.lr * config.subsetOptMult)

        return optimizer

    '''
    Computes gradients for all variables or subset of them, based on provided loss,
    using optimizer.
    '''
    def computeGradients(self, optimizer, loss, trainableVars = None): # tf.trainable_variables()
        with tf.variable_scope("computeGradients"):
            if config.trainSubset:
                trainableVars = []
                allVars = tf.trainable_variables()
                for var in allVars:
                    if any((s in var.name) for s in config.varSubset):
                        trainableVars.append(var)

            if config.subsetOpt:
                trainableVars = []
                subsetVars = []
                allVars = tf.trainable_variables()
                for var in allVars:
                    if any((s in var.name) for s in config.varSubset):
                        subsetVars.append(var)
                    else:
                        trainableVars.append(var)

            gradients_vars = optimizer.compute_gradients(loss, trainableVars)

            if config.subsetOpt:
                self.subset_gradients_vars = self.subsetOptimizer.compute_gradients(loss, subsetVars)
                self.subset_gradientVarsList.append(self.subset_gradients_vars)

        return gradients_vars

    '''
    Apply gradients. Optionally clip them, and update exponential moving averages
    for parameters.
    '''
    def addTrainingOp(self, optimizer, gradients_vars):
        with tf.variable_scope("train"):
            gradients, variables = zip(*gradients_vars)
            norm = tf.global_norm(gradients)

            # gradient clipping
            if config.clipGradients:
                clippedGradients, _ = tf.clip_by_global_norm(gradients, config.gradMaxNorm, use_norm = norm)
                gradients_vars = zip(clippedGradients, variables)

            # updates ops (for batch norm) and train op
            updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updateOps):
                train = optimizer.apply_gradients(gradients_vars, global_step = self.globalStep)

                if config.subsetOpt:
                    subsetTrain = self.subsetOptimizer.apply_gradients(self.subset_gradientVarsAll)
                    train = tf.group(train, subsetTrain)

            # exponential moving average
            if config.useEMA:
                ema = tf.train.ExponentialMovingAverage(decay = config.emaDecayRate)
                maintainAveragesOp = ema.apply(tf.trainable_variables())

                with tf.control_dependencies([train]):
                    trainAndUpdateOp = tf.group(maintainAveragesOp)

                train = trainAndUpdateOp

                self.emaDict = ema.variables_to_restore()

        return train, norm

    def averageAcrossTowers(self, gpusNum):
        if gpusNum == 1:
            self.lossAll = self.lossList[0]
            self.answerLossAll = self.answerLossList[0]
            self.answerAccAll = self.answerAccList[0]
            self.answerCorrectNumAll = self.correctNumList[0]
            self.answerPredsAll = self.answerPredsList[0]
            if config.tflSS or config.tflFK:
                self.taskLossAll = self.taskLossList[0]
                self.taskAccAll = self.taskAccList[0]
                self.taskCorrectNumAll = self.taskCorrectNumList[0]
                self.taskPredsAll = self.taskPredsList[0]
                self.logicLossAll = self.logicLossList[0]
            self.gradientVarsAll = self.gradientVarsList[0]

            if config.subsetOpt:
                self.subset_gradientVarsAll = self.subset_gradientVarsList[0]
        else:
            self.lossAll = tf.reduce_mean(tf.stack(self.lossList, axis = 0), axis = 0)
            self.answerLossAll = tf.reduce_mean(tf.stack(self.answerLossList, axis = 0), axis = 0)
            self.answerAccAll = tf.reduce_mean(tf.stack(self.answerAccList, axis = 0), axis = 0)
            self.correctNumAll = tf.reduce_sum(tf.stack(self.correctNumList, axis = 0), axis = 0)
            self.answerPredsAll = tf.concat(self.answerPredsList, axis = 0)
            self.taskLossAll = tf.reduce_mean(tf.stack(self.taskLossList, axis = 0), axis = 0)
            self.taskAccAll = tf.reduce_mean(tf.stack(self.taskAccList, axis = 0), axis = 0)
            self.taskCorrectNumAll = tf.reduce_sum(tf.stack(self.taskCorrectNumList, axis = 0), axis = 0)
            self.taskpredsAll = tf.concat(self.taskPredsList, axis = 0)
            self.logicLossAll = tf.reduce_mean(tf.stack(self.logicLossList, axis = 0), axis = 0)

            self.gradientVarsAll = []
            for grads_var in zip(*self.gradientVarsList):
                gradients, variables = zip(*grads_var)
                if gradients[0] != None:
                    avgGradient = tf.reduce_mean(tf.stack(gradients, axis = 0), axis = 0)
                else:
                    avgGradient = None
                var = variables[0]
                grad_var = (avgGradient, var)
                self.gradientVarsAll.append(grad_var)

            if config.subsetOpt:
                self.subset_gradientVarsAll = []
                for grads_var in zip(*self.subset_gradientVarsList):
                    gradients, variables = zip(*grads_var)
                    if gradients[0] != None:
                        avgGradient = tf.reduce_mean(tf.stack(gradients, axis = 0), axis = 0)
                    else:
                        avgGradient = None
                    var = variables[0]
                    grad_var = (avgGradient, var)
                    self.subset_gradientVarsAll.append(grad_var)

    def adasumAcrossDancers(self):
        # collect and observe data on all dancers
        self.lossAll = hvd.allreduce(self.lossAll, op=hvd.Average)
        self.answerLossAll = hvd.allreduce(self.answerLossAll, op=hvd.Average)
        self.answerAccAll = hvd.allreduce(self.answerAccAll, op=hvd.Average)
        self.answerCorrectNumAll = hvd.allreduce(self.answerCorrectNumAll, op=hvd.Sum)
        if config.tflSS or config.tflFK:
            self.taskLossAll = hvd.allreduce(self.taskLossAll, op=hvd.Average)
            self.taskAccAll = hvd.allreduce(self.taskAccAll, op=hvd.Average)
            self.taskCorrectNumAll = hvd.allreduce(self.taskCorrectNumAll, op=hvd.Sum)
            self.logicLossAll = hvd.allreduce(self.logicLossAll, op=hvd.Average)
        self.batchSizeAll = hvd.allreduce(self.batchSizeAll, op=hvd.Sum)
        if config.getPreds:
            self.answerPredsAll = hvd.allgather(self.answerPredsAll)
            self.taskPredsAll = hvd.allgather(self.taskPredsAll)

    def trim2DVectors(self, vectors, vectorsLengths):
        maxLength = np.max(vectorsLengths)
        return vectors[:,:maxLength]

    def trimData(self, data):
        data["questions"] = self.trim2DVectors(data["questions"], data["questionLengths"])
        return data

    '''
    Builds predictions JSON, by adding the model's predictions and attention maps
    back to the original data JSON.
    '''
    def buildPredsList(self, data, predictions, attentionMaps):
        predsList = []

        for i, instance in enumerate(data["instances"]):

            if predictions is not None:
                if config.ansFormat == "oe":
                    pred = self.answerDict.decodeId(predictions[i])
                else:
                    pred = instance["choices"][predictions[i]]
                instance["prediction"] = pred

            # aggregate np attentions of instance i in the batch into 2d list
            attMapToList = lambda attMap: [step[i].tolist() for step in attMap]
            if attentionMaps is not None:
                attentions = {k: attMapToList(attentionMaps[k]) for k in attentionMaps}
                instance["attentions"] = attentions

            predsList.append(instance)

        return predsList

    '''
    Processes a batch of data with the model.

    Args:
        sess: TF session

        data: Data batch. Dictionary that contains numpy array for:
        questions, questionLengths, answers.
        See preprocess.py for further information of the batch structure.

        images: batch of image features, as numpy array. images["images"] contains
        [batchSize, channels, h, w]

        train: True to run batch for training.

        getAtt: True to return attention maps for question and image (and optionally
        self-attention and gate values).

    Returns results: e.g. loss, accuracy, running time.
    '''
    def runBatch(self, sess, data, images, train, getUpdate=True, vBatchSize=0,
                 getPreds = False, getAtt = False, allData = None, progress=None):
        batchSizeOp = self.batchSizeAll
        numQuestionsOp = self.numQuestions \
            if config.batchStyle in ['family','hybrid'] and (config.tflSS or config.tflFK) \
            else self.noOp
        indicesOp = self.noOp

        trainOp = self.trainOp if train and getUpdate else self.noOp
        gradNormOp = self.gradNorm if train else self.noOp

        if config.tflSS or config.tflFK:
            lossValOp = (self.lossAll, self.answerLossAll, self.taskLossAll, self.logicLossAll)
        else:
            lossValOp = (self.lossAll, self.noOp, self.noOp)

        answerPredsOp = (self.answerPredsAll, self.answerCorrectNumAll, self.answerAccAll)
        if config.tflSS or config.tflFK:
            taskPredsOp = (self.taskPredsAll, self.taskCorrectNumAll, self.taskAccAll)
        else:
            taskPredsOp = (self.noOp , self.noOp, self.noOp)

        if config.tflSS or config.tflFK:
            constraintWeightsOp = self.constraintWeights
        else:
            constraintWeightsOp = self.noOp

        attOp = self.macCell.attentions if not config.useBaseline else (self.attentions if config.baselineNew else self.noOp)

        time0 = time.time()
        feed = self.createFeedDict(data, images, train, progress, vBatchSize)

        time1 = time.time()

        batchSize, numQuestions, indices, _, loss, answerPredsInfo, \
        taskPredsInfo, gradNorm, attentionMaps, constraintWeights = sess.run(
            [batchSizeOp, numQuestionsOp, indicesOp, trainOp, lossValOp,
             answerPredsOp, taskPredsOp, gradNormOp, attOp, constraintWeightsOp],
            feed_dict = feed)
        # sess.run(self.hookjoin)

        time2 = time.time()

        answerPredsList = []
        taskPredsList = []
        if getPreds:
            if data is None:
                data = [allData["instances"][i] for i in indices]
            answerPredsList = self.buildPredsList(data, answerPredsInfo[0], attentionMaps if getAtt else None)
            if config.tflSS or config.tflFK:
                taskPredsList = self.buildPredsList(data, taskPredsInfo[0], attentionMaps if getAtt else None)

        if not config.tflSS and not config.tflFK:
            taskPredsInfo = [np.nan, np.nan, np.nan]

        # float(predsInfo[1])/float(batchSize)
        return {"loss": loss[0],
                "answerLoss": loss[1] if config.tflSS or config.tflFK else -1,
                "taskLoss": loss[2] if config.tflSS or config.tflFK else -1,
                "logicLoss": loss[3] if config.tflSS or config.tflFK else -1,
                "answerCorrectNum": answerPredsInfo[1],
                "answerAcc": answerPredsInfo[2],
                "answerPreds": answerPredsList,
                "taskCorrectNum": taskPredsInfo[1],
                "taskAcc": taskPredsInfo[2],
                "taskPreds": taskPredsList,
                "gradNorm": gradNorm if train else -1.0,
                "readTime": time1 - time0,
                "trainTime": time2 - time1,
                "batchSize": batchSize,
                "numQuestions": numQuestions,
                "constraintWeights": constraintWeights}

    def build(self):
        self.addPlaceholders()
        self.optimizer = self.addOptimizerOp()

        self.gradientVarsList = []
        if config.subsetOpt:
            self.subset_gradientVarsList = []
        self.lossList = []

        self.answerLossList = []
        self.correctNumList = []
        self.answerAccList = []
        self.answerPredsList = []

        if config.tflSS or config.tflFK:
            self.logicLossList = []
            self.taskLossList = []
            self.taskCorrectNumList = []
            self.taskAccList = []
            self.taskPredsList = []

        ATmask = [1845,1893]

        with tf.variable_scope("macModel"):
            for i in range(config.gpusNum):
                with tf.device("/gpu:{}".format(i)):
                    with tf.name_scope("tower{}".format(i)) as scope:
                        self.initTowerBatch(i, config.gpusNum, self.batchSizeAll)

                        self.loss = tf.constant(0.0)

                        # embed questions words (and optionally answer words)
                        questionWords, choices = self.embeddingsOp(self.questionIndices,
                            self.choicesIndices, self.embeddingsInit)

                        projWords = projQuestion = ((config.encDim != config.ctrlDim) or config.encProj)
                        questionCntxWords, vecQuestions = self.encoder(questionWords,
                            self.questionLengths, projWords, projQuestion, config.ctrlDim)

                        # Image Input Unit (stem)
                        imageFeatures, imageDim = self.stem(self.images, self.imageInDim, config.memDim)

                        # baseline model
                        if config.useBaseline:
                            # inpImg = imageFeatures if config.baselineNew else self.images
                            # inpDim = imageDim if config.baselineNew else self.imageInDim
                            output, dim = self.baseline(vecQuestions, config.ctrlDim,
                                imageFeatures, imageDim, config.attDim) # self.images
                        # MAC model
                        else:
                            finalControl, finalMemory = self.MACnetwork(imageFeatures, vecQuestions,
                                questionWords, questionCntxWords, self.questionLengths)

                            # Output Unit - step 1 (preparing classifier inputs)
                            output, dim = self.outputOp(finalMemory, finalControl, vecQuestions,
                                self.images, self.imageInDim)

                        # Output Unit - step 2a (answer classifier)
                        ansLogit = self.classifier(output, dim, choices, self.choicesNums, outType='ans')

                        # Output Unit - step 2b (task classifier)
                        if config.checkVal or config.checkTrain:
                            tskLogit = tf.one_hot(
                                self.taskIndicesAll,depth=ATmask[1]-ATmask[0],
                                on_value=0.999,off_value=0.001,
                                dtype=tf.float32)
                        else:
                            if (config.batchStyle == 'hybrid'
                                or config.batchStyle == 'family') \
                                and (config.tflSS or config.tflFK):
                                    tskLogit = self.classifier(
                                        vecQuestions[0:self.numQuestions,:],
                                        config.ctrlDim, choices,
                                        self.choicesNums, outType='tsk')
                            else:
                                tskLogit = tf.stop_gradient(
                                    self.classifier(vecQuestions, config.ctrlDim, choices, self.choicesNums, outType='tsk'))

                        if config.tflSS or config.tflFK:
                            self.logits = tf.concat(
                                [ansLogit[:self.numQuestions,:],tskLogit],axis=1)

                        # compute loss, predictions, accuracy
                        if config.dataset == "VQA":
                            self.answerFreqs = self.aggregateFreqs(self.answerFreqLists, self.answerFreqNums)
                        else:
                            self.answerFreqs = None
                            self.answerFreqNums = None

                        # compute logic and point-wise loss
                        if config.tflSS or config.tflFK:

                            tfl.World.reset()
                            tfl.World._evaluation_mode = tfl.LOSS_MODE
                            if config.tflSS:
                                tfl.setTNorm(id=tfl.SS, p=config.tflLambda)
                            elif config.tflFK:
                                tfl.setTNorm(id=tfl.FRANK, p=config.tflLambda)

                            answerLoss, self.losses = self.addAnswerLossOp(
                                [ansLogit, tskLogit], self.answerIndices,
                                self.answerFreqs, self.answerFreqNums, self.taskIndices) # self.answerLosses

                            if config.batchStyle == 'hybrid':
                                logicLoss = self.addLogicLossOp(
                                    self.logits,
                                    vecQuestions[:self.numQuestions,:],
                                    self.answerIndices[:self.numQuestions],
                                    self.taskIndices
                                )
                            else:
                                logicLoss = self.addLogicLossOp(
                                    self.logits,vecQuestions)
                            self.loss += logicLoss + answerLoss
                            self.logicLossList.append(logicLoss)

                        elif config.batchStyle=='hybrid':
                            answerLoss, self.losses = self.addAnswerLossOp(
                                [ansLogit,tskLogit], self.answerIndices,
                                self.answerFreqs, self.answerFreqNums, self.taskIndices) # self.answerLosses
                            self.loss += answerLoss

                        else:
                            answerLoss, self.losses = self.addAnswerLossOp(
                                [ansLogit], self.answerIndices,
                                self.answerFreqs, self.answerFreqNums) # self.answerLosses
                            self.loss += answerLoss

                        self.answerPreds, self.answerCorrects, self.answerCorrectNum = \
                            self.addPredOp(ansLogit, self.answerIndices, predMask=[0,ATmask[0]]) # ,self.answerFreqs
                        self.answerPredsList.append(self.answerPreds)

                        if config.tflSS or config.tflFK:
                            self.taskPreds, self.taskCorrects, self.taskCorrectNum = \
                                self.addPredOp(tskLogit, self.taskIndices, predMask=[ATmask[0],ATmask[1]]) # ,self.answerFreqs
                            self.taskPredsList.append(self.taskPreds)

                        self.lossList.append(self.loss)

                        self.loss = self.loss / tf.cast(self.passes_per_step,tf.float32)

                        # compute gradients
                        gradient_vars = self.computeGradients(self.optimizer, self.loss, trainableVars = None)
                        self.gradientVarsList.append(gradient_vars)

                        # reuse variables in next towers
                        tf.get_variable_scope().reuse_variables()

        self.averageAcrossTowers(config.gpusNum) # gpusNum = 1 in all exps
        # if config.finalTest: self.answerPredsAll = hvd.allgather(self.answerPredsAll)

        self.trainOp, self.gradNorm = self.addTrainingOp(self.optimizer, self.gradientVarsAll)
        self.noOp = tf.no_op()
        self.hookjoin = tf.no_op()

        if config.hvdspeed:
            # if not config.train:
            # self.adasumAcrossDancers()
            self.hookjoin = hvd.join()
