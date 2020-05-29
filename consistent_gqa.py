from __future__ import print_function
import sys
sys.path.append('/home/catle/Projects/deepproblog/')
sys.path.append('/home/catle/Projects/lyrics_tfnorm/tf-logic/')
import pickle as pkl
import keras
from tensorflow.contrib.eager.python import tfe
eager = False
if eager: tfe.enable_eager_execution()
import tensorflow as tf
import tfl
import numpy as np
import os
from collections import OrderedDict

# MACNet
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")

import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue

from collections import defaultdict
from termcolor import colored, cprint
import datetime
import numpy as np
import threading
import random
import os
import time
import math
import h5py
import json

from macnetwork.config import config, loadDatasetConfig, parseArgs
from macnetwork.preprocess import Preprocesser, bold, bcolored, writeline, writelist
# from macnetwork.model import MACnet
from model_gqa import MACnet
from logger import Logger
from macnetwork.main import loadImageBatch, openImageFiles, setSession, setSavers, loadWeights
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def preprocessFamily(images, question_data, tier, imageIndex, preprocessor):
    """
    Create a question family
    """
    no_question = len(question_data); instances = []
    for i, an_item in enumerate(question_data):
        imageId = an_item['data']['imageId']; questionStr = an_item['data']['question']
        imageInfo = imageIndex[imageId]
        imageId = {"group": tier, "id": imageId, "idx": imageInfo["index"]} # int(imageId)

        question = preprocessor.encodeQuestionStr(questionStr)
        instances.append({
            "questionStr": questionStr,
            "question": question,
            "answer": an_item['data']['answer'], # Dummy answer
            "imageId": imageId,
            "objectsNum": imageInfo['objectsNum'],
            "tier": tier,
            "index": i
        })

    family_data = preprocessor.vectorizeData(instances)
    family_images = loadImageBatch(images,{"imageIds":[imageId]*no_question})
    return family_data, family_images

def openImageSplit(data,tier):
    """
    Open files of tier = 'train'
    """
    images = data['main'][tier]['images']
    imgsInfoFilename = config.imgsInfoFile(tier)
    with open(imgsInfoFilename, "r") as file:
        imageIndex = json.load(file)
    openImageFiles(images)
    return images, imageIndex


def main():
    # --------------------- TRAINING PARAMETERS----------------------------------
    '''
    with open('/home/catle/Projects/deepproblog/examples/A2I2/gqa/data/train_verify_global_balanced_600_answer.pkl','rb') as af:
        answerDict = pkl.load(af)
    '''
    max_iterations = 300
    data_augmentation = False
    subtract_pixel_mean = True
    use_logic = config.useLogicConstr; constr_weight = .001
    nextElement=None; evalTrain_ratio = .1
    dataOps = None
    minibatch_size = 1
    supervided_size = 10 # -1 means all of them

    print('Setting up logger ...')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_total_logger = Logger('train','total',current_time)
    train_constr_logger = Logger('train','constr',current_time)
    train_task_logger = Logger('train','task',current_time)
    train_ans_logger = Logger('train','ans',current_time)

    # -----------------------------DATA------------------------------------------
    # Load the GQA data.
    activation_map = list()
    with open('/home/catle/Projects/lyrics_tfnorm/data/train_verify_global_balanced_raw_300.pkl','rb') as fh:
        question_data = pkl.load(fh)
    num_families = len(question_data['question_dict'].keys())
    task_dict = question_data['task_dict']
    binary_answers = ['yes','no']
    open_answers = [ k for k in question_data['answer_list'] if k not in binary_answers]

    for i, answer in enumerate(open_answers):
        if i == 0:
            open_answers_str = '{}(q)'.format(answer)
        else:
            open_answers_str += ' or {}(q)'.format(answer)

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
    for k, v in question_data['task_dict'].items():
        preprocessor.answerDict.addToVocab(k)

    data , embeddings, answerDict, questionDict = preprocessor.preprocessData()

    train_images, trainImageIndex = openImageSplit(data,'train')
    # evalTrain_images, evalTrainImageIndex = openImageSplit(data,'evalTrain')

    num_answers = 1845
    num_tasks = len(answerDict.sym2id.keys())-num_answers
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))

    # -----------------------------MODEL------------------------------------------
    print("Creating model graph...")
    gqa_log = MACnet(embeddings, answerDict, questionDict, nextElement)

    # initializer
    # init = tf.global_variables_initializer()

    ## savers
    # savers = setSavers(gqa_log)
    # saver, emaSaver = savers["saver"], savers["emaSaver"]
    saver = tf.train.Saver()

    # sessionConfig
    sessionConfig = setSession()
    sess = tf.Session(config=sessionConfig)

    # restore / initialize weights, initialize epoch variable
    # epoch = loadWeights(sess, saver, init)

    # define tf-logic model
    '''
    activation_map = [1845,4]
    gqa = tfl.functions.FromTFModel(gqa_log, activation_map)
    '''

    # -------------------------------LOGIC------------------------------------------
    print("Creating logic component graph for consistent GQA...")
    predicates_dict = answerDict.sym2id # load from answer

    # Domains Definition
    class IsDiff(tfl.functions.AbstractFunction):
        def __call__(self, a, b):
            dist = tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1))
            return tf.where(dist > .0001*tf.ones_like(dist), tf.ones_like(dist),tf.zeros_like(dist))

    is_diff = IsDiff()

    """
    # ----------------------------- TRIAL ---------------------------------------
    tfl.World.reset()
    tfl.World._evaluation_mode = tfl.LOSS_MODE
    tfl.setTNorm(id=tfl.SS, p=1)

    question_ph = tf.placeholder(dtype=tf.float32,shape=(None,7))
    answer_ph = tf.placeholder(dtype=tf.int32,shape=(None,))
    task_ph = tf.placeholder(dtype=tf.int32,shape=(None,))
    questions = tfl.Domain("Question", data=question_ph, size=1)
    tfl.Predicate("isDiff", domains=["Question","Question"], function=is_diff)

    # Predicates Definition
    for k,v in predicates_dict.items():
        tfl.Predicate(k, domains=("Question",), function=tfl.functions.Slice(gqa, v))

    constraints = []
    constraints.append(tfl.constraint("forall q: verifyGlobalTrue(q) -> yes(q)"))
    constraints.append(tfl.constraint("forall q: verifyGlobalFalse(q) -> no(q)"))
    constraints.append(tfl.constraint("forall q: queryGlobal(q) <-> cloudy(q) or cloudless(q)"))
    constraints.append(tfl.constraint("forall q: chooseGlobal(q) <-> cloudy(q) or cloudless(q) "))
    constraints.append(tfl.constraint("forall p: forall q: isDiff(p,q) and (verifyGlobalTrue(p) -> verifyGlobalFalse(q))"))
    constraints.append(tfl.constraint("forall p: forall q: isDiff(p,q) and (verifyGlobalTrue(p) <-> queryGlobal(q))"))
    constraints.append(tfl.constraint("forall p: forall q: isDiff(p,q) and (queryGlobal(p) <-> chooseGlobal(q))")) # ??? same answer

    with tf.variable_scope("macModel",reuse=tf.AUTO_REUSE):
        for i in range(config.gpusNum):
            with tf.device("/gpu:{}".format(i)):
                with tf.name_scope("tower{}".format(i)) as scope:

                    answer_ohv = tf.one_hot(answer_ph, activation_map[0])
                    ans_logits = gqa_log.logits[:,:activation_map[0]]
                    ans_preds = tf.to_int32(tf.argmax(ans_logits, axis = -1))
                    ans_loss = tf.losses.softmax_cross_entropy(onehot_labels = answer_ohv, logits = ans_logits)

                    task_ohv = tf.one_hot(task_ph, activation_map[1])
                    task_logits = gqa_log.logits[:,activation_map[0]:sum(activation_map[0:2])]
                    task_loss = tf.losses.softmax_cross_entropy(onehot_labels = task_ohv, logits = task_logits)

                    constr_loss = tf.add_n(constraints)

                    contr_weight = tf.placeholder(tf.float32, shape = ())

                    gqa_log.loss += (ans_loss + task_loss + .001*constr_loss) # const_weight

                    gqa_log.lossList.append(gqa_log.loss)

                    gradient_vars = gqa_log.computeGradients(gqa_log.optimizer, gqa_log.loss, trainableVars=None)
                    gqa_log.gradientVarsList.append(gradient_vars)

                    # reuse variables in next towers
                    tf.get_variable_scope().reuse_variables()

    gqa_log.averageAcrossTowers(config.gpusNum)
    trainOp, gradNormOp = gqa_log.addTrainingOp(gqa_log.optimizer, gqa_log.gradientVarsAll)
    """
    # ------------------------EXECUTION----------------------------------- #
    family_index = 0
    evalTrain_ansAcc = []
    evalTrain_taskAcc = []

    for imageId, question_family in question_data['question_dict'].items():
        family_index += 1
        # create tf-logic world
        tfl.World.reset()
        tfl.World._evaluation_mode = tfl.LOSS_MODE
        tfl.setTNorm(id=tfl.SS, p=0)

        # define tf-logic model
        activation_map = [num_answers,num_tasks]
        gqa = tfl.functions.FromTFModel(gqa_log, activation_map)

        # prepare a train (family) dataset
        family_data, family_images = preprocessFamily(train_images, question_family, 'train', trainImageIndex, preprocessor)
        x_train = np.float32(family_data['questions'])
        y_train = np.int32(family_data['answers'])
        z_train = np.int32([predicates_dict[a_question['task']]-num_answers
                            for a_question in question_family])
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train,y_train,z_train)).batch(1).repeat()
        iter_train = dataset_train.make_one_shot_iterator()
        question_tr, answer_tr, task_tr = iter_train.get_next()


        no_questions = y_train.shape[0]
        questions = tfl.Domain("Question", data=question_tr, size=no_questions)

        tfl.Predicate("isDiff", domains=["Question","Question"], function=is_diff)

        # Predicates Definition
        for k,v in predicates_dict.items():
            tfl.Predicate(k, domains=("Question",), function=tfl.functions.Slice(gqa, v))


        # Compute loss
        answer_ohv = tf.one_hot(answer_tr, activation_map[0])
        ans_logits = gqa_log.logits[:,:activation_map[0]]
        ans_loss = tf.losses.softmax_cross_entropy(onehot_labels = answer_ohv, logits = ans_logits)
        # Compute prediction
        ans_preds = tf.to_int32(tf.argmax(ans_logits, axis = -1))
        ans_corrects = tf.to_float(tf.equal(ans_preds, answer_tr))
        ans_correctNum = tf.reduce_sum(ans_corrects)
        ans_accuracy = tf.reduce_mean(ans_corrects)

        family_name=str(family_index)+'_'+imageId
        ans_summary_ops = tuple([
            tf.summary.scalar(name='ans_loss',tensor=ans_loss,collections=['ans','loss'],family=family_name),
            tf.summary.scalar(name='ans_accuracy',tensor=ans_accuracy,collections=['ans','acc'],family=family_name)])

        if use_logic:
            constraints = []
            constraints.append(tfl.constraint("forall q: verifyGlobalTrue(q) -> yes(q)"))
            constraints.append(tfl.constraint("forall q: verifyGlobalFalse(q) -> no(q)"))
            # constraints.append(tfl.constraint("forall q: queryGlobal(q) <-> {}".format(open_answers_str)))
            # constraints.append(tfl.constraint("forall q: chooseGlobal(q) <-> {}".format(open_answers_str)))
            # constraints.append(tfl.constraint("forall q: queryGlobal(q) <-> not yes(q) and not no(q)"))
            # constraints.append(tfl.constraint("forall q: chooseGlobal(q) <-> not yes(q) and not no(q)"))
            constraints.append(tfl.constraint("forall p: forall q: isDiff(p,q) and (verifyGlobalTrue(p) -> verifyGlobalFalse(q))"))
            constraints.append(tfl.constraint("forall p: forall q: isDiff(p,q) and (verifyGlobalTrue(p) <-> queryGlobal(q))"))
            constraints.append(tfl.constraint("forall p: forall q: isDiff(p,q) and (queryGlobal(p) <-> chooseGlobal(q))")) # correct

            # Compute loss
            task_ohv = tf.one_hot(task_tr, activation_map[1])
            task_logits = gqa_log.logits[:,activation_map[0]:sum(activation_map[0:2])]
            task_loss = tf.losses.softmax_cross_entropy(onehot_labels = task_ohv, logits = task_logits)
            # Compute prediction
            task_preds = tf.to_int32(tf.argmax(task_logits, axis = -1))
            task_corrects = tf.to_float(tf.equal(task_preds, task_tr))
            task_correctNum = tf.reduce_sum(task_corrects)
            task_accuracy = tf.reduce_mean(task_corrects)

            constr_loss = tf.add_n(constraints)
            total_loss = ans_loss + task_loss + constr_weight*constr_loss # contr_weight
            preds_op = (task_preds, task_corrects, task_accuracy, ans_preds, ans_corrects, ans_accuracy)
            constr_summary_ops = tuple([
                tf.summary.scalar(name='constr_loss', tensor=constr_weight*constr_loss, collections=['constr','loss'], family=family_name),
            ])
            total_summary_ops = tuple([
                tf.summary.scalar(name='total_loss', tensor=total_loss, collections=['total','loss'], family=family_name),
            ])
            task_summary_ops = tuple([
                tf.summary.scalar(name='task_loss', tensor=task_loss, collections=['task','loss'], family=family_name),
                tf.summary.scalar(name='task_accuracy',tensor=task_accuracy, collections=['task','acc'], family=family_name)
            ])
        else:
            total_loss = ans_loss
            preds_op = (ans_preds, ans_corrects, ans_accuracy)
            total_summary_ops = tuple([
                tf.summary.scalar(name='total_loss', tensor=total_loss, collections=['total','loss'], family=family_name)
            ])

        # summary_ops = tuple(summary_ops)
        if family_index < (1-evalTrain_ratio)*num_families: # train
            train_flg = True
            train_op = tf.train.AdamOptimizer(0.001).minimize(total_loss)
        else: # evalTrain
            train_flg = False
            train_op = gqa_log.noOp

        print("Session initialization the model ...")
        sess.run(tf.global_variables_initializer())

        if family_index > 1 or train_flg==False:
            print("Load model {}".format(config.expName))
            saver.restore(sess,'./model/{}.ckpt'.format(config.expName))

        feed_dict = gqa_log.createFeedDict(family_data, family_images, train_flg)

        # for i in range(iterations):
        total_loss_value, diff_loss_value, prev_loss_value, iter_value = 2.3*no_questions*2, 1., 9999., 0
        ans_loss_value, task_loss_value = 3.5*no_questions, 2.2*no_questions

        if train_flg: # train
            print('Train on #Iid {}'.format(imageId))
            while True:
                iter_value += 1
                if use_logic:
                    # break condition constrained training tfl.setTNorm(id=tfl.SS, p=0)"""
                    '''
                    if (diff_loss_value < .15 and
                        ans_loss_value < .35*no_questions and
                        task_loss_value < .35*no_questions):
                        break
                    '''

                    _, total_loss_value, task_loss_value, ans_loss_value, preds_info, \
                        total_summary_info, task_summary_info,\
                        constr_summary_info, ans_summary_info = \
                        sess.run([
                            train_op, total_loss, task_loss, ans_loss, preds_op,
                            total_summary_ops, task_summary_ops,
                            constr_summary_ops, ans_summary_ops]
                            , feed_dict=feed_dict)
                    '''
                    print(
                        '#{} IId {} - {} Qs - Iter {} W-loss:{:.2f}, T-loss/acc:{:.2f}/{:.2f}, A-loss/acc:{:.2f}/{:.2f}'.format(
                        family_index,imageId,no_questions,iter_value,
                        total_loss_value,task_loss_value,preds_info[2],
                        ans_loss_value,preds_info[5]))
                    '''
                    train_constr_logger.put(constr_summary_info, iter_value)
                    train_task_logger.put(task_summary_info, iter_value)
                else:
                    if (diff_loss_value < .05 and
                        ans_loss_value < .35*no_questions): break

                    _, total_loss_value, ans_loss_value, preds_info, \
                        total_summary_info, ans_summary_info = sess.run([
                            train_op, total_loss, ans_loss, preds_op,
                            total_summary_ops, ans_summary_ops],
                            feed_dict=feed_dict)
                    '''
                    print(
                        '#{} ImageId {} - {} Qs - Iter {} W-loss:{:.2f}, A-loss/acc:{:.2f}/{:.2f}'.format(
                        family_index, imageId,no_questions,iter_value,
                        total_loss_value,ans_loss_value,preds_info[2]))
                    '''

                # record train into tensorboard log
                train_total_logger.put(total_summary_info, iter_value)
                train_ans_logger.put(ans_summary_info, iter_value)

                diff_loss_value = np.abs(total_loss_value-prev_loss_value)
                prev_loss_value = total_loss_value

                if iter_value > max_iterations:
                    break

            save_path = saver.save(sess, './model/{}.ckpt'.format(config.expName))
            print("Model saved in file: {}".format(save_path))
        else: # evalTrain
            print('evalTrain on #Iid {}'.format(imageId))
            # _, preds_info = sess.run([train_op, preds_op], feed_dict=feed_dict)
            _, total_loss_value, ans_loss_value, preds_info, \
                total_summary_info, ans_summary_info = sess.run([
                    train_op, total_loss, ans_loss, preds_op,
                    total_summary_ops, ans_summary_ops],
                    feed_dict=feed_dict)

            if use_logic:
                evalTrain_taskAcc.append(preds_info[2])
                evalTrain_ansAcc.append(preds_info[5])
            else:
                evalTrain_ansAcc.append(preds_info[2])

    if use_logic:
        print('EvalTrain: T-acc: {:.2f}, A-acc: {:.2f}'.format(
            np.mean(np.array(evalTrain_taskAcc)),
            np.mean(np.array(evalTrain_ansAcc))))
    else:
        print('EvalTrain: A-acc: {:.2f}'.format(
            np.mean(np.array(evalTrain_ansAcc))))

# -----------------------MAIN------------------------------------------ #
if __name__ == "__main__":
    parseArgs()
    loadDatasetConfig[config.dataset]()
    main()
