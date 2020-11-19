# This script is written to explore whether a GQA entailment is complete or fa
# missing - each scenegraph has different no of questions
import argparse
import pymongo
from pymongo import MongoClient
import pandas as pd
from pprint import pprint
import pickle as pkl
from collections import defaultdict
import json
from pprint import pprint
import multiprocessing as mp
import numpy as np
from functools import partial
import itertools
from itertools import chain, islice
from collections import ChainMap
from pymongo.errors import CursorNotFound
import random

pklDir = '/data/catle/Projects/lyrics_tfnorm/data/'

def init_database():
    try:
        client = MongoClient('localhost',27017)
        client.server_info()
    except:
        print("Connection Error: pls restart mongod service `mongod --dbpath /data/catle/mongod`")

    gqa_db = client.gqa
    return gqa_db

def generate_verifyGlobal(gqa_db, split, name, form, num_data=None, batch_size=None):
    """
    generate a dataset of all questions belong to verifyGlobal familiy in the same image
    """
    if split == 'train':
        col = gqa_db.train_all_questions
        if 'balanced' in name:
            semanticGlobal_family = gqa_db.train_balanced_questions.find(
                {'types.structural':'verify','types.semantic':'global'}
            ).batch_size(batch_size)

    elif split == 'val':
        col = gqa_db.val_all_questions
        if 'balanced' in name:
            semanticGlobal_family = gqa_db.val_balanced_questions.find(
                {'types.structural':'verify','types.semantic':'global'}
            ).batch_size(batch_size)

    name += '_' + form

    print('There are {} verifyGlobal questions {} split of GQA dataset'.format(semanticGlobal_family.count(), split))
    if num_data is not None:
        # semanticGlobal_family = semanticGlobal_family.aggregate([{'$sample': {'size': num_data}}])
        semanticGlobal_family.limit(num_data)
        name += '_' + str(num_data)
        print('...; however, we only use {} samples for train and evalTrain'.format(num_data))

    ############################################################
    # Generate all families of entailment for training
    semanticGlobal_family.rewind()
    questions_dict = dict()
    task_list = list()
    answer_list = list()
    for a_question in semanticGlobal_family:
        if len(a_question['entailed']) > 0: # check existences of entailed quests
            questions_list = []
            imageId=a_question['imageId']
            headAns=a_question['answer'],
            headTask=a_question['types']['structural']+\
                a_question['types']['semantic'].capitalize()
            if 'verify' in headTask: headTask += str(headAns=='yes')
            task_list.append(headTask); answer_list.append(headAns[0])
            headQuest='{}_{}'.format(a_question['key_id'],a_question['question'])

            if form == 'raw':
                questions_list.append({'data': a_question, 'task': headTask})
            else:
                questions_list.append({'data': headQuest, 'task': headTask})

            for key_id in a_question['entailed']:
                an_entailed_question=list(col.find({'key_id':key_id}))[0]
                tailAns=an_entailed_question['answer']
                tailTask=an_entailed_question['types']['structural']+\
                    an_entailed_question['types']['semantic'].capitalize()
                if 'verify' in tailTask: tailTask += str(tailAns=='yes')
                task_list.append(tailTask); answer_list.append(tailAns)
                tailQuest='{}_{}'.format(an_entailed_question['key_id'],an_entailed_question['question'])
                if form == 'raw':
                    questions_list.append({'data': an_entailed_question, 'task': tailTask})
                else:
                    questions_list.append({'data': tailQuest, 'task': tailTask})

            questions_dict[imageId] = questions_list

    answer_list = list(set(answer_list))
    task_list = list(set(task_list)); task_list.sort() # unique task list
    task_dict = { task: i for i, task in enumerate(task_list)}
    with open(pklDir + '{}_{}.pkl'.format(split,name),'wb') as fh:
        pkl.dump({'question_dict': questions_dict, 'task_dict': task_dict, 'answer_list': answer_list},fh)

def generate_entailed_data(split, name, form, num_data=None, batch_size=None, skip_limit=[0,0]):
    """
    generate a dataset of all questions belong to verifyGlobal familiy in the same image
    """
    gqa_db = init_database()
    if split == 'train' or split == 'train_balanced':
        col = gqa_db.train_all_questions
        if 'balanced' in split:
            semantic_family = gqa_db.train_balanced_questions.find(
                no_cursor_timeout=False, skip=int(skip_limit[0]),
                limit=int(skip_limit[1]))
        else:
            semantic_family = gqa_db.train_all_questions.find(
                no_cursor_timeout=False, skip=int(skip_limit[0]),
                limit=int(skip_limit[1]))
    elif split == 'val' or split == 'val_balanced':
        col = gqa_db.val_all_questions
        if 'balanced' in split:
            semantic_family = gqa_db.val_balanced_questions.find(
                no_cursor_timeout=False, skip=int(skip_limit[0]),
                limit=int(skip_limit[1]))
        else:
            semantic_family = gqa_db.val_all_questions.find(
                no_cursor_timeout=False, skip=int(skip_limit[0]),
                limit=int(skip_limit[1]))

    with open('./types_detailed.json','r') as json_fh:
        task_dict = json.load(json_fh)

    ############################################################
    # Generate all families of entailment for training
    semantic_family.rewind()
    questions_dict = dict()
    questions_lists = []
    try:
        for a_question in semantic_family:
            # if len(a_question['entailed']) > 0: # check existences of entailed quests
            print('Process SG-{}'.format(a_question['imageId']))
            questions_task_dict = dict()
            imageId=a_question['imageId']
            headAns=a_question['answer'],
            headTask = task_dict[a_question['types']['detailed']]
            headQuest='{}_{}'.format(a_question['key_id'],a_question['question'])


            if form == 'raw':
                # questions_list.append({'data': a_question, 'task': headTask})
                questions_task_dict[headTask] = [{'data': a_question, 'task': headTask}]
            else:
                # questions_list.append({'data': headQuest, 'task': headTask})
                questions_task_dict[headTask] = [{'data': headQuest, 'task': headTask}]

            for key_id in a_question['entailed']:
                an_entailed_question=list(col.find({'key_id':key_id}))[0]
                tailAns=an_entailed_question['answer']
                tailTask=task_dict[an_entailed_question['types']['detailed']]
                tailQuest='{}_{}'.format(an_entailed_question['key_id'],an_entailed_question['question'])
                if tailTask in questions_task_dict.keys():
                    if form == 'raw':
                        questions_task_dict[tailTask].append({'data': an_entailed_question, 'task': tailTask})
                    else:
                        questions_task_dict[tailTask].append({'data': tailQuest, 'task': tailTask})
                else:
                    if form == 'raw':
                        questions_task_dict[tailTask] = [{'data': an_entailed_question, 'task': tailTask}]
                    else:
                        questions_task_dict[tailTask] = [{'data': tailQuest, 'task': tailTask}]

            # generate list of unique tasks
            # questions_lists.append(questions_list)
            all_combs_list = list(itertools.product(*list(questions_task_dict.values())))
            if len(all_combs_list) > 1:
                # only take the first and the last element trick
                questions_lists += all_combs_list[::len(all_combs_list)-1]
            else:
                questions_lists += all_combs_list

    except CursorNotFound:
        print("Lost cursor. Should I retry ?")
        return questions_lists

    return questions_lists

def generate_data(split, name, form, num_data=None, batch_size=None, skip_limit = [0,0]):
    """
    generate a dataset of all questions in the same image
    """
    gqa_db = init_database()

    if split == 'train' or split == 'train_balanced':
        if split == 'train':
            col = gqa_db.train_all_questions
        elif split == 'train_balanced':
            col = gqa_db.train_balanced_questions
        # scene_graphs = gqa_db.train_sceneGraphs.find(no_cursor_timeout=True)
        scene_graphs = gqa_db.train_sceneGraphs.find(
            no_cursor_timeout=False, skip=int(skip_limit[0]),
            limit=int(skip_limit[1])).batch_size(batch_size)
    elif split == 'val' or split == 'val_balanced':
        if split == 'val':
            col = gqa_db.val_all_questions
        elif split == 'val_balanced':
            col = gqa_db.val_balanced_questions
        # scene_graphs = gqa_db.val_sceneGraphs.find(no_cursor_timeout=True)
        scene_graphs = gqa_db.val_sceneGraphs.find(
            no_cursor_timeout=False, skip=int(skip_limit[0]),
            limit=int(skip_limit[1])).batch_size(batch_size)

    '''
    semanticGlobal_family = gqa_db.val_all_questions.find(
        {'types.structural':'verify','types.semantic':'global'}
    )
    '''

    with open('./types_detailed.json','r') as json_fh:
        task_dict = json.load(json_fh)


    ############################################################
    # Generate all families of entailment for training
    scene_graphs.rewind()
    questions_dict = dict()
    task_list = list()
    answer_list = list()
    idx_scg = 0
    try:
        for a_sceneGraph in scene_graphs:
            imageId = a_sceneGraph['key_id']
            '''
            ,
                {'$match':{
                    'count':{'$gt': 5}
                }}
            '''
            semantic_families = list(col.aggregate(
                [{'$match': {'imageId':imageId}},
                {'$unwind': '$semantic'},
                {'$group': {
                    '_id': '$key_id',
                    'selectSemantic': {'$first' : '$semantic'}}},
                {'$group':{
                    '_id':'$selectSemantic.argument',
                    'key_ids': {'$push': '$_id'},
                    'count':{'$sum':1}}}]
            ))# 47

            if len(semantic_families) == 0:
                continue
            else:
                idx_scg +=1
                if num_data != None:
                    if idx_scg > num_data: break
                print('Process SG-{}'.format(imageId))

            questions_dict[imageId]=list()
            for a_group in semantic_families:
                # check whether a question family has more than 10 questions
                questions_list = []
                for question_id in a_group['key_ids']:
                    a_question = col.find({'key_id': question_id}).next()
                    imageId=a_question['imageId']
                    ans=a_question['answer']
                    task=task_dict[a_question['types']['detailed']]
                    task_list.append(task); answer_list.append(ans)
                    questions_list.append({'data': a_question, 'task': task})

                questions_dict[imageId].append(questions_list)
        scene_graphs.close()
    except CursorNotFound:
        print("Lost cursor. Should I retry ?")
        return questions_dict
    """
    answer_list = list(set(answer_list))
    # task_list = list(set(task_list)); task_list.sort() # unique task list
    task_list = list(set([v for k,v in task_dict.items()])); task_list.sort()
    task_dict = { task: i for i, task in enumerate(task_list)}
    with open(pklDir + '{}_{}.pkl'.format(split,name),'wb') as fh:
        pkl.dump({'question_dict': questions_dict, 'task_dict': task_dict, 'answer_list': answer_list},fh)

    if num_data is None:
        print('There are {} families in this dataset'.format(idx_scg))
    """
    return questions_dict

# takes a list and integer n as input and returns generator objects of n lengths from that list
"""
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
"""

def mpGenerateData(gqa_db, split, name, form, num_data=None, mgdb_batch_size=None, num_pkls=None):
    if split == 'train' or split == 'train_balanced':
        if split == 'train':
            col = gqa_db.train_all_questions
        elif split == 'train_balanced':
            col = gqa_db.train_balanced_questins
        # scene_graphs = gqa_db.train_sceneGraphs.find(no_cursor_timeout=True)

        if name == 'all_tasks':
            scene_graphs = gqa_db.train_sceneGraphs.find(no_cursor_timeout=False)#.batch_size(mgdb_batch_size)
        elif name == 'all_tasks_entailed':
            if 'balanced' in split:
                semantic_family = gqa_db.train_balanced_questions.find(
                    no_cursor_timeout=False)
            else:
                semantic_family = gqa_db.train_all_questions.find(
                    no_cursor_timeout=False)

    elif split == 'val' or split == 'val_balanced':
        if split == 'val':
            col = gqa_db.val_all_questions
        elif split == 'val_balanced':
            col = gqa_db.val_balanced_questions
        # scene_graphs = gqa_db.val_sceneGraphs.find(no_cursor_timeout=True)
        if name == 'all_tasks':
            scene_graphs = gqa_db.val_sceneGraphs.find(no_cursor_timeout=False)# .batch_size(mgdb_batch_size)
        elif name == 'all_tasks_entailed':
            if 'balanced' in split:
                semantic_family = gqa_db.val_balanced_questions.find(
                    no_cursor_timeout=False)
            else:
                semantic_family = gqa_db.val_all_questions.find(
                    no_cursor_timeout=False)

    if name == 'all_tasks':
        num_sgs = scene_graphs.count()
        print('There are {} scenegraphs in {} split of GQA dataset'.format(scene_graphs.count(), split))
        if num_data is not None:
            # semanticGlobal_family = semanticGlobal_family.aggregate([{'$sample': {'size': num_data}}])
            # scene_graphs.limit(num_data)
            assert num_data < scene_graphs.count(), 'There are not enough scene graphs in the dataset'
            num_sgs = num_data
            name += '_' + str(num_data)
            print('...; however, we only use {} images for train and evalTrain'.format(num_data))
    elif name == 'all_tasks_entailed':
        num_qst = semantic_family.count()
        print('There are {} questions {} split of GQA dataset'.format(num_qst, split))
        if num_data is not None:
            # semanticGlobal_family = semanticGlobal_family.aggregate([{'$sample': {'size': num_data}}])
            semantic_family.limit(num_data)
            name += '_' + str(num_data)
            print('...; however, we only use {} samples for train and evalTrain'.format(num_data))

    cpu_count = 128 # mp.cpu_count()
    mp_batch_size = 100 # np.int32(np.floor(num_sgs / cpu_count))
    if name == 'all_tasks':
        # mp_batch_size = np.int32(np.floor(num_sgs / cpu_count))
        mp_init_pos = range(0,num_sgs,mp_batch_size)
    elif name == 'all_tasks_entailed':
        # mp_batch_size = np.int32(np.floor(num_qst / cpu_count))
        mp_init_pos = range(0,num_qst,mp_batch_size)

    mp_skip_limit = [[mp_init_pos[i],mp_batch_size] for i in range(0,len(mp_init_pos))]

    if name == 'all_tasks':
        partialGenerateData = partial(
            generate_data, split, name, form,
            None, mgdb_batch_size)
    elif name == 'all_tasks_entailed':
        partialGenerateData = partial(
            generate_entailed_data, split, name, form,
            None, mgdb_batch_size)

    with mp.Pool(cpu_count) as pool: #cpu_count
        result = pool.map(partialGenerateData, mp_skip_limit)

    if name == 'all_tasks':
        # save by imageID - with overlap id problem
        questions_dict = dict(ChainMap(*result))

        name += '_' + form
        with open(pklDir + '{}_{}.pkl'.format(split,name),'wb') as fh:
            pkl.dump({'question_dict': questions_dict},fh)

        if num_pkls != None:
            def chunks(data, size=10000):
                it = iter(data)
                for i in range(0, len(data), size):
                    yield {k:data[k] for k in islice(it, size)}

            print('# samples is {}'.format(len(questions_dict.keys())))
            pkl_size = np.int32(np.floor(len(questions_dict.keys())/num_pkls))
            pkl_idx = 0
            for item in chunks(questions_dict,pkl_size):
                with open(pklDir + '{}_{}_{}.pkl'.format(split,name, pkl_idx),'wb') as fh:
                    pkl.dump({'question_dict': item},fh)
                pkl_idx+=1

    elif name == 'all_tasks_entailed':
        question_data= list(chain(*result))
        question_data.sort(key=len)
        print('# families is {}'.format(len(question_data)))

        name += '_' + form
        '''
        with open(pklDir + '{}_{}.pkl'.format(split,name),'wb') as fh:
            pkl.dump({'question_dict': question_data},fh)
        '''

        splits_list = [{} for i in range(num_pkls)]
        flag_change = False
        tmp = [] # store a question data with list_len size

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        while(len(question_data) > 0):
            if len(tmp) == 0 or len(question_data[0]) == len(tmp[0]):
                tmp.append(question_data.pop(0))
            else:
                list_len = len(tmp[0])
                tmp += random.choices(tmp,k=num_pkls - len(tmp) % num_pkls)
                tmp_chunk_size = np.int32(np.floor(len(tmp)/num_pkls))
                for i, item in enumerate(chunks(tmp, tmp_chunk_size)):
                    splits_list[i].update({str(list_len) : item})
                tmp = [] # reset

        for pkl_idx, a_split in enumerate(splits_list):
            with open(pklDir + '{}_{}_{}_sorted.pkl'.format(
                split,name,pkl_idx),'wb') as fh:
                pkl.dump({'question_dict': a_split},fh)


def main():
    #########################################################
    # get input and setting os variable
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', default=1,
                        type=str, help='train / val /train_balanced / val_balanced')
    parser.add_argument('-t', '--name', default='verify_global_balanced / all_tasks',
                        type=str, help='output filename')
    parser.add_argument('-f', '--form', default='raw',
                        type=str, help='item format')
    parser.add_argument('-n', '--nosamples', default=None,
                       type=int, help='number of samples, 0 means unlimited')
    parser.add_argument('-b', '--batchSize', default=128,
                       type=int, help='pymongo batchSize')
    parser.add_argument('--parallel', action='store_true',
                        help='enable multiprocesses')
    parser.add_argument('--numpkls', default=None, type=int,
                        help='number of pkl packages for saving')
    args = parser.parse_args()
    db = init_database()
    # verifyGlobal task
    if args.parallel:
        mpGenerateData(db, args.split, args.name, args.form,
                       num_data = args.nosamples, mgdb_batch_size = args.batchSize,
                       num_pkls=args.numpkls)
    else:
        if args.name == 'verify_global_balanced':
            generate_verifyGlobal(db,args.split,args.name,args.form,
                                  args.nosamples,args.batchSize)
        elif args.name == 'all_tasks_entailed':
            generate_entailed_data(args.split,args.name,args.form,
                        num_data=args.nosamples,batch_size=args.batchSize)
        elif args.name == 'all_tasks':
            generate_data(args.split,args.name,args.form,
                        num_data=args.nosamples,batch_size=args.batchSize)

    """
    elif args.name == 'all_unique_tasks':
        generate_data_uniqueTask(db,args.split,args.name,args.form,types=None,num_data=args.nosamples)
    """


if __name__ == '__main__':
    main()
