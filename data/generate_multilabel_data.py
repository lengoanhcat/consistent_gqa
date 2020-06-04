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

def init_database():
    try:
        client = MongoClient('localhost',27017)
        client.server_info()
    except:
        print("Connection Error: pls restart mongod service `mongod --dbpath /home/catle/mongod`")

    gqa_db = client.gqa
    return gqa_db

def generate_verifyGlobal(gqa_db, split, name, form, num_data=None):
    """
    generate a dataset of all questions belong to verifyGlobal familiy in the same image
    """
    if split == 'train':
        col = gqa_db.train_all_questions
        if 'balanced' in name:
            semanticGlobal_family = gqa_db.train_balanced_questions.find(
                {'types.structural':'verify','types.semantic':'global'}
            )

    elif split == 'val':
        col = gqa_db.val_all_questions
        if 'balanced' in name:
            semanticGlobal_family = gqa_db.val_balanced_questions.find(
                {'types.structural':'verify','types.semantic':'global'}
            )

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
    with open('./{}_{}.pkl'.format(split,name),'wb') as fh:
        pkl.dump({'question_dict': questions_dict, 'task_dict': task_dict, 'answer_list': answer_list},fh)

def generate_data(gqa_db, split, name, form, types=None, num_data=None):
    """
    generate a dataset of all questions in the same image
    """
    if split == 'train':
        col = gqa_db.train_all_questions
        scene_graphs = gqa_db.train_sceneGraphs.find(no_cursor_timeout=True)

    elif split == 'val':
        col = gqa_db.val_all_questions
        scene_graphs = gqa_db.val_sceneGraphs.find(no_cursor_timeout=True)

    '''
    semanticGlobal_family = gqa_db.val_all_questions.find(
        {'types.structural':'verify','types.semantic':'global'}
    )
    '''

    with open('./types_detailed.json','r') as json_fh:
        task_dict = json.load(json_fh)

    name += '_' + form

    print('There are {} scenegraphs in {} split of GQA dataset'.format(scene_graphs.count(), split))
    if num_data is not None:
        # semanticGlobal_family = semanticGlobal_family.aggregate([{'$sample': {'size': num_data}}])
        # scene_graphs.limit(num_data)
        name += '_' + str(num_data)
        print('...; however, we only use {} images for train and evalTrain'.format(num_data))

    ############################################################
    # Generate all families of entailment for training
    scene_graphs.rewind()
    questions_dict = dict()
    task_list = list()
    answer_list = list()
    idx_scg = 0
    for a_sceneGraph in scene_graphs:
        imageId = a_sceneGraph['key_id']
        semantic_families = list(col.aggregate(
            [{'$match': {'imageId':imageId}},
             {'$unwind': '$semantic'},
             {'$group': {
                 '_id': '$key_id',
                 'selectSemantic': {'$first' : '$semantic'}}},
             {'$group':{
                 '_id':'$selectSemantic.argument',
                 'key_ids': {'$push': '$_id'},
                 'count':{'$sum':1}}},
             {'$match':{
                 'count':{'$gt': 20}
            }}]
        ))

        if len(semantic_families) == 0:
            continue
        else:
            idx_scg +=1
            if idx_scg > num_data: break

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

    answer_list = list(set(answer_list))
    # task_list = list(set(task_list)); task_list.sort() # unique task list
    task_list = list(set([v for k,v in task_dict.items()])); task_list.sort()
    task_dict = { task: i for i, task in enumerate(task_list)}
    with open('./{}_{}.pkl'.format(split,name),'wb') as fh:
        pkl.dump({'question_dict': questions_dict, 'task_dict': task_dict, 'answer_list': answer_list},fh)

"""
def generate_data_uniqueTask(gqa_db, split, name, form, types=None, num_data=None):
    '''
    generate a dataset of all questions in the same image
    '''
    if split == 'train':
        col = gqa_db.train_all_questions
        scene_graphs = gqa_db.train_sceneGraphs.find(no_cursor_timeout=True)

    elif split == 'val':
        col = gqa_db.val_all_questions
        scene_graphs = gqa_db.val_sceneGraphs.find(no_cursor_timeout=True)

    '''
    semanticGlobal_family = gqa_db.val_all_questions.find(
        {'types.structural':'verify','types.semantic':'global'}
    )
    '''


    def createQuestionDict():
        all_tasks = ['queryNotObj','queryAttrObj', 'existAttrTrue', 'existAttrOrTrue',
                    'existNotOrTrue','queryObject', 'existNotTrue', 'existTrue',
                    'existOrTrue', 'existAndTrue', 'existRelTrue', 'verifyRelTrue',
                    'chooseRel', 'queryRel',

                    'verifyRelFalse',

                    'existOrFalse','existFalse', 'existNotFalse', 'existRelFalse',
                    'existAndFalse','existAttrFalse','existAttrOrFalse','existNotOrFalse',

                    'verifyAttrsTrue', 'verifyAttrAndTrue','verifyAttrTrue',
                    'queryAttr','verifyAttrFalse', 'verifyAttrAndFalse',
                    'chooseAttr', 'chooseNot',

                    'verifyGlobalTrue', 'verifyGlobalFalse', 'queryGlobal',
                    'chooseGlobal',

                    'compare', 'common', 'twoSimilarTrue', 'twoDiffFalse',

                    'twoSamelFalse', 'twoDiffTure',

                    'allSameTrue', 'allDiffFalse',

                    'allSameFalse', 'allDiffTrue'] # 47

        question_dict = defaultdict(list)
        for task in all_tasks: question_dict[task] = []
        return question_dict

    with open('./types_detailed.json','r') as json_fh:
        task_dict = json.load(json_fh)

    name += '_' + form

    print('There are {} scenegraphs in {} split of GQA dataset'.format(scene_graphs.count(), split))
    if num_data is not None:
        # semanticGlobal_family = semanticGlobal_family.aggregate([{'$sample': {'size': num_data}}])
        # scene_graphs.limit(num_data)
        name += '_' + str(num_data)
        print('...; however, we only use {} images for train and evalTrain'.format(num_data))

    ############################################################
    # Generate all families of entailment for training
    scene_graphs.rewind()
    questions_dict = dict()
    task_list = list()
    answer_list = list()
    idx_scg = 0
    for a_sceneGraph in scene_graphs:
        imageId = a_sceneGraph['key_id']
        semantic_families = list(col.aggregate(
            [{'$match': {'imageId':imageId}},
             {'$unwind': '$semantic'},
             {'$group': {
                 '_id': '$key_id',
                 'selectSemantic': {'$first' : '$semantic'}}},
             {'$group':{
                 '_id':'$selectSemantic.argument',
                 'key_ids': {'$push': '$_id'},
                 'count':{'$sum':1}}},
             {'$match':{
                 'count':{'$gt': 47}
            }}]
        ))

        if len(semantic_families) == 0:
            continue
        else:
            idx_scg +=1
            if idx_scg > num_data: break

        questions_dict[imageId]=list()
        for a_group in semantic_families:
            # check whether a question family has more than 10 questions
            questions_lists = [list()]
            tq_dict = createQuestionDict() # task-based questions
            for question_id in a_group['key_ids']:
                a_question = col.find({'key_id': question_id}).next()
                imageId=a_question['imageId']
                ans=a_question['answer']
                task=task_dict[a_question['types']['detailed']]
                task_list.append(task); answer_list.append(ans)
                # questions_list.append({'data': a_question, 'task': task})
                tq_dict[task].append(a_question)

            while len(tq_dict.keys()) > 0:
                task, current_list = tq_dict.popitem()
                if len(current_list) > 0:
                    '''
                    questions_lists = [prev_list.append(cur_list)
                                    for prev_list in questions_lists
                                    for cur_list in current_list]
                    '''
                    tmp = []
                    for prev_list in questions_lists:
                        for cur_item in current_list:
                            tmp.append(prev_list
                    questions_lists = tmp

            questions_dict[imageId].append(questions_lists)

    answer_list = list(set(answer_list))
    # task_list = list(set(task_list)); task_list.sort() # unique task list
    task_list = list(set([v for k,v in task_dict.items()])); task_list.sort()
    task_dict = { task: i for i, task in enumerate(task_list)}
    with open('./{}_{}.pkl'.format(split,name),'wb') as fh:
        pkl.dump({'question_dict': questions_dict, 'task_dict': task_dict, 'answer_list': answer_list},fh)
"""

def main():

    #########################################################
    # get input and setting os variable
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', default=1,
                        type=str, help='train / var split of data')
    parser.add_argument('-t', '--name', default='verify_global',
                        type=str, help='output filename')
    parser.add_argument('-f', '--form', default='raw',
                        type=str, help='item format')
    parser.add_argument('-n', '--nosamples', default=None,
                       type=int, help='number of samples')
    args = parser.parse_args()
    db = init_database()
    # verifyGlobal task
    if args.name == 'verify_global_balanced':
        generate_verifyGlobal(db,args.split,args.name,args.form,args.nosamples)
    elif args.name == 'all_tasks':
        generate_data(db,args.split,args.name,args.form,types=None,num_data=args.nosamples)
    """
    elif args.name == 'all_unique_tasks':
        generate_data_uniqueTask(db,args.split,args.name,args.form,types=None,num_data=args.nosamples)
    """


if __name__ == '__main__':
    main()
