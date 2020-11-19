#!/bin/bash
# generate train and val for balanced dataset
SPLITS=(balanced_all_tasks all_tasks)
PARTITIONS=(train val)

for SPLIT in ${SPLITS[@]}; do
    for PART in ${PARTITIONS[@]}; do
        # iterate through mongodb to find families of questions
        python generate_multilabel_data_parallel.py\
            -s ${PART} -t ${SPLIT} -f raw -b 128\
            --parallel --numpkls 16 &
	wait
        # sorted by the number of questions and split into # chunks
        python split_multilabel_data_into_chunks.py\
            -s ${PART} -t ${SPLIT} -f raw -n 16 &
	wait
    done
done
