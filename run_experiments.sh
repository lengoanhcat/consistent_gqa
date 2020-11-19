#!/bin/bash
inputTflParams() {
    if [ -z "$TFNTP" ] || [ -z "$TFNLD" ] || [ -z "$RULES" ] || [ -z "$TFNLW" ]; then
        echo -n "Enter tflType [SS/FK]:"
        read TFNTP
        echo -n "Enter the tflLambda [0.0]: "
        read TFNLD
        echo -n "Enter the rules [(B)inary/(G)lobal/(A)ttr/(O)ject/(R)elation/(Ex)ntailement/(Cx)onsistent]: "
        read RULES
        echo -n "Enter the tflLossweight [1.0]: "
        read TFNLW
        echo -n "Enter the weight mode (fw/sw): "
        read WM
    fi
}

inputHbrParams() {
    echo -n "Enter the cut-off Epoch [smo19, 24]: "
    read CTOEP
}

inputRestoreParams() {
    echo -n "Enter the restore epoch: "
    read -t 10 RSTEP
    if [ -z "$RSTEP" ]; then
        RSTEP=0
    fi
}

inputEvalParams() {
    if [ -z "$TESTS" ]; then
        echo -n "Enter subset for eval: "
        read -t 10 TESTS
        if [ -z "$TESTS" ]; then
            TESTS='all'
        fi
    fi
}

inputExecutingServers() {
    echo -n "Enter executing server [8gpu-3/8gpu-5/16gpu-1]: "
    read EXSRV
    if [ -z "$EXSRV" ]; then
        echo "No executing servers"
        exit 1
    elif [ "$EXSRV" == "8gpu-5" ]; then
        # EXSRV=8gpu-5-f1:8,8gpu-4-f1:8
        EXSRV=8gpu-5-f1:8
        NOGPU=8
    elif [ "$EXSRV" == "8gpu-3" ]; then
        EXSRV=8gpu-3-f1:8,8gpu-2-f1:8
        NOGPU=16
    elif [ "$EXSRV" == "16gpu-1" ]; then
        EXSRV=16gpu-1-f1:16
        NOGPU=16
    fi
}

setGPU() {
    if [[ "$EXSRV" =~ ^(8gpu-[3,5])$ ]]; then
        export CUDA_VISIBLE_DEVICES="$(( $RANDOM % 8 ))"
    elif [[ "$EXSRV" =~ ^(16gpu-[1])$ ]]; then
        export CUDA_VISIBLE_DEVICES="$(( $RANDOM % 16 ))"
    fi
}

echo -n "Enter the beginning steps [0,1,2,mc,it]: "
read ESTEP
echo -n "Enter the experiment number (expno_00[1,2,3,4,5,6]): "
read EXPNO
if (( "$EXPNO" > 1 )); then
    echo -n "Enter the family data style (unique / ununique): "
    read FAMST
fi
echo -n "Enter the data percentage: "
read DTPCT
echo -n "Enter the batch size: "
read BTCSZ
echo -n "Enter the MACnet/baseline architecture [64x8x4, 512x32x4, CNN, LSTM, LSTMCNN]): "
read MARCH
echo -n 'Enter extra option:'
read EXOPT
echo -n "Enter the hvd optimizer (average/adasum) : "
read HVDOP
echo -n "Enter learning rate (1e-6) : "
read OPLNR
echo -n "Enter # epoches [batch 95/ unique 6*n / not unique 38*n or 68*n] : "
read EPCNO
echo -n "Enter min # quests (1-6): "
read MINNQ
echo -n "Enter the experiment postfix [hb/or/bt/cr]_[un/nu]_[extra]:"
read EXPPF
inputExecutingServers

EVALHOME='/data/catle/Datasets/GQA/'
PCTST=$(printf "%03d" $DTPCT)
case 'expno_00'$EXPNO in
    expno_001)
    SCNPF=$MARCH
    EXPNAM='gqa3_horovod_original_'$SCNPF'_g'$HVDOP'_lr'$OPLNR'_q'$MINNQ'_'$EXPPF'_pc'$PCTST'_b'$BTCSZ
    ;;

    expno_002)
    SCNPF=$MARCH
    EXPNAM='gqa3_horovod_hybrid_'$SCNPF'_g'$HVDOP'_lr'$OPLNR'_q'$MINNQ'_'$EXPPF'_pc'$PCTST'_b'$BTCSZ
    ;;

    expno_003)
    inputTflParams
    SCNPF=$RULES'_'$MARCH'_'$WM
    EXPNAM='gqa3_horovod_family_'$SCNPF'_tfl'$TFNTP'_l'$TFNLD'_w'$TFNLW'_g'$HVDOP'_lr'$OPLNR'_q'$MINNQ'_'$EXPPF'_pc'$PCTST'_b'$BTCSZ
    ;;

    expno_004)
    SCNPF=$MARCH
    inputHbrParams
    EXPNAM='gqa3_horovod_hybrid_'$SCNPF'_g'$HVDOP'_lr'$OPLNR'_q'$MINNQ'_'$EXPPF'_pc'$PCTST'_b'$BTCSZ
    ;;

    expno_005|expno_006)
    inputTflParams
    SCNPF=$RULES'_'$MARCH'_'$WM
    # inputHbrParams
    EXPNAM='gqa3_horovod_hybrid_'$SCNPF'_tfl'$TFNTP'_l'$TFNLD'_w'$TFNLW'_g'$HVDOP'_lr'$OPLNR'_q'$MINNQ'_'$EXPPF'_pc'$PCTST'_b'$BTCSZ
    ;;

    *)
    echo -n 'Unknown experiments'
    ;;
esac

if [[ "$ESTEP" =~ ^[0-9]+$ ]]; then
    # train the model
    if (( "$ESTEP" <= 0 )); then
        echo 'Train model : '$EXPNAM
        case 'expno_00'$EXPNO in
            debug)
            inputTflParams
            EXPNAM='gqa3_horovod_family_'$SCNPF'_tflss_l'$TFNLD'_w'$TFNLW'_g'$HVDOP'_lr'$OPLNR'_q'$MINNQ
            echo -n "Enter GPU ID : "; read GPUID
            export CUDA_VISIBLE_DEVICES="$GPUID" \
            python main.py \
            --expName=$EXPNM \
            --train --testedNum=10000 --epochs=36 --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --tflSS --tflLambda=$TFNLD --tflLossWeight=$TFNLW \
            --netLength=4 --batchSize=$BTCSZ --batchStyle=family @configs/gqa/gqa_horovod.txt
            ;;

            # i.i.d batch - w/o logic --testedNum=10000 
            # --timeline-filename $EXPNAM'.json' 
            expno_001)
            horovodrun -np 16 -H $EXSRV \
            python main.py --expName=$EXPNAM --train \
            --lr=$OPLNR --epochs=$EPCNO --weightsToKeep=$EPCNO --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --datapct=$DTPCT --netLength=4 $EXOPT \
            --batchSize=$BTCSZ --batchStyle=original \
            @configs/gqa/gqa_horovod_$MARCH.txt
            ;;

            # family/hybrid batch - w/o logic
            expno_002)
            # echo -n "Enter the restore epoch: "
            # read RSTEP
            # -r --restoreEpoch=$RSTEP \
            # --timeline-filename $EXPNAM'.json' 
            horovodrun -np $NOGPU -H $EXSRV \
            python main.py --expName=$EXPNAM --train\
            --lr=$OPLNR --epochs=$EPCNO --weightsToKeep=$EPCNO --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --datapct=$DTPCT --netLength=4 $EXOPT \
            --batchSize=$BTCSZ --batchStyle=family --familyStyle=$FAMST \
            --dataSubset=balanced \
            @configs/gqa/gqa_horovod_$MARCH.txt
            ;;

            # family batch - w logic
            expno_003)
            horovodrun -np 16 -H $EXSRV \
            python main.py --expName=$EXPNAM --train \
            --lr=$OPLNR --epochs=$EPCNO --weightsToKeep=$EPCNO --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --datapct=$DTPCT '--tfl'$TFNTP --tflLambda=$TFNLD --tflLossWeight=$TFNLW \
            --tflRules=$RULES --tflWeightMode=$WM --netLength=4 $EXOPT \
            --batchSize=$BTCSZ --batchStyle=family --familyStyle=$FAMST \
            @configs/gqa/gqa_horovod_$MARCH.txt
            ;;

            # hybrid batch - w/o logic
            expno_004)
            horovodrun -np 16 -H $EXSRV \
            python main.py --expName=$EXPNAM --train\
            --lr=$OPLNR --epochs=$EPCNO --weightsToKeep=$EPCNO --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --datapct=$DTPCT --netLength=4 $EXOPT \
            --batchSize=$BTCSZ --batchStyle=hybrid --familyStyle=$FAMST \
            --dataSubset=balanced \
            @configs/gqa/gqa_horovod_$MARCH.txt
            ;;

            # family batch - w logic
            # --timeline-filename $EXPNAM'.json' 
            # echo -n "Enter the restore epoch: "
            # read RSTEP
            # -r --restoreEpoch=$RSTEP \
            expno_005)
            horovodrun -np 16 -H $EXSRV \
            python main.py --expName=$EXPNAM --train \
            --lr=$OPLNR --epochs=$EPCNO --weightsToKeep=$EPCNO --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --datapct=$DTPCT '--tfl'$TFNTP --tflLambda=$TFNLD --tflLossWeight=$TFNLW \
            --tflRules=$RULES --tflWeightMode=$WM --netLength=4 $EXOPT \
            --batchSize=$BTCSZ --batchStyle=hybrid --familyStyle=$FAMST \
            @configs/gqa/gqa_horovod_$MARCH.txt
            ;;

            expno_006)
            echo -n "Enter the restore epoch: "
            read RSTEP
            horovodrun -np 16 -H $EXSRV \
            python main.py --expName=$EXPNAM --train \
            --lr=$OPLNR --epochs=$EPCNO --weightsToKeep=$EPCNO --hvdoptim=$HVDOP \
            --minNoQuest=$MINNQ --datapct=$DTPCT '--tfl'$TFNTP --tflLambda=$TFNLD --tflLossWeight=$TFNLW \
            --tflRules=$RULES --tflWeightMode=$WM --netLength=4 $EXOPT \
            --batchSize=$BTCSZ --batchStyle=hybrid --familyStyle=$FAMST \
            -r --restoreEpoch=$RSTEP \
            @configs/gqa/gqa_horovod_$MARCH.txt
            ;;

            *)
            echo -n 'Unknown experiments'
            ;;
        esac
    fi

    # generate prediction 
    # --earlyStopping=3, --restoreEpoch=-9 
    if (( "$ESTEP" <= 1 )); then
        echo 'Generate prediction : '$EXPNAM
        export CUDA_VISIBLE_DEVICES="$(( $RANDOM % 8 ))"
        inputRestoreParams
        inputEvalParams
        # horovodrun -np 2 -H 16gpu-1-f1:2 \
        python main.py \
        --expName=$EXPNAM --finalTest --getPreds --getAtt \
        -r --restoreEpoch=$RSTEP --dataSubset=$TESTS --testAll \
        --lr=$OPLNR --epochs=$EPCNO --hvdoptim=$HVDOP --minNoQuest=$MINNQ \
        --netLength=4 $EXOPT \
        --batchStyle=original --batchSize=$BTCSZ \
        @configs/gqa/'gqa_horovod_'$MARCH'.txt'
    fi

    # produce evaluation metric
    if (( "$ESTEP" <= 2 )); then
        echo 'Evaluate metrics : '$EXPNAM
        inputEvalParams
        python $EVALHOME/eval/eval.py --tier val --consistency \
        --scenes $EVALHOME/val_sceneGraphs.json \
        --choices $EVALHOME/eval/val_choices.json \
        --questions $EVALHOME'/val_'$TESTS'_questions.json' \
        --predictions \
        "./preds/"$EXPNAM"/valPredictions-"$EXPNAM".json" \
        | tee -a \
        "./preds/"$EXPNAM"/valPredictions-report.txt"
    fi
# do model check
elif (( "$ESTEP" == "mc" )) ; then
    echo 'Model check with satisfaction score'
    inputTflParams
    inputRestoreParams
    # setGPU
    horovodrun -np 16 -H $EXSRV \
    python main.py --expName=$EXPNAM --checkVal \
    -r --restoreEpoch=$RSTEP --dataSubset=$TESTS \
    --lr=$OPLNR --epochs=$EPCNO --hvdoptim=$HVDOP --minNoQuest=4 \
    '--tfl'$TFNTP --tflLambda=$TFNLD --tflLossWeight=$TFNLW \
    --tflRules=$RULES --tflWeightMode=$WM --netLength=4 $EXOPT \
    --batchStyle=family --batchSize=$BTCSZ \
    @configs/gqa/'gqa_horovod_'$MARCH'.txt'
fi
