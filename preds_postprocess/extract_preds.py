# Extract prediction from valPredictions file
# original: 'gqa3_horovod_original_512x32x4_gadasum_lr1.56e-6_q1_or_pc100_b16'
# hybrid: 'gqa3_horovod_hybrid_512x32x4_gadasum_lr1.56e-6_q1_hb_un_pc020_b16'
# logic:
# 'gqa3_horovod_hybrid_E4_512x32x4_fw_tflSS_l1.0_w1e-1_gadasum_lr1.56e-6_q1_hb_un_pc020_b16'
import argparse
import json

def extract_preds(expname):
    with open('./{exp}/valPredictions-{exp}.json'.format(exp=expname),'r') as json_fh:
        predictions = json.load(json_fh)
    return {p['questionId']:p['prediction'] for p in predictions}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', default='',
                    type=str, help='path to valPredictions.json file')
    parser.add_argument('-o', '--outfile', default='',
                        type=str, help='path to valPredicts.json file')
    args = parser.parse_args()
    preds_dict = extract_preds(args.infile)
    with open(args.outfile,'w') as json_fh:
        json.dump(preds_dict,json_fh)

if __name__ == '__main__':
    main()
