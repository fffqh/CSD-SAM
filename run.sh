#!/bin/bash

DATA_NAME="csd"
##S0
ST_IDX=1
ED_IDX=201
##S3
P_V=10
P_LOF_K=5
P_LOF_V=2.5

ED_IDX=`python video2img.py --video_path 'data/csd.mp4' --out_path 'data/ori_imgs/csd'`

EXCODE=$?
if [ "$EXCODE" -ne "0" ]; then
    exit 1
fi
echo "Num of frames:"${ED_IDX}

cd S0_OpticalFlow
python visualize_flow.py --eval_type seq --root_dir '../data/ori_imgs' --seq_dir ${DATA_NAME} --viz_root_dir '../data/flw_imgs' --start_idx ${ST_IDX} --end_idx ${ED_IDX}
echo "=========== S0 Done. ===========\n"

cd ..
cd S1_Classification
python tools/detect.py --cfg 'experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml' --dataDir '../data/flw_imgs/'${DATA_NAME} --testModel './checkpoints/best_state.pth'
echo "=========== S1 Done. ===========\n"

cd ..
cd S2_BuildPrompts
python tools/inference.py --cfg 'experiments/csc/cscinf_hrnet_w18_kp10.yaml' --model-file 'hrnetv2_pretrained/csc_best10.pth' --input-dir '../data/flw_imgs/'${DATA_NAME}'/true' --output-dir '../data/flw_imgs/'${DATA_NAME}
echo "=========== S2 Done. ===========\n"

cd ..
cd S3_SAM_CSD
pip install e .
python csc_tool/csd_vis.py --csv-path '../data/flw_imgs/'${DATA_NAME}'/prompts.csv' --ori-root '../data/ori_imgs/'${DATA_NAME} --flw-root '../data/flw_imgs/'${DATA_NAME}'/true' --out-root '../data/output/'${DATA_NAME} --kpnum 10 --pV ${P_V} --plofk ${P_LOF_K} --plofv ${P_LOF_V}
echo "=========== S3 Done. ===========\n"

cd ..

python img2video.py --img_path 'data/output/csd' --out_path 'output.mp4'

