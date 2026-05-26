# <!-- {
#     "name": "评测",
#     "type": "debugpy",
#     "request": "launch",
#     "cwd":"${workspaceFolder}/TrackEval",
#     "program": "scripts/run_mmot_rgb.py",
#     "args": [
#         "--USE_PARALLEL", "False",
#         "--METRICS", "HOTA","CLEAR","Identity",
#         "--TRACKERS_TO_EVAL", "motr/e2e_motr_r50_train_mmot_rgb_23_l1_mmrotate_interval3",
#         "--TRACKER_SUB_FOLDER", "preds",
#         // "--TRACKERS_TO_EVAL", "motrv2/e2e_motr_r50_train_mmot_rgb_1",
#         // "--TRACKER_SUB_FOLDER", "submit"
#         // "--TRACKERS_TO_EVAL", "yolo11/predict/mmot_rgb_v11lobb_pretrainedweight9_predict_track",
#         // "--TRACKER_SUB_FOLDER", "preds"
#     ],
#     "justMyCode": false,
#     "env": {
#         "CUDA_VISIBLE_DEVICES": "0"
#     }

# } -->
# 第一个输入参数给Trackers
# 例如 motr/e2e_motr_r50_train_mmot_rgb_23_l1_mmrotate_interval3 
TRACKERS_TO_EVAL=$1
TRACKERS_SUB_FOLDER=$2

PWD=$(cd `dirname $0` && pwd)
cd $PWD/

TRACKERS_FOLDER_DEFAULT=$PWD/../exp/
# 检查输入参数数量
if [ $# -eq 3 ]; then
    TRACKERS_FOLDER=$3
else
    TRACKERS_FOLDER=$TRACKERS_FOLDER_DEFAULT
fi


DATASET=$PWD/../data 

python scripts/run_mmot_8ch.py \
    --USE_PARALLEL False \
    --METRICS HOTA CLEAR Identity \
    --TRACKERS_TO_EVAL ${TRACKERS_TO_EVAL} \
    --TRACKER_SUB_FOLDER ${TRACKERS_SUB_FOLDER} \
    --GT_FOLDER ${DATASET}/test/mot \
    --IMG_FOLDER ${DATASET}/test/npy \
    --TRACKERS_FOLDER ${TRACKERS_FOLDER}