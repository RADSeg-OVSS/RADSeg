CONFIG=$1
MODEL_VERSION=$2
LANG_MODEL=$3
SIM_SCALE=$4
WORKDIR=$5
AGG_BETA=$6
AGG_GAMMA=$7
SAM_REFINE=$8
python eval.py --config $CONFIG --model_version $MODEL_VERSION --lang_model $LANG_MODEL --sim_scale $SIM_SCALE --work-dir $WORKDIR --agg_beta $AGG_BETA --agg_gamma $AGG_GAMMA $SAM_REFINE
