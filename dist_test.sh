CONFIG=$1
MODEL_VERSION=$2
LANG_MODEL=$3
WORKDIR=$4
SCRA_SCALING=$5
SCGA_SCALING=$6
SAM_REFINE=$7
python eval.py --config $CONFIG --model_version $MODEL_VERSION --lang_model $LANG_MODEL --work-dir $WORKDIR --scga_scaling $SCGA_SCALING --scra_scaling $SCRA_SCALING $SAM_REFINE
