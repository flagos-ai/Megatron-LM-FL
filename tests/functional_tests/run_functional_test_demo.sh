# Won't work for now, because of compatibility issues 

export TRAINING_SCRIPT_PATH="pretrain_gpt.py"                              # training script
export TRAINING_PARAMS_PATH="tests/functional_tests/test_cases/gpt/gpt3_mcore_tp1_pp2/model_config.yaml"  # test configuration
export GOLDEN_VALUES_PATH="tests/functional_tests/test_cases/gpt/gpt3_mcore_tp1_pp2/golden_values_dev_dgx_a100.json"  # reference values
export OUTPUT_PATH="/tmp/megatron_test_output"                             # output directory
export TENSORBOARD_PATH="/tmp/megatron_test_tb"                            # TensorBoard logs
export CHECKPOINT_SAVE_PATH="/tmp/megatron_test_ckpt_save"                 # checkpoint save path
export CHECKPOINT_LOAD_PATH="/tmp/megatron_test_ckpt_load"                 # checkpoint load path
export DATA_PATH="/root/data"                      # data path (must be prepared in advance)
export DATA_CACHE_PATH="/tmp/megatron_data_cache"                          # data cache path
export ENABLE_LIGHTWEIGHT_MODE="true"                                      # lightweight mode (reduce training steps)
export GPUS_PER_NODE=8                                                     # number of GPUs per node
export NUM_NODES=1                                                         # number of nodes

mkdir -p $OUTPUT_PATH $TENSORBOARD_PATH $CHECKPOINT_SAVE_PATH $CHECKPOINT_LOAD_PATH $DATA_CACHE_PATH

cd /root/Megatron-LM-FL
bash tests/functional_tests/shell_test_utils/run_ci_test.sh