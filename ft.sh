export TRAINING_SCRIPT_PATH="pretrain_gpt.py"                              # 训练脚本
export TRAINING_PARAMS_PATH="tests/functional_tests/test_cases/gpt/gpt3_mcore_tp1_pp2/model_config.yaml"  # 测试配置
export GOLDEN_VALUES_PATH="tests/functional_tests/test_cases/gpt/gpt3_mcore_tp1_pp2/golden_values_dev_dgx_a100.json"  # 参考值
export OUTPUT_PATH="/tmp/megatron_test_output"                             # 输出目录
export TENSORBOARD_PATH="/tmp/megatron_test_tb"                            # TensorBoard 日志
export CHECKPOINT_SAVE_PATH="/tmp/megatron_test_ckpt_save"                 # checkpoint 保存路径
export CHECKPOINT_LOAD_PATH="/tmp/megatron_test_ckpt_load"                 # checkpoint 加载路径
export DATA_PATH="/root/data"                      # 数据路径 (需要提前准备)
export DATA_CACHE_PATH="/tmp/megatron_data_cache"                          # 数据缓存路径
export ENABLE_LIGHTWEIGHT_MODE="true"                                      # 轻量模式 (减少训练步数)
export GPUS_PER_NODE=8                                                     # 每节点GPU数量
export NUM_NODES=1                                                         # 节点数量

# 创建必要目录
mkdir -p $OUTPUT_PATH $TENSORBOARD_PATH $CHECKPOINT_SAVE_PATH $CHECKPOINT_LOAD_PATH $DATA_CACHE_PATH

# 运行测试
cd /root/Megatron-LM-FL
bash tests/functional_tests/shell_test_utils/run_ci_test.sh