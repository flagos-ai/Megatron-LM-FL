import os
from datetime import timedelta

import torch
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

import megatron.core.parallel_state as ps


def _get_device_backend():
    """根据可用硬件返回 (device_type, dist_backend)（使用 match/case 实现）"""
    # 从环境变量中获取平台类型，默认为 'cuda'，并转换为小写以便匹配
    #这里需要看一下哪里添加代码传MEGATRON_TEST_PLATFORM参数过来的，看看怎么改？？？？？
    platform = os.environ.get('MEGATRON_TEST_PLATFORM', 'cuda').strip().lower()
    
    # 使用 match/case 语法根据平台类型返回对应的设备类型和分布式后端
    match platform:
        case 'ascend':
            import torch_npu  # 华为昇腾
            return 'npu', 'hccl'
        case 'metax':
            import torch_musa  # 沐曦 MetaX（使用 MUSA 框架）
            return 'musa', 'mccl'
        case 'cuda':  # 匹配所有其他情况，默认为 CUDA
            return 'cuda', 'nccl'
        case _:  # 匹配所有非法平台，直接报错
            raise ValueError(
                f"不支持的平台：{platform}！"
                "目前支持 ascend/metax/cuda 三种平台，请添加适配其他平台代码"
            )
        #aa

# 全局初始化设备类型和分布式后端
DEVICE_TYPE, DIST_BACKEND = _get_device_backend()


class TestModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        bias: bool,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )
        if shared_embedding:
            self.layers[-1].weight.shared_embedding = True


class Utils:

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['LOCAL_RANK'])
    inited = False
    store = None

    @staticmethod
    def initialize_distributed():

        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f'Initializing torch.distributed with rank: {Utils.rank}, '
                f'world_size: {Utils.world_size}'
            )
            # 根据设备类型设置当前设备
            device_module = getattr(torch, DEVICE_TYPE)  # torch.cuda / torch.npu / torch.musa
            device_module.set_device(Utils.rank % device_module.device_count())
            # torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '6000')
            init_method += master_ip + ':' + master_port
            rendezvous_iterator = rendezvous(
                init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1)
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=1))

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                #修改这里的 backend 参数，改为根据设备类型动态设置
                # backend='nccl', world_size=Utils.world_size, rank=Utils.rank, store=store  
                backend=DIST_BACKEND, world_size=Utils.world_size, rank=Utils.rank, store=store
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['LOCAL_RANK'])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        if not Utils.inited:
            return
        torch.distributed.barrier()
        ps.destroy_model_parallel()
        Utils.inited = False

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        # Need to unset these variables to make sure previous
        # tests setting them doesn't interfere current test.
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        Utils.inited = True

    @staticmethod
    def fake_initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
    ):
        """Used for layer-wise UT as a proxy for NeMo-style intialization."""
        ps.set_tensor_model_parallel_world_size(tensor_model_parallel_size)
        ps.set_tensor_model_parallel_rank(0)

        ps.set_expert_model_parallel_world_size(expert_model_parallel_size)
        ps.set_expert_model_parallel_rank(0)
        if virtual_pipeline_model_parallel_size is not None:
            ps.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
        ps.set_virtual_pipeline_model_parallel_rank(0)

        ps.set_pipeline_model_parallel_world_size(pipeline_model_parallel_size)
