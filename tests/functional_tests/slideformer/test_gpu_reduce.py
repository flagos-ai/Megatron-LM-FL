"""Two/four-rank GPU reduction against an independent full all-reduce oracle.

Covers BF16/FP32 DMA, native FP32 accumulation, empty/uneven shards, partial
buckets, slot reuse and FP32 cross-microbatch sums. Run under an ownership guard.
"""
import os
from types import SimpleNamespace
visible=os.environ['SLIDEFORMER_GPU_IDS'].split(',')
os.environ['CUDA_VISIBLE_DEVICES']=visible[int(os.environ['LOCAL_RANK'])]
os.environ['LOCAL_RANK']='0'
import torch
import torch.distributed as dist
from slideformer_megatron.gpu_reduce import GPUReduceScatter
from slideformer_megatron.sharded_cpu_optimizer import ShardedCPUOptimizer

torch.cuda.set_device(0)
dist.init_process_group('nccl', device_id=torch.device('cuda:0'))
rank,size=dist.get_rank(),dist.get_world_size()
assert size in (2,4)
widths=[[64]*size,[67]*size,[64+r for r in range(size)],[0]*(size-1)+[3]]
bounds=[]
for ws in widths:
 points=[0]
 for w in ws:points.append(points[-1]+w)
 bounds.append(list(zip(points[:-1],points[1:])))
owners=[SimpleNamespace(layer_idx=i,_flat_grad_copy_supported=True,cpu_grad_flat=torch.empty(b[-1][1],dtype=torch.bfloat16),d2h_stream=torch.cuda.Stream()) for i,b in enumerate(bounds)]
sharded=SimpleNamespace(size=size,rank=rank,dp_group=dist.group.WORLD,owners=owners,special={i:[] for i in range(len(owners))},bounds=dict(enumerate(bounds)),owns=lambda owner:True)
checks=0
path_counts={'inplace':0,'bucketed':0}
try:
 for layout in ('auto','bucketed'):
  os.environ['MEGATRON_SLIDEFORMER_GPU_REDUCE_LAYOUT']=layout
  for bucket in ('0','0.0001220703125'):
   os.environ['MEGATRON_SLIDEFORMER_GPU_REDUCE_BUCKET_MB']=bucket
   for precision in ('bf16','fp32'):
    for native_fp32 in (False,True):
     for microbatches in (1,3):
      engine=SimpleNamespace(device=torch.device('cuda:0'),config=SimpleNamespace(gpu_reduce_dtype=precision,reduce_scatter_with_fp32_accumulation=native_fp32,chunk_size_mb=0.00005),_eager_scale=1/size,_expected_microbatches=microbatches,_layer_bwd_count={},traffic_counters={})
      reducer=GPUReduceScatter(engine,sharded)
      worker=SimpleNamespace(engine=engine,adam_owner=lambda owner,grad_slice:None,done=set(),phase_times={'layers':0,'barrier':0})
      for repeat in range(2):
       engine._layer_bwd_count.clear();expected_sums={}
       for micro in range(microbatches):
        for owner,b in zip(owners,bounds):
         gen=torch.Generator().manual_seed(123+rank+7*micro+31*repeat)
         # Random BF16 inputs exercise rounding, not only exactly summable integers.
         values = torch.randn(b[-1][1], generator=gen) if size == 2 else torch.randint(-40, 40, (b[-1][1],), generator=gen) / 8
         flat=values.to(device='cuda',dtype=torch.bfloat16)
         expected=flat.float() if native_fp32 or precision=='fp32' else flat.clone()
         dist.all_reduce(expected)
         expected=expected.to(reducer.transport_dtype)
         if microbatches==1:expected.mul_(1/size)
         lo,hi=b[rank];expected=expected[lo:hi].cpu()
         reducer.submit(owner,flat);owner._gpu_reduce_event.synchronize()
         if owner.layer_idx==0 and repeat==0 and micro==0 and layout=='auto' and precision=='bf16' and not native_fp32:
          assert all(x is None for x in reducer._padded+reducer._out), 'Native equal-shard reduction allocated scratch'
         actual=reducer.sacc[owner.layer_idx] if microbatches==1 else reducer.stmp[owner.layer_idx]
         assert actual.dtype==reducer.transport_dtype
         torch.testing.assert_close(actual,expected,atol=0,rtol=0)
         if microbatches>1:
          ShardedCPUOptimizer._gpu_job(worker,owner,owner._gpu_reduce_event,micro==0,micro==microbatches-1,None,reducer)
          if micro==0:expected_sums[owner.layer_idx]=expected.float()
          else:expected_sums[owner.layer_idx].add_(expected)
          assert reducer.sacc[owner.layer_idx].dtype==torch.float32
          torch.testing.assert_close(reducer.sacc[owner.layer_idx],expected_sums[owner.layer_idx],atol=0,rtol=0)
         engine._layer_bwd_count[owner.layer_idx]=micro+1;checks+=1
      path_counts['inplace']+=engine.traffic_counters.get('gpu_reduce_inplace_layers',0)
      path_counts['bucketed']+=engine.traffic_counters.get('gpu_reduce_bucketed_layers',0)
      dist.destroy_process_group(reducer.group)
 assert path_counts['inplace']>0 and path_counts['bucketed']>0
 print(f'GPU_REDUCE_PARITY rank={rank} checks={checks} max_abs_error=0 paths={path_counts}',flush=True)
finally:dist.destroy_process_group()
