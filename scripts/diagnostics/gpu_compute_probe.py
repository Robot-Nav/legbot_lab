"""Bounded CUDA allocation, memory integrity and multi-stream compute probe (no Isaac)."""
import argparse
import json
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=float, default=180)
parser.add_argument('--bursty', action='store_true')
args = parser.parse_args()
torch.cuda.set_device(0)
print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0), 'seconds':args.seconds}), flush=True)
if args.bursty:
    # Exercise short kernels separated by idle gaps as a distinct workload.
    x=torch.ones(64,64,device='cuda')
    h=torch.ones(128,128,device='cuda',dtype=torch.float16)
    indices=torch.arange(32,device='cuda')
    started=last=time.monotonic(); bursts=0
    while time.monotonic()-started < args.seconds:
        for _ in range(50):
            y=(x@x).add(1).relu()
            z=y.index_select(0,indices).sum()
            q=(h@h).mean()
            u=x.sigmoid().mul(x)
        torch.cuda.synchronize()
        assert z.item()==65*32*64 and q.item()==128, 'Small-kernel result corruption'
        bursts+=1
        time.sleep(.03)
        if time.monotonic()-last>=15:
            print(json.dumps({'elapsed_s':round(time.monotonic()-started,1),'bursts':bursts}),flush=True)
            last=time.monotonic()
    print(json.dumps({'passed':True,'bursty':True,'bursts':bursts}),flush=True)
    raise SystemExit(0)
# Persistent allocations plus live temporaries approach the training memory footprint.
resident = [torch.full((256*1024*1024,), 90, dtype=torch.uint8, device='cuda') for _ in range(16)]
streams = [torch.cuda.Stream() for _ in range(4)]
a = torch.randn(4096,4096,device='cuda',dtype=torch.float16)
b = torch.randn_like(a)
expected = (a@b).float().sum().item()
torch.cuda.synchronize()
start = last = time.monotonic()
count = 0
while time.monotonic()-start < args.seconds:
    values=[]
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            x=torch.full((64*1024*1024,), float(i+count%7),device='cuda')
            y=x.square().add_(3)
            values.append((x,y,y.sum(), (float(i+count%7)**2+3)*y.numel()))
    result=a@b
    torch.cuda.synchronize()
    for x,y,total,target in values:
        assert total.item()==target, (count,total.item(),target)
    actual=result.float().sum().item()
    assert abs(actual-expected)<max(abs(expected)*1e-5, 1e-2), (actual,expected)
    del x,y,total,values,result
    count+=1
    if count%10==0:
        for buffer in resident:
            assert buffer.sum().item()==90*buffer.numel(), 'Persistent GPU memory corrupted'
        torch.cuda.empty_cache()
    now=time.monotonic()
    if now-last>=15:
        print(json.dumps({'elapsed_s':round(now-start,1),'iterations':count,
                          'peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20)}),flush=True)
        last=now
print(json.dumps({'passed':True,'iterations':count,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
