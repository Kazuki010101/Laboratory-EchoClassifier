import argparse, json, statistics, time
from pathlib import Path
import torch
from models.PatchEchoClassifier import PatchReservoir as PRC
from models.TeacherGuidedEvidenceCondensation import PatchReservoir as RoutedPRC

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--input-size',type=int,default=496); p.add_argument('--reservoir-size',type=int,default=1000)
    p.add_argument('--patch-size',type=int,default=16); p.add_argument('--keep-ratio',type=float,default=.5); a=p.parse_args()
    device=torch.device(a.device); x=torch.randn(1,3,a.input_size,device=device)
    models={'prc_full':PRC(3,a.patch_size,a.patch_size,a.reservoir_size,8),
            'aps_prc':RoutedPRC(3,a.patch_size,a.patch_size,a.reservoir_size,8,a.keep_ratio,'aps'),
            'tg_skip_prc':RoutedPRC(3,a.patch_size,a.patch_size,a.reservoir_size,8,a.keep_ratio,'teacher_skip'),
            'tgec_prc':RoutedPRC(3,a.patch_size,a.patch_size,a.reservoir_size,8,a.keep_ratio,'tgec')}
    result={}
    for name,m in models.items():
        m.to(device).eval()
        with torch.no_grad():
            for _ in range(10): m(x)
            times=[]
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            for _ in range(100):
                t=time.perf_counter(); m(x)
                if device.type=='cuda': torch.cuda.synchronize()
                times.append((time.perf_counter()-t)*1000)
        stats=m.get_routing_stats() if hasattr(m,'get_routing_stats') else {'patch_count':31,'keep_count':31}
        recurrent_tokens = int(stats['keep_count']) + (1 if name == 'tgec_prc' else 0) + 2
        # Dominant reservoir MACs: one R x R recurrence plus D x R input per token.
        estimated_macs = recurrent_tokens * (a.reservoir_size ** 2 + a.patch_size * a.reservoir_size)
        result[name]={'parameters':sum(q.numel() for q in m.parameters()),'latency_ms_median':statistics.median(times),
                      'latency_ms_mean':statistics.mean(times),'peak_memory_mb':(torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None),
                      'recurrent_tokens_including_cls_dist': recurrent_tokens,
                      'estimated_dominant_macs': estimated_macs, **stats}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
