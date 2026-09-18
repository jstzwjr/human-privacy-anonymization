import json,time,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRSegMedium,RFDETRSegLarge,RFDETRSeg2XLarge
p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
cls,f={'M':(RFDETRSegMedium,'medium'),'L':(RFDETRSegLarge,'large'),'2XL':(RFDETRSeg2XLarge,'xxlarge')}[args.model]
m=cls(pretrain_weights=str(root/'model'/f'rf-detr-seg-{f}.pt'),device='cuda')
images={n:Image.open(root/'test_images'/n).convert('RGB') for n in ['bus.jpg','zidane.jpg','coco_val2017_000000000785.jpg']}
device_check={}
def hook(module,inputs):
    x=inputs[0]
    if hasattr(x,'tensors'):x=x.tensors
    device_check.update(input_device=str(x.device),input_dtype=str(x.dtype),parameter_device=str(next(module.parameters()).device),parameter_dtype=str(next(module.parameters()).dtype),training=module.training,inference_mode=torch.is_inference_mode_enabled())
h=m.model.model.register_forward_pre_hook(hook)
m.predict(next(iter(images.values())),threshold=.3);h.remove()
print(json.dumps({'device_check':device_check}),flush=True)
references={};reports=[]
for mode in ['baseline','no_source','optimized_fp32']:
    if mode=='optimized_fp32':m.inference(compile=False,dtype=torch.float32)
    for name,image in images.items():
        kw={'threshold':.3,'include_source_image':mode=='baseline'}
        for _ in range(5):m.predict(image,**kw)
        times=[]
        for _ in range(20):
            torch.cuda.synchronize();start=time.perf_counter();d=m.predict(image,**kw);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        if mode=='baseline':references[name]=d
        ref=references[name]
        same_n=len(d)==len(ref)
        report={'model':args.model,'mode':mode,'image':name,'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'samples_ms':times,'classes_equal':np.array_equal(d.class_id,ref.class_id),'masks_equal':np.array_equal(d.mask,ref.mask),'boxes_equal':np.array_equal(d.xyxy,ref.xyxy),'scores_equal':np.array_equal(d.confidence,ref.confidence),'max_score_diff':float(np.max(np.abs(d.confidence-ref.confidence))) if same_n else None,'mask_changed_pixels':int(np.count_nonzero(d.mask!=ref.mask)) if d.mask.shape==ref.mask.shape else None}
        reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='samples_ms'}),flush=True)
output=root/'outputs/speed_audit';output.mkdir(exist_ok=True,parents=True)
(output/f'{args.model}.json').write_text(json.dumps({'device_check':device_check,'reports':reports,'torch':torch.__version__,'cuda':torch.version.cuda,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32},indent=2)+'\n')
