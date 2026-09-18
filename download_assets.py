from pathlib import Path
import concurrent.futures, hashlib, json, subprocess, time

root = Path(__file__).resolve().parent
assets = [
 ('model/rf-detr-seg-medium.pt','https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth','a49af1562c3719227ad43d0ca53b4c7a','RF-DETR-Seg-M'),
 ('model/rf-detr-seg-large.pt','https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth','275f7b094909544ed2841c94a677d07e','RF-DETR-Seg-L'),
 ('model/rf-detr-seg-xxlarge.pt','https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth','040bc3412af840fa8a47e0ff69b552ba','RF-DETR-Seg-2XL'),
 ('test_images/bus.jpg','https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg',None,'街道行人与公交车，观察多人全身覆盖'),
 ('test_images/zidane.jpg','https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg',None,'近景人物，观察局部人体与头部覆盖'),
 ('test_images/coco_val2017_000000000139.jpg','https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000139.jpg',None,'COCO 验证集样例 139，检查室内人物'),
 ('test_images/coco_val2017_000000000785.jpg','https://s3.amazonaws.com/images.cocodataset.org/val2017/000000000785.jpg',None,'COCO 验证集样例 785，检查运动人物'),
]
def download(asset):
    rel,url,expected,label=asset
    dest=root/rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial=dest.with_suffix(dest.suffix+'.part')
    cmd=['curl','--fail','--location','--retry','5','--retry-all-errors','--connect-timeout','20','--max-time','1800','--silent','--show-error','--output',str(partial),url]
    if dest.exists() and expected and hashlib.md5(dest.read_bytes()).hexdigest()==expected:
        partial.write_bytes(dest.read_bytes())
    else:
        subprocess.run(cmd,check=True)
    md5=hashlib.md5(); sha=hashlib.sha256()
    with partial.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): md5.update(chunk); sha.update(chunk)
    if expected and md5.hexdigest()!=expected: raise RuntimeError('MD5 mismatch: '+rel)
    if rel.startswith('test_images'):
        with partial.open('rb') as f:
            if f.read(2)!=b'\xff\xd8': raise RuntimeError('Not JPEG: '+rel)
    partial.replace(dest)
    print('OK',rel,dest.stat().st_size,flush=True)
    return dict(path=rel,url=url,label=label,bytes=dest.stat().st_size,md5=md5.hexdigest(),sha256=sha.hexdigest(),official_md5=expected)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=list(pool.map(download,assets))
(root/'assets_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
(root/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in results))
print('ALL DOWNLOADS VERIFIED',flush=True)
