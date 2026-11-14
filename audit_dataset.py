"""Audit ImageFolder class order, image dimensions and cross-split content leakage."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from PIL import Image
EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}

def digest(path: Path) -> str:
    with Image.open(path) as image:
        rgb=image.convert('RGB'); h=hashlib.sha256()
        h.update(f'{rgb.width}x{rgb.height}:'.encode('ascii')); h.update(rgb.tobytes())
        return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',type=Path,default=Path('dataset'))
    p.add_argument('--output',type=Path,default=Path('dataset_audit.json')); p.add_argument('--image-size',type=int,default=224)
    a=p.parse_args(); root=a.dataset_dir
    splits=('train','val','test'); by_hash=defaultdict(list); counts={}; classes_by_split={}; bad_dimensions=[]; image_count=0
    for split in splits:
        base=root/split
        if not base.is_dir(): raise FileNotFoundError(base)
        classes_by_split[split]=sorted(x.name for x in base.iterdir() if x.is_dir()); counts[split]={}
        for cls in classes_by_split[split]:
            files=[x for x in (base/cls).rglob('*') if x.is_file() and x.suffix.lower() in EXTS]
            counts[split][cls]=len(files)
            for f in files:
                with Image.open(f) as im:
                    size=im.size; im.verify()
                image_count+=1
                if size != (a.image_size,a.image_size): bad_dimensions.append({'path':str(f),'size':size})
                by_hash[digest(f)].append({'split':split,'class_name':cls,'path':str(f)})
    class_lists_match=len(set(tuple(v) for v in classes_by_split.values()))==1
    mapping_path=root.parent/'class_indices.json'; mapping_ok=False
    if mapping_path.exists():
        mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
        expected={name:i for i,name in enumerate(classes_by_split['train'])}
        mapping_ok=mapping==expected
    overlaps={h:items for h,items in by_hash.items() if len({x['split'] for x in items})>1}
    cross_class={h:items for h,items in by_hash.items() if len({x['class_name'] for x in items})>1}
    empty_classes={s:[c for c,n in counts[s].items() if n==0] for s in splits}
    passed=class_lists_match and mapping_ok and not overlaps and not cross_class and not bad_dimensions and not any(empty_classes.values())
    result={'dataset_dir':str(root.resolve()),'expected_image_size':[a.image_size,a.image_size],
        'class_count':len(classes_by_split['train']),'class_lists_match':class_lists_match,
        'class_indices_mapping_matches_ImageFolder':mapping_ok,'split_counts':{s:sum(counts[s].values()) for s in splits},
        'per_class_counts':counts,'empty_classes':empty_classes,'images_checked':image_count,
        'wrong_dimensions':len(bad_dimensions),'wrong_dimension_examples':bad_dimensions[:20],
        'unique_content_hashes':len(by_hash),'duplicate_hash_groups':sum(1 for v in by_hash.values() if len(v)>1),
        'cross_split_duplicate_groups':len(overlaps),'cross_split_duplicates':list(overlaps.values()),
        'cross_class_duplicate_groups':len(cross_class),'cross_class_duplicates':list(cross_class.values()),
        'status':'PASS' if passed else 'FAIL'}
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_class_counts','cross_split_duplicates','cross_class_duplicates','wrong_dimension_examples')},indent=2))
    if not passed: raise SystemExit(2)
if __name__=='__main__': main()
