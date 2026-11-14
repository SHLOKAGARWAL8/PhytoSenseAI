"""Create dataset class-distribution and sample-image exploration figures."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset-dir',type=Path,default=Path('dataset'))
    p.add_argument('--output-dir',type=Path,default=Path('results/dataset_exploration')); a=p.parse_args()
    root=a.dataset_dir; out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    for cls in sorted((root/'train').iterdir()):
        if cls.is_dir(): counts[cls.name]=sum(1 for f in cls.rglob('*') if f.is_file() and f.suffix.lower() in EXTS)
    if not counts: raise FileNotFoundError(f'No class folders found under {root / "train"}')
    names=list(counts); vals=[counts[n] for n in names]
    fig,ax=plt.subplots(figsize=(16,8)); ax.bar(range(len(names)),vals); ax.set_xticks(range(len(names)),names,rotation=90,fontsize=7)
    ax.set(title='PlantVillage training-set class distribution',xlabel='Disease class',ylabel='Training images'); fig.tight_layout()
    fig.savefig(out/'class_distribution.png',dpi=160); plt.close(fig)
    samples=[]
    for cls in sorted((root/'train').iterdir()):
        if cls.is_dir():
            files=[f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
            if files: samples.append((cls.name,files[0]))
    samples=samples[:min(12,len(samples))]
    cols=4; rows=(len(samples)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(14,rows*3)); axes=list(getattr(axes,'flat',[axes]))
    for ax,(label,path) in zip(axes,samples):
        with Image.open(path) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(label.replace('___','\n'),fontsize=7); ax.axis('off')
    for ax in axes[len(samples):]: ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'sample_leaf_images.png',dpi=160); plt.close(fig)
    print(f'Wrote {out / "class_distribution.png"} and {out / "sample_leaf_images.png"}')
if __name__=='__main__': main()
