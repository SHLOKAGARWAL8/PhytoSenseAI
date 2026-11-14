"""Canonical two-phase PyTorch training pipeline for all AgroDetect classifiers.

Examples:
  python train_models.py --model all --data-dir dataset --phase1-epochs 5 --phase2-epochs 10
  python train_models.py --model efficientnet_b0 --phase1-epochs 5 --phase2-epochs 10
  python train_models.py --model all --no-pretrained

Outputs: models/<model>_best.pth and results/<model>/metrics.json, reports,
confusion matrices and learning curves. Test evaluation always includes the
final partial batch (drop_last=False).
"""
from __future__ import annotations
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED

ROOT = Path(__file__).resolve().parent
MEAN, STD = (0.485,0.456,0.406), (0.229,0.224,0.225)
MODEL_NAMES = ('resnet50','efficientnet_b0','mobilenet_v2','vgg16')

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256), nn.ReLU(inplace=True),
                                 nn.Dropout(0.4), nn.Linear(256, num_classes))
    elif name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                         nn.Linear(model.classifier[1].in_features, num_classes))
    elif name == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(256, num_classes))
    elif name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, num_classes))
    else: raise ValueError(f'Unknown model: {name}')
    return model

def set_trainable(model: nn.Module, name: str, phase: int):
    for p in model.parameters(): p.requires_grad = False
    if name == 'resnet50':
        for p in model.fc.parameters(): p.requires_grad = True
        if phase == 2:
            for p in model.layer4.parameters(): p.requires_grad = True
    elif name == 'efficientnet_b0':
        for p in model.classifier.parameters(): p.requires_grad = True
        if phase == 2:
            for p in model.features[6:].parameters(): p.requires_grad = True
    elif name == 'mobilenet_v2':
        for p in model.classifier.parameters(): p.requires_grad = True
        if phase == 2:
            for p in model.features[14:].parameters(): p.requires_grad = True
    elif name == 'vgg16':
        for p in model.classifier.parameters(): p.requires_grad = True
        if phase == 2:
            for p in model.features[24:].parameters(): p.requires_grad = True

def loaders(data_dir: Path, batch: int, workers: int):
    train_tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(brightness=.2,contrast=.2),
        transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    train = datasets.ImageFolder(data_dir/'train', transform=train_tf)
    val = datasets.ImageFolder(data_dir/'val', transform=eval_tf)
    test = datasets.ImageFolder(data_dir/'test', transform=eval_tf)
    expected = list(CLASS_NAMES)
    for label, ds in [('train',train),('val',val),('test',test)]:
        if ds.classes != expected: raise ValueError(f'{label} classes/order mismatch. Re-run preprocessing using shared config.py.')
    common = dict(batch_size=batch, num_workers=workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers>0, drop_last=False)
    return (DataLoader(train,shuffle=True,**common), DataLoader(val,shuffle=False,**common),
            DataLoader(test,shuffle=False,**common))

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None; model.train(training)
    loss_sum=0.0; correct=0; total=0; y_true=[]; y_pred=[]
    for images, labels in loader:
        images, labels = images.to(device,non_blocking=True), labels.to(device,non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                logits=model(images); loss=criterion(logits,labels)
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        predictions=logits.argmax(1)
        n=labels.size(0); loss_sum += loss.item()*n; correct += (predictions==labels).sum().item(); total += n
        y_true.extend(labels.detach().cpu().numpy().tolist()); y_pred.extend(predictions.detach().cpu().numpy().tolist())
    macro_f1=float(f1_score(y_true,y_pred,average='macro',zero_division=0)) if y_true else 0.0
    return loss_sum/max(total,1), correct/max(total,1), macro_f1

def evaluate(model, loader, device):
    model.eval(); y_true=[]; y_pred=[]; elapsed=0.0; seen=0
    with torch.inference_mode():
        for images, labels in loader:
            images=images.to(device,non_blocking=True)
            if device.type=='cuda': torch.cuda.synchronize()
            t=time.perf_counter(); logits=model(images)
            if device.type=='cuda': torch.cuda.synchronize()
            elapsed += time.perf_counter()-t; seen += len(labels)
            y_true.extend(labels.numpy().tolist()); y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.asarray(y_true),np.asarray(y_pred),elapsed/max(seen,1)*1000

def plot_results(name, history, y_true, y_pred, out_dir):
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(history['train_loss'],label='train'); axes[0].plot(history['val_loss'],label='validation'); axes[0].set(title='Loss',xlabel='Epoch',ylabel='Loss'); axes[0].legend()
    axes[1].plot(history['train_accuracy'],label='train'); axes[1].plot(history['val_accuracy'],label='validation'); axes[1].set(title='Accuracy',xlabel='Epoch',ylabel='Accuracy'); axes[1].legend()
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out_dir/'training_curves.png',dpi=140); plt.close(fig)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(CLASS_NAMES))))
    fig,ax=plt.subplots(figsize=(16,14)); im=ax.imshow(cm,interpolation='nearest',cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set(title=f'{name} confusion matrix',xlabel='Predicted',ylabel='Actual')
    ax.set_xticks(range(len(CLASS_NAMES)),CLASS_NAMES,rotation=90,fontsize=5); ax.set_yticks(range(len(CLASS_NAMES)),CLASS_NAMES,fontsize=5)
    fig.tight_layout(); fig.savefig(out_dir/'confusion_matrix.png',dpi=150); plt.close(fig)

def train_one(name,args,device):
    train_loader,val_loader,test_loader=loaders(args.data_dir,args.batch_size,args.workers)
    model=build_model(name,len(CLASS_NAMES),not args.no_pretrained).to(device)
    criterion=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history={'train_loss':[],'train_accuracy':[],'train_f1_macro':[],'val_loss':[],'val_accuracy':[],'val_f1_macro':[],'phase':[]}
    best_val_f1=-1.0; best_val_accuracy=0.0; best_state=None; scaler=torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    phases=[(1,args.phase1_epochs),(2,args.phase2_epochs)]
    for phase,epochs in phases:
        if epochs<=0: continue
        set_trainable(model,name,phase)
        params=[p for p in model.parameters() if p.requires_grad]
        lr=args.head_lr if phase==1 else args.finetune_lr
        optimizer=AdamW(params,lr=lr,weight_decay=args.weight_decay)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=2,factor=.3)
        print(f'[{name}] Phase {phase}: {sum(p.numel() for p in params):,} trainable parameters; {epochs} epochs')
        for epoch in range(epochs):
            tr_loss,tr_acc,tr_f1=run_epoch(model,train_loader,criterion,device,optimizer,scaler)
            va_loss,va_acc,va_f1=run_epoch(model,val_loader,criterion,device)
            scheduler.step(va_loss)
            for k,v in [('train_loss',tr_loss),('train_accuracy',tr_acc),('train_f1_macro',tr_f1),('val_loss',va_loss),('val_accuracy',va_acc),('val_f1_macro',va_f1)]: history[k].append(float(v))
            history['phase'].append(phase)
            if va_f1>best_val_f1:
                best_val_f1=va_f1; best_val_accuracy=va_acc; best_state=copy.deepcopy(model.state_dict())
                (ROOT/'models').mkdir(exist_ok=True)
                torch.save(best_state,ROOT/'models'/f'{name}_best.pth')
            print(f'  epoch {epoch+1:02d}/{epochs}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f} val_loss={va_loss:.4f}')
    if best_state is None: raise RuntimeError(f'{name}: no training epochs executed')
    model.load_state_dict(best_state)
    y_true,y_pred,inference_ms=evaluate(model,test_loader,device)
    result_dir=ROOT/'results'/name; result_dir.mkdir(parents=True,exist_ok=True)
    report=classification_report(y_true,y_pred,labels=list(range(len(CLASS_NAMES))),target_names=CLASS_NAMES,zero_division=0,digits=5)
    (result_dir/'classification_report.txt').write_text(report,encoding='utf-8')
    plot_results(name,history,y_true,y_pred,result_dir)
    weight_path=ROOT/'models'/f'{name}_best.pth'
    metrics={'model':name,'framework':'PyTorch','classes':CLASS_NAMES,'test_samples':int(len(y_true)),
        'test_accuracy':float(accuracy_score(y_true,y_pred)),'test_precision_macro':float(precision_score(y_true,y_pred,average='macro',zero_division=0)),
        'test_recall_macro':float(recall_score(y_true,y_pred,average='macro',zero_division=0)),
        'test_f1_macro':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
        'best_validation_accuracy':float(best_val_accuracy),'best_validation_f1_macro':float(best_val_f1),'inference_ms_per_image':float(inference_ms),
        'model_size_mb':round(weight_path.stat().st_size/(1024*1024),3),'checkpoint':str(weight_path.relative_to(ROOT)),
        'history':history,'evaluation_note':'Computed on the complete test loader; drop_last=False.'}
    (result_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    print(f'[{name}] test_acc={metrics["test_accuracy"]:.5f} macro_f1={metrics["test_f1_macro"]:.5f} samples={len(y_true)}')
    return metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=[*MODEL_NAMES,'all'],default='all')
    p.add_argument('--data-dir',type=Path,default=ROOT/'dataset')
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--phase1-epochs',type=int,default=5); p.add_argument('--phase2-epochs',type=int,default=10)
    p.add_argument('--head-lr',type=float,default=1e-3); p.add_argument('--finetune-lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--label-smoothing',type=float,default=.1)
    p.add_argument('--seed',type=int,default=SEED); p.add_argument('--no-pretrained',action='store_true')
    args=p.parse_args(); seed_everything(args.seed)
    if not args.data_dir.is_dir(): raise FileNotFoundError(f'Dataset not found: {args.data_dir}. Run preprocess_dataset.py first.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Device: {device}')
    selected=MODEL_NAMES if args.model=='all' else (args.model,); all_metrics=[]
    for name in selected: all_metrics.append(train_one(name,args,device))
    if all_metrics:
        summary=[{k:m[k] for k in ('model','test_samples','best_validation_accuracy','best_validation_f1_macro','test_accuracy','test_f1_macro','model_size_mb','inference_ms_per_image')} for m in all_metrics]
        out=ROOT/'results'; out.mkdir(exist_ok=True)
        (out/'model_comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        try:
            import csv
            with (out/'model_comparison.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
        except OSError: pass
        print('Comparison written to results/model_comparison.json and .csv')
if __name__=='__main__': main()
