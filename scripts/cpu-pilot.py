"""Small real-data CPU feasibility test. Never counts as an official baseline."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def training_seconds(samples_per_second, train_samples, epochs, seeds):
    if not math.isfinite(samples_per_second) or samples_per_second<=0:
        raise ValueError('Positive measured throughput required')
    return train_samples*epochs*seeds/samples_per_second


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.batch_size<2 or args.steps<2 or args.threads<1:
        parser.error('batch-size and steps must be at least two; threads positive')
    import numpy as np
    import torch
    import torchvision
    import medmnist
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from tensorboardX import SummaryWriter

    torch.set_num_threads(args.threads)
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'runs'/'smoke'/f'chestmnist-cpu-{stamp}'
    out.mkdir(parents=True,exist_ok=False)
    record=dict(kind='smoke-only',status='running',started_utc=stamp,pid=os.getpid(),
        seed=17,device='cpu',threads=args.threads,batch_size=args.batch_size,
        steps=args.steps,source_size=28,input_size=224,official_runs_completed=0)
    def save():
        temp=out/'status.tmp'
        temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        temp.replace(out/'status.json')
    save()
    print(f'Pilot output: {out}',flush=True)
    try:
        data_root=ROOT/'data'; data_root.mkdir(exist_ok=True)
        transform=transforms.Compose([transforms.Resize((224,224),interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),transforms.Normalize(mean=[.5],std=[.5])])
        dataset=medmnist.ChestMNIST(split='train',root=str(data_root),download=True,as_rgb=True,size=28,transform=transform)
        digest=hashlib.md5((data_root/'chestmnist.npz').read_bytes()).hexdigest()
        if digest!=medmnist.INFO['chestmnist']['MD5']:
            raise ValueError('Downloaded dataset checksum mismatch')
        source=ROOT/'external/medmnist-experiments/MedMNIST2D'
        sys.path.insert(0,str(source))
        spec=importlib.util.spec_from_file_location('official_2d',source/'train_and_eval_pytorch.py')
        official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
        model=torchvision.models.resnet50(weights=None,num_classes=14)
        criterion=torch.nn.BCEWithLogitsLoss()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        official.iteration=0
        durations=[]
        with SummaryWriter(str(out/'tensorboard')) as writer:
            for step in range(args.steps):
                indices=range(step*args.batch_size,(step+1)*args.batch_size)
                loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,list(indices)),batch_size=args.batch_size)
                start=time.perf_counter()
                loss=official.train(model,loader,'multi-label, binary-class',criterion,optimizer,torch.device('cpu'),writer)
                seconds=time.perf_counter()-start
                if not math.isfinite(loss): raise ValueError('Nonfinite training loss')
                durations.append(seconds)
                print(f'Smoke step {step+1}/{args.steps}: {seconds:.2f}s, loss {loss:.6f}',flush=True)
        throughput=args.batch_size/float(np.median(durations[1:]))
        model.eval()
        with torch.no_grad():
            prediction=torch.sigmoid(model(dataset[0][0].unsqueeze(0)))
        if prediction.shape!=(1,14) or not torch.isfinite(prediction).all():
            raise ValueError('Invalid smoke prediction')
        # Timing extrapolation is not a training result and excludes evaluation/thermal effects.
        record.update(status='completed',step_seconds=durations,samples_per_second=throughput,
            chest_five_runs_training_only_days=training_seconds(throughput,len(dataset),100,5)/86400,
            dataset_md5=digest,torch=torch.__version__,torchvision=torchvision.__version__,
            medmnist=medmnist.__version__,python=platform.python_version(),
            upstream_experiments_commit=subprocess.check_output(['git','-C',str(ROOT/'external/medmnist-experiments'),'rev-parse','HEAD'],text=True).strip(),
            pilot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            caveat='Small-batch extrapolation only. Excludes full-split evaluations, thermal throttling and batch-128 memory costs. Not an official result.',
            ended_utc=datetime.now(timezone.utc).isoformat())
        (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
        save(); print(json.dumps(record,indent=2),flush=True)
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        save(); raise


if __name__=='__main__':
    main()
