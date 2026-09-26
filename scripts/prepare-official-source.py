"""Copy pinned official code to a run environment and apply recorded fixes."""
import argparse
import shutil
from pathlib import Path

from experiments import ROOT, patch_3d_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    if args.destination.exists():
        parser.error('destination must not already exist')
    source=ROOT/'external/medmnist-experiments'
    shutil.copytree(source,args.destination,ignore=shutil.ignore_patterns('.git'))
    target=args.destination/'MedMNIST3D/train_and_eval_pytorch.py'
    target.write_text(patch_3d_checkpoint(target.read_text(encoding='utf-8')),encoding='utf-8')
    (args.destination/'LOCAL_PATCHES.md').write_text(
        '# Local compatibility patches\n\n'
        '- The 3D training script now saves `best_model.state_dict()` because that is the model selected by validation AUC and used for final evaluation. Upstream saved the final-epoch model under `best_model.pth`.\n',
        encoding='utf-8')
    print(args.destination)


if __name__=='__main__':
    main()
