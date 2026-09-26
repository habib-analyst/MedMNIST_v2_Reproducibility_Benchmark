"""Run all three joint-model architecture checks without claiming metrics."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import specs_from_plan
from src.smoke import architecture_smoke


def main():
    plan_path = ROOT / 'configs' / 'experiment-plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    specs = specs_from_plan(plan)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'runs' / 'smoke' / f'joint-architecture-{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    records = [architecture_smoke(mode, specs) for mode in ('2d', '3d', 'unified')]
    result = {
        'kind': 'architecture-smoke-only',
        'status': 'completed',
        'created_utc': stamp,
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'records': records,
        'official_runs_completed': 0,
        'caveat': 'No dataset performance was measured. These are synthetic forward/backward checks.',
    }
    target = output / 'status.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
