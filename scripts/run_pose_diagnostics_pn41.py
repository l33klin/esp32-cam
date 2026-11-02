"""Run untimed diagnostics after a timing batch; then VGA finalist profiles."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-manifest', type=Path, required=True)
    parser.add_argument('--vga-models', nargs='+', default=['lightning', 'thunder'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    deadline = time.monotonic()+3600
    while True:
        try:
            finished = json.loads((root/args.wait_manifest).read_text()).get('finished')
        except (OSError, ValueError):
            finished = False
        if finished:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Previous batch unfinished')
        time.sleep(5)
    compose = ['sudo', '-n', 'docker', 'compose', '--env-file', str(root/'deploy/.env'),
               '-f', str(root/'deploy/compose.yaml'), '-f', str(root/'deploy/compose.pn41.yaml')]
    running = subprocess.check_output(['sudo', '-n', 'docker', 'inspect', '--format',
        '{{.State.Running}}', 'indoorcam-analytics-1'], text=True).strip() == 'true'
    subprocess.run(compose+['stop', 'analytics'], check=True)
    status = {}
    try:
        for module in ('probe_detector_sanity', 'probe_pose_oracle'):
            print('START', module, flush=True)
            cmd = ['sudo', '-n', 'docker', 'run', '--rm', '--user', '0', '--network', 'none',
                   '--device', '/dev/apex_0', '--device', '/dev/apex_1', '-v', f'{root}:/work',
                   '-w', '/work', '--entrypoint', 'python', 'indoorcam:pn41', '-m', f'scripts.{module}']
            status[module] = subprocess.run(cmd, cwd=root, timeout=300).returncode
    finally:
        if running:
            subprocess.run(compose+['start', 'analytics'], check=True)
        (root/'evaluations/pose-comparison/diagnostics-status.json').write_text(json.dumps(status, indent=2)+'\n')
    if args.vga_models:
        subprocess.run(['python3', '-m', 'scripts.run_pose_comparison_pn41',
                        '--output', 'evaluations/pose-comparison/vga-results',
                        '--input-max-width', '640', '--models', *args.vga_models], cwd=root, check=True)


if __name__ == '__main__':
    main()
