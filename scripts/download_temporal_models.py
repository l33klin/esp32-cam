"""Fetch pinned research weights/source into an isolated offline-study directory."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/temporal-comparison'))
    p.add_argument('--manifest',type=Path,default=Path('config/temporal-models.json'))
    args=p.parse_args();root=args.root.resolve()
    for name,item in json.loads(args.manifest.read_text()).items():
        target=(root/name).resolve()
        if not target.is_relative_to(root):raise ValueError('Path outside research root')
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
            print('verified',name);continue
        with urlopen(item['url'],timeout=60) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise RuntimeError('Hash mismatch: '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(target)
        print('downloaded',name)


if __name__=='__main__':main()
