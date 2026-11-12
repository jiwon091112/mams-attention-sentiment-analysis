"""Download the recorded public inputs without changing the experiment code."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/master/data/MAMS-ATSA/raw/'
GLOVE = 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz'


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure(url, path, expected, algorithm, check_only):
    if not path.exists():
        if check_only:
            raise FileNotFoundError(path)
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            with urlopen(url, timeout=120) as response, temporary.open('wb') as f:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(chunk)
            if digest(temporary, algorithm) != expected:
                raise ValueError(f'Checksum mismatch: {path.name}; source may have changed')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(path, algorithm) != expected:
        raise ValueError(f'Checksum mismatch: {path.name}')
    print('Verified:', path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'raw')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'recorded/data/manifest.json').read_text())
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for split, expected in manifest['raw_sha256'].items():
        ensure(SOURCE + split + '.xml', args.raw_dir / (split + '.xml'), expected, 'sha256', args.check_only)
    ensure(GLOVE, args.raw_dir / 'glove-wiki-gigaword-100.gz', manifest['pretrained_md5'], 'md5', args.check_only)
    if not args.check_only:
        for folder in ['data', 'results']:
            (ROOT / folder).mkdir(exist_ok=True)


if __name__ == '__main__':
    main()
