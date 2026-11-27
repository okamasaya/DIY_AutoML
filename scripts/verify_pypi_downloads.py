"""Download a published wheel and sdist and verify them against the local originals."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


INDEXES = {"testpypi": "https://test.pypi.org", "pypi": "https://pypi.org"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_downloads(index, name, version, wheel, sdist, output):
    originals = ((wheel, "bdist_wheel"), (sdist, "sdist"))
    expected = {path.name: sha256(path) for path, _ in originals}
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new download directory: {output}")

    api_url = f"{INDEXES[index]}/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
    with urlopen(api_url, timeout=30) as response:
        entries = {item["filename"]: item for item in json.load(response)["urls"]}

    urls = {}
    for path, kind in originals:
        entry = entries.get(path.name, {})
        if entry.get("packagetype") != kind or entry.get("yanked"):
            raise ValueError(f"The index has no active {kind} named {path.name}")
        if entry.get("digests", {}).get("sha256") != expected[path.name]:
            raise ValueError(f"Published SHA-256 differs for {path.name}")
        url = entry.get("url", "")
        if urlsplit(url).scheme != "https":
            raise ValueError(f"Expected an HTTPS download URL for {path.name}")
        urls[path.name] = url

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "verified"
        staged.mkdir()
        for filename, url in urls.items():
            target = staged / filename
            with urlopen(url, timeout=30) as response, target.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha256(target) != expected[filename]:
                raise ValueError(f"Downloaded SHA-256 differs for {filename}")
        staged.rename(output)

    for filename, digest in expected.items():
        print(f"{digest}  {output / filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, choices=INDEXES)
    parser.add_argument("--name", required=True, help="Published distribution name")
    parser.add_argument("--version", required=True, help="Exact published version")
    parser.add_argument("--wheel", type=Path, required=True, help="Original wheel")
    parser.add_argument("--sdist", type=Path, required=True, help="Original source distribution")
    parser.add_argument("--output", type=Path, required=True, help="New verified-download directory")
    args = parser.parse_args()
    try:
        verify_downloads(args.index, args.name, args.version, args.wheel, args.sdist, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Release download verification failed: {exc}\n")


if __name__ == "__main__":
    main()
