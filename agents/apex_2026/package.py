"""Package only the standalone inference file; never include development tools."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

PROHIBITED_IMPORTS = {"ctypes", "importlib", "multiprocessing", "os", "pathlib",
                      "resource", "shutil", "signal", "socket", "subprocess", "sys"}
PROHIBITED_CALLS = {"compile", "eval", "exec", "__import__"}


def build_package(source, output):
    source, output = Path(source), Path(output)
    data = source.read_bytes()
    tree = ast.parse(data.decode("utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        if PROHIBITED_IMPORTS.intersection(names):
            raise ValueError("prohibited inference import")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in PROHIBITED_CALLS:
            raise ValueError("prohibited inference call")
    if len(data) > 500 * 1024 * 1024:
        raise ValueError("inference source exceeds official member size limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    member = zipfile.ZipInfo("agent.py", date_time=(2026, 10, 4, 0, 0, 0))
    member.compress_type = zipfile.ZIP_DEFLATED
    member.external_attr = 0o100644 << 16
    # Exclusive creation preserves archives already produced for the user.
    with zipfile.ZipFile(output, "x") as archive:
        archive.writestr(member, data)
    if output.stat().st_size > 500 * 1024 * 1024:
        output.unlink()
        raise ValueError("archive exceeds official ZIP size limit")
    with zipfile.ZipFile(output) as archive:
        entry = archive.getinfo("agent.py")
        if entry.file_size / max(1, entry.compress_size) > 100:
            output.unlink()
            raise ValueError("archive exceeds official compression-ratio limit")
        if archive.testzip() is not None or archive.read("agent.py") != data:
            raise ValueError("archive integrity verification failed")
    return {"archive": str(output.resolve()), "size_bytes": output.stat().st_size,
            "zip_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "member_sha256": {"agent.py": hashlib.sha256(data).hexdigest()},
            "official_linux_certification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("agent.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.output.with_suffix(".package.json")
    if args.output.exists() or metadata.exists():
        parser.error("archive or metadata exists; use a new output path")
    receipt = build_package(args.source, args.output)
    with metadata.open("x", encoding="utf-8") as file:
        json.dump(receipt, file, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
