"""Apply runtime fixes to a separately obtained PixelGen checkout."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

PINNED_COMMIT = "0ccec6029f4c011590331be82f7e4eca8fb9ac4a"
RELATIVE_FILE = Path("src/models/conditioner/qwen3_text_encoder.py")
SOURCE_SHA256 = "85691f48e1c325872da17259fa383cf54cfe68bb16d77971e51e7181d8a71c42"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def transform(source):
    """Guard compile; send token tensors to the model's device."""
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Source file does not match the expected pinned commit SHA-256.")
    text = source.decode("utf-8")
    tree = ast.parse(text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen3TextEncoder"]
    if len(classes) != 1:
        raise ValueError("Expected one conditioner class")
    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    compile_calls = [n for n in methods["__init__"].body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                     and n.value.func.attr == "compile"]
    placements = [n for n in methods["_impl_condition"].body if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                  and n.value.func.attr == "cuda"]
    if len(compile_calls) != 1 or len(placements) != 2:
        raise ValueError("Unexpected pinned-source structure")
    lines = text.splitlines(keepends=True)
    replacements = []
    n = compile_calls[0]
    indent = " " * n.col_offset
    guarded = (indent + 'compile_enabled = os.getenv("PIXELGEN_QWEN3_COMPILE", "0").lower() in {"1", "true", "yes"}\n'
               + indent + 'if compile_enabled:\n' + '    ' + lines[n.lineno-1])
    replacements.append((n.lineno-1, n.end_lineno, guarded))
    for index, n in enumerate(placements):
        n.value.func.attr = "to"
        n.value.args = [ast.Name(id="model_device", ctx=ast.Load())]
        indent = " " * n.col_offset
        replacement = indent + ast.unparse(n) + "\n"
        if index == 0:
            replacement = indent + "model_device = next(self.model.parameters()).device\n" + replacement
        replacements.append((n.lineno-1, n.end_lineno, replacement))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    insertion = imports[2].end_lineno
    replacements.append((insertion, insertion, "import os\n"))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = ("".join(lines).rstrip("\n")+"\n").encode("utf-8")
    compile(result, str(RELATIVE_FILE), "exec")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pixelgen-root", type=Path, required=True)
    p.add_argument("--apply", action="store_true", help="Write fixes; otherwise only validate/preview")
    args = p.parse_args()
    target = args.pixelgen_root / RELATIVE_FILE
    if target.is_symlink():
        raise ValueError("Refusing a symlink target")
    original = target.read_bytes()
    updated = transform(original)
    report = {"pinned_commit": PINNED_COMMIT, "relative_file": str(RELATIVE_FILE),
              "source_sha256": digest(original), "updated_sha256": digest(updated),
              "changes": ["compile disabled by default; opt in through PIXELGEN_QWEN3_COMPILE",
                          "token tensors follow the encoder model device"], "applied": False}
    if args.apply:
        backup = target.with_name(target.name + ".before_research_fixes")
        if backup.exists():
            raise FileExistsError("Backup already exists; source is unchanged")
        with backup.open("xb") as f:
            f.write(original)
        if target.read_bytes() != original:
            raise ValueError("Source changed during preparation; no update written")
        target.write_bytes(updated)
        report["applied"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
