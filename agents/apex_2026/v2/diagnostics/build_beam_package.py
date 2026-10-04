"""Deterministic /tmp-only standalone preflight builder; never selects/freezes.

Input bytes must match the two explicitly authorized archived source hashes.
Only AST-bound shadow _Road identifiers and the exact beam helper import change.
All other source text, including numeric literals and MIT notice, is preserved.
"""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize

BEAM_SHA='aef90410f8ad7f3f022a7c5a07c9ba87b997304964a63e5ca0601d8e60e7e8fa'
SHADOW_SHA='8d5c21ce445bda2f192e4815739253ebf46d1099f635d966c12fe9e877990a56'
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BEAM=ROOT/'results/beam_sources/aef90410f8ad.py'
DEFAULT_SHADOW=ROOT/'results/shadow-sources/r2/shadow_physics.py'
R6_BEAM_SHA='cc0543c2bebbec0f2c1917a60b4e506cdd3da5bd562131e7090ae2bb55d65425'
R6_BEAM=ROOT/'results/beam_refined_sources/cc0543c2bebb.py'
PINNED_BEAMS={'r3':(BEAM_SHA,DEFAULT_BEAM),'r6':(R6_BEAM_SHA,R6_BEAM)}

def digest(data):return hashlib.sha256(data).hexdigest()

def _replace(source,edits):
    lines=source.splitlines(keepends=True)
    starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    for (row,column),old,new in sorted(edits,reverse=True):
        offset=starts[row-1]+column
        if source[offset:offset+len(old)]!=old:raise ValueError('AST/token source span mismatch')
        source=source[:offset]+new+source[offset+len(old):]
    return source

def rename_shadow_road(source):
    tree=ast.parse(source)
    definitions=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Road']
    if len(definitions)!=1:raise ValueError('Expected one module shadow _Road binding')
    if any((isinstance(n,ast.Name) and n.id=='_ShadowRoad') or
           (isinstance(n,ast.ClassDef) and n.name=='_ShadowRoad') for n in ast.walk(tree)):
        raise ValueError('Shadow replacement name already bound')
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ShadowCar')
    initializer=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    uses=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='_Road']
    scoped=[n for n in ast.walk(initializer) if isinstance(n,ast.Name) and n.id=='_Road']
    if len(uses)!=1 or uses!=scoped or not isinstance(uses[0].ctx,ast.Load):
        raise ValueError('Unexpected _Road binding/use; manual review required')
    token=next(t for t in tokenize.generate_tokens(io.StringIO(source).readline)
               if t.type==tokenize.NAME and t.string=='_Road' and t.start[0]==definitions[0].lineno)
    return _replace(source,[(token.start,'_Road','_ShadowRoad'),
                            ((uses[0].lineno,uses[0].col_offset),'_Road','_ShadowRoad')])

def remove_project_import(source):
    tree=ast.parse(source)
    matches=[n for n in tree.body if isinstance(n,ast.ImportFrom) and
             n.module=='agents.apex_2026.v2.shadow_physics']
    if len(matches)!=1:raise ValueError('Expected exact beam helper import')
    node=matches[0]
    if node.level or [(x.name,x.asname) for x in node.names]!=[('ShadowCar',None),('decode_wheel_omega',None)]:
        raise ValueError('Unexpected beam import bindings')
    if node.lineno!=node.end_lineno:raise ValueError('Unexpected multiline import')
    lines=source.splitlines(keepends=True)
    if lines[node.lineno-1].strip()!=ast.get_source_segment(source,node):
        raise ValueError('Import shares line with other source')
    lines[node.lineno-1]='\n'
    return ''.join(lines)

def build(beam_path,shadow_path,output,*,variant='r3'):
    if variant not in PINNED_BEAMS:raise ValueError('Unknown pinned beam variant')
    beam_sha,_=PINNED_BEAMS[variant]
    beam_bytes=Path(beam_path).read_bytes();shadow_bytes=Path(shadow_path).read_bytes()
    if digest(beam_bytes)!=beam_sha:raise ValueError('Beam source hash mismatch')
    if digest(shadow_bytes)!=SHADOW_SHA:raise ValueError('Shadow source hash mismatch')
    output=Path(output).resolve()
    if not output.is_relative_to(Path('/tmp')) or output==Path('/tmp'):
        raise ValueError('Preflight output must be an explicit subdirectory of /tmp')
    shadow=rename_shadow_road(shadow_bytes.decode('utf-8'))
    beam=remove_project_import(beam_bytes.decode('utf-8'))
    header='# Standalone packaging preflight only; no candidate selection or freeze.\n'
    header+=f'# Beam SHA256 {beam_sha}\n# Shadow SHA256 {SHADOW_SHA}\n'
    packaged=(header+shadow.rstrip('\n')+'\n\n'+beam.rstrip('\n')+'\n').encode('utf-8')
    tree=ast.parse(packaged)
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):modules=[x.name for x in n.names]
        elif isinstance(n,ast.ImportFrom):modules=[n.module or '']
        else:continue
        if any(m.split('.')[0] not in {'math','heapq','cv2','numpy','Box2D'} for m in modules):
            raise ValueError('Unexpected standalone dependency')
    manifest=dict(schema_version=1,scope='packaging_preflight_only_no_selection_or_freeze',
        beam_sha256=beam_sha,shadow_sha256=SHADOW_SHA,agent_sha256=digest(packaged),
        transformations=['shadow module _Road binding and sole ShadowCar.__init__ load renamed _ShadowRoad',
                         'beam exact project helper import removed; archived texts concatenated'],
        files=['agent.py','manifest.json'])
    files={'agent.py':packaged,'manifest.json':(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}
    for name,data in files.items():
        path=output/name
        if path.is_symlink():raise ValueError('Output artifact must not be a symlink')
        if path.exists() and path.read_bytes()!=data:raise ValueError('Existing preflight artifact differs')
    output.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():(output/name).write_bytes(data)
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=tuple(PINNED_BEAMS),default='r3')
    parser.add_argument('--beam',type=Path,default=None)
    parser.add_argument('--shadow',type=Path,default=DEFAULT_SHADOW)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    beam=args.beam if args.beam is not None else PINNED_BEAMS[args.variant][1]
    print(json.dumps(build(beam,args.shadow,args.output,variant=args.variant),indent=2))
if __name__=='__main__':main()
