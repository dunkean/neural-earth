"""Import the pinned Magna Urbis Satellite material, without its finite-map host.

Usage: python tools/assets/import_magna_satellite.py --source ../magna-urbis
The generated WGSL functions are copied, namespaced and bounded to the material
entry points. The planetary adapter supplies the climate and immutable tiles.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def functions(source):
    out = {}
    for match in re.finditer(r'^fn (\w+)\(', source, re.M):
        start = source.index('{', match.end())
        end, depth = start + 1, 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        out[match[1]] = source[match.start():end]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Import a NEW working-tree snapshot; omit to rebuild the retained snapshot')
    args = parser.parse_args()
    if args.source is None:
        retained = ROOT/'native/magna_surface'
        emit_runtime((retained/'satellite.wgsl').read_text(encoding='utf-8'), json.loads((retained/'source.json').read_text(encoding='utf-8')))
        return
    source_root = args.source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
    raw = (source_root/'web/src/ui/terrainWgsl.generated.ts').read_text(encoding='utf-8')
    source = json.loads(re.search(r'export const terrainFs = ("(?:[^"\\]|\\.)*");', raw)[1])
    funcs = functions(source)
    # Only host boundaries are replaced: the material/BRDF/mean-canopy functions
    # below are the upstream generated WGSL, rather than a rewritten palette.
    replacements = {
        'forestField': 'fn forestField(p:vec2<f32>,lod:f32)->vec4<f32>{let f=r3Aux(p,2,lod);return vec4<f32>(f.rgb,v2CanopyTone);}',
        'forestBankCover': 'fn forestBankCover(p:vec2<f32>,fp:f32)->f32{return 0.;}',
        'forestWaterCoverage': 'fn forestWaterCoverage(p:vec2<f32>)->f32{return magnaWaterCover(p);}',
        'groundField': 'fn groundField(p:vec3<f32>,seed:u32,fp:f32)->vec4<f32>{var clasts=0.;return groundFieldWithClasts(p.xz,seed,fp,&clasts);}',
        'groundSlopeVariance': 'fn groundSlopeVariance(fp:f32)->f32{return groundSlopeVariance(fp);}',
        'v2Pattern': 'fn v2Pattern(p:vec3<f32>,scale:f32,fp:f32,salt:u32)->f32{return magnaPattern(p,scale,fp,salt);}',
        # Magna Urbis' optional manually selected climate-zone art direction is
        # disabled: Neural Earth already supplies spatial seasonal climate.
        'v2Zoned': 'fn v2Zoned()->bool{return false;}',
        'v2Zone': 'fn v2Zone()->i32{return 0;}',
        'v2ZoneOpen': 'fn v2ZoneOpen(c:vec3<f32>,e:V2Env,p:vec2<f32>,fp:f32)->vec3<f32>{return c;}',
        'v2SeasonHerb': 'fn v2SeasonHerb()->vec3<f32>{return vec3<f32>(1.);}',
    }
    seen, pending = set(), ['v2GroundEnv', 'v2ShadeGround', 'v2DisplayShoulder']
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in replacements:
            pending.extend(n for n in re.findall(r'\b(\w+)\s*\(', funcs[name]) if n in funcs and n not in seen)
    header = source[:source.index('fn bw(')]
    header = re.sub(r'^diagnostic[^;]+;\s*', '', header)
    header = re.sub(r'@group\(\d+\)\s*@binding\(\d+\)\s*var(?:<[^>]+>)?\s+\w+:[^;]+;', '', header)
    header = re.sub(r'@id\(\d+\) override (\w+): i32 = 0i;', lambda m: f'const {m[1]}:i32 = '+('1' if m[1] in ('uForest','uRenderV2_') else '3' if m[1]=='uMode' else '0')+';', header)
    header += '\nvar<private> global: TerrainParams;\n'
    # Prefix all source-level symbols, while leaving structure field names alone.
    symbols = set(re.findall(r'\b(?:struct|const|var(?:<[^>]+>)?)\s+(\w+)', header)) | set(funcs)
    rename = lambda text: re.sub(r'\b('+'|'.join(sorted(symbols, key=len, reverse=True))+r')\b', lambda m:'mu_'+m[0], text)
    chunks = []
    for name in funcs:
        if name not in seen:
            continue
        if name in replacements:
            # Namespace the declaration/type references, not the host callback.
            text = replacements[name]
            signature, body = text.split('{', 1)
            chunks.append(rename(signature)+'{'+body.replace('e:V2Env','e:mu_V2Env'))
        else:
            text = funcs[name].replace('textureDimensions(uForestField, 0i)', 'textureDimensions(aux, 0)')
            text = text.replace('textureDimensions(uForestField, 0)', 'textureDimensions(aux, 0)')
            chunks.append(rename(text))
    shader = rename(header)+'\n'+'\n\n'.join(chunks)
    # Evaluate the upstream finite-map microdetail in overlapping, globally
    # anchored domains. The host avoids both tile-local phases and f32 metre
    # coordinates at planetary distances; material functions stay upstream.
    shader = shader.replace('fn mu_surfaceMicro(', 'fn mu_surfaceMicroLocal(')
    shader += '\nfn mu_surfaceMicro(p:vec2<f32>,s:f32,fp:f32,r:f32,g:f32,seed:u32)->vec3<f32>{return magnaMicro(p,s,fp,r,g,seed);}\n'
    if re.search(r'\btexture\w*\(u\w+', shader):
        raise ValueError('Unadapted finite-map texture input')
    destination = ROOT/'native/magna_surface'
    destination.mkdir(exist_ok=True)
    (destination/'satellite.wgsl').write_text(shader, encoding='utf-8', newline='\n')
    license_path = next(p for p in (source_root/'LICENSE', source_root/'LICENSE.md') if p.exists())
    shutil.copyfile(license_path, destination/'LICENSE')
    # Retain the exact CPU reference implementations for provenance and
    # independent numerical comparisons of the Python presentation sampler.
    for name in ('web/src/ui/terrain3dNoise.ts', 'rust/bridge/terrainGround.ts'):
        shutil.copyfile(source_root/name, destination/Path(name).name)
    manifest = dict(repository='https://github.com/dunkean/magna-urbis', revision=revision,
        sources={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in
                 ('web/src/ui/terrainWgsl.generated.ts','rust/bridge/terrainRenderV2.ts','rust/bridge/terrainGround.ts','web/src/ui/terrain3dNoise.ts')},
        entry_points=['v2GroundEnv','v2ShadeGround','v2DisplayShoulder'], functions=len(seen))
    manifest['dirty_sources'] = [name for name in manifest['sources'] if hashlib.sha256(subprocess.check_output(['git','-C',str(source_root),'show',revision+':'+name])).hexdigest() != manifest['sources'][name]]
    manifest['snapshot_id'] = hashlib.sha256(json.dumps(manifest['sources'], sort_keys=True).encode()).hexdigest()
    (destination/'source.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    emit_runtime(shader, manifest)


def emit_runtime(shader, manifest):
    revision = manifest['revision']
    runtime = ROOT/'web/terrain_magna_satellite.js'
    adapter = (ROOT/'tools/assets/magna_satellite_adapter.wgsl').read_text(encoding='utf-8')
    coarse = shader
    coarse = coarse.replace('return magnaMicro(p,s,fp,r,g,seed);', 'return vec3<f32>(1.);')
    coarse_replacements = {
        'mu_forestField': 'return magnaCoarsePopulation;',
        'mu_forestWaterCoverage': 'return 0.;',
        'mu_groundField': 'return vec4<f32>(0.);',
        'mu_groundSlopeVariance': 'return '+repr(sum((a/w)**2*(1+1/r**2)*v for w,a,r,v in ((8,.17,.35,.2),(2.2,.04,.24,1.8),(.32,.012,.85,1.3),(.055,.002,.72,.4))))+';',
        'mu_v2Pattern': 'return 0.;',
    }
    for name, body in coarse_replacements.items():
        original = functions(coarse)[name]
        coarse = coarse.replace(original, original.split('{', 1)[0]+'{'+body+'}')
    coarse = coarse.replace('textureDimensions(aux, 0)', 'textureDimensions(elevation, 0)')
    coarse += '\n'+functions(adapter)['magnaEnvironment'].replace('m.seed=r3Seed()', 'm.seed=u32(surfaceMeta(6))')
    coarse += '\n'+(ROOT/'tools/assets/magna_satellite_coarse.wgsl').read_text(encoding='utf-8')
    runtime.write_text('/* Generated by tools/assets/import_magna_satellite.py. Magna Urbis AGPL-3.0. */\n'+
        '(root=>{root.TerrainMagnaSatellite={revision:'+json.dumps(revision)+',snapshot:'+json.dumps(manifest.get('snapshot_id',revision))+',shader:'+json.dumps(shader+'\n'+adapter)+',coarseShader:'+json.dumps(coarse)+'};})(typeof window===\'undefined\'?globalThis:window);\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(revision=revision, functions=manifest['functions'], shader_bytes=len(shader))))


if __name__ == '__main__':
    main()
