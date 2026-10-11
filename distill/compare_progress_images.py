"""Export completed physical app comparisons into the user's flat _base folder."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from distill.common import DATA, atomic_json, atomic_write, external_path
from distill.native_progress_images import LABELS, shade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DATA/'_base')
    args = parser.parse_args()
    root, output = external_path(args.comparison), external_path(args.output)
    report = json.loads((root/'report.json').read_text())
    fingerprint = report['checkpoint_digests']['native_student']
    step, sha = fingerprint['training_step'], fingerprint['base']
    reference = {(r['site'], r['lod']): r for r in report['variants']['reference'] if 'error' not in r}
    student = {(r['site'], r['lod']): r for r in report['variants']['native_student'] if 'error' not in r}
    sites = report['sites']
    if len(student) != len(sites)*2 or reference.keys() != student.keys():
        raise ValueError('Export only a complete, paired two-LOD comparison.')
    groups = ({kind: [s for s in sites if s['kind'] == kind] for kind in dict.fromkeys(s['kind'] for s in sites)}
              if report.get('site_manifest_digest') else {'historical': sites})
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    size, left, pad, height, header = 512, 280, 12, 550, 130
    with np.load(root/'arrays.npz', allow_pickle=False) as arrays:
        for label, group in groups.items():
            for lod in (3, 0):
                image = Image.new('RGB', (left+2*(size+pad), header+height*len(group)), (24, 24, 24))
                draw = ImageDraw.Draw(image)
                draw.text((12, 10), f'Etape {step} | Chaine application | {LABELS.get(label, label)} | LOD {lod}', fill='white', font=font)
                draw.text((12, 39), 'Meme coarse4 et decoder200k. Mondes neufs, sans cache. Ombrage fixe sans lissage.', fill='white', font=small)
                draw.text((12, 64), f'Checkpoint {sha[:12]} | 30 x 2^{lod} metres/pixel ; agrandissement au plus proche.', fill='white', font=small)
                draw.text((left, 100), 'Teacher base original', fill='white', font=font)
                draw.text((left+size+pad, 100), f'Eleve base, etape {step}', fill='white', font=font)
                rows = []
                for index, site in enumerate(group):
                    name = site['name']; row = student[name, lod]; error = row['vs_reference']
                    y = header+index*height
                    title = '\n'.join(textwrap.wrap(name, width=29, break_on_hyphens=True))
                    caption = f'{title}\nSeed {site["seed"]}\n{site["profile"]}\nErreur : {error["mae"]:.2f} m\nCote : {100*error["coast"]:.2f} %'
                    draw.multiline_text((12, y+130), caption, fill='white', font=small, spacing=7)
                    for column, variant in enumerate(('reference', 'native_student')):
                        elevation = arrays[f'{variant}|{name}|{lod}']
                        panel = shade(elevation, 30*2**lod, 3. if lod == 3 else 1.5).resize((size, size), Image.Resampling.NEAREST)
                        image.paste(panel, (left+column*(size+pad), y))
                    rows.append(dict(step=step, site=name, seed=site['seed'], profile=site['profile'],
                        x_m=site['x'], y_m=site['y'], lod=lod, mae_m=error['mae'], coast_disagreement=error['coast'],
                        reference_slope_mean_deg=reference[name, lod]['slope_mean'], student_slope_mean_deg=row['slope_mean']))
                stem = f'{step:06d}-app-{label}-lod{lod}'
                description = ('Comparaison complete de la chaine application sur les memes sites et le meme coarse4/decoder200k. '
                               f'Etape {step}, LOD {lod}. Donnees et erreurs dans {stem}.csv. '
                               'Les secondes de ces evaluations simultanees ne sont pas une mesure de performance.')
                meta = PngImagePlugin.PngInfo(); meta.add_text('Description', description); meta.add_text('CheckpointSHA256', sha)
                path = output/f'{stem}.png'
                if path.exists():
                    with Image.open(path) as existing:
                        if existing.info.get('CheckpointSHA256') != sha:
                            raise ValueError('Refuse to overwrite images of another model at the same step.')
                atomic_write(path, lambda f: image.save(f, format='PNG', pnginfo=meta))
                with (output/f'{stem}.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                (output/f'{stem}.txt').write_text(description+'\n', encoding='utf-8')
                atomic_json(output/f'{stem}.json', dict(checkpoint_sha256=sha, comparison=str(root),
                    report_sha256=hashlib.sha256((root/'report.json').read_bytes()).hexdigest(), rows=rows, accepted=False))
                print(json.dumps(dict(step=step, image=str(path))), flush=True)


if __name__ == '__main__':
    main()
