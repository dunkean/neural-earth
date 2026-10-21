"""Six readable crops of real paired terrain plates, retaining source metadata."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from distill.common import atomic_json, external_path


def publish(report_path, output):
    report = json.loads(report_path.read_text())
    if report['representation'] != 'packed-nf4' or report['weight_bytes'] >= 100000000:
        raise ValueError('An actual sub100MB packed NF4 evaluation is required.')
    output = external_path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(report_path.with_suffix('.png')) as source:
        plate = source.convert('RGB')
    if plate.size != (1293, 125+len(report['rows'])*556):
        raise ValueError('Unknown source plate geometry.')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    titles = ['Plaine désertique — aride', 'Plaine tempérée', 'Côte découpée — humide',
              'Plaine basse et mer', 'Haute montagne humide — A', 'Haute montagne humide — B']
    slugs = ['01-desert', '02-plaine-temperee', '03-cote-humide', '04-plaine-mer',
             '05-montagne-humide-a', '06-montagne-humide-b']
    cards, entries = [], []
    for scene, (title, slug) in enumerate(zip(titles, slugs, strict=True)):
        detail = Image.new('RGB', (1060, 1240), (24, 24, 24))
        draw = ImageDraw.Draw(detail)
        draw.text((12, 10), title, fill='white', font=font)
        draw.text((12, 44), f'NF4 {report["weight_bytes"]/1e6:.1f} Mo • étape {report["step"]} • {report["selected_weights"]}', fill='white', font=small)
        draw.text((12, 72), 'Même seed et bruit • decoder original • relief ×1,5 • sans lissage', fill='white', font=small)
        draw.text((12, 102), 'Original BF16 — 2 passes', fill='white', font=small)
        draw.text((536, 102), 'Base NF4 — 1 passe', fill='white', font=small)
        fine, rows = [], []
        for row_number, lod in enumerate((3, 0)):
            index = scene*2+row_number
            row = report['rows'][index]
            if row['lod'] != lod:
                raise ValueError('Unexpected LOD ordering.')
            y, dy = 125+index*556, 156+row_number*542
            draw.text((12, dy-24), f'LOD {lod} • seed {row["seed"]} • écart moyen {row["mae_m"]:.2f} m', fill='white', font=small)
            for column, x in enumerate((245, 769)):
                panel = plate.crop((x, y, x+512, y+512))
                detail.paste(panel, (12+column*524, dy))
                if lod == 0:
                    fine.append(panel)
            rows.append(row)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('CheckpointSHA256', report['checkpoint_sha256'])
        metadata.add_text('PackedWeightsSHA256', report['packed_weights_sha256'])
        metadata.add_text('Description', 'Real packed runtime. LOD3:32px native;LOD0:256px native. Nearest enlargement;no denoising.')
        detail.save(output/(slug+'.png'), pnginfo=metadata)
        card = Image.new('RGB', (612, 394), (24, 24, 24))
        draw = ImageDraw.Draw(card)
        draw.text((10, 8), title, fill='white', font=font)
        draw.text((10, 40), 'Original (2 passes)', fill='white', font=small)
        draw.text((316, 40), 'NF4 (1 passe)', fill='white', font=small)
        for column, panel in enumerate(fine):
            card.paste(panel.resize((296, 296), Image.Resampling.NEAREST), (10+column*306, 66))
        draw.text((10, 368), f'LOD0 • seed {rows[1]["seed"]} • écart {rows[1]["mae_m"]:.1f} m', fill='white', font=small)
        cards.append(card)
        entries.append(dict(title=title, file=slug+'.png', rows=rows))
    contact = Image.new('RGB', (1248, 1318), (15, 15, 15))
    draw = ImageDraw.Draw(contact)
    draw.text((12, 10), f'Base une passe NF4 — {report["weight_bytes"]/1e6:.1f} Mo — étape {report["step"]}', fill='white', font=font)
    draw.text((12, 44), f'{report["selected_weights"]} • {report["rows"][0]["coarse"]} • original à gauche, NF4 à droite', fill='white', font=small)
    draw.text((12, 72), 'Six scènes calculées • relief ×1,5 • même bruit • sans lissage ajouté', fill='white', font=small)
    for index, card in enumerate(cards):
        contact.paste(card, (12+(index%2)*612, 112+(index//2)*398))
    contact.save(output/'six-scenes-nf4.png')
    atomic_json(output/'gallery.json', dict(source_report=str(report_path), step=report['step'],
        checkpoint_sha256=report['checkpoint_sha256'], packed_weights_sha256=report['packed_weights_sha256'],
        weight_bytes=report['weight_bytes'], scenes=entries))
    print(json.dumps(dict(image=str(output/'six-scenes-nf4.png'), step=report['step'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    publish(args.report, args.output)


if __name__ == '__main__':
    main()
