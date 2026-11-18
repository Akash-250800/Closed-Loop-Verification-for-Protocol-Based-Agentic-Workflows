"""Fix ONLY Figure_4_9.png in the existing Spyder-style dissertation folder.

This does not run experiments or regenerate any other figures.
The image reproduces a dark Spyder-like editor view, without artificial
Spyder menus, toolbars, reports, or added explanatory footers.
The displayed text and line numbers are taken verbatim from the actual
redo_v4_final_verification/FINAL_REDO_CHECK.txt source file.

Run from Spyder (F5) or `python 10_fix_figure_4_9_editor_view.py`.
Requires Pillow.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parent
# Optional: set a specific existing folder name here. None uses the newest
# FINAL_DISSERTATION_FIGURES_SPYDER_STYLE folder in the project root.
TARGET_FOLDER_NAME = None
REPORT_RELATIVE = 'redo_v4_final_verification/FINAL_REDO_CHECK.txt'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def locate_source():
    path = PROJECT / REPORT_RELATIVE
    if path.is_file():
        return path
    matches = [p for p in PROJECT.glob('*/' + REPORT_RELATIVE) if p.is_file()]
    require(len(matches) == 1, 'Expected exactly one final report: ' + REPORT_RELATIVE)
    return matches[0]


def locate_output_folder():
    if TARGET_FOLDER_NAME:
        folder = PROJECT / TARGET_FOLDER_NAME
        require(folder.is_dir(), 'Folder not found: ' + str(folder))
        return folder
    candidates = [p for p in PROJECT.glob('FINAL_DISSERTATION_FIGURES_SPYDER_STYLE*')
                  if p.is_dir() and (p / 'Figure_4_9.png').is_file()]
    require(candidates, 'No existing Spyder-style figure folder containing Figure_4_9.png')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def font(size, monospace=False, bold=False):
    # Use installed Windows fonts when present, and portable fallback elsewhere.
    if os.name == 'nt':
        if monospace:
            names = ['C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/lucon.ttf']
        elif bold:
            names = ['C:/Windows/Fonts/segoeuib.ttf', 'C:/Windows/Fonts/arialbd.ttf']
        else:
            names = ['C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf']
    else:
        names = []
    if monospace:
        names += ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']
    elif bold:
        names += ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
    else:
        names += ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for n in names:
        if Path(n).is_file():
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()


def file_hash(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def render(report):
    lines = report.read_text(encoding='utf-8-sig').splitlines()
    assert len(lines) >= 34
    # Display a true contiguous section, NOT selected lines masquerading as a screenshot.
    checked = {
        4: 'FINAL STATUS: PASS',
        18: '- BFCL: 20 unique tasks, 60 policy-runs',
        19: '- 30-country two-source policy matrix: 450',
        21: '- GRAND TOTAL: 1268',
        25: '- Corrected Phase C evidence comes from redo_v4_multiref_corrected only.',
        29: '- Phase C no verification complete: 24/150',
        30: '- Phase C naive retry complete: 57/150',
        31: '- Phase C closed loop complete: 145/150',
    }
    for line_no, expected in checked.items():
        require(lines[line_no - 1] == expected,
                'The final verification report is different at line ' + str(line_no))

    W, H = 1920, 1080
    img = Image.new('RGB', (W, H), '#19232e')
    draw = ImageDraw.Draw(img)

    # Minimal Spyder editor chrome based on the user's real screenshot.
    draw.rectangle((0, 0, W, 36), fill='#f6f6f6')
    draw.text((30, 5), 'Editor', font=font(23), fill='#101010')
    draw.text((W - 132, 4), '—   □   ×', font=font(23), fill='#222222')

    draw.rectangle((0, 36, W, 83), fill='#17232e')
    report_display_path = str(report).replace('/', '\\')
    path_font = font(18)
    max_path_width = W - 28
    if draw.textlength(report_display_path, font=path_font) > max_path_width:
        report_display_path = '…\\' + '\\'.join(report_display_path.split('\\')[-3:])
    draw.text((17, 47), report_display_path, font=path_font, fill='#e7edf4')
    draw.line((5, 83, W - 5, 83), fill='#546476', width=1)

    draw.rectangle((0, 84, W, 145), fill='#152230')
    draw.rounded_rectangle((72, 102, 397, 141), radius=6, fill='#445e75')
    draw.rectangle((72, 139, 397, 143), fill='#49a4d0')
    draw.text((88, 111), 'FINAL_REDO_CHECK.txt', font=font(20), fill='#eef3f9')
    draw.text((364, 111), '×', font=font(22), fill='#eef3f9')

    draw.rounded_rectangle((6, 148, W - 7, H - 9), radius=5,
                           fill='#19232e', outline='#3b4b5d', width=2)
    draw.rectangle((8, 150, 99, H - 12), fill='#182735')
    draw.line((99, 150, 99, H - 12), fill='#34485c', width=1)
    draw.rectangle((W - 26, 150, W - 7, H - 12), fill='#26384c')
    draw.rounded_rectangle((W - 23, 181, W - 10, 237), radius=6, fill='#698096')

    mono = font(21, monospace=True)
    nums = font(20, monospace=True)
    row_y, step = 166, 26
    for n, content in enumerate(lines[:34], start=1):
        y = row_y + (n - 1) * step
        # Highlight the verified status and grand total lines without altering source text.
        if n in (4, 21, 31):
            draw.rectangle((101, y - 2, W - 32, y + step - 1), fill='#283844')
        draw.text((34, y), str(n).rjust(3), font=nums,
                  fill='#b0bed0' if n in (4, 21, 31) else '#8699ad')
        # For .txt source, a plain text editor display is appropriate.
        ink = '#dce8f3'
        if n in (4, 21):
            ink = '#c9f391'
        elif n in (29, 30, 31):
            ink = '#e8f1ff'
        draw.text((118, y), content, font=mono, fill=ink)
        require(draw.textbbox((118, y), content, font=mono)[2] < W - 40,
                'One report line exceeds editor width: line ' + str(n))
    return img


def patch_manifest(folder, new_hash):
    path = folder / 'FIGURE_MANIFEST.csv'
    if not path.is_file():
        return
    with path.open(newline='', encoding='utf-8-sig') as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames
        rows = list(rdr)
    require(fields is not None, 'Manifest lacks header')
    hits = 0
    for row in rows:
        if row.get('file') == 'Figure_4_9.png':
            row['output_sha256'] = new_hash
            row['type'] = 'Actual verification report text in dark editor styled rendering'
            hits += 1
    require(hits == 1, 'Manifest does not contain exactly one Figure_4_9.png entry')
    # Manifest is staged on the same drive, preventing Windows WinError 17.
    handle, staged = tempfile.mkstemp(prefix='.figure_4_9_manifest_', suffix='.csv', dir=folder)
    os.close(handle)
    try:
        with open(staged, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def main():
    report = locate_source()
    folder = locate_output_folder()
    target = folder / 'Figure_4_9.png'
    # Hash all other figures before patch.
    before = {p.name: file_hash(p) for p in folder.glob('Figure_*.png') if p.name != target.name}
    require(len(before) == 26, 'Expected 26 other dissertation figures in ' + str(folder))
    new_image = render(report)
    handle, staged = tempfile.mkstemp(prefix='.new_figure_4_9_', suffix='.png', dir=folder)
    os.close(handle)
    try:
        new_image.save(staged, format='PNG', optimize=True)
        require(Path(staged).stat().st_size > 20000, 'Generated image too small')
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    patch_manifest(folder, file_hash(target))
    after = {p.name: file_hash(p) for p in folder.glob('Figure_*.png') if p.name != target.name}
    require(before == after, 'A different figure was unexpectedly modified')
    print('FIGURE 4.9 CORRECTION: PASS')
    print('Updated:', target)
    print('Other 26 PNG figures: unchanged')
    print('Source:', report)
    print('Figure manifest: updated')


if __name__ == '__main__':
    main()
