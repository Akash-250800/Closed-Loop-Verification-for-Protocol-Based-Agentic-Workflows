"""Replace only the code-editor figures inside an existing
FINAL_DISSERTATION_FIGURES_SPYDER_STYLE folder.

This script does NOT generate all figures again.
It only replaces the code screenshot style figures with a darker editor-only
layout that looks closer to the user's Spyder screenshot.

Replaced files
- Figure_4_2.png
- Figure_4_3.png
- Figure_4_4.png
- Figure_4_5.png
- Figure_4_6.png
- Figure_4_7.png
- Figure_4_8.png
- Figure_4_11.png
"""
from __future__ import annotations

import hashlib
import json
import keyword
import re
import tokenize
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

PROJECT = Path(__file__).resolve().parent

# Optional manual override. Leave as None to auto-detect the newest folder.
TARGET_FOLDER_NAME = None

CAPTIONS = {
 '4_2': 'Inventory MCP Tool Contracts',
 '4_3': 'Model Proposal Normalization',
 '4_4': 'Parsing and Schema Checks',
 '4_5': 'Semantic and Safety Checks',
 '4_6': 'Feedback Rendering Formats',
 '4_7': 'Controlled Failure Injection',
 '4_8': 'Bounded Recovery and Termination',
 '4_11': 'Historical MCP Tool Interfaces',
}


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _font(size: int, bold: bool = False, mono: bool = False):
    if mono:
        family = 'DejaVu Sans Mono'
    else:
        family = 'DejaVu Sans'
    prop = FontProperties(family=family, weight='bold' if bold else 'normal')
    return ImageFont.truetype(findfont(prop), size=size)


def _text(draw, xy, txt, font, fill):
    draw.text(xy, txt, font=font, fill=fill)


def code_file(name, must_contain):
    candidates = [PROJECT / name, PROJECT / 'src' / name, PROJECT / 'mcp_thesis' / name]
    candidates += [p for p in PROJECT.glob('*/' + name) if p.is_file()]
    valid = []
    for p in dict.fromkeys(candidates):
        if p.is_file():
            text = p.read_text(encoding='utf-8-sig', errors='ignore')
            if must_contain in text:
                valid.append(p)
    require(valid, 'Cannot locate actual implementation: ' + name)
    valid.sort(key=lambda p: (len(p.parts), len(str(p))))
    return valid[0]


def locate(relative):
    direct = PROJECT / relative
    if direct.is_file():
        return direct
    matches = [p for p in PROJECT.glob('*/' + relative) if p.is_file()]
    require(len(matches) == 1, 'Required file not uniquely found: ' + relative)
    return matches[0]


def detect_target_folder() -> Path:
    if TARGET_FOLDER_NAME:
        folder = PROJECT / TARGET_FOLDER_NAME
        require(folder.is_dir(), 'Target folder not found: ' + str(folder))
        return folder
    candidates = [p for p in PROJECT.glob('FINAL_DISSERTATION_FIGURES_SPYDER_STYLE*') if p.is_dir()]
    require(candidates, 'No FINAL_DISSERTATION_FIGURES_SPYDER_STYLE folder found in project root')
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _color_tokens(line):
    base = '#d7e3f1'
    out = [base] * len(line)
    specials = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(line + '\n').readline):
            if tok.type == tokenize.ENDMARKER:
                continue
            col = tok.start[1]
            end = min(len(line), tok.end[1])
            if tok.type == tokenize.STRING:
                color = '#b8e986'
            elif tok.type == tokenize.COMMENT:
                color = '#8b95a7'
            elif tok.type == tokenize.NUMBER:
                color = '#f0a6ff'
            elif tok.string in keyword.kwlist:
                color = '#d98cff'
            elif tok.string in ('True', 'False', 'None'):
                color = '#ffb86c'
            elif tok.type == tokenize.OP:
                color = '#d7e3f1'
            else:
                continue
            if 0 <= col < end:
                specials.append((col, end, color))
    except (tokenize.TokenError, IndentationError):
        pass
    for start, stop, color in specials:
        out[start:stop] = [color] * (stop - start)
    return out


def _spyder_editor_canvas(script_name: str):
    W, H = 2380, 1450
    im = Image.new('RGB', (W, H), '#0f1b29')
    d = ImageDraw.Draw(im)

    # Windows title bar
    d.rectangle((0, 0, W, 38), fill='#f6f6f6')
    _text(d, (20, 7), 'Editor', _font(28), '#101010')
    _text(d, (W - 170, 5), '▁   ☐   ✕', _font(26), '#2f2f2f')

    # Path bar
    d.rectangle((0, 38, W, 86), fill='#132235')
    full_path = str(PROJECT / script_name).replace('/', '\\')
    _text(d, (15, 50), full_path, _font(22), '#f0f4f9')
    d.line((0, 86, W, 86), fill='#46607b', width=2)

    # Tab bar
    d.rectangle((0, 86, W, 156), fill='#112132')
    d.rounded_rectangle((66, 110, 530, 151), radius=7, fill='#3f5870')
    label = script_name
    if len(label) > 34:
        label = label[:15] + '...' + label[-16:]
    _text(d, (95, 119), label, _font(19), '#ecf2f8')
    _text(d, (498, 116), '✕', _font(18, bold=True), '#ecf2f8')
    _text(d, (31, 116), '◫', _font(18), '#ecf2f8')
    _text(d, (W - 40, 114), '≡', _font(24), '#ecf2f8')

    # Main editor region
    top = 156
    d.rectangle((0, top, W, H), fill='#122131')
    d.rectangle((54, top + 14, W - 34, H - 30), outline='#445668', width=1)
    d.rectangle((54, top + 14, 94, H - 30), fill='#152434')
    d.rectangle((94, top + 14, W - 34, H - 30), fill='#132231')

    # Current line highlight strip near top
    d.rectangle((108, top + 22, W - 70, top + 52), fill='#3f4b59')

    # Scroll bar
    d.rectangle((W - 28, top + 16, W - 12, H - 34), fill='#24374a')
    d.rounded_rectangle((W - 27, top + 36, W - 13, top + 98), radius=7, fill='#6d879f')
    d.rectangle((W - 28, top + 74, W - 12, top + 97), fill='#f28c28')

    # Warning markers margin, decorative only
    for y in (top + 400, top + 520, top + 760):
        d.polygon([(17, y + 8), (30, y + 32), (4, y + 32)], fill='#f5a623')
        _text(d, (13, y + 12), '!', _font(14, bold=True), '#263238')

    return im, d, (top + 18)


def render_editor_figure(out_path: Path, source_path: Path, title: str, lines_with_numbers):
    im, d, code_top = _spyder_editor_canvas(source_path.name)
    W, H = im.size
    mono = _font(24, mono=True)
    numfont = _font(24, mono=True)
    line_height = 40
    y = code_top
    for number, text in lines_with_numbers:
        if number is None:
            y += line_height
            continue
        # line number
        _text(d, (67 - max(0, len(str(number)) - 1) * 8, y), str(number), numfont, '#dbe6f2')
        # code segments with token color
        x = 108
        colors = _color_tokens(text)
        if not text:
            y += line_height
            continue
        group_start = 0
        for j in range(1, len(text) + 1):
            if j == len(text) or colors[j] != colors[group_start]:
                frag = text[group_start:j]
                _text(d, (x, y), frag, mono, colors[group_start])
                x += d.textlength(frag, font=mono)
                group_start = j
        y += line_height
    im.save(out_path, optimize=True)


def excerpt_from_anchor(path: Path, anchor: str, start_offset: int, lines_count: int):
    lines = path.read_text(encoding='utf-8-sig', errors='ignore').splitlines()
    matches = [i for i, ln in enumerate(lines) if anchor in ln]
    require(matches, f'Missing source anchor {anchor!r} in {path.name}')
    start = max(0, matches[0] + start_offset)
    rows = [(j + 1, lines[j]) for j in range(start, min(len(lines), start + lines_count))]
    return rows


def source_signatures(path: Path, names):
    lines = path.read_text(encoding='utf-8-sig', errors='ignore').splitlines()
    rows = []
    for idx, name in enumerate(names):
        matched = [j for j, l in enumerate(lines) if re.match(r'\s*def ' + re.escape(name) + r'\(', l)]
        require(len(matched) == 1, 'Missing actual signature ' + name)
        if idx:
            rows.append((None, ''))
        start = matched[0]
        for j in range(start, min(start + 3, len(lines))):
            rows.append((j + 1, lines[j]))
    return rows


def update_manifest(target_folder: Path, changed_tags):
    manifest_path = target_folder / 'FIGURE_MANIFEST.csv'
    if not manifest_path.is_file():
        return
    rows = []
    lines = manifest_path.read_text(encoding='utf-8-sig').splitlines()
    if not lines:
        return
    header = lines[0].split(',')
    import csv
    from io import StringIO
    reader = csv.DictReader(StringIO('\n'.join(lines)))
    for row in reader:
        m = re.search(r'Figure\s+(\d+)\.(\d+)', row.get('figure', ''))
        if not m:
            rows.append(row)
            continue
        tag = f'{m.group(1)}_{m.group(2)}'
        if tag in changed_tags:
            file_path = target_folder / row['file']
            row['type'] = 'Exact code excerpt, Spyder-like editor view'
            row['output_sha256'] = sha(file_path)
        rows.append(row)
    with manifest_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    target = detect_target_folder()

    p_server = code_file('server.py', 'def reserve_stock(')
    p_agents = code_file('agents.py', 'class OllamaAgent:')
    p_verify = code_file('verification.py', 'def parse_proposal(')
    p_feedback = code_file('feedback.py', 'def render_feedback(')
    p_exp = code_file('experiment.py', 'async def run_trial(')
    p_main = locate('02_run_redo_experiments_FINAL.py')

    jobs = {
        '4_2': ('Inventory MCP Tool Contract', p_server, excerpt_from_anchor(p_server, 'def reserve_stock(', -1, 17)),
        '4_3': ('Model Proposal Normalization', p_agents, excerpt_from_anchor(p_agents, '        data = response.json()', 0, 22)),
        '4_4': ('Parsing and Proposal Structure', p_verify, excerpt_from_anchor(p_verify, 'def parse_proposal(', 0, 16)),
        '4_5': ('Evidence Grounding Validation', p_main, excerpt_from_anchor(p_main, 'def semantic_preflight(', 0, 18)),
        '4_6': ('Three Feedback Representations', p_feedback, excerpt_from_anchor(p_feedback, 'def render_feedback(', 0, 20)),
        '4_7': ('Historical Fault Injection', p_main, excerpt_from_anchor(p_main, 'def inject_multiref_fault(', 0, 18)),
        '4_8': ('Bounded Runner and Retry Path', p_exp, excerpt_from_anchor(p_exp, 'async def run_trial(', 0, 20)),
        '4_11': ('Historical MCP Tool Definitions', p_main, source_signatures(p_main, ['resolve_country_by_capital', 'lookup_wikidata_president', 'lookup_hdata_president', 'submit_presidential_decision'])),
    }

    changed = []
    for tag, (title, path, rows) in jobs.items():
        out_path = target / f'Figure_{tag}.png'
        require(out_path.parent.is_dir(), 'Target folder missing: ' + str(target))
        render_editor_figure(out_path, path, title, rows)
        changed.append(tag)

    update_manifest(target, set(changed))

    print('CODE SCREENSHOT PATCH: PASS')
    print('Target folder:', target)
    print('Replaced files:')
    for tag in changed:
        print('  Figure_' + tag + '.png')
    print('No other figures were changed.')


if __name__ == '__main__':
    main()
