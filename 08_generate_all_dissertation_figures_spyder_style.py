"""Generate 27 dissertation figures in a NEW directory, with Spyder style code views.

Run this single file in Spyder (F5), from the MCP-Thesis project root.
Dependencies: matplotlib, pandas, Pillow (Pillow is installed with matplotlib).
All chart values come from the verified trial CSVs. All displayed code is read
from the actual project source. Phase D evidence uses real saved agent events.
This creates styled editor views, not literal screen captures of the Spyder app.
Does not modify old figures, original files, or experimental records.
"""
from __future__ import annotations

import csv
import keyword
import hashlib
import json
import math
import os
import re
import textwrap
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Spyder compatible, saves figures without showing 27 windows
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects
import pandas as pd

# Spyder users can optionally set PROJECT to an absolute project folder.
# The default works when this script is copied into the MCP-Thesis root.
PROJECT = Path(__file__).resolve().parent
def fresh_directory(root):
    base = 'FINAL_DISSERTATION_FIGURES_SPYDER_STYLE'
    for i in range(1,1000):
        name = base if i == 1 else f'{base}_{i:02d}'
        target = root / name
        if not target.exists():
            return target
    raise RuntimeError('No free figure output directory')

OUTPUT = fresh_directory(PROJECT)
DPI = 260
POLICIES = ['no_verification', 'naive_retry', 'closed_loop']
P_NAMES = {'no_verification': 'No verification', 'naive_retry': 'Naive retry', 'closed_loop': 'Closed loop'}
COLORS = {'no_verification': '#62758b', 'naive_retry': '#cc9551', 'closed_loop': '#168d91'}
NAVY = '#18314a'
WHITE = '#ffffff'
LIGHT = '#edf3f5'
GREY = '#556879'

REQUIRED = {
 'inventory': ('redo_v3_inventory_baseline/trials.csv', 480),
 'feedback': ('redo_v3_feedback/trials.csv', 72),
 'retry': ('redo_v3_retry/trials.csv', 176),
 'bfcl': ('redo_v3_bfcl/trials.csv', 60),
 'historical': ('redo_v4_multiref_corrected/trials.csv', 450),
 'two_agent': ('redo_v3_two_agent/trials.csv', 30),
}
CAPTIONS = {
 '3_1': 'Experimental Comparison Design',
 '3_2': 'Two Source Evidence Workflow',
 '4_1': 'GVEF Execution Architecture',
 '4_2': 'Inventory MCP Tool Contracts',
 '4_3': 'Model Proposal Normalization',
 '4_4': 'Parsing and Schema Checks',
 '4_5': 'Semantic and Safety Checks',
 '4_6': 'Feedback Rendering Formats',
 '4_7': 'Controlled Failure Injection',
 '4_8': 'Bounded Recovery and Termination',
 '4_9': 'Local Validation Evidence',
 '4_10': 'Dependent Historical Workflow',
 '4_11': 'Historical MCP Tool Interfaces',
 '4_12': 'Two Agent Source Selection',
 '4_13': 'Evidence Merge and Final Decision',
 '4_14': 'Agent Evidence and Proposal Trace',
 '5_1': 'Distribution of Experimental Trials',
 '5_2': 'Inventory Completion Rates',
 '5_3': 'Recovery Across Inventory Faults',
 '5_4': 'Controlled Inventory Recovery',
 '5_5': 'BFCL Completion Comparison',
 '5_6': 'Historical Fault Recovery',
 '5_7': 'Natural Versus Controlled Completion',
 '5_8': 'Two Agent Selection and Completion',
 '5_9': 'Natural Completion and Token Cost',
 '5_10': 'Feedback Format Comparison',
 '5_11': 'Completion Across Retry Budgets',
}
manifest = []
source_paths = set()
figures_made = set()


def require(cond, message):
    if not cond:
        raise RuntimeError('EVIDENCE CHECK FAILED: ' + message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def locate(relative):
    direct = PROJECT / relative
    if direct.is_file():
        source_paths.add(direct)
        return direct
    # Only use exact relative paths at root or inside one enclosing folder.
    # Never silently substitute an old Phase C directory for the corrected one.
    matches = [p for p in PROJECT.glob('*/' + relative) if p.is_file()]
    require(len(matches) == 1, 'Required file not uniquely found: ' + relative)
    source_paths.add(matches[0]); return matches[0]


def code_file(name, must_contain):
    candidates = [PROJECT / name, PROJECT / 'src' / name, PROJECT / 'mcp_thesis' / name]
    candidates += [p for p in PROJECT.glob('*/' + name) if p.is_file()]
    valid = []
    for p in dict.fromkeys(candidates):
        if p.is_file() and must_contain in p.read_text(encoding='utf-8-sig'):
            valid.append(p)
    require(len(valid) >= 1, 'Cannot locate actual implementation: ' + name)
    valid.sort(key=lambda p: len(p.parts))
    source_paths.add(valid[0]); return valid[0]


def read_csv_file(relative, expected):
    path = locate(relative)
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [str(c).lstrip('\ufeff') for c in df.columns]
    require(len(df) == expected, '%s contains %d rows; expected %d' % (relative,len(df),expected))
    need = {'task_completed','condition','fault','total_tokens','latency_seconds'}
    require(need.issubset(df.columns), 'Required columns absent from ' + relative)
    for col in ['task_completed','total_tokens','latency_seconds']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    require(df.task_completed.notna().all(), 'Missing completion values in '+relative)
    require(df.task_completed.isin([0, 1]).all(), 'Completion must be zero or one in '+relative)
    require(df.total_tokens.notna().all() and df.latency_seconds.notna().all(), 'Missing token or time values in '+relative)
    return df


def check_final_report(datasets):
    path = locate('redo_v4_final_verification/FINAL_REDO_CHECK.txt')
    txt = path.read_text(encoding='utf-8-sig')
    require('PASS' in txt.upper(), 'Consolidated verification report does not state PASS')
    require(sum(len(v) for v in datasets.values()) == 1268, 'Total is not 1268')
    checks = {
        'inventory': [7, 67, 148],
        'feedback': [68],
        'bfcl': [9, 9, 9],
        'historical': [24, 57, 145],
        'two_agent': [25],
    }
    for group in ('inventory','bfcl','historical'):
        actual = [int(datasets[group].loc[datasets[group].condition.eq(p), 'task_completed'].sum()) for p in POLICIES]
        require(actual == checks[group], '%s policy successes %r differ from verified %r' % (group,actual,checks[group]))
    require(int(datasets['feedback'].task_completed.sum()) == 68, 'Feedback successes differ from 68')
    require(int(datasets['two_agent'].task_completed.sum()) == 25, 'Two agent successes differ from 25')
    require(len(datasets['historical'].query("fault == 'none'")) == 90, 'Natural historical subset must contain 90 trials')
    require('controlled_seed' in datasets['historical'].columns, 'No controlled seed indicator in corrected Phase C')
    require(int(datasets['historical'].controlled_seed.sum()) == 360, 'Corrected Phase C controls must be 360')
    require('agents_selected_independent_sources' in datasets['two_agent'].columns, 'Missing two agent independence evidence')
    require(int(datasets['two_agent'].agents_selected_independent_sources.sum()) == 29, 'Two agent independent source total differs from 29')


def preflight():
    ds = {k: read_csv_file(rel,n) for k,(rel,n) in REQUIRED.items()}
    check_final_report(ds)
    return ds


def save(fig, tag, sources, detail):
    require(tag in CAPTIONS, 'Undocumented figure ' + tag)
    require(tag not in figures_made, 'Figure generated twice: '+tag)
    output = OUTPUT / ('Figure_' + tag + '.png')
    fig.savefig(output, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    require(output.stat().st_size > 20000, 'Blank or tiny figure: '+str(output))
    figures_made.add(tag)
    for p in sources: source_paths.add(Path(p))
    manifest.append({'figure': 'Figure '+tag.replace('_','.'), 'caption': CAPTIONS[tag], 'file': output.name,
                     'evidence': '; '.join(sorted(set(str(Path(p).relative_to(PROJECT)) if Path(p).is_relative_to(PROJECT) else str(p) for p in sources))),
                     'type': detail, 'output_sha256': sha(output)})


def styled_ax(fig, title, subtitle=''):
    ax=fig.add_subplot(111)
    ax.set_title(title, loc='left', fontsize=17, fontweight='bold', color=NAVY, pad=30)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=GREY, va='bottom')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.17)
    ax.set_axisbelow(True)
    return ax


def bar_policy(tag, data, title, subtitle, denominator=None):
    fig=plt.figure(figsize=(11.8,6.4))
    ax=styled_ax(fig,title,subtitle)
    counts=[int(data.loc[data.condition.eq(p),'task_completed'].sum()) for p in POLICIES]
    denoms=[int(data.condition.eq(p).sum()) for p in POLICIES]
    if denominator is not None: require(denoms == [denominator]*3, 'Wrong denominator in '+tag)
    rates=[100*a/b for a,b in zip(counts,denoms)]
    bars=ax.bar([P_NAMES[p] for p in POLICIES],rates,color=[COLORS[p] for p in POLICIES],width=.58)
    ax.set_ylabel('Completed trials (%)');ax.set_ylim(0, max(103,max(rates)+18))
    for b,n,d,r in zip(bars,counts,denoms,rates):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+2.1, f'{n}/{d}   {r:.1f}%',ha='center',fontsize=11,weight='bold',color=NAVY)
    fig.tight_layout();save(fig,tag,[locate(rel) for rel,_ in REQUIRED.values() if (tag=='5_5' and 'bfcl' in rel) or (tag=='5_2' and 'inventory_baseline' in rel) or (tag=='5_4' and 'inventory_baseline' in rel)],'Calculated policy completion')


def matrix_chart(tag, df, title, subtitle, faults):
    subset=df.loc[df.fault.isin(faults)]
    vals=[]; labels=[]; nums=[]
    for fault in faults:
        group=[]; text=[]
        for policy in POLICIES:
            part=subset[(subset.fault==fault)&(subset.condition==policy)]
            require(len(part)>0, 'Missing fault policy cell '+fault+' '+policy)
            n=int(part.task_completed.sum());d=len(part)
            group.append(n/d*100);text.append(f'{n}/{d}')
        vals.append(group);nums.append(text);labels.append(fault.replace('_',' ').title())
    fig=plt.figure(figsize=(11.7,7.1));ax=fig.add_subplot(111)
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('tealgray',['#e4ebf0','#b8dcd9','#118a91'])
    im=ax.imshow(vals,vmin=0,vmax=100,cmap=cmap,aspect='auto')
    ax.set_xticks(range(3),[P_NAMES[p] for p in POLICIES],fontsize=10)
    ax.set_yticks(range(len(faults)),labels,fontsize=10)
    for i, row in enumerate(nums):
        for j,t in enumerate(row):
            ax.text(j,i,t,ha='center',va='center',fontsize=11,weight='bold',color='white' if vals[i][j]>65 else NAVY)
    ax.set_title(title,loc='left',fontsize=17,weight='bold',color=NAVY,pad=28)
    fig.text(.12,.92,subtitle,fontsize=10,color=GREY)
    fig.colorbar(im,ax=ax,shrink=.85,label='Completion (%)');fig.tight_layout()
    save(fig,tag,[locate('redo_v3_inventory_baseline/trials.csv') if tag=='5_3' else locate('redo_v4_multiref_corrected/trials.csv')], 'Calculated fault by policy comparison')


def note_fig(tag,title,subtitle,nodes,edges,source_files):
    """Draw conceptual workflow with boxes. Labels describe code verified elsewhere."""
    fig,ax=plt.subplots(figsize=(12.8,7.0))
    ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis('off')
    ax.text(.3,6.72,title,size=18,weight='bold',color=NAVY,va='center')
    ax.text(.3,6.34,subtitle,size=10,color=GREY,va='center')
    for key,x,y,w,h,label,desc in nodes:
        box=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.09,rounding_size=.13',facecolor=LIGHT,edgecolor='#aac6cd',linewidth=1.5)
        ax.add_patch(box)
        ax.text(x+w/2,y+h*.63,label,ha='center',va='center',weight='bold',fontsize=11,color=NAVY)
        if desc: ax.text(x+w/2,y+h*.25,desc,ha='center',va='center',fontsize=9,color=GREY)
    nodemap={n[0]:n for n in nodes}
    for a,b,how in edges:
        A=nodemap[a];B=nodemap[b]
        axa=A[1]+A[3]/2;aya=A[2]+A[4]/2
        bxa=B[1]+B[3]/2;bya=B[2]+B[4]/2
        dx=bxa-axa;dy=bya-aya
        if abs(dx)>abs(dy):
            sx=A[1]+(A[3]+.12 if dx>0 else -.12);sy=aya
            tx=B[1]-(.12 if dx>0 else -B[3]-.12);ty=bya
        else:
            sx=axa;sy=A[2]+(A[4]+.12 if dy>0 else -.12)
            tx=bxa;ty=B[2]-(.12 if dy>0 else -B[4]-.12)
        arrow=FancyArrowPatch((sx,sy),(tx,ty),arrowstyle='-|>', mutation_scale=16,color='#436b7c',linewidth=1.8,connectionstyle='arc3,rad=0')
        ax.add_patch(arrow)
        if how:
            ax.text((sx+tx)/2,(sy+ty)/2+.15,how,ha='center',fontsize=8,color=GREY,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    save(fig,tag,source_files,'Code supported conceptual diagram')


def _font(size, bold=False, mono=False):
    """Choose installed fonts, with fallback to a bundled matplotlib font."""
    choices = []
    if os.name == 'nt':
        font_dir = Path(os.environ.get('WINDIR','C:/Windows')) / 'Fonts'
        if mono:
            choices = [font_dir/'consola.ttf', font_dir/'lucon.ttf']
        elif bold:
            choices = [font_dir/'segoeuib.ttf', font_dir/'arialbd.ttf']
        else:
            choices = [font_dir/'segoeui.ttf', font_dir/'arial.ttf']
    choices.append(Path(findfont(FontProperties(family='DejaVu Sans Mono' if mono else 'DejaVu Sans',weight='bold' if bold else 'normal'))))
    for path in choices:
        if path.exists():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default()


def _text(draw, xy, value, font, fill='#28394b'):
    draw.text(xy, str(value), font=font, fill=fill)


def _color_tokens(line):
    """Only syntax-color the ACTUAL source string. Never change its characters."""
    import io, tokenize
    specials = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(line+'\n').readline):
            if tok.type == tokenize.ENDMARKER: continue
            col = tok.start[1]
            end = min(len(line),tok.end[1])
            if tok.type == tokenize.STRING: color='#2b9259'
            elif tok.type == tokenize.COMMENT: color='#8993a5'
            elif tok.type == tokenize.NUMBER: color='#9a4cb5'
            elif tok.string in keyword.kwlist: color='#9652b7'
            elif tok.string in ('True','False','None'): color='#bb581b'
            else: continue
            if 0 <= col < end:
                specials.append((col,end,color))
    except (tokenize.TokenError,IndentationError):
        pass
    # Each character keeps exactly its original content; colors are decorative.
    out = ['#1d374d']*len(line)
    for start,stop,color in specials:
        out[start:stop] = [color]*(stop-start)
    return out


def _spyder_canvas(title, tab, mode='editor'):
    W,H=2380,1450
    im=Image.new('RGB',(W,H),'#eef1f5');d=ImageDraw.Draw(im)
    f_ui=_font(28); f_bold=_font(29,bold=True);f_small=_font(22)
    d.rectangle((0,0,W,70),fill='#e7e9ed')
    _text(d,(30,15),'Spyder',_font(36,bold=True),'#28475c')
    _text(d,(228,23),'Python development environment',f_ui,'#53616e')
    d.rectangle((0,70,W,129),fill='#f8f9fb')
    _text(d,(36,84),'File     Edit     Search     Source     Run     Debug     Consoles     Projects     Tools     View     Help',f_ui,'#283a4b')
    d.rectangle((0,129,W,198),fill='#f1f3f6')
    for idx,(symbol,label) in enumerate([('○','Open'),('▣','Save'),('▶','Run'),('■','Stop'),('⌕','Find')]):
        x=34+idx*205
        _text(d,(x,142),symbol,_font(29,bold=True),'#2b8a77' if label=='Run' else '#607083')
        _text(d,(x+42,146),label,_font(24),'#526578')
    d.rectangle((17,216,W-17,H-68),fill='#ffffff',outline='#cbd2dc',width=2)
    d.rectangle((17,216,W-17,283),fill='#e6e9f0')
    _text(d,(48,234),tab,f_bold,'#28445c')
    d.line((17,282,W-17,282),fill='#cbd2dc',width=2)
    d.rectangle((0,H-62,W,H),fill='#e9edf3')
    _text(d,(34,H-48),'Python 3     |     UTF-8     |     Source based research evidence',f_small,'#5b6a7a')
    return im,d


def spyder_code_view(tag,title,path,segments):
    """Spyder-style editor figure. segments = (real line number, exact text)."""
    allrows=[]
    for num,content in segments:
        if num is None:
            allrows.append((None,'')); continue
        # Wrap visually without truncation. Wrapped lines keep their true line number only once.
        maxlen=112
        parts=[content[i:i+maxlen] for i in range(0,len(content),maxlen)] or ['']
        for j,part in enumerate(parts):
            allrows.append((num if j==0 else '',part))
    require(len(allrows) <= 31,'Source view too dense for a single readable figure '+tag)
    im,d=_spyder_canvas(title,path.name)
    W,H=im.size
    d.rectangle((27,283,148,H-76),fill='#edf1f6')
    d.line((148,283,148,H-76),fill='#d5dce6',width=2)
    _text(d,(182,305),title,_font(28,bold=True),'#2e5570')
    _text(d,(182,354),f'{path.name}   •   exact source line numbers',_font(23),'#708394')
    y0=405
    line_height=min(40, max(32, 945//max(1,len(allrows))))
    mono=_font(26,mono=True)
    numfont=_font(25,mono=True)
    for i,(number,line) in enumerate(allrows):
        y=y0+i*line_height
        if number is None:
            d.line((183,y+16,W-88,y+16),fill='#dbe2e8',width=2)
            continue
        if (line.lstrip().startswith(('def ','class ','async def ','@'))):
            d.rectangle((150,y-4,W-55,y+line_height),fill='#f2f8fd')
        if isinstance(number,int):
            _text(d,(55,y),str(number).rjust(4),numfont,'#758899')
        # Colored token runs, without changing exact source text.
        colors=_color_tokens(line)
        x=181
        group_start=0
        for j in range(1,len(line)+1):
            if j==len(line) or (j<len(line) and colors[j]!=colors[group_start]):
                fragment=line[group_start:j]
                _text(d,(x,y),fragment,mono,colors[group_start])
                x+=d.textlength(fragment,font=mono)
                group_start=j
        require(x < W-35, 'Code line still exceeds figure width for '+tag)
    dst=OUTPUT/('Figure_'+tag+'.png')
    im.save(dst,optimize=True)
    save_external(tag,dst,[path],'Exact code excerpt, styled editor view')


def save_external(tag,output,sources,detail):
    require(tag in CAPTIONS and tag not in figures_made,'Repeated or undocumented figure '+tag)
    require(output.is_file() and output.stat().st_size>20000,'Blank output: '+str(output))
    figures_made.add(tag)
    for p in sources: source_paths.add(Path(p))
    manifest.append({'figure':'Figure '+tag.replace('_','.'),'caption':CAPTIONS[tag],
                     'file':output.name,
                     'evidence':'; '.join(sorted(set(str(Path(p).relative_to(PROJECT)) if Path(p).is_relative_to(PROJECT) else str(p) for p in sources))),
                     'type':detail,'output_sha256':sha(output)})


def src_excerpt(tag,title,path,anchor,start_offset=0,lines_count=18):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    matches=[i for i,ln in enumerate(lines) if anchor in ln]
    require(matches,'Missing source anchor '+repr(anchor)+' in '+path.name)
    a=max(0,matches[0]+start_offset)
    excerpt=[(j+1,lines[j]) for j in range(a,min(len(lines),a+lines_count))]
    spyder_code_view(tag,title,path,excerpt)


def source_signatures(tag,title,path,names):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    segments=[]
    for k,name in enumerate(names):
        matched=[j for j,l in enumerate(lines) if re.match(r'\s*def '+re.escape(name)+r'\(',l)]
        require(len(matched)==1,'Missing actual signature '+name)
        if k: segments.append((None,''))
        start=matched[0]
        for j in range(start,min(start+3,len(lines))):
            segments.append((j+1,lines[j]))
    spyder_code_view(tag,title,path,segments)


def verification_evidence():
    p=locate('redo_v4_final_verification/FINAL_REDO_CHECK.txt')
    original=p.read_text(encoding='utf-8-sig').splitlines()
    terms=('FINAL STATUS: PASS','- BFCL: 20 unique','- 30-country two-source','- GRAND TOTAL: 1268',
           '- Corrected Phase C evidence','- Completed two-agent','- Independent-source selections','- pytest: PASS')
    selections=[]
    for term in terms:
        matches=[(i+1,ln) for i,ln in enumerate(original) if ln.strip().startswith(term)]
        require(len(matches)==1,'Final evidence report missing unique line '+term)
        selections.append(matches[0])
    spyder_code_view('4_9','Verified Final Experiment Report',p,selections)


def trace_evidence():
    p=locate('redo_v3_two_agent/records.json')
    records=json.loads(p.read_text(encoding='utf-8-sig'))
    require(len(records)==30,'Two agent records must contain 30 trials')
    require(sum(int(r.get('task_completed',0)) for r in records)==25,'Wrong Phase D completion total')
    require(sum(int(r.get('agents_selected_independent_sources',0)) for r in records)==29,'Wrong source choice total')
    matches=[]
    for r in records:
        if not (r.get('task_completed') and r.get('agents_selected_independent_sources')): continue
        steps={}
        for e in r.get('events',[]):
            if e.get('result')!='success':continue
            k=e.get('step')
            if k not in (1,2,3,4) or k in steps: continue
            try: call=json.loads(e.get('proposal','{}'))
            except (TypeError,ValueError): continue
            out=e.get('output')
            if isinstance(out,dict):steps[k]=(e,call,out)
        if len(steps)!=4:continue
        if [steps[k][0].get('agent_role') for k in (1,2,3,4)]!=['A','A','B','A']:continue
        tools=[steps[k][1].get('tool') for k in (1,2,3,4)]
        if tools[0]!='resolve_country_by_capital' or tools[3]!='submit_presidential_decision':continue
        if set(tools[1:3])!={'lookup_wikidata_president','lookup_hdata_president'}:continue
        if any(not steps[k][2].get('president_name') for k in (2,3,4)):continue
        matches.append((r,steps))
    require(matches,'No genuine completed distinct-source Phase D trace')
    record,steps=sorted(matches,key=lambda pair:pair[0]['country_code'])[0]
    im,d=_spyder_canvas('Recorded two agent MCP trace','Variable explorer  |  two_agent_record')
    W,H=im.size
    _text(d,(60,314),'Recorded Agent Roles and MCP Results',_font(36,bold=True),'#2d536c')
    context='Country '+str(record.get('country_name'))+'   |   Code '+str(record.get('country_code'))+'   |   Target date '+str(record.get('target_date'))
    _text(d,(60,374),context,_font(27),'#546c80')
    xs=[65,230,450,1460]
    headers=['Step','Agent','Executed MCP tool','Returned evidence']
    d.rectangle((52,445,W-52,541),fill='#dce8f4',outline='#bdcddd',width=2)
    for x,h in zip(xs,headers):_text(d,(x,475),h,_font(29,bold=True),'#284760')
    for index,k in enumerate((1,2,3,4)):
        e,proposal,out=steps[k]
        t=proposal.get('tool','')
        evidence = ((str(out.get('country_code',''))+'  '+str(out.get('country_name',''))) if k==1
                     else str(out.get('president_name','')))
        y=541+index*172
        d.rectangle((52,y,W-52,y+172),fill='#ffffff' if index%2 else '#f5f8fc',outline='#d9e1e9',width=2)
        for x,v in zip(xs,[str(k),e.get('agent_role',''),t,evidence]):
            font=_font(29,bold=True) if x==230 else _font(27)
            # Long displayed evidence wraps, while retaining full content.
            avail={65:150,230:160,450:960,1460:790}[x]
            pieces=[];segment=''
            for word in str(v).split(' '):
                candidate=(segment+' '+word).strip()
                if segment and d.textlength(candidate,font=font)>avail:
                    pieces.append(segment);segment=word
                else: segment=candidate
            if segment: pieces.append(segment)
            require(len(pieces)<=3,'Recorded trace cell too tall')
            for j,piece in enumerate(pieces):_text(d,(x,y+51+j*39),piece,font,'#263f58')
    dst=OUTPUT/'Figure_4_14.png';im.save(dst,optimize=True)
    save_external('4_14',dst,[p],'Real Phase D two agent trace, styled evidence explorer')


def draw_plots(ds):
    inv,fb,retry,bfcl,hist,agent=(ds[k] for k in ('inventory','feedback','retry','bfcl','historical','two_agent'))
    # 5.1 six suite coverage
    items=[('Inventory',len(inv)),('Feedback',len(fb)),('Retry budgets',len(retry)),('Adapted BFCL',len(bfcl)),('Historical evidence',len(hist)),('Two agents',len(agent))]
    fig=plt.figure(figsize=(12,7));ax=styled_ax(fig,'Verified Research Trial Coverage','Six suites   total 1,268 distinct trial rows')
    y=list(range(6));bars=ax.barh(y,[n for _,n in items],color=['#6b8596','#9bb6c0','#b3cacb','#648a9c','#168d91','#cc9551'])
    ax.set_yticks(y,[name for name,_ in items]);ax.invert_yaxis();ax.set_xlim(0,560);ax.set_xlabel('Number of recorded trial rows')
    for bar,(name,n) in zip(bars,items): ax.text(n+8,bar.get_y()+bar.get_height()/2,str(n),va='center',color=NAVY,weight='bold')
    fig.tight_layout();save(fig,'5_1',[locate(rel) for rel,_ in REQUIRED.values()],'Counts from six trial CSV files')
    # 5.2 inventory
    bar_policy('5_2',inv,'Inventory Completion by Execution Policy','All inventory conditions   160 trials per policy',160)
    # 5.3 fault heatmap
    faults=sorted(x for x in inv.fault.dropna().unique() if x!='none')
    require(len(faults)==7,'Expected seven inventory fault categories, got '+repr(faults))
    matrix_chart('5_3',inv,'Completion Under Inventory Faults','Cell labels are completed trials divided by all assigned trials',faults)
    # 5.4 controlled inventory
    controlled=inv[inv.fault!='none'].copy()
    bar_policy('5_4',controlled,'Matched Controlled Inventory Recovery','Prepared valid calls with one injected fault   140 trials per policy',140)
    # 5.5 BFCL
    bar_policy('5_5',bfcl,'Adapted BFCL Completion','20 supported single call tasks per policy   not an official BFCL score',20)
    # 5.6 historical fault matrix
    hf=sorted(hist.fault.dropna().unique(),key=lambda x: 0 if x=='none' else 1)
    require(len(hf)==5,'Expected five Phase C fault conditions')
    matrix_chart('5_6',hist,'Corrected Historical Trial Outcomes','30 countries per policy and fault setting',hf)
    # 5.7 natural compared with prepared controls
    fig=plt.figure(figsize=(13.2,7));ax=styled_ax(fig,'Natural and Prepared Historical Trials','Natural groups have 30 cases each   controlled groups have 120 each')
    x=range(3);w=.32
    for i,policy in enumerate(POLICIES):
        vals=[];labs=[]
        for natural in (True,False):
            sub=hist[(hist.condition==policy)&((hist.fault=='none') if natural else (hist.fault!='none'))]
            require(len(sub)==(30 if natural else 120),'Invalid historical subset')
            nums=int(sub.task_completed.sum());vals.append(nums/len(sub)*100);labs.append(f'{nums}/{len(sub)}')
        for j,val in enumerate(vals):
            px=i+(j-.5)*w
            ax.bar(px,val,width=w,color=COLORS[policy] if j==0 else '#b8c7d0',edgecolor='white')
            ax.text(px,val+2,labs[j],ha='center',fontsize=10,color=NAVY)
    ax.set_xticks(list(x),[P_NAMES[p] for p in POLICIES]);ax.set_ylim(0,112);ax.set_ylabel('Completion (%)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],lw=8,color=COLORS['closed_loop'],label='Natural, policy color'),Line2D([0],[0],lw=8,color='#b8c7d0',label='Controlled, gray')],loc='upper left',frameon=False)
    fig.tight_layout();save(fig,'5_7',[locate('redo_v4_multiref_corrected/trials.csv')],'Natural and controlled groups separately calculated')
    # 5.8 agents
    independent=int(agent.agents_selected_independent_sources.sum()); complete=int(agent.task_completed.sum())
    require((independent,complete)==(29,25),'Unexpected two agent outcomes')
    fig=plt.figure(figsize=(10.7,6));ax=styled_ax(fig,'Two Agent Evidence Outcomes','Separate measures from 30 country trials')
    bars=ax.bar(['Distinct source choices','Completed final decisions'],[independent/30*100,complete/30*100],color=['#168d91','#cc9551'],width=.6)
    ax.set_ylabel('Trials meeting criterion (%)');ax.set_ylim(0,112)
    for bar,count in zip(bars,[independent,complete]):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{count}/30',ha='center',size=12,weight='bold')
    fig.tight_layout();save(fig,'5_8',[locate('redo_v3_two_agent/trials.csv')],'Independent source and completion counts')
    # 5.9 Genuine natural completion and tokens, separated for readability
    fig,axes=plt.subplots(2,2,figsize=(14.6,9.6),layout='constrained')
    fig.suptitle('Natural Completion and Model Token Cost',fontsize=19,weight='bold',color=NAVY)
    for j,(family,df,denom) in enumerate([('Inventory',inv,20),('Historical',hist,30)]):
        sub=df[df.fault.eq('none')]
        groups=[sub[sub.condition.eq(policy)] for policy in POLICIES]
        require(all(len(group)==denom for group in groups),'Incorrect natural trial denominator '+family)
        done=[int(group.task_completed.sum()) for group in groups]
        toks=[float(group.total_tokens.mean()) for group in groups]
        rates=[100*n/denom for n in done]
        labels=[P_NAMES[p] for p in POLICIES]
        col=[COLORS[p] for p in POLICIES]
        a=axes[0,j]
        bars=a.bar(range(3),rates,color=col,width=.58)
        a.set_title(f'{family} natural trials',weight='bold',color=NAVY,fontsize=13)
        a.set_ylim(0,111);a.set_ylabel('Completion (%)')
        a.set_xticks(range(3),labels);a.tick_params(axis='x',labelsize=9)
        a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
        for bar,n in zip(bars,done):a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2.3,f'{n}/{denom}',ha='center',fontsize=11,color=NAVY,weight='bold')
        b=axes[1,j]
        bars=b.bar(range(3),toks,color=col,width=.58)
        b.set_ylim(0,max(toks)*1.25);b.set_ylabel('Mean model tokens')
        b.set_xticks(range(3),labels);b.tick_params(axis='x',labelsize=9)
        b.spines[['top','right']].set_visible(False);b.grid(axis='y',alpha=.15);b.set_axisbelow(True)
        for bar,n in zip(bars,toks):b.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(toks)*.03,f'{n:,.1f}',ha='center',fontsize=10,color=NAVY,weight='bold')
    save(fig,'5_9',[locate('redo_v3_inventory_baseline/trials.csv'),locate('redo_v4_multiref_corrected/trials.csv')],'Natural completion and token cost, separate charts')
    # 5.10 feedback formats
    format_order=['raw_trace','schema_diff','natural_language']
    fig=plt.figure(figsize=(11.2,6.3));ax=styled_ax(fig,'Completion by Feedback Representation','24 assigned trials per message format')
    rates=[];counts=[]
    for f in format_order:
        z=fb[fb.feedback_format==f];require(len(z)==24,'Feedback format has missing trials: '+f)
        n=int(z.task_completed.sum());rates.append(n/24*100);counts.append(n)
    bars=ax.bar(['Raw trace','Schema difference','Natural language'],rates,color=['#168d91','#758fa4','#cc9551'],width=.6)
    ax.set_ylabel('Task completion (%)');ax.set_ylim(0,112)
    for b,n in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{n}/24',ha='center',size=12,weight='bold')
    fig.tight_layout();save(fig,'5_10',[locate('redo_v3_feedback/trials.csv')],'Feedback completion by actual format')
    # 5.11 budgets, no-verification only budget zero
    fig=plt.figure(figsize=(12.5,6.6));ax=styled_ax(fig,'Retry Budget and Completion','Each plotted budget has 16 assigned trials per policy')
    key='scheduled_retry_budget' if 'scheduled_retry_budget' in retry.columns else 'retry_budget'
    retry[key]=pd.to_numeric(retry[key],errors='coerce')
    for policy in POLICIES:
        part=retry[retry.condition==policy]
        grouping=part.groupby(key).task_completed.agg(['sum','count']).sort_index()
        exp=1 if policy=='no_verification' else 5
        require(len(grouping)==exp,'Unexpected retry budget groups for '+policy)
        xs=list(map(int,grouping.index));ys=(grouping['sum']/grouping['count']*100).tolist()
        ax.plot(xs,ys,marker='o',markersize=9,linewidth=2.4,label=P_NAMES[policy],color=COLORS[policy])
        for x,y,n,d in zip(xs,ys,grouping['sum'],grouping['count']):
            require(int(d)==16,'Budget cell not 16 trials')
            ax.annotate(f'{int(n)}/{int(d)}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks([0,1,2,3,4]);ax.set_xlabel('Maximum permitted corrective attempts');ax.set_ylabel('Task completion (%)');ax.set_ylim(-5,115)
    ax.legend(frameon=False,loc='lower right')
    fig.tight_layout();save(fig,'5_11',[locate('redo_v3_retry/trials.csv')],'Retry budget cell completion')


def workflows(ds):
    evidence=[locate('02_run_redo_experiments_FINAL.py'),locate('redo_v4_multiref_corrected/trials.csv')]
    allcsv=[locate(x) for x,_ in REQUIRED.values()]
    node=lambda k,x,y,w,h,t,s:(k,x,y,w,h,t,s)
    # Chapter 3 design
    note_fig('3_1','Research Experiment Design','Six verified suites with separate denominators',[
     node('policy',.6,4.15,2.3,1,'Three policies','Comparison basis'),
     node('inventory',3.9,5.0,3.2,.85,'Inventory baseline','480 trials'),
     node('feedback',3.9,3.85,3.2,.85,'Feedback formats','72 trials'),
     node('retry',3.9,2.7,3.2,.85,'Retry budget','176 trials'),
     node('bfcl',8.1,5,3.0,.85,'Adapted BFCL','60 policy runs'),
     node('hist',8.1,3.85,3.0,.85,'Historical evidence','450 trials'),
     node('agents',8.1,2.7,3.0,.85,'Two agent study','30 trials')],
     [('policy','inventory',''),('policy','feedback',''),('policy','retry','')],allcsv)
    # two source data flow
    note_fig('3_2','Two Independent Historical Sources','Country resolution supports two independent presidential lookups',[
      node('task',.5,4.1,2.5,1.2,'Capital and target date','Benchmark task'),
      node('resolve',4.1,4.1,3.1,1.2,'Resolve country','MCP country reference'),
      node('wiki',8.25,4.9,3.15,1,'Wikidata lookup','Presidential term'),
      node('hdata',8.25,2.8,3.15,1,'H DATA lookup','Presidential term'),
      node('decision',4.1,1.2,3.1,1.0,'Grounded decision','Both source names required')],
      [('task','resolve',''),('resolve','wiki','Code and date'),('resolve','hdata','Code and date'),('wiki','decision','Evidence'),('hdata','decision','Evidence')],evidence)
    # 4.1 GVEF architecture
    note_fig('4_1','GVEF Framework Architecture','Verification before execution and evaluation after execution',[
      node('generate',.35,4.1,2.25,1.1,'Generate','Model proposes call'),
      node('verify',3.4,4.1,2.35,1.1,'Verify','Schema and grounding'),
      node('mcp',6.55,4.1,2.25,1.1,'MCP execution','Discovered tools'),
      node('eval',9.45,4.1,2.2,1.1,'Evaluate','Task requirements'),
      node('feedback',3.4,1.55,2.35,1.05,'Feedback','Correct or stop')],
      [('generate','verify',''),('verify','mcp','Valid call'),('mcp','eval','Result'),('verify','feedback','Fault'),('eval','feedback','Failed step'),('feedback','generate','New proposal')],evidence)
    # 4.10 historical chain
    note_fig('4_10','Dependent Historical MCP Workflow','All decisions use the target date and previous returned evidence',[
       node('step1',.5,4.1,2.2,1.2,'Step 1','Resolve capital'),
       node('step2',3.35,4.1,2.2,1.2,'Step 2','Wikidata lookup'),
       node('step3',6.2,4.1,2.2,1.2,'Step 3','H DATA lookup'),
       node('step4',9.05,4.1,2.45,1.2,'Step 4','Submit decision'),
       node('verify',4.6,1.5,3.0,1.1,'Evidence checks','Code, date and names')],
       [('step1','step2','Code'),('step2','step3','Evidence'),('step3','step4','Both sources'),('step4','verify','Final validation')],evidence)
    # 4.12 two agent
    note_fig('4_12','Two Agent Source Orchestration','Both agents can see the same MCP tool catalog',[
       node('task',.45,4.2,2.1,1.1,'Country task','Capital and date'),
       node('A',3.1,4.2,2.5,1.1,'Agent A','Resolve and source 1'),
       node('B',6.4,4.2,2.5,1.1,'Agent B','Select other source'),
       node('merge',9.55,4.2,2.0,1.1,'Agent A','Merge decision'),
       node('checker',5.4,1.6,3.0,1.1,'Evaluator','Distinct sources and match')],
       [('task','A',''),('A','B','First source'),('B','merge','Second source'),('merge','checker','Verify')],evidence)
    # 4.13 merge concept
    note_fig('4_13','Grounded Evidence Merge','Submitted names checked against observed MCP responses',[
       node('wiki',.5,4.4,3.0,1.1,'Wikidata response','President and date'),
       node('hdata',.5,2.4,3.0,1.1,'H DATA response','President and date'),
       node('merge',4.5,3.45,2.75,1.15,'Merge proposal','Candidate final fields'),
       node('check',8.25,3.45,3.0,1.15,'Grounding validator','Check code, date and names')],
       [('wiki','merge','Evidence'),('hdata','merge','Evidence'),('merge','check','Proposed result')],evidence)


def code_and_logs():
    p_server=code_file('server.py','def reserve_stock(')
    p_agents=code_file('agents.py','class OllamaAgent:')
    p_verify=code_file('verification.py','def parse_proposal(')
    p_feedback=code_file('feedback.py','def render_feedback(')
    p_exp=code_file('experiment.py','async def run_trial(')
    p_main=locate('02_run_redo_experiments_FINAL.py')
    src_excerpt('4_2','Inventory MCP Tool Contract',p_server,'def reserve_stock(',start_offset=-1,lines_count=17)
    src_excerpt('4_3','Model Proposal Normalization',p_agents,'        data = response.json()',start_offset=0,lines_count=22)
    src_excerpt('4_4','Parsing and Proposal Structure',p_verify,'def parse_proposal(',start_offset=0,lines_count=16)
    src_excerpt('4_5','Evidence Grounding Validation',p_main,'def semantic_preflight(',start_offset=0,lines_count=18)
    src_excerpt('4_6','Three Feedback Representations',p_feedback,'def render_feedback(',start_offset=0,lines_count=20)
    src_excerpt('4_7','Historical Fault Injection',p_main,'def inject_multiref_fault(',start_offset=0,lines_count=18)
    src_excerpt('4_8','Bounded Runner and Retry Path',p_exp,'async def run_trial(',start_offset=0,lines_count=20)
    verification_evidence()
    source_signatures('4_11','Historical MCP Tool Definitions',p_main,['resolve_country_by_capital','lookup_wikidata_president','lookup_hdata_president','submit_presidential_decision'])
    trace_evidence()


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    data=preflight()  # Fail before writing new figures when data is incomplete.
    workflows(data)
    code_and_logs()
    draw_plots(data)
    require(figures_made == set(CAPTIONS),'Missing figures: '+str(sorted(set(CAPTIONS)-figures_made)))
    with (OUTPUT/'FIGURE_MANIFEST.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['figure','caption','file','evidence','type','output_sha256']);writer.writeheader();writer.writerows(sorted(manifest,key=lambda r:tuple(map(int,re.findall(r'\d+',r['figure'])))))
    with (OUTPUT/'SOURCE_HASHES.json').open('w',encoding='utf-8') as f:
        json.dump({str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p):sha(p) for p in sorted(source_paths)},f,indent=2)
    print('\nVERIFIED FIGURE GENERATION: PASS')
    print('All 27 figures generated from original code, final trial rows and verified logs.')
    print('Files saved in:', OUTPUT)
    print('Metadata:',OUTPUT/'FIGURE_MANIFEST.csv')
    print('No trials were rerun. No experimental values were invented.')
    print('Source views are Spyder styled recreations using actual project code, not screen captures.')


if __name__=='__main__':
    main()
