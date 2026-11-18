"""Correct ONLY Figures 4.9, 4.14 and 5.9 in the existing figure collection.

Run from Spyder (F5) or with:
    python 07_correct_three_dissertation_figures.py

Copy this file into the MCP-Thesis project root, beside the redo_v3_* and
redo_v4_* folders. Do not change the existing experimental records.
Requires matplotlib and pandas. It does not rerun any experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# Spyder note: set PROJECT to your project root here ONLY if you keep the
# script outside the MCP-Thesis root. Otherwise the automatic path is correct.
PROJECT = Path(__file__).resolve().parent
FIGURES = PROJECT / 'FINAL_DISSERTATION_FIGURES'
DPI = 260
NAVY = '#18314a'
GREY = '#526579'
PALETTE = {'no_verification':'#657d95', 'naive_retry':'#cf9650', 'closed_loop':'#168d91'}
POLICIES = ('no_verification', 'naive_retry', 'closed_loop')
NAMES = {'no_verification':'No verification', 'naive_retry':'Naive retry', 'closed_loop':'Closed loop'}
CORRECTED = ('Figure_4_9.png', 'Figure_4_14.png', 'Figure_5_9.png')
SOURCES = {
    'Figure_4_9.png': ['redo_v4_final_verification/FINAL_REDO_CHECK.txt'],
    'Figure_4_14.png': ['redo_v3_two_agent/records.json'],
    'Figure_5_9.png': ['redo_v3_inventory_baseline/trials.csv', 'redo_v4_multiref_corrected/trials.csv'],
}
CAPTIONS = {
    'Figure_4_9.png': 'Local Validation Evidence',
    'Figure_4_14.png': 'Agent Evidence and Proposal Trace',
    'Figure_5_9.png': 'Natural Completion and Token Cost',
}


def check(condition, explanation):
    if not condition:
        raise RuntimeError('SOURCE VERIFICATION FAILED. ' + explanation)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_file(relative):
    p = PROJECT / relative
    check(p.is_file(), 'Missing file ' + str(p))
    return p


def check_folder():
    check(FIGURES.is_dir(), 'Expected existing FINAL_DISSERTATION_FIGURES folder. Extract your completed figures ZIP first.')
    names = {p.name for p in FIGURES.glob('Figure_*.png')}
    check(len(names) == 27, 'Expected 27 original figures. Found ' + str(len(names)))
    check(set(CORRECTED).issubset(names), 'One or more target PNGs are missing')
    manifest = FIGURES / 'FIGURE_MANIFEST.csv'
    check(manifest.is_file(), 'Existing FIGURE_MANIFEST.csv was not found')
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames
    check(columns and {'figure','caption','file','evidence','type','output_sha256'}.issubset(columns), 'Original manifest columns do not match')
    check(len(rows) == 27 and len({r['file'] for r in rows}) == 27, 'Manifest must list exactly 27 unique figures')
    check(all(sum(r['file'] == name for r in rows) == 1 for name in CORRECTED), 'Manifest target entries are missing or duplicated')
    # Do not touch other figures, and catch an incomplete or altered collection.
    for r in rows:
        if r['file'] not in CORRECTED:
            p = FIGURES / r['file']
            check(p.is_file(), 'Manifest figure is missing ' + r['file'])
            check(file_hash(p) == r['output_sha256'], 'Other figure differs from its original manifest ' + r['file'])
    return manifest, rows, columns


def report_lines():
    path = get_file(SOURCES['Figure_4_9.png'][0])
    original = path.read_text(encoding='utf-8-sig').splitlines()
    check(any(s.strip() == 'FINAL STATUS: PASS' for s in original), 'Final report does not declare final PASS')
    # Match entire original report lines without modifying reported values.
    prefixes = [
        'FINAL STATUS: PASS',
        '- BFCL: 20 unique tasks, 60 policy-runs',
        '- 30-country two-source policy matrix: 450',
        '- GRAND TOTAL: 1268',
        '- Corrected Phase C evidence comes from',
        '- Completed two-agent trials:',
        '- Independent-source selections:',
        '- pytest: PASS',
    ]
    selected=[]
    for prefix in prefixes:
        matches=[s.strip() for s in original if s.strip().startswith(prefix)]
        check(len(matches) == 1, 'Expected one report line starting with ' + prefix)
        selected.append(matches[0])
    return path, selected


def two_agent_example():
    path = get_file(SOURCES['Figure_4_14.png'][0])
    records = json.loads(path.read_text(encoding='utf-8-sig'))
    check(isinstance(records, list) and len(records) == 30, 'Two agent evidence must contain 30 country records')
    completed = sum(int(r['task_completed']) for r in records)
    distinct = sum(int(r['agents_selected_independent_sources']) for r in records)
    check((completed, distinct) == (25, 29), 'Two agent counts differ from final verified results')
    examples=[]
    for r in records:
        if not (r.get('task_completed') == 1 and r.get('agents_selected_independent_sources') == 1):
            continue
        steps={}
        for e in r.get('events', []):
            if e.get('result') != 'success':
                continue
            step=e.get('step')
            if step not in (1,2,3,4) or step in steps:
                continue
            try:
                proposal=json.loads(e.get('proposal',''))
            except (ValueError,TypeError):
                continue
            out=e.get('output')
            if not isinstance(out,dict):
                continue
            steps[step]=(e,proposal,out)
        if len(steps) != 4:
            continue
        roles=[steps[k][0].get('agent_role') for k in (1,2,3,4)]
        tools=[steps[k][1].get('tool') for k in (1,2,3,4)]
        if roles != ['A','A','B','A']:
            continue
        if tools[0] != 'resolve_country_by_capital' or tools[3] != 'submit_presidential_decision':
            continue
        if set(tools[1:3]) != {'lookup_wikidata_president','lookup_hdata_president'}:
            continue
        if not (steps[2][2].get('president_name') and steps[3][2].get('president_name')):
            continue
        if steps[1][2].get('country_code') != r.get('country_code'):
            continue
        if steps[4][2].get('president_name') is None:
            continue
        examples.append((r,steps))
    check(examples, 'Could not find a genuine, successful, independent two agent trace')
    return path, sorted(examples, key=lambda v:v[0]['country_code'])[0]


def natural_data():
    sources=[get_file(s) for s in SOURCES['Figure_5_9.png']]
    results={}
    for family,path,n_expected in [('inventory',sources[0],20),('historical',sources[1],30)]:
        df=pd.read_csv(path, encoding='utf-8-sig')
        check({'condition','fault','task_completed','total_tokens'}.issubset(df.columns), 'Missing required columns ' + str(path))
        sub=df[df['fault'].eq('none')]
        vals={}
        for policy in POLICIES:
            group=sub[sub['condition'].eq(policy)]
            check(len(group) == n_expected, family + ' natural denominator unexpected for ' + policy)
            successes=pd.to_numeric(group['task_completed'],errors='coerce')
            tokens=pd.to_numeric(group['total_tokens'],errors='coerce')
            check(successes.notna().all() and successes.isin([0,1]).all(), family + ' invalid completion field')
            check(tokens.notna().all() and (tokens>=0).all(), family + ' invalid tokens')
            vals[policy]=(int(successes.sum()), len(group), float(tokens.mean()))
        results[family]=vals
    actual_inv=[results['inventory'][p][0] for p in POLICIES]
    actual_hist=[results['historical'][p][0] for p in POLICIES]
    check(actual_inv == [7,7,8] and actual_hist == [24,27,27],
          'Natural success counts differ from confirmed results ' + repr((actual_inv,actual_hist)))
    return sources, results


def rounded_box(ax, x, y, width, height, fill='#f0f5f7'):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.008,rounding_size=0.018',
                                facecolor=fill,edgecolor='#d1e0e5',linewidth=1.1,transform=ax.transAxes))


def render_4_9(lines, destination):
    fig,ax=plt.subplots(figsize=(14.5,8.4),facecolor='white')
    ax.axis('off')
    ax.text(.045,.94,'Final Consolidated Verification',transform=ax.transAxes,fontsize=20,weight='bold',color=NAVY)
    ax.text(.045,.885,'Selected exact lines from FINAL_REDO_CHECK.txt',transform=ax.transAxes,fontsize=11,color=GREY)
    left=lines[:4]; right=lines[4:]
    for j,group in enumerate((left,right)):
        x=.045 if j==0 else .515
        for i,line in enumerate(group):
            y=.725-i*.159
            rounded_box(ax,x,y,.438,.137,fill='#edf5f5' if i==0 and j==0 else '#f2f6f8')
            # Text is unchanged except for display wrapping at whitespace.
            chunks=textwrap.wrap(line,width=54 if j==0 else 48,break_long_words=False,break_on_hyphens=False)
            check(len(chunks)<=3, 'Report line exceeds available figure panel')
            ax.text(x+.022,y+.067,'\n'.join(chunks),transform=ax.transAxes,ha='left',va='center',
                    family='DejaVu Sans Mono',fontsize=10.1 if len(chunks)==1 else 9.4,
                    color=NAVY,linespacing=1.55,weight='bold' if i==0 and j==0 else 'normal')
    ax.text(.045,.064,'Selected report evidence, not a simulated Spyder or terminal screenshot.',
            transform=ax.transAxes,fontsize=10,color=GREY)
    fig.savefig(destination,dpi=DPI,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def render_4_14(record,steps,destination):
    fig,ax=plt.subplots(figsize=(14.8,8.2),facecolor='white')
    ax.axis('off')
    ax.text(.04,.945,'Recorded Two Agent MCP Trace',transform=ax.transAxes,fontsize=19,weight='bold',color=NAVY)
    ax.text(.04,.887,f"{record['country_name']}    {record['country_code']}    {record['target_date']}    {record['task_id']}",
            transform=ax.transAxes,fontsize=11,color=GREY)
    headings=('Step','Agent','Actual MCP tool','Returned MCP evidence')
    xs=(.06,.14,.26,.65)
    widths=(.07,.11,.37,.30)
    rounded_box(ax,.04,.749,.92,.075,fill='#dcebef')
    for x,h in zip(xs,headings):
        ax.text(x,.786,h,transform=ax.transAxes,fontsize=10.4,color=NAVY,weight='bold',va='center')
    for idx,k in enumerate((1,2,3,4)):
        e,proposal,out=steps[k]
        tool=proposal['tool']; role=e['agent_role']
        if k==1:
            ev=out.get('country_code','')+'  '+out.get('country_name','')
        elif k in (2,3):
            ev=out['president_name']
        else:
            ev=out.get('president_name','')
        y=.623-idx*.142
        rounded_box(ax,.04,y,.92,.122,fill='white' if idx%2 else '#f2f6f8')
        for x,value in zip(xs,(str(k),role,tool,ev)):
            text=ax.text(x,y+.060,value,transform=ax.transAxes,va='center',fontsize=9.8,
                         color=NAVY,weight='bold' if x==.14 else 'normal')
    ax.text(.048,.065,'Agent A resolves, selects first source, and submits. Agent B selects the other source.',
            transform=ax.transAxes,fontsize=10,color=GREY)
    # No made up evidence. Every displayed code, name, role and tool comes from one record.
    fig.savefig(destination,dpi=DPI,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def render_5_9(results,destination):
    fig,axes=plt.subplots(2,2,figsize=(14.3,9.2),facecolor='white',layout='constrained')
    fig.suptitle('Natural Completion and Model Token Cost',fontsize=20,color=NAVY,weight='bold')
    labels=[NAMES[p] for p in POLICIES]
    colors=[PALETTE[p] for p in POLICIES]
    for column,(family,title) in enumerate((('inventory','Inventory natural trials'),('historical','Historical natural trials'))):
        group=results[family]
        counts=[group[p][0] for p in POLICIES]
        denoms=[group[p][1] for p in POLICIES]
        toks=[group[p][2] for p in POLICIES]
        rates=[100*n/d for n,d in zip(counts,denoms)]
        a=axes[0,column]
        bars=a.bar(range(3),rates,color=colors,width=.55)
        a.set_title(title,fontsize=13.5,color=NAVY,pad=13,weight='bold')
        a.set_xticks(range(3),labels,fontsize=9.5)
        a.set_ylabel('Completed trials (%)' if column==0 else 'Completion (%)')
        a.set_ylim(0,110)
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='y',alpha=.17);a.set_axisbelow(True)
        for bar,n,d in zip(bars,counts,denoms):
            a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2.0,f'{n}/{d}',ha='center',fontsize=11.2,color=NAVY,weight='bold')
        b=axes[1,column]
        bars=b.bar(range(3),toks,color=colors,width=.55)
        b.set_xticks(range(3),labels,fontsize=9.5)
        b.set_ylabel('Mean tokens per attempted trial' if column==0 else 'Mean model tokens')
        b.set_ylim(0,max(toks)*1.24)
        b.spines[['top','right']].set_visible(False)
        b.grid(axis='y',alpha=.17);b.set_axisbelow(True)
        for bar,mean in zip(bars,toks):
            b.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(toks)*.025,f'{mean:,.1f}',ha='center',fontsize=10.5,color=NAVY,weight='bold')
    fig.savefig(destination,dpi=DPI,facecolor='white',bbox_inches='tight')
    plt.close(fig)


def atomic_write_csv(path,rows,columns):
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)


def main():
    manifest_path,manifest_rows,columns=check_folder()
    rpt,lines=report_lines()
    agent_file,(record,steps)=two_agent_example()
    chart_files,natural=natural_data()
    # Source checks finish before any original image is replaced.
    # Keep staging on the destination drive so os.replace works on Windows.
    with tempfile.TemporaryDirectory(prefix='correct_figures_', dir=FIGURES) as temp:
        stage=Path(temp)
        render_4_9(lines, stage/CORRECTED[0])
        render_4_14(record,steps,stage/CORRECTED[1])
        render_5_9(natural,stage/CORRECTED[2])
        for name in CORRECTED:
            check((stage/name).is_file() and (stage/name).stat().st_size>25_000,
                  'Generated output is blank or unexpectedly small '+name)
        # All 3 new images exist before any original images are replaced.
        for name in CORRECTED:
            os.replace(stage/name,FIGURES/name)
    for row in manifest_rows:
        name=row['file']
        if name in CORRECTED:
            row['caption']=CAPTIONS[name]
            row['evidence']='; '.join(SOURCES[name])
            row['type']={CORRECTED[0]:'Selected exact final verification report lines',
                         CORRECTED[1]:'Observed Phase D two agent event trace',
                         CORRECTED[2]:'Calculated genuine natural trial completion and token usage'}[name]
            row['output_sha256']=file_hash(FIGURES/name)
    atomic_write_csv(manifest_path,manifest_rows,columns)
    hashes_path=FIGURES/'SOURCE_HASHES.json'
    if hashes_path.is_file():
        hashes=json.loads(hashes_path.read_text(encoding='utf-8'))
        for name in CORRECTED:
            for rel in SOURCES[name]:hashes[rel]=file_hash(PROJECT/rel)
        temp=hashes_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(dict(sorted(hashes.items())),indent=2)+'\n',encoding='utf-8')
        os.replace(temp,hashes_path)
    print('\nTHREE FIGURE CORRECTION: PASS')
    print('Only Figure_4_9.png, Figure_4_14.png and Figure_5_9.png were regenerated.')
    print('Figure 4.14 example: ',record['task_id'],record['country_name'])
    print('Actual two agent totals: 25/30 completed; 29/30 independent source choices')
    print('Other 24 figure files were not changed.')
    print('Figure folder:',FIGURES)
    print('Manifest updated:',manifest_path)
    print('No experiment was rerun or altered.')


if __name__ == '__main__':
    main()
