"""Generate all 27 dissertation figures from actual project evidence.

Run in Spyder from the existing MCP-Thesis project, or use
    python 06_generate_dissertation_figures.py

Requires matplotlib and pandas. Generates figures only; it never runs experiments,
modifies trial files, or simulates results. Stops if source evidence is incomplete.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import textwrap
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
OUTPUT = PROJECT / 'FINAL_DISSERTATION_FIGURES'
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
    ax.text(.3,.22,'System workflow reconstructed from verified implementation. Not a Spyder screen capture.',fontsize=9,color=GREY)
    save(fig,tag,source_files,'Code supported conceptual diagram')


def src_excerpt(tag,title,path,anchor,start_offset=0,lines_count=18):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    matches=[i for i,l in enumerate(lines) if anchor in l]
    require(matches, f'Anchor {anchor!r} absent in {path.name}')
    a=max(0,matches[0]+start_offset)
    excerpt=lines[a:a+lines_count]
    rendered=[]
    for number,line in enumerate(excerpt,a+1):
        # Wrap for legibility but preserve every character of the source line.
        width=93
        if not line: pieces=['']
        else: pieces=[line[x:x+width] for x in range(0,len(line),width)]
        for j,piece in enumerate(pieces):
            rendered.append('%4s %s' % (number if j==0 else ' ',piece))
    require(len(rendered)<36,'Source excerpt is too dense for an academic figure: '+tag)
    height=max(6.3,.34*len(rendered)+2.5)
    fig,ax=plt.subplots(figsize=(14.1,height))
    fig.patch.set_facecolor('white');ax.axis('off')
    ax.text(.025,.972,title,transform=ax.transAxes,va='top',color=NAVY,weight='bold',size=17)
    ax.text(.025,.926,f'Exact source excerpt   {path.name}   lines {a+1} to {a+len(excerpt)}',transform=ax.transAxes,va='top',size=10,color=GREY)
    bg=FancyBboxPatch((.018,.095),.964,.788,transform=ax.transAxes,boxstyle='round,pad=.007',facecolor='#f2f6f9',edgecolor='#cbdbe2')
    ax.add_patch(bg)
    font_size=min(10.1, max(8.0, 11.7- max(0,len(rendered)-20)*.13))
    ax.text(.039,.863,'\n'.join(rendered),transform=ax.transAxes,va='top',family='DejaVu Sans Mono',fontsize=font_size,color='#142e42',linespacing=1.43)
    ax.text(.024,.048,'Source preserved as written. This is a generated code excerpt, not a Spyder screenshot.',transform=ax.transAxes,fontsize=9,color=GREY)
    save(fig,tag,[path],'Exact line numbered source excerpt')



def source_signatures(tag,title,path,names):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    shown=[]
    for name in names:
        matched=[i for i,l in enumerate(lines) if re.match(r'\s*def '+re.escape(name)+r'\(',l)]
        require(len(matched)==1,'Expected one real MCP tool signature '+name)
        start=matched[0]
        # Preserve line numbers from the original source for each exact excerpt.
        for j in range(start,min(start+3,len(lines))):
            shown.append('%4d %s' % (j+1,lines[j]))
        shown.append('')
    fig,ax=plt.subplots(figsize=(14,7.6));ax.axis('off')
    ax.text(.025,.96,title,transform=ax.transAxes,va='top',color=NAVY,weight='bold',size=17)
    ax.text(.025,.905,'Four exact function definitions from '+path.name,transform=ax.transAxes,va='top',color=GREY,size=10)
    bg=FancyBboxPatch((.02,.115),.96,.74,transform=ax.transAxes,boxstyle='round,pad=.007',facecolor='#f2f6f9',edgecolor='#cbdbe2')
    ax.add_patch(bg)
    ax.text(.045,.823,'\n'.join(shown),transform=ax.transAxes,va='top',family='DejaVu Sans Mono',fontsize=10.5,linespacing=1.65,color=NAVY)
    ax.text(.025,.055,'Selected noncontiguous source excerpts with original line numbers. Not a Spyder screenshot.',transform=ax.transAxes,fontsize=9,color=GREY)
    save(fig,tag,[path],'Four exact source signatures')


def verification_evidence():
    p=locate('redo_v4_final_verification/FINAL_REDO_CHECK.txt')
    s=p.read_text(encoding='utf-8-sig').splitlines()
    relevant=[l.strip() for l in s if ('PASS' in l or '1268' in l or '1,268' in l or '450' in l or '30 countries' in l or '20 unique' in l)]
    require(relevant, 'Final verification report does not include readable result lines')
    relevant=relevant[:12]
    fig,ax=plt.subplots(figsize=(13,5.6));ax.axis('off')
    ax.text(.04,.94,'Final Consolidated Verification',transform=ax.transAxes,weight='bold',size=18,color=NAVY)
    ax.text(.04,.84,'Verbatim lines from FINAL_REDO_CHECK.txt',transform=ax.transAxes,color=GREY,size=10)
    body='\n\n'.join(relevant)
    ax.text(.05,.7,body,transform=ax.transAxes,color=NAVY,va='top',family='DejaVu Sans Mono',size=11)
    ax.text(.04,.06,'Recorded report excerpt. No terminal or Spyder interface has been simulated.',transform=ax.transAxes,color=GREY,size=9)
    save(fig,'4_9',[p],'Verbatim final verification report')


def trace_evidence():
    p=locate('redo_v4_multiref_corrected/records.json')
    records=json.loads(p.read_text(encoding='utf-8-sig'))
    candidates=[r for r in records if r.get('fault')=='none' and r.get('condition')=='closed_loop' and r.get('task_completed')==1 and r.get('controlled_seed')==0]
    require(candidates,'No completed natural closed loop trial in corrected evidence')
    # Stable source evidence, not a fabricated illustrative trace.
    r=sorted(candidates,key=lambda r:r['country_code'])[0]
    rows=[]
    for e in r.get('events',[]):
        if e.get('result') != 'success': continue
        try: proposal=json.loads(e.get('proposal','{}'))
        except (ValueError,TypeError): continue
        out=e.get('output') or {}
        rows.append((e.get('step'),proposal.get('tool',''),out.get('country_code',''),out.get('president_name',out.get('country_name','')),e.get('result')))
    require(len(rows)>=4,'Recorded natural success does not show four successful calls')
    fig,ax=plt.subplots(figsize=(14.1,6.0));ax.axis('off')
    ax.text(.035,.93,'Observed Natural MCP Tool Sequence',transform=ax.transAxes,color=NAVY,weight='bold',size=17)
    ax.text(.035,.85,f"Country {r.get('country_name')}    Condition {r.get('condition')}    Task {r.get('task_id')}",transform=ax.transAxes,color=GREY,fontsize=10)
    heads=['Step','Executed tool','Country code','Retrieved evidence','Status']
    cells=[[str(x) for x in row] for row in rows[:5]]
    tb=ax.table(cellText=cells,colLabels=heads,cellLoc='left',colLoc='left',loc='center',colWidths=[.08,.36,.14,.28,.12],bbox=[.03,.22,.94,.53])
    tb.auto_set_font_size(False);tb.set_fontsize(9.5)
    for (i,j),cell in tb.get_celld().items():
        cell.set_edgecolor('#d3e1e5');cell.set_facecolor('#dcebed' if i==0 else 'white')
        if i==0:cell.set_text_props(weight='bold',color=NAVY)
    ax.text(.04,.1,'Directly extracted from the corrected Phase C natural trial event trace.',transform=ax.transAxes,fontsize=9,color=GREY)
    save(fig,'4_14',[p],'Real recorded sequence, no simulated output')


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
    # 5.9 mean tokens versus observed completion for natural groups only
    fig=plt.figure(figsize=(12.5,7));ax=styled_ax(fig,'Natural Completion and Token Cost','Each point represents 20 inventory or 30 historical natural trials')
    offsets={'inventory':(9,6),'historical':(9,-13)}
    for family,df,marker in [('inventory',inv,'o'),('historical',hist,'s')]:
        subset=df[df.fault=='none']
        for policy in POLICIES:
            v=subset[subset.condition==policy]
            require(len(v)==(20 if family=='inventory' else 30),'Natural subset count differs: '+family)
            x=float(v.total_tokens.mean());y=float(v.task_completed.mean()*100)
            ax.scatter(x,y,color=COLORS[policy],s=150,marker=marker,edgecolor='white',linewidth=1.2,zorder=4)
            xx,yy=offsets[family]
            ax.annotate(P_NAMES[policy]+'   '+family,(x,y),xytext=(xx,yy),textcoords='offset points',fontsize=9,color=NAVY)
    ax.set_xlabel('Mean model tokens per attempted natural trial');ax.set_ylabel('Natural task completion (%)');ax.set_ylim(22,105)
    fig.tight_layout();save(fig,'5_9',[locate('redo_v3_inventory_baseline/trials.csv'),locate('redo_v4_multiref_corrected/trials.csv')],'Natural trial completion and mean tokens')
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
    print('Source panels are NOT screenshots of the Spyder application.')


if __name__=='__main__':
    main()
