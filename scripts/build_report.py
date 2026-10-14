"""Build report directly from run artifacts; no hand-entered performance values."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sensor-ssl-mpl')
import csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
METHODS=['supervised','time','channel','both']; FRACTIONS=[.01,.05,.1,1.]; SEEDS=[0,1,2]
NAMES={'supervised':'Supervised','time':'SSL-time','channel':'SSL-channel','both':'SSL-combined'}
records={}; flat=[]
for m in METHODS:
 for f in FRACTIONS:
  for s in SEEDS:
   folder=ROOT/'runs/full'/f'{m}_f{f}_s{s}'
   r=json.loads((folder/'results.json').read_text());meta=json.loads((folder/'metadata.json').read_text());hist=json.loads((folder/'history.json').read_text())
   assert not r['synthetic'] and meta['config']['epochs']==30 and meta['config']['pretrain_epochs']==30
   assert len([h for h in hist if h['stage']=='finetune'])==30
   assert set(meta['train_subjects']).isdisjoint(meta['val_subjects'])
   assert set(meta['train_subjects']+meta['val_subjects']).isdisjoint(meta['test_subjects'])
   for cond in ['none','acc','gyro']:
    cm=np.array(r['test'][cond]['confusion_matrix']);assert cm.sum()==2947
    f1=(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)).mean()
    assert abs(f1-r['test'][cond]['macro_f1'])<1e-12
    flat.append({'method':m,'fraction':f,'seed':s,'condition':cond,'macro_f1':f1,'accuracy':r['test'][cond]['accuracy'],'labeled_count':meta['labeled_count']})
   records[m,f,s]=(r,meta,hist)
for f in FRACTIONS:
 for s in SEEDS:
  indices=records['supervised',f,s][1]['labeled_indices']
  assert all(records[m,f,s][1]['labeled_indices']==indices for m in METHODS)
assert len({r[1]['data_sha256'] for r in records.values()})==1
assert len({r[1]['git_commit'] for r in records.values()})==1
with (OUT/'all_evaluations.csv').open('w',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
def vals(m,f,cond='none'):return np.array([records[m,f,s][0]['test'][cond]['macro_f1'] for s in SEEDS])*100
def fmt(v):return f'{v.mean():.2f} +/- {v.std(ddof=1):.2f}'
def table_data(cond):return [['Labels','Supervised','SSL-time','SSL-channel','SSL-combined']]+[[f'{f:.0%}']+[fmt(vals(m,f,cond)) for m in METHODS] for f in FRACTIONS]
paired=[]
for cond in ['none','acc','gyro']:
 for f in FRACTIONS:
  for m in METHODS[1:]:
   delta=vals(m,f,cond)-vals('supervised',f,cond)
   paired.append({'condition':cond,'fraction':f,'method':m,'mean_delta_pp':delta.mean(),'std_delta_pp':delta.std(ddof=1),'seeds_positive':int((delta>0).sum())})
with (OUT/'paired_differences.csv').open('w',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colorset=['#44546a','#1f77b4','#dd8452','#269a88']
fig,axes=plt.subplots(1,3,figsize=(13,4.1),sharey=True)
for ax,cond,title in zip(axes,['none','acc','gyro'],['Complete signals','Accelerometer unavailable','Gyroscope unavailable']):
 for m,c in zip(METHODS,colorset):
  ax.errorbar(range(4),[vals(m,f,cond).mean() for f in FRACTIONS],yerr=[vals(m,f,cond).std(ddof=1) for f in FRACTIONS],marker='o',capsize=3,color=c,label=NAMES[m])
 ax.set_xticks(range(4),['1%','5%','10%','100%']);ax.set_ylim(0,100);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_xlabel('Training-label budget')
axes[0].set_ylabel('Test Macro-F1 (%)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(OUT/'comparison.png',dpi=220);fig.savefig(OUT/'comparison.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for m,c in zip(METHODS,colorset):
 ax.errorbar(range(4),[vals(m,f).mean() for f in FRACTIONS],yerr=[vals(m,f).std(ddof=1) for f in FRACTIONS],marker='o',capsize=4,label=NAMES[m],color=c)
ax.set_xticks(range(4),['1%','5%','10%','100%']);ax.set_ylabel('Test Macro-F1 (%)');ax.set_xlabel('Training-label budget');ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'clean.png',dpi=220);plt.close(fig)
# Validation-only descriptive selection, kept separate from test leaders.
selected={f:max(METHODS,key=lambda m:np.mean([records[m,f,s][0]['best_val_macro_f1'] for s in SEEDS])) for f in FRACTIONS}
selection_rows=[]
for f,m in selected.items():
 selection_rows.append({'fraction':f,'selected_method':m,'mean_validation_macro_f1':np.mean([records[m,f,s][0]['best_val_macro_f1'] for s in SEEDS]),'mean_test_macro_f1':vals(m,f).mean()/100})
with (OUT/'validation_selection.csv').open('w',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=list(selection_rows[0]));w.writeheader();w.writerows(selection_rows)
low=vals('both',.01)-vals('supervised',.01)
abstract=(f'We compare supervised training with three masked-reconstruction pretraining objectives on subject-disjoint UCI HAR data. Forty-eight training configurations cover four label budgets and three random seeds; each checkpoint is evaluated under three signal conditions. At a 1% training-label budget, supervised training obtains {fmt(vals("supervised",.01))}% Macro-F1, temporal masking obtains {fmt(vals("time",.01))}% ({(vals("time",.01)-vals("supervised",.01)).mean():+.2f} percentage points), and combined temporal-channel masking obtains {fmt(vals("both",.01))}%. These findings apply to one fixed subject split and do not establish a new algorithm or state-of-the-art performance. Full validation labels are used for selection and are excluded from the stated training-label budgets.')
clean_rows=table_data('none')
md='# Experimental results\n\n'+abstract+'\n\n'
for cond,title in [('none','Complete signals'),('acc','Accelerometer unavailable'),('gyro','Gyroscope unavailable')]:
 rows=table_data(cond);md+=f'## {title}\n\nMacro-F1 in percent, mean +/- sample SD over three seeds.\n\n'+'| '+' | '.join(rows[0])+' |\n|'+'|'.join(['---']*len(rows[0]))+'|\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows[1:])+'\n'
md+='## Paired clean-condition differences\n\n| Labels | SSL-time | SSL-channel | SSL-combined |\n|---|---|---|---|\n'
for f in FRACTIONS:md+='| '+f'{f:.0%}'+' | '+' | '.join(fmt(vals(m,f)-vals('supervised',f)) for m in METHODS[1:])+' |\n'
md+='\nDifferences are percentage points and use paired seeds. Three seeds are not sufficient for strong significance claims. Test-score leaders are descriptive only; model choice must use validation results.\n'
bootstrap=list(csv.DictReader((OUT/'subject_bootstrap.csv').open()))
md+='\n## Exploratory paired subject bootstrap\n\n95% percentile intervals, 5,000 resamples of nine test subjects; conditional on the fitted models, without multiplicity correction.\n\n| Labels | Method | Difference (pp) | 95% interval (pp) |\n|---|---|---|---|\n'
for row in bootstrap:
 md+=f"| {float(row['fraction']):.0%} | {NAMES[row['method']]} | {float(row['delta_macro_f1_pp']):+.2f} | [{float(row['cluster_ci_low_pp']):+.2f}, {float(row['cluster_ci_high_pp']):+.2f}] |\n"
(OUT/'RESULTS.md').write_text(md)
base=(ROOT/'docs/PAPER_BASE.md').read_text().replace('–','-').replace('—','-')
paper=base.replace('## 1. Introduction','## Abstract\n\n'+abstract+'\n\n## 1. Introduction')
result_section='### Complete signals'+md.split('## Complete signals',1)[1]
result_section=result_section.replace('\n## ','\n### ')
start,end=paper.index('## 5. Results'),paper.index('## 6. Limitations')
paper=paper[:start]+'## 5. Results\n\n'+result_section+'\n'+paper[end:]
(ROOT/'docs/PAPER_DRAFT.md').write_text(paper)
# PDF: intentionally explicit page breaks for predictable print layout.
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=22,leading=27,textColor=colors.HexColor('#15364a'),spaceAfter=15));styles.add(ParagraphStyle(name='BodyCustom',fontSize=10,leading=14,spaceAfter=9));styles.add(ParagraphStyle(name='SmallCustom',fontSize=8,leading=11,spaceAfter=6))
story=[]
def para(text,style='BodyCustom'):story.append(Paragraph(escape(text).replace('\n','<br/>'),styles[style]))
def heading(text):story.append(Paragraph(escape(text),styles['Heading2']))
def tab(rows,widths=None,font=8):
 data=[[Paragraph(escape(str(cell)),styles['SmallCustom']) for cell in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9ab0bc')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,10))
def page():story.append(PageBreak())
def section(num):
 start=base.index(f'## {num}.');end=base.find('\n## ',start+1)
 return base[start:end if end!=-1 else None].split('\n\n')[1:]
para('SENSOR SSL / RESEARCH REPORT','SmallCustom');para('Label Budget and Sensor-Outage Sensitivity of Masked Reconstruction in Human Activity Recognition','TitleCustom');para('14 October 2026 | Empirical study for advisor review','SmallCustom')
heading('Abstract');para(abstract)
heading('1. Research question');para(section(1)[0]);heading('2. Positioning against prior work');para(section(2)[0]);para('No author names or institutional affiliations are asserted. The manuscript requires researcher and advisor review before submission.','SmallCustom')
page();heading('3. Data and experimental design')
for text in section(3):para(text)
tab([['Partition','Subjects','Windows'],['Training','17','5,844'],['Validation','4','1,508'],['Test','9','2,947']],[180,130,180])
tab([['Nominal training labels','Actual labeled windows'],*[[f'{f:.0%}',str(records['supervised',f,0][1]['labeled_count'])] for f in FRACTIONS]],[245,245])
para('Archive SHA-256: '+records['supervised',.01,0][1]['data_sha256'],'SmallCustom')
page();heading('4. Architecture and optimization')
for text in section(4)[:-1]:para(text)
tab([['Pretraining objective','Masked fraction','Geometry'],['Time','25%','32 contiguous time steps, all channels'],['Channel','16.67%','One of six channels, all time steps'],['Combined','37.5%','Union of time and channel masks']],[150,100,240])
para(section(4)[-1]);para('The masked MSE objective is averaged over masked values only. Reconstruction quality is not itself an activity-recognition metric.','SmallCustom')
page();heading('5. Clean-signal results');para('Macro-F1 (%), mean +/- sample standard deviation across seeds 0, 1 and 2. Every row uses the same labeled examples across methods.');tab(clean_rows,[50,110,110,110,110]);story.append(Image(str(OUT/'clean.png'),width=490,height=245))
heading('Paired effect of SSL relative to supervised training');tab([['Labels','SSL-time','SSL-channel','SSL-combined'],*[[f'{f:.0%}']+[fmt(vals(m,f)-vals('supervised',f)) for m in METHODS[1:]] for f in FRACTIONS]],[55,145,145,145]);para('Paired differences are percentage points. Standard deviation is computed on the three per-seed differences, not by combining independent standard deviations.','SmallCustom')
page();heading('6. Simulated sensor-outage results');para('The selected clean-validation checkpoint is reused without further training. Each unavailable sensor group is replaced by its training mean.');heading('Accelerometer unavailable');tab(table_data('acc'),[50,110,110,110,110]);heading('Gyroscope unavailable');tab(table_data('gyro'),[50,110,110,110,110]);story.append(Image(str(OUT/'comparison.png'),width=490,height=155));para('Error bars show sample standard deviation across seeds. Channels are grouped into two physical sensor types. These stress tests do not represent every real-world missingness mechanism.','SmallCustom')
page();heading('7. Exploratory test-subject uncertainty')
para('We resample the nine test subjects with replacement 5,000 times, retaining all windows within each selected subject. Subject draws are paired across methods and seeds. The statistic is the mean of three seed-level differences in pooled-window Macro-F1. Percentile intervals are conditional on the fixed training split and fitted models.')
tab([['Labels','Method','Difference (pp)','95% interval (pp)'],*[[f"{float(row['fraction']):.0%}",NAMES[row['method']],f"{float(row['delta_macro_f1_pp']):+.2f}",f"[{float(row['cluster_ci_low_pp']):+.2f}, {float(row['cluster_ci_high_pp']):+.2f}]"] for row in bootstrap]],[55,145,120,170])
para('This supplementary analysis was specified while the experiment matrix was running. It does not alter model fitting or selection. Only clean-condition differences are assessed. Nine subjects are few, no multiple-comparison correction is applied, and these intervals do not include training-split uncertainty. They must not be interpreted as a guarantee of population performance.')
page();heading('8. Interpretation and limitations')
for f in FRACTIONS:
 best=max(METHODS,key=lambda m:vals(m,f).mean());d=vals(best,f).mean()-vals('supervised',f).mean()
 para(f'At {f:.0%} training labels, the highest observed mean clean Macro-F1 is {vals(best,f).mean():.2f}% ({NAMES[best]}), {d:+.2f} percentage points relative to supervised training. This is a descriptive test-set comparison, not a model-selection rule.')
for text in section(6):para(text)
heading('Conclusion');para('This controlled experiment quantifies label-budget sensitivity and simulated sensor-outage behavior for a compact reconstruction-based SSL family. Any practical preference should consider validation evidence, seed variability, additional pretraining cost and the intended missing-sensor scenario. Broader claims require external baselines and independent datasets.')
page();heading('9. Reproducibility and submission status')
for text in section(7):para(text)
para('Training completed on CPU with Python '+records['supervised',.01,0][1]['python']+', PyTorch '+records['supervised',.01,0][1]['torch']+' and NumPy '+records['supervised',.01,0][1]['numpy']+'.')
para('Training source commit: '+records['supervised',.01,0][1]['git_commit'],'SmallCustom')
heading('References')
refs=base.split('## References\n\n')[1].split('\n\n')
for ref in refs:para(ref,'SmallCustom')
heading('What remains before formal submission');para('Confirm the target venue and author details; agree on a defensible contribution with the advisor; add modern external baselines, additional datasets and subject splits; audit AI-use disclosure and source licensing. This package is a complete bounded experiment, not a guarantee of thesis completion or publication acceptance.')
page();heading('Appendix A. Complete clean-test runs');para('One row per configuration. Outage scores, per-class F1 and confusion matrices are included in the machine-readable artifacts.','SmallCustom')
rows=[['Method','Labels','Seed','Macro-F1','Accuracy','Val F1']]
for m in METHODS:
 for f in FRACTIONS:
  for s in SEEDS:
   r=records[m,f,s][0];rows.append([NAMES[m],f'{f:.0%}',s,f'{r["test"]["none"]["macro_f1"]*100:.2f}',f'{r["test"]["none"]["accuracy"]*100:.2f}',f'{r["best_val_macro_f1"]*100:.2f}'])
# compact appendix table; 48 rows fits one page at 12-point row height
appendix=Table(rows,colWidths=[125,65,50,85,85,80],repeatRows=1)
appendix.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),7),('LEADING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
story.append(appendix)
def footer(canvas,doc):
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#667785'));canvas.drawString(45,27,'Sensor SSL | Reproducible empirical study | Advisor-review manuscript');canvas.drawRightString(A4[0]-45,27,str(doc.page))
SimpleDocTemplate(str(OUT/'research-report.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=40,bottomMargin=45,title='Sensor SSL: label budget and sensor-outage sensitivity',author='').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'audit.json').write_text(json.dumps({'completed_configurations':48,'test_evaluations':144,'seeds':SEEDS,'all_confusion_matrices_total':2947,'metrics_recomputed':True,'paired_label_subsets_verified':True,'subject_disjoint_verified':True,'single_data_hash_verified':True,'single_training_commit_verified':True,'synthetic_excluded':True},indent=2))
thai='# สรุปผลการทดลองจริง\n\n'
thai+='รันครบ 48 configurations: 4 วิธี × 4 สัดส่วนป้ายกำกับ × 3 seed และประเมิน 3 สภาพสัญญาณ รวม 144 ผลประเมิน\n\n'
thai+='ข้อมูล UCI HAR: train 5,844 หน้าต่าง, validation 1,508 หน้าต่าง, test 2,947 หน้าต่าง แยกผู้ใช้ไม่ซ้ำกัน\n\n'
for condition,title in [('none','สัญญาณครบ'),('acc','ไม่มี accelerometer'),('gyro','ไม่มี gyroscope')]:
 thai+=f'## {title}\n\nMacro-F1 (%): ค่าเฉลี่ย +/- sample SD จาก 3 seed\n\n'
 rows=table_data(condition);thai+='| '+' | '.join(rows[0])+' |\n|'+'|'.join(['---']*len(rows[0]))+'|\n'+''.join('| '+' | '.join(row)+' |\n' for row in rows[1:])+'\n'
thai+='## ผลที่ตีความได้\n\n'
for f in FRACTIONS:
 delta=vals('both',f)-vals('supervised',f)
 thai+=f'- ที่ training labels {f:.0%}: SSL-combined ต่างจาก supervised เฉลี่ย {delta.mean():+.2f} จุดเปอร์เซ็นต์ Macro-F1 (SD ของผลต่าง {delta.std(ddof=1):.2f})\n'
thai+='\nค่าบวกหมายถึง SSL-combined ดีกว่า baseline ในเงื่อนไขนี้ ค่าลบหมายถึงแย่กว่า ไม่ควรสรุปว่า SSL ชนะทุกสถานการณ์หรือว่าดีกว่าวิธีล่าสุด เพราะยังไม่ได้ทดลอง baseline จากงานอื่น\n\n'
thai+='ผล bootstrap ผู้ใช้ 9 คนอยู่ใน subject_bootstrap.csv เป็นการวิเคราะห์เพิ่มเติมที่มีข้อจำกัดด้านจำนวนคนและไม่มีการแก้ปัญหา multiple comparisons\n\n'
thai+='## สถานะการส่งงาน\n\nชุดการทดลองตาม protocol นี้เสร็จครบ พร้อมรายงานให้อาจารย์ตรวจ แต่ยังไม่ใช่การยืนยันความใหม่หรือรับรองการตีพิมพ์ ก่อนส่งจริงต้องใส่ผู้เขียน/สังกัดและรูปแบบเป้าหมาย เพิ่ม external baselines และข้อมูลอิสระตาม contribution ที่อาจารย์เห็นชอบ\n'
(OUT/'RESULTS_TH.md').write_text(thai)
print(abstract)
print('REPORT',OUT/'research-report.pdf')
