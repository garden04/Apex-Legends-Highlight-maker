"""Sparse damage OCR experiment. Equal readings are hypotheses, never truth labels."""
import csv
import io
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from bisect import bisect_left,bisect_right
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

DEFAULT_ROI = (.88, .075, .94, .12)
_digit_reader = None

def neural_digits(image,cancel):
    global _digit_reader
    check(cancel)
    if _digit_reader is None:
        import torch
        import easyocr
        torch.set_num_threads(2)
        _digit_reader=easyocr.Reader(['ko','en'],gpu=False,detector=False,
            download_enabled=False,verbose=False)
    check(cancel)
    rows=_digit_reader.recognize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),
        allowlist='0123456789',detail=1)
    check(cancel)
    if len(rows)!=1:return None,0
    text=rows[0][1]
    if not text.isascii() or not text.isdigit() or len(text)>5:return None,0
    return int(text),float(rows[0][2])*100

class Cancelled(Exception): pass

def check(cancel):
    if cancel.is_set(): raise Cancelled()

BUNDLED_TESSERACT = Path(__file__).resolve().parents[2] / 'tools' / 'tesseract' / 'tesseract.exe'

def tesseract_path():
    if os.environ.get('TESSERACT_CMD'): return os.environ['TESSERACT_CMD']
    if BUNDLED_TESSERACT.is_file(): return str(BUNDLED_TESSERACT)
    return shutil.which('tesseract') or 'C:/Program Files/Tesseract-OCR/tesseract.exe'

@dataclass
class Options:
    source: str
    output: str
    interval: float = 60
    detail_interval: float = 10
    roi: tuple = DEFAULT_ROI
    confidence: float = 30
    padding: float = 3
    start: float = 0
    duration: float = 0
    decoder: str = 'd3d11va'
    tesseract: str = ''
    baseline: str = ''

    def validate(self):
        if not Path(self.source).is_file(): raise ValueError('영상 파일을 선택하세요.')
        if not self.output.strip(): raise ValueError('결과 저장 폴더를 선택하세요.')
        values = [self.interval, self.detail_interval, self.confidence, self.padding, self.start, self.duration, *self.roi]
        if not all(math.isfinite(x) for x in values): raise ValueError('설정에 유효한 숫자를 입력하세요.')
        if not 0 < self.detail_interval <= self.interval or self.interval < 1: raise ValueError('세부 간격은 0 초과, 기본 간격 이하여야 합니다.')
        x1,y1,x2,y2=self.roi
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1: raise ValueError('데미지 숫자 영역을 지정하세요.')
        if not 0 <= self.confidence <= 100 or min(self.start,self.duration,self.padding)<0: raise ValueError('설정 범위를 확인하세요.')
        if self.decoder not in ('cpu','d3d11va'): raise ValueError('디코더를 확인하세요.')
        if not Path(self.tesseract or tesseract_path()).is_file(): raise ValueError('Tesseract 실행 파일을 찾지 못했습니다. 경로를 지정하세요.')
        if self.baseline and not Path(self.baseline).is_file(): raise ValueError('비교할 녹다운 기록 파일을 확인하세요.')

def probe(path):
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or fps<=0 or count<=0: raise ValueError('영상 정보를 읽을 수 없습니다.')
        return dict(duration=count/fps,fps=fps,width=int(cap.get(3)),height=int(cap.get(4)))
    finally: cap.release()

def run_process(args,cancel,timeout=45):
    check(cancel)
    env=os.environ.copy();env['OMP_THREAD_LIMIT']='1'
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    started=time.monotonic()
    try:
        while True:
            check(cancel)
            try:
                out,err=p.communicate(timeout=.1)
                if p.returncode: raise RuntimeError(err.decode('utf-8',errors='replace')[-2000:])
                return out
            except subprocess.TimeoutExpired:
                if time.monotonic()-started>timeout: raise TimeoutError(f'처리 제한 시간 {timeout}초 초과: {Path(args[0]).name}')
    finally:
        if p.poll() is None:
            p.kill();p.communicate()
        p.stdout.close();p.stderr.close()

def get_frame(source,at,decoder,cancel):
    def command(hw):
        args=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin']
        if hw!='cpu': args+=['-hwaccel',hw]
        # Sparse input seeks; do not decode the entire preceding video.
        return args+['-ss',f'{at:.6f}','-i',str(source),'-map','0:v:0','-an','-sn','-dn',
            '-frames:v','1','-c:v','png','-f','image2pipe','pipe:1']
    fallback=None
    try: data=run_process(command(decoder),cancel)
    except (RuntimeError,TimeoutError) as exc:
        if decoder=='cpu': raise
        fallback=str(exc);data=run_process(command('cpu'),cancel)
    frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR) if data else None
    if frame is None: raise ValueError(f'{at:.2f}초 화면을 추출하지 못했습니다.')
    return frame,fallback

def crop_roi(frame,roi):
    h,w=frame.shape[:2];x1,y1,x2,y2=roi
    crop=frame[int(y1*h):max(int(y1*h)+1,int(y2*h)),int(x1*w):max(int(x1*w)+1,int(x2*w))]
    if crop.size==0: raise ValueError('숫자 영역이 비어 있습니다.')
    return crop

def save_png(path,frame):
    ok,encoded=cv2.imencode('.png',frame)
    if not ok: raise ValueError('PNG 저장 실패')
    Path(path).write_bytes(encoded.tobytes())

def parse_tsv(data):
    words=[];conf=[]
    for row in csv.DictReader(io.StringIO(data.decode('utf-8',errors='replace')),delimiter='\t'):
        text=(row.get('text') or '').strip()
        if not text: continue
        if not text.isascii() or not text.isdigit(): return None,0,text
        words.append(text);conf.append(float(row['conf']))
    text=''.join(words)
    if not text or len(text)>5: return None,0,text
    return int(text),min(conf),text

def number_region(crop):
    """Find a row of digit-sized white components, ignoring HUD borders/icons."""
    mask=cv2.inRange(crop,np.array([180]*3,np.uint8),np.array([255]*3,np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    components=[]
    for x,y,w,h,area in stats[1:]:
        if h>=6 and .15<=w/h<=1.2 and area/(w*h)>=.25:
            components.append((int(x),int(y),int(w),int(h)))
    groups=[]
    for seed in components:
        x,y,w,h=seed
        row=sorted([c for c in components if max(c[3],h)/min(c[3],h)<=1.12 and abs(c[1]+c[3]-y-h)<=min(h,c[3])*.18])
        group=[]
        for c in row:
            if group and c[0]-(group[-1][0]+group[-1][2])>h*.7:
                if seed in group:break
                group=[]
            group.append(c)
        if seed in group and 1<=len(group)<=5:groups.append(group)
    if not groups:return None
    group=max(groups,key=lambda g:(len(g),g[-1][0]+g[-1][2]))
    x=min(c[0] for c in group);y=min(c[1] for c in group)
    right=max(c[0]+c[2] for c in group);bottom=max(c[1]+c[3] for c in group)
    # A clipped glyph is not a trustworthy number.
    if x==0 or right>=crop.shape[1] or y==0 or bottom>=crop.shape[0]:return None
    return crop[max(0,y-1):bottom+1,max(0,x-1):right+1]

def recognize(crop,directory,key,o,cancel):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    # Two preprocessings must agree. Empty recognition remains unknown, never zero.
    mask=cv2.inRange(cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC),
        np.array([165,165,165],np.uint8),np.array([255,255,255],np.uint8))
    reads=[]
    def pair(variants,name):
        current=[]
        for index,variant in enumerate(variants):
            padded=cv2.copyMakeBorder(variant,20,20,20,20,cv2.BORDER_CONSTANT,value=255)
            file=directory/f'{key}_{name}{index}.png';save_png(file,padded)
            raw=run_process([o.tesseract or tesseract_path(),str(file),'stdout','-l','eng','--psm','7',
                '-c','tessedit_char_whitelist=0123456789','tsv'],cancel,timeout=15)
            value,confidence,text=parse_tsv(raw)
            current.append(dict(value=value,confidence=confidence,text=text,preprocessing=name))
        reads.extend(current)
        if (all(r['value'] is not None and r['confidence']>=o.confidence for r in current)
                and current[0]['value']==current[1]['value']):
            return current[0]['value'],min(r['confidence'] for r in current)
        return None,0
    number=number_region(crop)
    value,confidence=None,0
    if number is not None:
        save_png(directory/f'{key}_number.png',number)
        isolated_gray=cv2.cvtColor(number,cv2.COLOR_BGR2GRAY)
        isolated_mask=cv2.inRange(number,np.array([180]*3,np.uint8),np.array([255]*3,np.uint8))
        value,confidence=pair([cv2.resize(255-v,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC) for v in (isolated_gray,isolated_mask)],'number')
        if value is None:
            for tag,img in [('native',number),('large',cv2.resize(number,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)),
                    ('inverse',255-cv2.resize(number,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))]:
                neural_value,neural_confidence=neural_digits(img,cancel)
                reads.append(dict(value=neural_value,confidence=neural_confidence,text=str(neural_value),preprocessing='easyocr_'+tag))
            accepted={}
            for index,r in enumerate(reads):
                if not r['preprocessing'].startswith('easyocr_') or r['value'] is None or r['confidence']<max(85,o.confidence):continue
                agrees=any(j!=index and q['value']==r['value'] and q['confidence']>=(60 if q['preprocessing'].startswith('easyocr_') else 20)
                    for j,q in enumerate(reads))
                if agrees:accepted[r['value']]=max(accepted.get(r['value'],0),r['confidence'])
            if len(accepted)==1:value,confidence=next(iter(accepted.items()))
    if value is None:value,confidence=pair([255-gray,255-mask],'ocr')
    if value is None:
        # Remove small background specks and a clipped icon on the left edge.
        # Retry only: retain valid readings from the original crop.
        h,w=crop.shape[:2]
        native_mask=cv2.inRange(crop,np.array([165]*3,np.uint8),np.array([255]*3,np.uint8))
        _,labels,stats,_=cv2.connectedComponentsWithStats(native_mask)
        clean=np.zeros_like(native_mask)
        for k,(x,y,cw,ch,area) in enumerate(stats[1:],1):
            if ch>=h*.4 and not (x<=1 and cw<w*.3):clean[labels==k]=255
        x,y,cw,ch=cv2.boundingRect(clean)
        if cw and ch:
            variants=[255-cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)[y:y+ch,x:x+cw],255-clean[y:y+ch,x:x+cw]]
            value,confidence=pair([cv2.resize(v,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC) for v in variants],'clean')
    return dict(damage=value,state='read' if value is not None else 'unknown',confidence=confidence,reads=reads)

def classify(samples):
    values=[s['damage'] for s in samples if s['damage'] is not None]
    if any(b<a for a,b in zip(values,values[1:])): return 'reset_suspected'
    if len(values)!=len(samples): return 'unknown'
    if len(set(values))==1: return 'equal_candidate'
    return 'increased'

LABELS={'unknown':'판독 불가 · 유지','reset_suspected':'경기 전환/오인식 의심 · 유지',
    'equal_candidate':'동일 데미지 · 제외 후보(검증 필요)','increased':'데미지 증가 · 유지'}

def sample_times(start,end,step,last_margin=.1):
    last=max(start,end-last_margin)
    times=[start+i*step for i in range(int((last-start)//step)+1)]
    if last-times[-1]>.001: times.append(last)
    return times

def merge_windows(windows):
    out=[]
    for start,end in sorted(windows):
        if end<=start: continue
        if out and start<=out[-1][1]: out[-1][1]=max(end,out[-1][1])
        else: out.append([start,end])
    return out

def proposed_windows(intervals,start,end,padding,exclude_unknown=False,samples=None):
    known=sorted((s for s in (samples or []) if s['damage'] is not None),key=lambda s:s['time'])
    changes=[(a['time'],b['time']) for a,b in zip(known,known[1:]) if a['damage']!=b['damage']]
    def keep(item):
        if item['classification']=='equal_candidate':return False
        if exclude_unknown and item['classification']=='unknown':
            # Missing readings between different damage values remain protected.
            return any(a<item['end'] and b>item['start'] for a,b in changes)
        return True
    return merge_windows([[max(start,i['start']-padding),min(end,i['end']+padding)]
        for i in intervals if keep(i)])

def connect_equal_gaps(samples,intervals):
    """Infer only bounded unknown runs; preserve raw OCR and observed changes."""
    ordered=sorted(samples,key=lambda s:s['time']);times=[s['time'] for s in ordered]
    for s in ordered:
        s['analysis_damage']=s['damage'];s['inferred']=False
        s['inference_before']=None;s['inference_after']=None
    known=[i for i,s in enumerate(ordered) if s['damage'] is not None]
    for left,right in zip(known,known[1:]):
        if right==left+1 or ordered[left]['damage']!=ordered[right]['damage']:continue
        for s in ordered[left+1:right]:
            s.update(analysis_damage=ordered[left]['damage'],inferred=True,
                inference_before=ordered[left]['time'],inference_after=ordered[right]['time'])
    out=[]
    for original in intervals:
        item=dict(original)
        section=ordered[bisect_left(times,item['start']):bisect_right(times,item['end'])]
        values=[s['analysis_damage'] for s in section]
        complete=bool(section) and section[0]['time']==item['start'] and section[-1]['time']==item['end']
        same=complete and None not in values and len(set(values))==1
        item['analysis_damage']=values[0] if same else None
        item['bridged']=False
        # A detected reset remains protected even when surrounding values repeat.
        if item['classification'] in ('unknown','equal_candidate') and same:
            item['classification']='equal_candidate'
            item['bridged']=any(s['inferred'] for s in section)
        if (out and item['classification']=='equal_candidate' and out[-1]['classification']=='equal_candidate'
                and out[-1]['end']==item['start'] and out[-1]['analysis_damage']==item['analysis_damage']):
            out[-1].update(end=item['end'],end_damage=item['end_damage'],bridged=out[-1]['bridged'] or item['bridged'])
        else:out.append(item)
    for item in out:
        section=ordered[bisect_left(times,item['start']):bisect_right(times,item['end'])]
        item['checked_samples']=len(section)
        item['inferred_samples']=sum(s['inferred'] for s in section)
        if item['classification']=='equal_candidate':
            item['label']='앞뒤 동일 · 판독 불가 연결 · 제외 후보' if item['bridged'] else LABELS['equal_candidate']
    return out

def load_baseline(path,source):
    file=Path(path)
    if file.suffix.lower()=='.json':
        data=json.loads(file.read_text(encoding='utf-8-sig'))
        if not isinstance(data,dict): raise ValueError('녹다운 JSON의 형식을 확인하세요.')
        rows=data.get('events',data.get('analysis',{}).get('events'))
        if rows is None: raise ValueError('JSON에 events 기록이 없습니다.')
        declared=data.get('source') or data.get('options',{}).get('source')
    else:
        with file.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or 'time' not in reader.fieldnames: raise ValueError('time 열이 있는 knockdowns.csv를 선택하세요.')
            rows=list(reader)
        declared=None
    sources=[r.get('source') for r in rows if r.get('source')]
    if declared: sources.append(declared)
    if any(Path(p).resolve()!=Path(source).resolve() for p in sources): raise ValueError('녹다운 기록의 원본 영상이 선택한 영상과 다릅니다.')
    times=[float(r['time']) for r in rows]
    if any(not math.isfinite(t) or t<0 for t in times): raise ValueError('잘못된 녹다운 시각입니다.')
    return times,bool(sources)

def write_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def analyze(o,cancel=None,progress=lambda d:None):
    o.validate();cancel=cancel or threading.Event();check(cancel);started=time.perf_counter()
    info=probe(o.source)
    if o.start>=info['duration']: raise ValueError('시작 시각이 영상 끝보다 큽니다.')
    end=min(info['duration'],o.start+(o.duration or info['duration']))
    baseline=None
    if o.baseline: baseline=load_baseline(o.baseline,o.source)
    root=Path(o.output).resolve();root.mkdir(parents=True,exist_ok=True)
    directory=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);directory.mkdir()
    images=directory/'images';images.mkdir();samples={};intervals=[];extract_seconds=0.;ocr_seconds=0.
    result=dict(status='running',directory=str(directory),source=str(Path(o.source).resolve()),
        options=asdict(o),info=info,start=o.start,end=end,shadow_only=True,
        limitation='동일 데미지는 제외 후보일 뿐입니다. 샘플 사이 경기 전환과 OCR 오독을 배제하지 못합니다. 실제 영상은 제외하지 않습니다.')
    def emit(stage,done,total):
        elapsed=time.perf_counter()-started
        progress(dict(stage=stage,done=done,total=total,elapsed=elapsed,
            remaining=elapsed/done*(total-done) if done else None,directory=str(directory)))
    def sample(at,phase):
        nonlocal extract_seconds,ocr_seconds
        check(cancel);key=round(at,6)
        if key in samples: return samples[key]
        tick=time.perf_counter();frame,fallback=get_frame(o.source,at,o.decoder,cancel)
        extract_seconds+=time.perf_counter()-tick
        name=f'{at:013.3f}';crop=crop_roi(frame,o.roi)
        save_png(images/f'{name}_damage.png',crop)
        h,w=frame.shape[:2];thumb=cv2.resize(frame,(min(w,1280),round(h*min(w,1280)/w)))
        save_png(images/f'{name}_screen.png',thumb)
        tick=time.perf_counter();reading=recognize(crop,images,name,o,cancel);ocr_seconds+=time.perf_counter()-tick
        s=dict(time=at,phase=phase,**reading,image=f'images/{name}_screen.png',
            crop=f'images/{name}_damage.png',fallback=fallback)
        samples[key]=s
        return s
    try:
        times=sample_times(o.start,end,o.interval,max(.1,1/info['fps']))
        for index,at in enumerate(times):
            emit('기본 간격 데미지 검사',index,len(times));sample(at,'primary')
        pairs=list(zip(times,times[1:]))
        for index,(a,b) in enumerate(pairs):
            check(cancel);emit('동일 구간 세부 확인',index,len(pairs))
            readings=[samples[round(a,6)],samples[round(b,6)]]
            if classify(readings)=='equal_candidate':
                t=a+o.detail_interval
                while t<b-.001:
                    readings.append(sample(t,'detail'));t+=o.detail_interval
            readings.sort(key=lambda x:x['time'])
            state=classify(readings)
            # A reset anywhere keeps the parent interval. Otherwise retain only
            # the smaller unknown/increased intervals, without filling missing OCR.
            groups=[readings] if state=='reset_suspected' or len(readings)==2 else list(zip(readings,readings[1:]))
            for group in groups:
                classification=classify(group)
                intervals.append(dict(start=group[0]['time'],end=group[-1]['time'],classification=classification,label=LABELS[classification],
                    start_damage=group[0]['damage'],end_damage=group[-1]['damage'],checked_samples=len(group)))
        intervals=connect_equal_gaps(list(samples.values()),intervals)
        # Never discard the unsampled tail (and very short inputs).
        tail=times[-1]
        if tail<end: intervals.append(dict(start=tail,end=end,classification='unknown',label='끝부분 미확인 구간 · 유지',
            start_damage=samples[round(tail,6)]['damage'],end_damage=None,checked_samples=1))
        windows=proposed_windows(intervals,o.start,end,o.padding)
        kept=sum(b-a for a,b in windows)
        result.update(status='completed',proposed_windows=windows,proposed_kept_seconds=kept,
            proposed_saved_seconds=end-o.start-kept,proposed_saved_ratio=1-kept/(end-o.start))
        if baseline:
            times_base,verified=baseline
            relevant=[t for t in times_base if o.start<=t<end]
            missed=[t for t in relevant if not any(a<=t<=b for a,b in windows)]
            result['comparison']=dict(total=len(relevant),would_miss=missed,source_verified=verified,
                note='기존 감지 시점 기준 비교이며 사람이 확인한 정답이나 전체 정확도 보장은 아닙니다.')
    except Cancelled: result['status']='cancelled'
    except Exception as exc: result.update(status='failed',error=str(exc))
    finally:
        result.update(samples=sorted(samples.values(),key=lambda s:s['time']),intervals=intervals,
            total_seconds=time.perf_counter()-started,extract_seconds=extract_seconds,ocr_seconds=ocr_seconds)
        (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        write_csv(directory/'damage_samples.csv',result['samples'],['time','phase','damage','analysis_damage','inferred','inference_before','inference_after','state','confidence','image','crop','fallback'])
        write_csv(directory/'intervals.csv',intervals,['start','end','classification','label','start_damage','end_damage','analysis_damage','bridged','inferred_samples','checked_samples'])
        write_csv(directory/'proposed_windows.csv',[dict(start=a,end=b) for a,b in result.get('proposed_windows',[])],['start','end'])
    return result
