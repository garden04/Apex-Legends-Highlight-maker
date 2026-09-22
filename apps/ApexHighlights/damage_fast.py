"""Damage prefilter followed by the editor's unchanged template matcher."""
import json,math,os,tempfile,threading,time
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
import damage_filter
from core import Settings,Event,Cancelled,check_cancel,metadata

def plan_windows(windows,duration,fps):
    out=[];gap=max(.8,2.1/fps)
    for a,b in sorted(windows):
        a=max(0,math.floor(a*fps)/fps);b=min(duration,math.ceil(b*fps)/fps)
        if b<=a:continue
        if out and a<=out[-1][1]+gap:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def analyze_fast(analyzer,path,settings,progress,cancel,force,stage_progress):
    from pipeline_analyzer import DecoderFailure
    started=time.perf_counter();settings.validate();cancel=cancel or threading.Event();check_cancel(cancel)
    def report(p,message,stage='image'):
        progress(p,message)
        if stage_progress:stage_progress(stage,p,message)
    normal=Settings(**asdict(settings));normal.damage_fast=False;normal.audio_fast=False
    def fallback(reason):
        report(0,reason+' · 전체 영상 분석으로 진행합니다.','setup')
        info,events=analyzer.analyze(path,normal,progress,cancel,force,stage_progress)
        info['analysis'].update(damage_fast=False,damage_fallback=reason,seconds=time.perf_counter()-started)
        return info,events
    if settings.detector!='template':return fallback('데미지 고속 모드는 기준 이미지 감지에서 지원됩니다')
    report(0,'영상 정보 확인 중','setup');info=metadata(path)
    cache=analyzer.data_dir/('damage-fast-v1-'+analyzer.cache_key(path,settings)+'.json')
    if cache.exists() and not force:
        try:
            saved=json.loads(cache.read_text(encoding='utf-8'))
            events=[Event(**e) for e in saved['events']]
            info['analysis']={**saved['stats'],'cached':True,'seconds':time.perf_counter()-started}
            report(1,'저장된 데미지 고속 분석 결과 불러옴','cache');return info,events
        except (OSError,ValueError,TypeError,KeyError):pass
    o=damage_filter.Options(str(path),str(analyzer.data_dir/'damage_checks'),roi=tuple(settings.damage_roi))
    def damage_progress(d):
        detailed=d['stage']=='동일 구간 세부 확인'
        p=(.65 if detailed else 0)+(.35 if detailed else .65)*d['done']/max(1,d['total'])
        report(p,f"고속 모드 · {d['stage']} {d['done']}/{d['total']}",'damage')
    report(0,'고속 모드 · 데미지 변화 확인 중','damage')
    try:
        result=damage_filter.analyze(o,cancel,damage_progress)
    except damage_filter.Cancelled:raise Cancelled()
    except (ValueError,OSError,RuntimeError) as exc:return fallback(f'데미지 필터 준비 실패: {exc}')
    if result['status']=='cancelled':raise Cancelled()
    if result['status']!='completed':return fallback('데미지 필터 실패: '+result.get('error','판독 실패'))
    check_cancel(cancel)
    windows=damage_filter.proposed_windows(result['intervals'],0,info['duration'],3,
        settings.damage_exclude_unknown,result['samples'])
    windows=plan_windows(windows,info['duration'],settings.sample_fps)
    # Record exactly which ranges the editor will inspect, including frame-grid padding.
    policy=dict(exclude_unknown=settings.damage_exclude_unknown,windows=windows,
        rule='앞뒤 동일 제외, 다른 값 사이 유지, 나머지 판독 불가만 옵션에 따라 제외')
    (Path(result['directory'])/'editor_windows.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2),encoding='utf-8')
    report(1,f'데미지 확인 완료 · 검사할 구간 {len(windows)}개','damage')
    patterns=[]
    for template in settings.templates:
        image=cv2.imdecode(np.fromfile(template['path'],np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or image.std()<3:raise ValueError('기준 이미지를 확인하세요.')
        patterns.append((template['kind'],image,Path(template['path']).stem))
    if not patterns:raise ValueError('기준 이미지가 없습니다.')
    events=[];count=0;match_seconds=0.;completed=0.;fallbacks=[]
    total=sum(b-a for a,b in windows)
    for index,(a,b) in enumerate(windows):
        check_cancel(cancel)
        def subreport(p,message):
            report((completed+p*(b-a))/max(.001,total),f'고속 모드 · 구간 {index+1}/{len(windows)} · {message}')
        try:found,stats=analyzer._stream(path,info,normal,patterns,cancel,subreport,'d3d11va' if os.name=='nt' else 'cpu',a,b)
        except DecoderFailure as exc:
            if os.name!='nt':raise
            fallbacks.append(str(exc));subreport(0,'GPU 추출 실패 · 해당 구간 CPU 재시도')
            found,stats=analyzer._stream(path,info,normal,patterns,cancel,subreport,'cpu',a,b)
        events.extend(found);count+=stats['analyzed_frames'];match_seconds+=stats['match_seconds'];completed+=b-a
    check_cancel(cancel)
    stats=dict(note='데미지 변화 필터 + FFmpeg PNG/OpenCV',damage_fast=True,audio_fast=False,
        damage_exclude_unknown=settings.damage_exclude_unknown,damage_windows=windows,
        damage_report=result['directory'],damage_seconds=result['total_seconds'],
        inspected_seconds=total,source_seconds=info['duration'],analyzed_frames=count,
        extracted_frames=count,match_seconds=match_seconds,fallbacks=fallbacks,
        seconds=time.perf_counter()-started,cached=False)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=analyzer.data_dir,suffix='.tmp',delete=False) as out:
            temporary=Path(out.name);json.dump(dict(events=[asdict(e) for e in events],stats=stats),out,ensure_ascii=False)
        temporary.replace(cache)
    finally:
        if temporary:temporary.unlink(missing_ok=True)
    info['analysis']=stats
    report(1,f'고속 분석 완료 · 검사 {count}장 · 후보 {len(events)}개 · {stats["seconds"]:.1f}초')
    return info,events
