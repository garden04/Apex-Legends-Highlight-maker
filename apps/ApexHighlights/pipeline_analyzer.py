"""Editor adapter: FFmpeg PNG extraction overlaps existing template matching."""
import json,math,os,subprocess,tempfile,threading,time
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from core import Analyzer,Event,Tracker,Cancelled,check_cancel,metadata,ffmpeg,normalized_frame
from image_io import read_completed_image,retry_access,ImageTemporaryDirectory

class DecoderFailure(RuntimeError):pass

class PipelineAnalyzer(Analyzer):
    def analyze(self,path,settings,progress=lambda p,m:None,cancel=None,force=False,stage_progress=None):
        if settings.damage_fast:
            from damage_fast import analyze_fast
            return analyze_fast(self,path,settings,progress,cancel,force,stage_progress)
        # Preserve optional OCR modes through their established implementation.
        if settings.detector!='template':
            # Historical audio settings never reactivate the replaced fast mode.
            safe=type(settings)(**asdict(settings));safe.audio_fast=False
            return super().analyze(path,safe,progress,cancel,force,stage_progress)
        started=time.perf_counter();settings.validate();check_cancel(cancel)
        def report(p,message,stage='image'):
            progress(p,message)
            if stage_progress:stage_progress(stage,p,message)
        report(0,'영상 정보 확인 중','setup');info=metadata(path)
        cache=self.data_dir/('png-stream-v1-'+self.cache_key(path,settings)+'.json')
        if cache.exists() and not force:
            try:
                saved=json.loads(cache.read_text(encoding='utf-8'));events=[Event(**e) for e in saved['events']]
                info['analysis']={**saved['stats'],'cached':True,'seconds':time.perf_counter()-started}
                report(1,'저장된 동시 분석 결과 불러옴','cache');return info,events
            except (OSError,ValueError,TypeError,KeyError):pass
        patterns=[]
        for template in settings.templates:
            image=cv2.imdecode(np.fromfile(template['path'],np.uint8),cv2.IMREAD_GRAYSCALE)
            if image is None or image.std()<3:raise ValueError('기준 이미지를 확인하세요.')
            patterns.append((template['kind'],image,Path(template['path']).stem))
        if not patterns:raise ValueError('기준 이미지가 없습니다.')
        fallback=None
        try:
            events,stats=self._stream(path,info,settings,patterns,cancel,report,'d3d11va' if os.name=='nt' else 'cpu')
        except DecoderFailure as error:
            if os.name!='nt':raise
            fallback=str(error);report(0,'GPU 추출 실패 · CPU로 처음부터 다시 분석합니다.')
            events,stats=self._stream(path,info,settings,patterns,cancel,report,'cpu')
        check_cancel(cancel)
        stats.update(seconds=time.perf_counter()-started,cached=False,audio_fast=False,audio_windows=None,fallback=fallback)
        info['analysis']=stats
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.data_dir,suffix='.tmp',delete=False) as out:
                temporary=Path(out.name);json.dump(dict(events=[asdict(e) for e in events],stats=stats),out,ensure_ascii=False)
            temporary.replace(cache)
        finally:
            if temporary:temporary.unlink(missing_ok=True)
        report(1,f"분석 완료 · 검사 {stats['analyzed_frames']}장 · 녹다운/처치 {len(events)}개 · {stats['seconds']:.1f}초")
        return info,events

    def _stream(self,path,info,settings,patterns,cancel,report,decoder,start=0,end=None):
        base=self.data_dir.resolve();base.mkdir(parents=True,exist_ok=True)
        tracker=Tracker(max(.8,2.1/settings.sample_fps));count=0;match_seconds=0.;overlapped=False
        length=(info['duration'] if end is None else end)-start
        estimate=max(1,math.ceil(length*settings.sample_fps));state={'frames':0};last_report=0
        with ImageTemporaryDirectory(prefix='png_stream_',dir=base) as folder:
            directory=Path(folder).resolve()
            if not directory.is_relative_to(base):raise ValueError('임시 저장 경로 오류')
            args=[ffmpeg(),'-hide_banner','-nostdin','-n','-loglevel','warning','-stats_period','0.5','-progress','pipe:1','-nostats']
            if decoder!='cpu':args+=['-hwaccel',decoder]
            if start:args+=['-ss',f'{start:.6f}']
            args+=['-i',str(path),'-map','0:v:0','-an','-sn','-dn','-vf',f"setpts=PTS-STARTPTS,fps=fps={settings.sample_fps}:start_time=0:round=near,scale=w='min(iw,1280)':h=-2",'-fps_mode','passthrough','-c:v','png','-compression_level','3','-atomic_writing','1','-start_number','1',str(directory/'frame_%08d.png')]
            if end is not None:args[-1:-1]=['-t',f'{length:.6f}']
            begin=time.perf_counter();proc=None;reader=None
            with (directory/'ffmpeg.log').open('w+b') as log:
                try:
                    proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    def read_progress():
                        for raw in iter(proc.stdout.readline,b''):
                            key,_,value=raw.decode(errors='replace').strip().partition('=')
                            if key=='frame':
                                try:state['frames']=int(value)
                                except ValueError:pass
                    reader=threading.Thread(target=read_progress,daemon=True);reader.start()
                    while True:
                        check_cancel(cancel);image_path=directory/f'frame_{count+1:08d}.png'
                        if image_path.exists():
                            tick=time.perf_counter();frame=read_completed_image(image_path,lambda:check_cancel(cancel))
                            if frame is None:raise ValueError('추출된 PNG를 읽지 못했습니다.')
                            if frame.shape[1]!=1280:frame=normalized_frame(frame)
                            h,w=frame.shape[:2];x1,y1,x2,y2=settings.roi;crop=cv2.cvtColor(frame[int(y1*h):int(y2*h),int(x1*w):int(x2*w)],cv2.COLOR_BGR2GRAY)
                            candidates=[]
                            for kind,pattern,name in patterns:
                                if crop.shape[0]>=pattern.shape[0] and crop.shape[1]>=pattern.shape[1]:
                                    score=float(cv2.minMaxLoc(cv2.matchTemplate(crop,pattern,cv2.TM_CCOEFF_NORMED))[1])
                                    if score>=max(.8,settings.confidence):candidates.append((kind,f'기준 이미지: {name}',score))
                            tracker.feed(start+count/settings.sample_fps,candidates);count+=1
                            match_seconds+=time.perf_counter()-tick;overlapped|=proc.poll() is None
                            retry_access(image_path.unlink,lambda:check_cancel(cancel))
                        elif proc.poll() is not None:
                            # Check once more after process exit to cover a final rename.
                            if image_path.exists():continue
                            break
                        else:
                            if cancel:cancel.wait(.03)
                            else:time.sleep(.03)
                        if time.monotonic()-last_report>.25:
                            extracted=max(count,state['frames']);message=f'{Path(path).name} · 추출 {extracted}장 / 검사 {count}장 · 후보 {len(tracker.events)}개'
                            report(min(.999,count/estimate),message);last_report=time.monotonic()
                    reader.join(timeout=3);code=proc.wait()
                    if code:
                        log.seek(0);detail=log.read().decode('utf-8',errors='replace')[-1500:]
                        raise DecoderFailure(f'FFmpeg {decoder} 종료 {code}: {detail}')
                    if count==0 or count!=state['frames']:raise ValueError(f'추출·검사 수 불일치: {state["frames"]}/{count}')
                finally:
                    if proc is not None:
                        if proc.poll() is None:
                            proc.terminate()
                            try:proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        if reader:reader.join(timeout=3)
                        proc.stdout.close()
        return tracker.events,dict(note='FFmpeg PNG 추출 + OpenCV 동시 검사',decoder=decoder,analyzed_frames=count,extracted_frames=count,total_samples=estimate,match_seconds=match_seconds,overlapped=overlapped,pipeline_seconds=time.perf_counter()-begin)

