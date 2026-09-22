"""Offline video analysis and non-destructive highlight export."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

VERSION = 6
RESOLUTIONS = {"FHD": (1920, 1080), "QHD": (2560, 1440), "4K": (3840, 2160)}
FRAME_RATES = ("source", "30", "60", "120")


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    sample_fps: float = 2.0
    before: float = 6.5
    after: float = 1.0
    merge_gap: float = 3.0
    confidence: float = .65
    roi: list[float] = field(default_factory=lambda: [.25, .50, .80, .84])
    mode: str = "both"
    detector: str = "ocr"
    language: str = "ko"
    templates: list[dict] = field(default_factory=list)
    output_resolution: str = "FHD"
    output_fps: str = "60"
    audio_fast: bool = False
    audio_threshold_db: float = -24.0
    audio_padding: float = 3.0
    damage_fast: bool = False
    damage_exclude_unknown: bool = False
    damage_roi: list[float] = field(default_factory=lambda: [.88,.075,.94,.12])

    def validate(self):
        values = [self.sample_fps, self.before, self.after, self.merge_gap, self.confidence, self.audio_threshold_db, self.audio_padding, *self.roi]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("설정에는 유한한 숫자만 입력하세요.")
        if not .5 <= self.sample_fps <= 30:
            raise ValueError("분석 빈도는 초당 0.5~30장이어야 합니다.")
        if min(self.before, self.after, self.merge_gap) < 0 or self.before + self.after <= 0:
            raise ValueError("앞뒤 길이는 0 이상이고 합계는 0보다 커야 합니다.")
        if not 0 < self.confidence <= 1:
            raise ValueError("인식 기준은 0 초과, 1 이하여야 합니다.")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("알림 영역을 다시 지정하세요.")
        if self.mode not in ("both", "knock", "kill") or self.detector not in ("ocr", "template", "both"):
            raise ValueError("지원하지 않는 분석 설정입니다.")
        validate_output(self.output_resolution, self.output_fps)
        if not isinstance(self.audio_fast, bool):
            raise ValueError("오디오 고속 모드 설정이 잘못되었습니다.")
        if not -60<=self.audio_threshold_db<=0 or not 0<=self.audio_padding<=10:
            raise ValueError('소리 기준은 -60~0 dBFS, 앞뒤 여유는 0~10초여야 합니다.')
        if not isinstance(self.damage_fast,bool) or not isinstance(self.damage_exclude_unknown,bool):
            raise ValueError('데미지 고속 모드 설정이 잘못되었습니다.')
        if len(self.damage_roi)!=4 or not all(math.isfinite(x) for x in self.damage_roi):raise ValueError('데미지 영역을 확인하세요.')
        a,b,c,d=self.damage_roi
        if not (0<=a<c<=1 and 0<=b<d<=1):raise ValueError('데미지 영역을 다시 지정하세요.')


def validate_output(resolution, fps):
    if resolution not in RESOLUTIONS or fps not in FRAME_RATES:
        raise ValueError("해상도는 FHD·QHD·4K, FPS는 원본 유지·30·60·120 중 선택하세요.")


@dataclass
class Event:
    time: float
    kind: str
    text: str
    confidence: float
    last_seen: float | None = None


@dataclass
class Segment:
    source: str
    start: float
    end: float


def check_cancel(cancel):
    if cancel and cancel.is_set():
        raise Cancelled("작업을 취소했습니다.")


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(cap.get(3)), int(cap.get(4))
        if not cap.isOpened() or fps <= 0 or count <= 0 or min(width, height) <= 0:
            raise ValueError(f"영상을 읽을 수 없습니다: {path}")
        return dict(fps=fps, duration=count/fps, width=width, height=height)
    finally:
        cap.release()


def read_frame(path, seconds):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds)*1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError("이 시점의 화면을 읽지 못했습니다. 시간을 조금 앞당겨 주세요.")
        return frame
    finally:
        cap.release()


def normalized_frame(frame):
    return cv2.resize(frame, (1280, max(1, round(frame.shape[0]*1280/frame.shape[1]))))


def classify(text):
    flat = re.sub(r"\s+", "", text).upper()
    for kind, words in (("knock", ("KNOCKEDDOWN", "KNOCKDOWN", "녹다운", "기절시킴")),
                        ("kill", ("ELIMINATED", "처치", "제거했습니다"))):
        if any(word in flat for word in words):
            return kind
    return None


class Tracker:
    """Confirm a single hit and extend repeated observations of the same notification."""
    def __init__(self, gap):
        self.gap = gap
        self.tracks = {}
        self.events = []

    def feed(self, time, candidates):
        for key in list(self.tracks):
            if time - self.tracks[key]["last"] > self.gap:
                del self.tracks[key]
        unique = {}
        for kind, text, score in candidates:
            key = (kind, re.sub(r"[^\w]", "", text).upper())
            if key not in unique or score > unique[key][2]:
                unique[key] = (kind, text, score)
        for key, (kind, text, score) in unique.items():
            track = self.tracks.get(key)
            if track is None:
                self.tracks[key] = dict(first=time, last=time, hits=1, emitted=True, event_index=len(self.events))
                self.events.append(Event(time,kind,text,score,time))
                continue
            track["hits"] += 1
            track["last"] = time
            if track["emitted"]:
                self.events[track["event_index"]].last_seen = time



def make_segments(source, events, duration, settings):
    settings.validate()
    ranges = []
    for e in sorted(events, key=lambda x: x.time):
        if settings.mode != "both" and e.kind != settings.mode:
            continue
        # A template may stay visible across consecutive opponents. Keep the whole
        # notification interval instead of truncating later action after its first onset.
        tail = e.last_seen if e.last_seen is not None else e.time
        a, b = max(0, e.time-settings.before), min(duration, max(e.time, tail)+settings.after)
        if b <= a:
            continue
        if ranges and a <= ranges[-1].end + settings.merge_gap:
            ranges[-1].end = max(ranges[-1].end, b)
        else:
            ranges.append(Segment(str(source), a, b))
    return ranges


def recommended_settings():
    """Calibrated Korean HUD preset; user-drawn profiles still override it."""
    root = Path(__file__).resolve().parent
    preset = root / "profiles" / "default.json"
    if not preset.exists():
        return Settings()
    payload = json.loads(preset.read_text(encoding="utf-8"))
    settings = Settings(**payload)
    for template in settings.templates:
        template["path"] = str((root / template["path"]).resolve())
    settings.validate()
    return settings


class FrameSampler:
    """Decode each frame at most once in normal mode; convert only selected frames.

    The selected frame number matches OpenCV's time-seek rounding. Fast mode may
    seek once over a long excluded interval, never for every sample.
    """
    def __init__(self, path, info, cancel=None, allow_seek=False):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = info["fps"]
        self.frame_count = round(info["duration"] * self.fps)
        self.current = -1
        self.cancel = cancel
        self.allow_seek = allow_seek
        self.seeks = 0
        self.grabbed = 0

    def read(self, seconds):
        target = round(seconds * self.fps)
        if target >= self.frame_count:
            return False, None
        if target < self.current:
            raise ValueError("분석 시점은 시간 순서대로 처리해야 합니다.")
        if self.allow_seek and target-self.current > max(2*self.fps, 1):
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                raise ValueError("영상의 후보 구간으로 이동하지 못했습니다.")
            self.current = target-1
            self.seeks += 1
        while self.current < target:
            check_cancel(self.cancel)
            if not self.cap.grab():
                return False, None
            self.current += 1
            self.grabbed += 1
        return self.cap.retrieve()

    def close(self):
        self.cap.release()


class WaveSamples:
    """Read only the PCM samples needed for one analysis batch."""
    def __init__(self,audio):
        if audio.getsampwidth()!=2 or audio.getnchannels()!=1:
            raise ValueError('분석용 오디오는 16비트 모노여야 합니다.')
        self.audio=audio

    def __len__(self):return self.audio.getnframes()

    def __getitem__(self,region):
        start,stop,step=region.indices(len(self))
        if step!=1:raise ValueError('연속 오디오 구간만 읽을 수 있습니다.')
        self.audio.setpos(start)
        return np.frombuffer(self.audio.readframes(stop-start),dtype='<i2').astype(np.float32)/32768


def audio_windows(samples, sample_rate, duration, cancel=None, progress=lambda p,m:None):
    """Broad transient / band-energy gate, not a classifier of kill sounds.

    Return None if silence, uninformative sound, or excessive coverage makes a
    filter unreliable/unhelpful. Windows are on the original video timeline.
    """
    check_cancel(cancel)
    if len(samples) < sample_rate//10:
        return None, "오디오가 너무 짧거나 읽기 어려워 전체 분석"
    peak = 0
    # 32 ms windows, 16 ms steps. Process batches to bound memory on long clips.
    size, hop = 512, 256
    if len(samples) < size:
        return None, "오디오 길이 부족 → 전체 분석"
    frame_count = (len(samples)-size)//hop+1
    window = np.hanning(size).astype(np.float32)
    bands = [(200, 1200), (1200, 3500), (3500, 7500)]
    frequencies = np.fft.rfftfreq(size, 1/sample_rate)
    energies, fluxes = [], []
    previous = None
    last_report = 0
    for start in range(0, frame_count, 2048):
        check_cancel(cancel)
        count=min(2048,frame_count-start)
        block=samples[start*hop:(start+count-1)*hop+size]
        if not np.isfinite(block).all():return None, "오디오가 너무 짧거나 읽기 어려워 전체 분석"
        peak=max(peak,float(np.max(np.abs(block))))
        frames=np.lib.stride_tricks.sliding_window_view(block,size)[::hop]
        spectrum = np.abs(np.fft.rfft(frames*window, axis=1))
        compressed = np.log1p(spectrum*10)
        prior = np.vstack([compressed[0] if previous is None else previous, compressed[:-1]])
        fluxes.append(np.maximum(0, compressed-prior).mean(axis=1))
        previous = compressed[-1]
        energies.append(np.column_stack([np.sqrt(np.mean(spectrum[:,(frequencies>=lo)&(frequencies<hi)]**2,axis=1)) for lo,hi in bands]))
        if time.monotonic()-last_report >= .5:
            progress(.9*min(1,(start+2048)/frame_count), f'오디오 후보 분석 중 · {min(duration,(start+count)*hop/sample_rate):.0f} / {duration:.0f}초')
            last_report = time.monotonic()
    tail=samples[(frame_count-1)*hop+size:len(samples)]
    if len(tail):
        if not np.isfinite(tail).all():return None, "오디오가 너무 짧거나 읽기 어려워 전체 분석"
        peak=max(peak,float(np.max(np.abs(tail))))
    if peak < .0001:return None, "무음 또는 매우 작은 소리 → 전체 분석"
    energy = np.concatenate(energies)
    flux = np.concatenate(fluxes)
    metrics = np.column_stack([flux, energy])
    active = np.zeros(frame_count, dtype=bool)
    for metric_index,metric in enumerate(metrics.T):
        check_cancel(cancel)
        progress(.9+.02*metric_index,'오디오 후보 기준 계산 중')
        median = np.median(metric)
        mad = np.median(np.abs(metric-median))
        threshold = max(median+3*mad, np.percentile(metric,80), float(metric.max())*.06, 1e-7)
        # Include strong band energy as well as onsets to cover sustained combat.
        active |= metric > threshold
    indices = np.flatnonzero(active)
    if not len(indices):
        return None, "뚜렷한 소리 후보 없음 → 전체 분석"
    intervals = []
    for index in indices:
        if index%4096==0:check_cancel(cancel)
        t = (index*hop+size/2)/sample_rate
        a, b = max(0,t-3), min(duration,t+3)
        if intervals and a <= intervals[-1][1]+.25:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a,b])
    coverage = sum(b-a for a,b in intervals)/max(duration,.001)
    progress(1,'오디오 후보 분석 완료')
    if coverage >= .85:
        return None, "소리 후보가 영상 대부분을 차지해 전체 분석"
    return intervals, f"오디오 후보 {len(intervals)}구간 · 화면 분석 범위 약 {coverage:.0%}"


def peak_envelope(samples, sample_rate, cancel=None, progress=lambda p,m:None):
    """20 ms sample peaks, with bounded reads and no spectral analysis."""
    block=max(1,round(sample_rate*.02)); count=math.ceil(len(samples)/block)
    peaks=np.empty(count,dtype=np.float32)
    last_report=0
    for offset in range(0,count,1600):
        check_cancel(cancel)
        n=min(1600,count-offset)
        chunk=np.asarray(samples[offset*block:min(len(samples),(offset+n)*block)],dtype=np.float32)
        if not np.isfinite(chunk).all():raise ValueError('invalid audio samples')
        if len(chunk)<n*block:chunk=np.pad(chunk,(0,n*block-len(chunk)))
        peaks[offset:offset+n]=np.abs(chunk).reshape(n,block).max(axis=1)
        if time.monotonic()-last_report>=.5:
            progress((offset+n)/max(count,1),f'음량 피크 분석 중 · {min(len(samples),(offset+n)*block)/sample_rate:.0f} / {len(samples)/sample_rate:.0f}초')
            last_report=time.monotonic()
    check_cancel(cancel)
    progress(1,'음량 피크 분석 완료')
    return peaks


def peak_windows(peaks,duration,threshold_db=-24.0,padding=3.0,cancel=None):
    check_cancel(cancel)
    active=np.asarray(peaks)>=10**(threshold_db/20)
    boundaries=np.diff(np.r_[False,active,False].astype(np.int8))
    starts=np.flatnonzero(boundaries==1);ends=np.flatnonzero(boundaries==-1)
    intervals=[]
    for i,(start,end) in enumerate(zip(starts,ends)):
        if i%4096==0:check_cancel(cancel)
        a=max(0,float(start)*.02-padding);b=min(duration,float(end)*.02+padding)
        if b<=a:continue
        if intervals and a<=intervals[-1][1]+.25:intervals[-1][1]=max(intervals[-1][1],b)
        else:intervals.append([a,b])
    if not intervals:return None,'음량 기준을 넘는 후보 없음 → 전체 분석'
    coverage=sum(b-a for a,b in intervals)/max(duration,.001)
    if coverage>=.85:return None,'큰 소리 구간이 영상 대부분을 차지해 전체 분석'
    return intervals,f'음량 후보 {len(intervals)}구간 · 화면 분석 범위 약 {coverage:.0%}'


def build_audio_windows(path, duration, cancel=None, stage_progress=lambda stage,p,m:None, cache_dir=None, force=False, threshold_db=-24.0, padding=3.0):
    check_cancel(cancel)
    if not math.isfinite(threshold_db) or not -60 <= threshold_db <= 0 or not math.isfinite(padding) or not 0 <= padding <= 10:
        raise ValueError('오디오 기준 범위를 확인하세요.')
    cache=None
    if cache_dir is not None:
        source=Path(path).resolve();stat=source.stat()
        key=hashlib.sha256(json.dumps(['peak-v1',str(source),stat.st_size,stat.st_mtime_ns,duration,16000,.02]).encode()).hexdigest()
        cache=Path(cache_dir)/(key+'.npz')
        if not force and cache.exists():
            try:
                with np.load(cache,allow_pickle=False) as saved:
                    peaks=saved['peaks']
                if peaks.ndim!=1 or len(peaks)!=math.ceil(duration/.02) or not np.isfinite(peaks).all() or np.any((peaks<0)|(peaks>1)):
                    raise ValueError('invalid peak cache')
                result=peak_windows(peaks,duration,threshold_db,padding,cancel)
                stage_progress('audio_scan',1,'저장된 음량으로 후보 계산 완료')
                return result
            except (OSError,ValueError,KeyError,TypeError,EOFError):pass
    try:
        if not has_audio(path):
            return None, "오디오 없음 → 전체 분석"
        with tempfile.TemporaryDirectory(prefix="apex_audio_") as scratch:
            audio_path = Path(scratch)/"analysis.wav"
            def extraction_progress():
                seconds=max(0,audio_path.stat().st_size-80)/32000 if audio_path.exists() else 0
                stage_progress('audio_extract',min(.999,seconds/max(duration,.001)),f'오디오 추출 중 · {min(seconds,duration):.0f} / {duration:.0f}초')
            stage_progress('audio_extract',0,'오디오 추출 준비 중')
            # Padding by timestamps preserves delayed starts and discontinuities.
            run_ffmpeg(["-i",str(path),"-map","0:a:0","-vn","-af","aresample=16000:async=1:first_pts=0,apad",
                        "-t",str(duration),"-ac","1","-ar","16000","-c:a","pcm_s16le",str(audio_path)], cancel, heartbeat=extraction_progress)
            stage_progress('audio_extract',1,'오디오 추출 완료')
            stage_progress('audio_scan',0,'분석용 오디오를 읽는 중')
            with wave.open(str(audio_path),'rb') as audio:
                sample_rate = audio.getframerate()
                peaks=peak_envelope(WaveSamples(audio),sample_rate,cancel,lambda p,m:stage_progress('audio_scan',p,m))
                result=peak_windows(peaks,duration,threshold_db,padding,cancel)
            check_cancel(cancel)
            if cache is not None:
                temporary=None
                try:
                    cache.parent.mkdir(parents=True,exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode='wb',dir=cache.parent,suffix='.tmp',delete=False) as output:
                        temporary=Path(output.name);np.savez(output,peaks=peaks)
                    temporary.replace(cache)
                except OSError:pass
                finally:
                    if temporary is not None:temporary.unlink(missing_ok=True)
            stage_progress('audio_scan',1,'오디오 후보 분석 완료')
            return result
    except Cancelled:
        raise
    except (RuntimeError, OSError, wave.Error, subprocess.TimeoutExpired):
        return None, "오디오를 읽지 못해 전체 분석"


class Analyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reader = None
        self.language = None

    def cache_key(self, path, settings):
        stat = Path(path).stat()
        config = asdict(settings)
        for name in ("before", "after", "merge_gap", "mode", "output_resolution", "output_fps"):
            config.pop(name)
        template_hashes = [hashlib.sha256(Path(t["path"]).read_bytes()).hexdigest() for t in settings.templates]
        raw = json.dumps([VERSION, str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, config, template_hashes], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def score_cache_path(self,path,settings):
        config=Settings(**asdict(settings))
        config.confidence=.5
        return self.data_dir/('template-scores-v1-'+self.cache_key(path,config)+'.json')

    def replay_scores(self,cache,settings,duration,cancel,progress):
        try:
            saved=json.loads(cache.read_text(encoding='utf-8'))
            rows=saved['rows'];stats=saved['stats']
            if not isinstance(rows,list) or len(rows)!=stats['analyzed_frames']:return None
            tracker=Tracker(max(.8,2.1/settings.sample_fps));previous=-1
            for index,(at,raw) in enumerate(rows):
                check_cancel(cancel)
                if not math.isfinite(at) or not previous<at<duration:return None
                previous=at
                if not isinstance(raw,list):return None
                for kind,text,score in raw:
                    if kind not in ('knock','kill') or not isinstance(text,str) or not math.isfinite(score) or not -1.001<=score<=1.001:return None
                tracker.feed(at,[(kind,text,score) for kind,text,score in raw if score>=max(.8,settings.confidence)])
                if index%256==0:progress((index+1)/max(1,len(rows)),'저장된 이미지 점수로 녹다운 판정 중')
            return tracker.events,stats
        except (OSError,ValueError,KeyError,TypeError):
            return None

    def analyze(self, path, settings, progress=lambda p, m: None, cancel=None, force=False, stage_progress=None):
        started = time.perf_counter()
        original_progress=progress
        stage='setup'
        def progress(p,m):
            original_progress(p,m)
            if stage_progress is not None:stage_progress(stage,p,m)
        progress(0,'분석 준비 중')
        settings.validate()
        info = metadata(path)
        cache = self.data_dir / (self.cache_key(path, settings)+".json")
        check_cancel(cancel)
        if cache.exists() and not force:
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                events = [Event(**e) for e in saved["events"]]
                info["analysis"] = {**saved["stats"], "cached": True, "seconds": time.perf_counter()-started}
                stage='cache';progress(1, "저장된 분석 결과 불러옴")
                return info, events
            except (ValueError, TypeError, KeyError):
                pass
        score_cache=self.score_cache_path(path,settings) if settings.detector=='template' and not settings.audio_fast else None
        if score_cache is not None and not force and score_cache.exists():
            replay=self.replay_scores(score_cache,settings,info['duration'],cancel,progress)
            if replay is not None:
                events,stats=replay
                info['analysis']={**stats,'cached':True,'score_cached':True,'seconds':time.perf_counter()-started,'note':'저장된 이미지 점수로 재판정'}
                cache.write_text(json.dumps(dict(events=[asdict(e) for e in events],stats=info['analysis']),ensure_ascii=False),encoding='utf-8')
                stage='cache';progress(1,'저장된 이미지 점수로 재판정 완료')
                return info,events
        score_rows=[]
        intervals, audio_note = None, "일반 모드 · 전체 구간 순차 분석"
        if settings.audio_fast:
            progress(0,"오디오 후보를 찾는 중")
            intervals, audio_note = build_audio_windows(path,info["duration"],cancel,stage_progress=stage_progress or (lambda stage,p,m:None),cache_dir=self.data_dir/'audio',force=force,threshold_db=settings.audio_threshold_db,padding=settings.audio_padding)
            progress(0,audio_note)
        use_ocr = settings.detector in ("ocr", "both")
        if use_ocr and (self.reader is None or self.language != settings.language):
            progress(0, "문자 인식 준비 중 · 첫 실행에는 인식 모델을 내려받습니다")
            import easyocr
            self.reader = easyocr.Reader(["ko", "en"] if settings.language == "ko" else ["en"], gpu=False,
                model_storage_directory=str(self.data_dir / "models"), verbose=False)
            self.language = settings.language
        patterns = []
        if settings.detector in ("template", "both"):
            for template in settings.templates:
                img = cv2.imdecode(np.fromfile(template["path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None or img.std() < 3:
                    raise ValueError("기준 이미지가 비어 있거나 단색입니다. 알림 글자를 포함해 다시 지정하세요.")
                patterns.append((template["kind"], img, Path(template["path"]).stem))
            if not patterns and not use_ocr:
                raise ValueError("미리보기에서 알림 기준 이미지를 먼저 등록하세요.")
        stage='image';progress(0,'화면 분석 준비 중')
        tracker = Tracker(max(.8, 2.1/settings.sample_fps))
        if os.name=='nt' and intervals is None and info['width']>=1920 and info['duration']>=30:
            from hardware_sampler import CompatibleSampler
            progress(0,'영상 디코더 준비 및 첫 화면 일치 확인 중')
            sampler=CompatibleSampler(path,info,cancel,settings.sample_fps)
        else:
            sampler = FrameSampler(path,info,cancel,allow_seek=intervals is not None)
        progress(0,f"{getattr(sampler,'backend','CPU')} · 화면 분석 시작")
        step = 1/settings.sample_fps
        samples = max(1, math.ceil(info["duration"]*settings.sample_fps))
        good_frames = 0
        requested_frames = 0
        interval_index = 0
        try:
            for i in range(samples):
                check_cancel(cancel)
                t = i*step
                if intervals is not None:
                    while interval_index < len(intervals) and t > intervals[interval_index][1]:
                        interval_index += 1
                    inside = interval_index < len(intervals) and intervals[interval_index][0] <= t <= intervals[interval_index][1]
                    # Follow an active notification beyond the audio window so its
                    # last_seen and the exported tail are not cut short by the gate.
                    tracking = any(t-track["last"] <= tracker.gap for track in tracker.tracks.values())
                    if not inside and not tracking:
                        if i % max(1,round(settings.sample_fps)) == 0:
                            progress((i+1)/samples,f"{Path(path).name} · 소리 후보 밖 건너뛰는 중 · {t:.1f}초")
                        continue
                requested_frames += 1
                ok, frame = sampler.read(t)
                if not ok:
                    continue
                good_frames += 1
                frame = normalized_frame(frame)
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = settings.roi
                crop = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
                candidates = []
                if use_ocr:
                    results = self.reader.readtext(crop, detail=1, paragraph=False)
                    # Group boxes into notification lines so names distinguish consecutive kills.
                    lines = []
                    for box, text, score in sorted(results, key=lambda r: (r[0][0][1], r[0][0][0])):
                        if score < settings.confidence:
                            continue
                        cy = sum(p[1] for p in box)/4
                        height = max(p[1] for p in box)-min(p[1] for p in box)
                        line = next((line for line in lines if abs(line[0]-cy) < max(10, height*.6)), None)
                        if line is None:
                            line = [cy, []]
                            lines.append(line)
                        line[1].append((box[0][0], text, float(score)))
                    for _, pieces in lines:
                        text = " ".join(p[1] for p in sorted(pieces))
                        kind = classify(text)
                        if kind:
                            candidates.append((kind, text, min(p[2] for p in pieces)))
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                raw_scores=[]
                for kind, pattern, name in patterns:
                    if gray.shape[0] >= pattern.shape[0] and gray.shape[1] >= pattern.shape[1]:
                        score = float(cv2.minMaxLoc(cv2.matchTemplate(gray, pattern, cv2.TM_CCOEFF_NORMED))[1])
                        if score_cache is not None:raw_scores.append((kind,f"기준 이미지: {name}",score))
                        if score >= max(.8, settings.confidence):
                            candidates.append((kind, f"기준 이미지: {name}", score))
                if score_cache is not None:score_rows.append((t,raw_scores))
                tracker.feed(t, candidates)
                progress((i+1)/samples, f"{Path(path).name} · {t:.1f} / {info['duration']:.1f}초 · 후보 {len(tracker.events)}개")
        finally:
            sampler.close()
        if requested_frames == 0 or good_frames < requested_frames*.8:
            raise ValueError("영상 프레임을 충분히 읽지 못했습니다. Steam에서 MP4로 다시 내보내 주세요.")
        check_cancel(cancel)
        # No detected notification is inconclusive: recover with full coverage.
        if intervals is not None and not tracker.events:
            progress(0,"오디오 후보에서 녹다운을 찾지 못해 전체 분석으로 재확인")
            fallback = Settings(**asdict(settings))
            fallback.audio_fast = False
            info, events = self.analyze(path,fallback,progress,cancel,force)
            info["analysis"] = {**info["analysis"], "audio_fast": True, "audio_windows": intervals,
                                "note": "후보에서 감지 없음 → 전체 분석으로 재확인", "seconds": time.perf_counter()-started}
            cache.write_text(json.dumps(dict(events=[asdict(e) for e in events],stats=info["analysis"]),ensure_ascii=False),encoding="utf-8")
            return info, events
        stats = dict(audio_fast=settings.audio_fast, audio_windows=intervals, note=audio_note,
                     requested_frames=requested_frames, analyzed_frames=good_frames, total_samples=samples,
                     decoder=getattr(sampler,'backend','CPU'), seeks=sampler.seeks, decoded_frames=sampler.grabbed, seconds=time.perf_counter()-started, cached=False)
        if score_cache is not None:
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.data_dir,suffix='.tmp',delete=False) as output:
                    temporary=Path(output.name)
                    json.dump(dict(rows=score_rows,stats=stats),output,ensure_ascii=False)
                check_cancel(cancel)
                temporary.replace(score_cache)
            except OSError:pass
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
        info["analysis"] = stats
        cache.write_text(json.dumps(dict(events=[asdict(e) for e in tracker.events],stats=stats), ensure_ascii=False), encoding="utf-8")
        progress(1,f"분석 완료 · 화면 {good_frames}/{samples}장 · {stats['seconds']:.1f}초 · {audio_note}")
        return info, tracker.events


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args, cancel=None, *, heartbeat=None):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen([ffmpeg(), "-hide_banner", "-nostdin", "-y", *args], stdout=subprocess.DEVNULL,
                                stderr=log, creationflags=flags)
        last_heartbeat=0
        try:
            while True:
                check_cancel(cancel)
                if heartbeat is not None and time.monotonic()-last_heartbeat>=.5:
                    heartbeat();last_heartbeat=time.monotonic()
                try:
                    code = proc.wait(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        log.seek(0)
        output = log.read().decode("utf-8", errors="replace")
        if code:
            raise RuntimeError("영상 출력에 실패했습니다.\n"+output[-2500:])


def has_audio(path):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True,
                       creationflags=flags, timeout=30)
    return bool(re.search(rb"Stream #.*Audio:", r.stderr))


def validate_folder_name(name):
    name = name.strip()
    if not name:
        return ''
    reserved = {'CON','PRN','AUX','NUL','CONIN$','CONOUT$',
                *('COM'+n for n in '123456789¹²³'), *('LPT'+n for n in '123456789¹²³')}
    if (name in {'.','..'} or name.endswith('.') or
            any(c in '<>:"/\\|?*' or ord(c)<32 for c in name) or
            name.split('.')[0].rstrip().upper() in reserved):
        raise ValueError('폴더 이름에 사용할 수 없는 이름이나 문자가 있습니다. 경로 대신 폴더 이름만 입력하세요.')
    return name


def create_export_folder(destination, name):
    name = validate_folder_name(name)
    if not name:
        return Path(tempfile.mkdtemp(prefix="highlights_", dir=destination))
    index = 1
    while True:
        folder = Path(destination) / (name if index == 1 else f'{name} ({index})')
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            index += 1


def export_segments(segments, destination, combined=True, progress=lambda p, m: None, cancel=None,
                    *, resolution="FHD", output_fps="60", folder_name=""):
    folder_name = validate_folder_name(folder_name)
    validate_output(resolution, output_fps)
    if not segments:
        raise ValueError("내보낼 구간이 없습니다.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_info = {}
    for s in segments:
        if s.source not in source_info:
            source_info[s.source] = metadata(s.source)
        if not all(math.isfinite(v) for v in (s.start, s.end)) or not 0 <= s.start < s.end <= source_info[s.source]["duration"]+.05:
            raise ValueError(f"구간 시간이 잘못되었습니다: {s.start}~{s.end}")
    first = source_info[segments[0].source]
    width, height = RESOLUTIONS[resolution]
    outputs = []
    # Every export gets an independent folder; originals and previous exports are never overwritten.
    session = create_export_folder(destination, folder_name)
    try:
        with tempfile.TemporaryDirectory(prefix=".render_", dir=session) as scratch:
            scratch = Path(scratch)
            audio_cache = {}
            for i, s in enumerate(segments):
                check_cancel(cancel)
                source_fps = first["fps"] if combined else source_info[s.source]["fps"]
                fps = source_fps if output_fps == "source" else int(output_fps)
                progress(i/(len(segments)+int(combined)), f"구간 {i+1}/{len(segments)} · {resolution} / {fps:g} FPS 출력 중")
                if s.source not in audio_cache:
                    audio_cache[s.source] = has_audio(s.source)
                args = ["-ss", str(s.start), "-i", s.source]
                audio = audio_cache[s.source]
                if not audio:
                    args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
                args += ["-t", str(s.end-s.start), "-map", "0:v:0", "-map", "0:a:0" if audio else "1:a:0",
                         "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                         "-r", str(fps), "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                         "-af", "aresample=48000:async=1:first_pts=0,apad", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                         "-movflags", "+faststart"]
                part = scratch / f"part_{i:04d}.mp4"
                run_ffmpeg([*args, str(part)], cancel)
                if not combined:
                    target = session / f"{i+1:03d}_{Path(s.source).stem}.mp4"
                    part.replace(target)
                    outputs.append(str(target))
            if combined:
                check_cancel(cancel)
                progress(len(segments)/(len(segments)+1), "합본 영상 연결 중")
                listing = scratch / "concat.txt"
                listing.write_text("".join(f"file 'part_{i:04d}.mp4'\n" for i in range(len(segments))), encoding="utf-8")
                final = scratch / "highlight.mp4"
                run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(final)], cancel)
                target = session / "highlight.mp4"
                final.replace(target)
                outputs.append(str(target))
        (session / "cuts.json").write_text(json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")
        (session / "output_settings.json").write_text(json.dumps(dict(resolution=resolution, width=width, height=height,
            fps_mode=output_fps, combined=combined,
            fps_by_segment=[(first["fps"] if combined else source_info[s.source]["fps"]) if output_fps == "source" else int(output_fps)
                            for s in segments]), indent=2), encoding="utf-8")
        progress(1, "내보내기 완료")
        return outputs
    except BaseException:
        # Keep completed individual clips on cancellation. Incomplete temporary files are cleaned above.
        if not any(session.iterdir()):
            session.rmdir()
        raise
