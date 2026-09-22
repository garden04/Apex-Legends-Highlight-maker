"""Exact-grid D3D11 decoding; selected frames retain the legacy CPU conversion."""
import os,queue,subprocess,tempfile,threading,time
import cv2
import numpy as np

class HardwareUnavailable(RuntimeError):pass

class HardwareSampler:
    def __init__(self,path,info,sample_fps,cancel=None):
        from core import ffmpeg,check_cancel
        check_cancel(cancel)
        step=info['fps']/sample_fps
        if abs(step-round(step))>1e-8 or step<1:raise HardwareUnavailable('non-integer sampling grid')
        self.step=round(step);self.fps=info['fps'];self.info=info;self.cancel=cancel
        self.grabbed=0;self.seeks=0;self.current=-1;self.frame=None;self.backend='D3D11 GPU'
        self.proc=None;self.stop=threading.Event();self.queue=queue.Queue(2);self.log=None
        # FFmpeg reports source range; only the tested 8-bit 4:2:0 path is enabled.
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        probe=subprocess.run([ffmpeg(),'-hide_banner','-i',str(path)],capture_output=True,creationflags=flags,timeout=15)
        video=next((line for line in probe.stderr.decode(errors='replace').splitlines() if 'Video:' in line),'')
        if not any(codec in video for codec in ('Video: h264','Video: hevc')) or not any(fmt in video for fmt in ('yuv420p','yuvj420p')) or '10le' in video:
            raise HardwareUnavailable('unsupported codec/pixel format')
        rng='pc' if 'yuvj420p' in video or '(pc' in video else 'tv'
        vf=f'select=not(mod(n\\,{self.step})),hwdownload,format=nv12,format=yuv420p,scale=in_color_matrix=bt601:in_range={rng}:out_range=pc:flags=bicubic,format=bgr24'
        args=[ffmpeg(),'-v','error','-nostdin','-hwaccel','d3d11va','-hwaccel_output_format','d3d11','-i',str(path),'-map','0:v:0','-an','-sn','-vf',vf,'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','bgr24','-']
        self.log=tempfile.TemporaryFile()
        try:
            self.proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=self.log,creationflags=flags)
            self.thread=threading.Thread(target=self._pump,daemon=True);self.thread.start()
            first=self._next()
            cap=cv2.VideoCapture(str(path))
            try:ok,reference=cap.read()
            finally:cap.release()
            if not ok or first is None or not np.array_equal(first,reference):raise HardwareUnavailable('first-frame pixels differ')
            self.frame=first;self.current=0;self.grabbed=1
        except BaseException:
            self.close();raise

    def _put(self,item):
        while not self.stop.is_set():
            try:self.queue.put(item,timeout=.1);return
            except queue.Full:pass

    def _pump(self):
        size=self.info['width']*self.info['height']*3
        try:
            while not self.stop.is_set():
                data=bytearray(size);view=memoryview(data);offset=0
                while offset<size:
                    count=self.proc.stdout.readinto(view[offset:])
                    if not count:break
                    offset+=count
                if offset==0:break
                if offset!=size:raise HardwareUnavailable('incomplete GPU frame')
                self._put(np.frombuffer(data,np.uint8).reshape(self.info['height'],self.info['width'],3))
            if not self.stop.is_set():
                code=self.proc.wait(timeout=10)
                if code:raise HardwareUnavailable(f'GPU decoder exit {code}')
                self._put(None)
        except Exception as error:self._put(error)

    def _next(self):
        from core import check_cancel
        started=time.monotonic()
        while True:
            check_cancel(self.cancel)
            try:
                result=self.queue.get(timeout=.1)
                if isinstance(result,Exception):raise HardwareUnavailable(str(result))
                return result
            except queue.Empty:
                if time.monotonic()-started>30:raise HardwareUnavailable('GPU decoder timeout')

    def read(self,seconds):
        target=round(seconds*self.fps)
        if target<self.current or target%self.step:raise HardwareUnavailable('sampling grid mismatch')
        if target>=round(self.info['duration']*self.fps):return False,None
        while self.current<target:
            frame=self._next()
            if frame is None:raise HardwareUnavailable('GPU stream ended early')
            self.frame=frame;self.current+=self.step;self.grabbed=self.current+1
        return True,self.frame

    def close(self):
        self.stop.set()
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            if hasattr(self,'thread'):self.thread.join(timeout=3)
            self.proc.stdout.close()
        if self.log is not None:self.log.close()

class CompatibleSampler:
    def __init__(self,path,info,cancel,sample_fps):
        from core import FrameSampler,Cancelled
        self.path=path;self.info=info;self.cancel=cancel
        try:self.reader=HardwareSampler(path,info,sample_fps,cancel)
        except Cancelled:raise
        except Exception:self.reader=FrameSampler(path,info,cancel)
    @property
    def backend(self):return getattr(self.reader,'backend','CPU (호환 경로)')
    @property
    def grabbed(self):return self.reader.grabbed
    @property
    def seeks(self):return self.reader.seeks
    def read(self,seconds):
        from core import FrameSampler,Cancelled
        try:return self.reader.read(seconds)
        except Cancelled:raise
        except Exception:
            if not isinstance(self.reader,HardwareSampler):raise
            self.reader.close();self.reader=FrameSampler(self.path,self.info,self.cancel)
            return self.reader.read(seconds)
    def close(self):self.reader.close()
