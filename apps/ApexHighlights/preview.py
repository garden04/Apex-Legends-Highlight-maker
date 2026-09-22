"""Native Qt playback; no Python frame decoding on the playback path."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QUrl, QTimer, Signal, QRectF, QSignalBlocker
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QSlider,QLabel,
                              QComboBox,QStackedWidget)
from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from core import metadata, run_ffmpeg, check_cancel


def proxy_path(source, directory):
    p=Path(source).resolve();stat=p.stat()
    identity=json.dumps([1,str(p),stat.st_size,stat.st_mtime_ns,'1080p60-h264'])
    return Path(directory)/(hashlib.sha256(identity.encode()).hexdigest()+'.mp4')


def make_proxy(source,directory,cancel=None,progress=lambda p,m:None):
    destination=proxy_path(source,directory)
    check_cancel(cancel)
    if destination.exists():
        return str(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    info=metadata(source)
    scale=min(1,1920/info['width'],1080/info['height'])
    w=max(2,int(info['width']*scale)//2*2);h=max(2,int(info['height']*scale)//2*2)
    fps=min(60,info['fps'])
    partial=destination.with_suffix('.partial.mp4')
    progress(0,'1080p 미리보기 준비 중 · 다음부터는 저장된 미리보기를 사용합니다')
    try:
        run_ffmpeg(['-i',str(source),'-map','0:v:0','-map','0:a:0?',
                    '-vf',f'scale={w}:{h}:flags=bilinear,setsar=1','-r',str(fps),
                    '-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p',
                    '-c:a','aac','-b:a','160k','-af','aresample=48000:async=1:first_pts=0',
                    '-movflags','+faststart',str(partial)],cancel)
        check_cancel(cancel)
        # Never use a broken proxy silently.
        result=metadata(partial)
        if abs(result['duration']-info['duration']) > max(.15,2/fps):
            raise ValueError('미리보기와 원본 길이가 다릅니다. 원본 재생을 사용하세요.')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    progress(1,'1080p 미리보기 준비 완료')
    return str(destination)


class CalibrationCanvas(QWidget):
    selected=Signal(str,list)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.image=QImage();self.roi=[.38,.69,.75,.78]
        self.mode=None;self.start=None;self.end=None
        self.setMinimumSize(320,180)

    def set_frame(self,frame,roi,mode):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self.roi=list(roi);self.mode=mode;self.start=self.end=None;self.update()

    def image_rect(self):
        if self.image.isNull():return QRectF()
        size=self.image.size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height())

    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor('#090e16'))
        rect=self.image_rect()
        if rect.isEmpty():return
        painter.drawImage(rect,self.image)
        def box(coords,color):
            a,b,c,d=coords
            painter.setPen(QPen(QColor(color),2))
            painter.drawRect(QRectF(rect.x()+a*rect.width(),rect.y()+b*rect.height(),(c-a)*rect.width(),(d-b)*rect.height()))
        box(self.roi,'#53d4a5')
        if self.start and self.end:
            box([min(self.start[0],self.end[0]),min(self.start[1],self.end[1]),max(self.start[0],self.end[0]),max(self.start[1],self.end[1])],'#ffb657')

    def point(self,event):
        rect=self.image_rect();p=event.position()
        return (max(0,min(1,(p.x()-rect.x())/max(1,rect.width()))),max(0,min(1,(p.y()-rect.y())/max(1,rect.height()))))

    def mousePressEvent(self,event):
        if self.mode and event.button()==Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
            self.start=self.end=self.point(event);self.update()

    def mouseMoveEvent(self,event):
        if self.start:self.end=self.point(event);self.update()

    def mouseReleaseEvent(self,event):
        if not self.start:return
        self.end=self.point(event)
        a,b=self.start;c,d=self.end
        box=[min(a,c),min(b,d),max(a,c),max(b,d)]
        self.start=self.end=None
        if box[2]-box[0]>=.015 and box[3]-box[1]>=.01:
            self.selected.emit(self.mode,box)
        self.update()


class Preview(QWidget):
    message=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.player=QMediaPlayer(self)
        self.audio=QAudioOutput(self);self.audio.setVolume(.5)
        self.player.setAudioOutput(self.audio)
        self.video=QVideoWidget();self.video.setMinimumSize(320,180)
        self.player.setVideoOutput(self.video)
        self.calibration=CalibrationCanvas()
        self.stack=QStackedWidget();self.stack.addWidget(self.video);self.stack.addWidget(self.calibration)
        self.pending=None;self.expected_source=QUrl();self.segment_start=0;self.segment_end=None
        self.dragging=False;self.resume_after_drag=False
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.stack,1)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(0,0);self.slider.setSingleStep(100);self.slider.setPageStep(1000)
        layout.addWidget(self.slider)
        row=QHBoxLayout();layout.addLayout(row)
        self.play_button=QPushButton('재생');self.play_button.clicked.connect(self.toggle)
        row.addWidget(self.play_button)
        self.time_label=QLabel('0.00 / 0.00초');row.addWidget(self.time_label);row.addStretch(1)
        self.speed=QComboBox();self.speed.addItems(['0.5×','1×','1.5×','2×']);self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(lambda i:self.player.setPlaybackRate([.5,1,1.5,2][i]))
        row.addWidget(self.speed)
        self.mute=QPushButton('음소거');self.mute.setCheckable(True);self.mute.toggled.connect(self.audio.setMuted)
        row.addWidget(self.mute)
        self.volume=QSlider(Qt.Orientation.Horizontal);self.volume.setRange(0,100);self.volume.setValue(50);self.volume.setMaximumWidth(90)
        self.volume.valueChanged.connect(lambda v:self.audio.setVolume(v/100));row.addWidget(self.volume)
        self.slider.sliderPressed.connect(self._drag_start)
        self.slider.sliderReleased.connect(self._drag_end)
        self.slider.sliderMoved.connect(lambda ms:self._time(ms))
        self.slider.valueChanged.connect(self._value_changed)
        self.player.positionChanged.connect(self._position)
        self.player.durationChanged.connect(lambda ms:self.slider.setRange(0,ms))
        self.player.mediaStatusChanged.connect(self._status)
        self.player.playbackStateChanged.connect(lambda state:self.play_button.setText('일시정지' if state==QMediaPlayer.PlaybackState.PlayingState else '재생'))
        self.player.errorOccurred.connect(lambda error,text:self.message.emit('재생 오류: '+text+' · 1080p 미리보기를 사용할 수 있습니다.'))
        self.end_timer=QTimer(self);self.end_timer.setInterval(20);self.end_timer.timeout.connect(self._check_end);self.end_timer.start()

    def set_media(self,path,start=0,end=None,autoplay=False):
        self.stack.setCurrentWidget(self.video)
        self.player.pause()
        self.segment_start=round(start*1000);self.segment_end=None if end is None else round(end*1000)
        url=QUrl.fromLocalFile(str(Path(path).resolve()))
        self.expected_source=url
        if self.player.source()==url and self.player.mediaStatus()==QMediaPlayer.MediaStatus.LoadingMedia:
            self.pending=(self.segment_start,autoplay)
        elif self.player.source()==url and self.player.mediaStatus() not in (QMediaPlayer.MediaStatus.NoMedia,QMediaPlayer.MediaStatus.InvalidMedia):
            self.pending=None;self.player.setPosition(self.segment_start)
            if autoplay:self.player.play()
        else:
            self.pending=(self.segment_start,autoplay)
            self.player.setSource(url)

    def _status(self,status):
        if self.pending and self.player.source()==self.expected_source and status in (QMediaPlayer.MediaStatus.LoadedMedia,QMediaPlayer.MediaStatus.BufferedMedia):
            position,autoplay=self.pending;self.pending=None
            self.player.setPosition(position)
            if autoplay:self.player.play()
        if status==QMediaPlayer.MediaStatus.EndOfMedia:self.play_button.setText('재생')

    def toggle(self):
        if self.player.source().isEmpty():return
        self.stack.setCurrentWidget(self.video)
        if self.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState:self.player.pause()
        else:
            if self.segment_end is not None and (self.player.position()<self.segment_start or self.player.position()>=self.segment_end-30):
                self.player.setPosition(self.segment_start)
            elif self.player.position()>=self.player.duration()-30:self.player.setPosition(0)
            if self.pending:self.pending=(self.pending[0],True)
            else:self.player.play()

    def _time(self,ms):
        self.time_label.setText(f'{ms/1000:.2f} / {self.player.duration()/1000:.2f}초')

    def _position(self,ms):
        if not self.dragging:
            with QSignalBlocker(self.slider):self.slider.setValue(ms)
            self._time(ms)
        self._check_end()

    def _check_end(self):
        if self.segment_end is not None and self.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState and self.player.position()>=self.segment_end:
            self.player.pause();self.player.setPosition(self.segment_end)

    def seek_to(self,ms):
        self.stack.setCurrentWidget(self.video)
        self.segment_end=None
        if self.pending:self.pending=(ms,self.pending[1])
        else:self.player.setPosition(ms)

    def _drag_start(self):
        self.dragging=True
        self.resume_after_drag=self.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState
        self.player.pause()

    def _drag_end(self):
        self.dragging=False;self.seek_to(self.slider.value())
        if self.resume_after_drag:self.player.play()

    def _value_changed(self,ms):
        if not self.dragging:self.seek_to(ms)

    def clear(self):
        self.pending=None;self.segment_end=None;self.player.stop();self.player.setSource(QUrl())
        self.stack.setCurrentWidget(self.video);self._time(0)
