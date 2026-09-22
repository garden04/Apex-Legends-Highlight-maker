"""Qt desktop UI. Detection, segmentation and export use unchanged core.py."""
from __future__ import annotations
import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import cv2
from PySide6.QtCore import Qt,QTimer,QSignalBlocker
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QLabel,QPushButton,QListWidget,QDoubleSpinBox,QComboBox,QCheckBox,QTableWidget,QTableWidgetItem,QLineEdit,
    QAbstractItemView,QHeaderView,QSplitter,QScrollArea,QFileDialog,QMessageBox,QProgressBar)
from core import Analyzer,Cancelled,Event,Segment,Settings,export_segments,make_segments,metadata,normalized_frame,read_frame,recommended_settings,validate_folder_name
from preview import Preview,make_proxy,proxy_path
from analysis_eta import AnalysisETA
from pipeline_analyzer import PipelineAnalyzer

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
MODES={'녹다운 + 처치':'both','녹다운':'knock','처치':'kill'}
DETECTORS={'문자 인식':'ocr','기준 이미지':'template','문자 + 이미지':'both'}
OUTPUT_SIZES={'FHD · 1920×1080':'FHD','QHD · 2560×1440':'QHD','4K · 3840×2160':'4K'}
OUTPUT_FPS={'원본 유지':'source','30 FPS':'30','60 FPS':'60','120 FPS':'120'}
VIDEO_EXTENSIONS={'.mp4','.mkv','.mov','.avi'}


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Apex Highlights · 자동 컷 편집')
        self.setAcceptDrops(True)
        self.resize(1320,900);self.setMinimumSize(1080,760)
        DATA.mkdir(parents=True,exist_ok=True)
        self.settings=recommended_settings();self.clips=[];self.segments=[];self.current=None
        self.defaults=recommended_settings()
        self.busy=False;self.cancel=threading.Event();self.messages=queue.Queue()
        self.analysis_eta=None
        self.analyzer=PipelineAnalyzer(DATA/'analysis');self.calibration_frame=None;self.last_output=None
        self.locked_widgets=[]
        self._build()
        self.poll_timer=QTimer(self);self.poll_timer.setInterval(80);self.poll_timer.timeout.connect(self.poll);self.poll_timer.start()

    def _button(self,text,callback,accent=False):
        button=QPushButton(text)
        if accent:button.setObjectName('accent')
        button.clicked.connect(lambda checked=False:self.guard(callback))
        return button

    def _heading(self,text):
        label=QLabel(text);label.setObjectName('heading');return label

    def _combo(self,mapping,value):
        box=QComboBox()
        for name,data in mapping.items():box.addItem(name,data)
        box.setCurrentIndex(max(0,box.findData(value)))
        return box

    def _with_reset(self,widget,key,label,default=None):
        value=getattr(self.defaults,key) if default is None else default
        if isinstance(widget,QDoubleSpinBox):
            display=f'{value:g}';reset=lambda:widget.setValue(value)
        elif isinstance(widget,QComboBox):
            display=widget.itemText(widget.findData(value));reset=lambda:widget.setCurrentIndex(widget.findData(value))
        else:
            display='켜짐' if value else '꺼짐';reset=lambda:widget.setChecked(value)
        container=QWidget();row=QHBoxLayout(container);row.setContentsMargins(0,0,0,0);row.setSpacing(5)
        row.addWidget(widget,1)
        button=self._button('기본값',reset);button.setObjectName('reset_'+key)
        button.setToolTip(f'{label} 기본값: {display}');button.setAccessibleName(f'{label} 기본값으로 복원: {display}')
        button.setStyleSheet('padding:4px 6px;font-size:12px;');row.addWidget(button)
        return container

    def _build(self):
        self.setStyleSheet('''
            QMainWindow,QWidget {background:#111722;color:#edf2f7;font-family:"Malgun Gothic";font-size:13px;}
            QLabel#title {font-family:"Segoe UI";font-size:27px;font-weight:700;}
            QLabel#heading {font-size:15px;font-weight:700;margin-top:5px;margin-bottom:4px;}
            QLabel#hint {color:#9cafc7;font-size:12px;}
            QPushButton {background:#273449;border:1px solid #34445b;border-radius:5px;padding:7px 10px;}
            QPushButton:hover {background:#38516c;} QPushButton:disabled {color:#64748b;background:#192231;}
            QPushButton:checked {background:#38516c;} QPushButton#accent {background:#b94436;border-color:#d25848;}
            QComboBox,QDoubleSpinBox,QLineEdit#outputFolder {background:#25334a;border:1px solid #34445b;border-radius:4px;padding:4px;}
            QDoubleSpinBox {padding-right:26px;min-height:26px;}
            QDoubleSpinBox::up-button {subcontrol-origin:border;subcontrol-position:top right;width:22px;height:17px;background:#30435e;border-left:1px solid #34445b;}
            QDoubleSpinBox::down-button {subcontrol-origin:border;subcontrol-position:bottom right;width:22px;height:17px;background:#30435e;border-left:1px solid #34445b;}
            QDoubleSpinBox::up-button:hover,QDoubleSpinBox::down-button:hover {background:#466284;}
            QDoubleSpinBox::up-button:pressed,QDoubleSpinBox::down-button:pressed {background:#587da5;}
            QDoubleSpinBox::up-arrow {image:url("__ASSETS__/spin_up.svg");width:10px;height:6px;}
            QDoubleSpinBox::down-arrow {image:url("__ASSETS__/spin_down.svg");width:10px;height:6px;}
            QDoubleSpinBox::up-button:disabled,QDoubleSpinBox::down-button:disabled {background:#192231;}
            QListWidget,QTableWidget {background:#192231;border:1px solid #273449;selection-background-color:#374c68;}
            QHeaderView::section {background:#25334a;color:#d7e2ef;border:0;padding:6px;}
            QCheckBox {spacing:7px;} QProgressBar {border:0;background:#25334a;max-height:8px;}
            QProgressBar::chunk {background:#53d4a5;} QScrollArea {border:0;}
        '''.replace('__ASSETS__',(ROOT/'assets').as_posix()))
        central=QWidget();self.setCentralWidget(central);layout=QVBoxLayout(central);layout.setContentsMargins(18,16,18,16);layout.setSpacing(9)
        header=QHBoxLayout();title=QLabel('APEX  /  HIGHLIGHTS');title.setObjectName('title');header.addWidget(title);header.addStretch()
        for text,callback in [('프로젝트 열기',self.load_project),('프로젝트 저장',self.save_project)]:
            b=self._button(text,callback);header.addWidget(b);self.locked_widgets.append(b)
        layout.addLayout(header)
        split=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(split,1)
        left=QWidget();left_layout=QVBoxLayout(left);left_layout.setContentsMargins(0,0,10,0);left_layout.setSpacing(8)
        left_layout.addWidget(self._heading('01  원본 클립'))
        drop_hint=QLabel('영상 파일을 창에 드래그해서 추가하세요.');drop_hint.setObjectName('hint');drop_hint.setWordWrap(True);left_layout.addWidget(drop_hint)
        row=QHBoxLayout();row.addWidget(self._button('파일 추가',self.add_files));row.addWidget(self._button('선택 삭제',self.remove_clip));left_layout.addLayout(row)
        self.clip_list=QListWidget();self.clip_list.setMinimumHeight(80);self.clip_list.setMaximumHeight(140)
        self.clip_list.currentRowChanged.connect(lambda index:self.guard(lambda:self.show_clip(index)) if index>=0 else None)
        left_layout.addWidget(self.clip_list)
        left_layout.addWidget(self._heading('02  추출 설정'))
        form=QFormLayout();self.vars={}
        for key,label in [('before','이벤트 이전 (초)'),('after','이벤트 이후 (초)'),('merge_gap','구간 병합 간격 (초)'),('sample_fps','초당 분석 화면 수'),('confidence','인식 기준 (0~1)')]:
            spin=QDoubleSpinBox();spin.setDecimals(2);spin.setRange(0,3600);spin.setValue(getattr(self.settings,key));spin.setMaximumWidth(100)
            if key=='sample_fps':spin.setRange(.5,30);spin.setSingleStep(.5)
            elif key=='confidence':spin.setRange(.01,1);spin.setSingleStep(.05)
            self.vars[key]=spin;form.addRow(label,self._with_reset(spin,key,label))
        self.mode=self._combo(MODES,self.settings.mode);self.detector=self._combo(DETECTORS,self.settings.detector)
        self.language=self._combo({'한글 + 영문':'ko','영문':'en'},self.settings.language)
        for key,label in [('mode','감지 대상'),('detector','감지 방법'),('language','게임 언어')]:
            form.addRow(label,self._with_reset(getattr(self,key),key,label))
        left_layout.addLayout(form)
        self.damage_fast=QCheckBox('고속 모드 · 데미지 변화');self.damage_fast.setChecked(self.settings.damage_fast)
        left_layout.addWidget(self._with_reset(self.damage_fast,'damage_fast','데미지 고속 모드'))
        self.damage_exclude_unknown=QCheckBox('판독 불가도 제외');self.damage_exclude_unknown.setChecked(self.settings.damage_exclude_unknown)
        self.damage_exclude_unknown.setToolTip('앞뒤 데미지가 다른 구간은 항상 유지합니다. 양쪽 값을 확인할 수 없는 구간만 추가로 제외합니다.')
        self.damage_exclude_unknown.setEnabled(self.damage_fast.isChecked());self.damage_fast.toggled.connect(self.damage_exclude_unknown.setEnabled)
        left_layout.addWidget(self._with_reset(self.damage_exclude_unknown,'damage_exclude_unknown','판독 불가 제외'))
        damage_hint=QLabel('기준 이미지 감지에서 적용 · 1분 간격 확인 + 같은 구간 10초 확인\n앞뒤 데미지가 같으면 연결해 제외하고, 변화한 구간은 유지합니다.');damage_hint.setObjectName('hint');damage_hint.setWordWrap(True);left_layout.addWidget(damage_hint)
        damage_row=QHBoxLayout();damage_row.addWidget(self._button('데미지 영역 지정',lambda:self.calibrate('damage_roi')))
        damage_row.addWidget(self._button('영역 기본값',self.reset_damage_roi));left_layout.addLayout(damage_row)
        left_layout.addWidget(self._button('데미지 판독 기록 열기',self.open_damage_report))
        self.force=QCheckBox('저장된 결과 대신 다시 분석');left_layout.addWidget(self._with_reset(self.force,'force','다시 분석',False))
        left_layout.addWidget(self._button('클립 전체 분석',self.analyze,True))
        left_layout.addWidget(self._button('현재 결과로 컷 길이 다시 계산',self.recalculate));left_layout.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(left);scroll.setMinimumWidth(350);split.addWidget(scroll);self.locked_widgets.append(scroll)
        right=QWidget();right_layout=QVBoxLayout(right);right_layout.setContentsMargins(8,0,0,0);right_layout.setSpacing(6);split.addWidget(right);split.setSizes([365,915]);split.setStretchFactor(1,1)
        top=QHBoxLayout();top.addWidget(self._heading('03  미리보기'));top.addStretch()
        self.proxy=QCheckBox('가벼운 미리보기 (1080p)');self.proxy.toggled.connect(lambda checked:self.guard(self.change_proxy));top.addWidget(self.proxy);right_layout.addLayout(top);self.locked_widgets.append(self.proxy)
        self.preview=Preview();self.preview.message.connect(self.set_status);right_layout.addWidget(self.preview,1);self.locked_widgets.append(self.preview)
        calibration_tools=QWidget();row=QHBoxLayout(calibration_tools);row.setContentsMargins(0,0,0,0)
        row.addWidget(self._button('알림 영역 지정',lambda:self.calibrate('roi')))
        # Keep template registration available internally without exposing its controls.
        for text,mode in [('녹다운 이미지 등록','knock'),('처치 이미지 등록','kill')]:
            button=self._button(text,lambda mode=mode:self.calibrate(mode));row.addWidget(button);button.hide()
        row.addWidget(self._button('재생 화면으로',lambda:self.preview.stack.setCurrentWidget(self.preview.video)));right_layout.addWidget(calibration_tools);self.locked_widgets.append(calibration_tools)
        template_tools=QWidget();row=QHBoxLayout(template_tools);row.setContentsMargins(0,0,0,0)
        self.template_label=QLabel();self.template_label.setObjectName('hint');row.addWidget(self.template_label);row.addStretch();reset=self._button('이미지 초기화',self.clear_templates);row.addWidget(reset);self.locked_widgets.append(reset);right_layout.addWidget(template_tools);template_tools.hide()
        self.preview.calibration.selected.connect(lambda mode,box:self.guard(lambda:self.apply_calibration(mode,box)))
        right_layout.addWidget(self._heading('04  추출 구간'))
        self.segment_hint=QLabel();self.segment_hint.setObjectName('hint');self.segment_hint.setWordWrap(True);self.segment_hint.hide();right_layout.addWidget(self.segment_hint)
        self.table=QTableWidget(0,4);self.table.setHorizontalHeaderLabels(['원본','시작 (초)','끝 (초)','길이'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for column in range(1,4):self.table.setColumnWidth(column,90)
        self.table.setMinimumHeight(100);self.table.setMaximumHeight(180);self.table.itemSelectionChanged.connect(lambda:self.guard(self.select_segment))
        right_layout.addWidget(self.table);self.locked_widgets.append(self.table)
        editor=QWidget();row=QHBoxLayout(editor);row.setContentsMargins(0,0,0,0)
        self.start=QDoubleSpinBox();self.end=QDoubleSpinBox()
        for name,spin in [('시작',self.start),('끝',self.end)]:spin.setRange(0,86400);spin.setDecimals(2);spin.setMaximumWidth(95);row.addWidget(QLabel(name));row.addWidget(spin)
        for text,callback in [('수정',self.edit_segment),('수동 추가',self.add_segment),('삭제',self.delete_segments),('선택 구간 재생',self.preview_segment)]:row.addWidget(self._button(text,callback))
        right_layout.addWidget(editor);self.locked_widgets.append(editor)
        exports=QWidget();row=QHBoxLayout(exports);row.setContentsMargins(0,0,0,0);row.addWidget(self._heading('05  내보내기'))
        self.output_size=self._combo(OUTPUT_SIZES,self.settings.output_resolution);self.output_rate=self._combo(OUTPUT_FPS,self.settings.output_fps)
        row.addWidget(QLabel('해상도'));row.addWidget(self.output_size);row.addWidget(QLabel('프레임률'));row.addWidget(self.output_rate);row.addStretch()
        row.addWidget(self._button('개별 내보내기',lambda:self.export(False)));row.addWidget(self._button('합본 내보내기',lambda:self.export(True),True));layout.addWidget(exports);self.locked_widgets.append(exports)
        folder_row=QHBoxLayout();folder_row.addWidget(QLabel('출력 폴더 이름'))
        self.output_folder=QLineEdit();self.output_folder.setObjectName('outputFolder');self.output_folder.setPlaceholderText('비워두면 자동 생성 · 예: 오늘의 하이라이트');self.output_folder.setMaxLength(100)
        folder_row.addWidget(self.output_folder,1);folder_hint=QLabel('같은 이름이 있으면 번호를 붙입니다.');folder_hint.setObjectName('hint');folder_row.addWidget(folder_hint)
        layout.addLayout(folder_row);self.locked_widgets.append(self.output_folder)
        hint=QLabel('내보내기는 원본 영상에서 처리합니다. 원본보다 높은 규격은 화면 확대·프레임 복제로 처리합니다.');hint.setObjectName('hint');layout.addWidget(hint)
        row=QHBoxLayout();self.status=QLabel('준비됨 · 한글 녹다운 기준 적용');self.status.setWordWrap(True);row.addWidget(self.status,1)
        cancel=QPushButton('작업 취소');cancel.clicked.connect(lambda checked=False:self.cancel.set());row.addWidget(cancel);layout.addLayout(row)
        self.bar=QProgressBar();self.bar.setRange(0,100);self.bar.setValue(0);self.bar.setTextVisible(False);layout.addWidget(self.bar)
        self.eta_label=QLabel();self.eta_label.setObjectName('hint');self.eta_label.setWordWrap(True);layout.addWidget(self.eta_label)
        self.update_template_label()

    def set_status(self,text):self.status.setText(text)

    def guard(self,callback):
        if self.busy:self.set_status('현재 작업이 끝난 뒤 실행해 주세요. 중단하려면 작업 취소를 누르세요.');return
        try:callback()
        except Exception as error:QMessageBox.warning(self,'확인 필요',str(error))

    def confirm(self,text):return QMessageBox.question(self,'현재 편집 결과',text)==QMessageBox.StandardButton.Yes

    def get_settings(self):
        s=Settings(**asdict(self.settings))
        for key,spin in self.vars.items():setattr(s,key,spin.value())
        s.mode=self.mode.currentData();s.detector=self.detector.currentData();s.language=self.language.currentData()
        s.output_resolution=self.output_size.currentData();s.output_fps=self.output_rate.currentData();s.audio_fast=False
        s.damage_fast=self.damage_fast.isChecked();s.damage_exclude_unknown=self.damage_exclude_unknown.isChecked()
        s.validate();self.settings=s;return s

    def dropped_paths(self,mime):
        return [url.toLocalFile() for url in mime.urls()
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTENSIONS
                and Path(url.toLocalFile()).is_file()]

    def dragEnterEvent(self,event):
        if not self.busy and self.dropped_paths(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction);event.accept()
        else:event.ignore()

    def dragMoveEvent(self,event):
        self.dragEnterEvent(event)

    def dropEvent(self,event):
        paths=self.dropped_paths(event.mimeData())
        if self.busy or not paths:
            event.ignore();return
        event.setDropAction(Qt.DropAction.CopyAction);event.accept()
        self.guard(lambda:self.add_files(paths))

    def add_files(self,paths=None):
        if paths is None:paths=QFileDialog.getOpenFileNames(self,'원본 클립 추가','','영상 파일 (*.mp4 *.mkv *.mov *.avi)')[0]
        errors=[]
        for path in paths:
            path=str(Path(path).resolve())
            if any(c['path']==path for c in self.clips):continue
            try:self.clips.append(dict(path=path,info=metadata(path),events=[],analyzed=False))
            except Exception as error:errors.append(str(error))
        self.refresh_clips()
        if self.clips and self.current is None:self.show_clip(0)
        if errors:raise ValueError('\n'.join(errors))

    def refresh_clips(self):
        with QSignalBlocker(self.clip_list):
            self.clip_list.clear()
            for c in self.clips:
                state=f"후보 {len(c['events'])}개" if c['analyzed'] else '분석 전'
                self.clip_list.addItem(f"{Path(c['path']).name} · {state}");self.clip_list.item(self.clip_list.count()-1).setToolTip(c['path'])
            if self.current is not None and self.current<len(self.clips):self.clip_list.setCurrentRow(self.current)

    def remove_clip(self):
        index=self.clip_list.currentRow()
        if index<0:return
        path=self.clips.pop(index)['path'];self.segments=[s for s in self.segments if s.source!=path]
        self.preview.clear();self.current=None;self.calibration_frame=None;self.refresh_clips();self.refresh_segments()
        if self.clips:self.show_clip(0)

    def show_clip(self,index,at=0,end=None,autoplay=False):
        if not 0<=index<len(self.clips):return
        self.current=index;self.calibration_frame=None
        with QSignalBlocker(self.clip_list):self.clip_list.setCurrentRow(index)
        source=self.clips[index]['path']
        if self.proxy.isChecked():
            cached=proxy_path(source,DATA/'preview_cache')
            if cached.exists():self.preview.set_media(cached,at,end,autoplay)
            else:self.worker(lambda:make_proxy(source,DATA/'preview_cache',self.cancel,self.progress),lambda path:self.preview.set_media(path,at,end,autoplay))
        else:self.preview.set_media(source,at,end,autoplay)

    def change_proxy(self):
        if self.current is not None:
            from PySide6.QtMultimedia import QMediaPlayer
            playing=self.preview.player.playbackState()==QMediaPlayer.PlaybackState.PlayingState
            self.show_clip(self.current,self.preview.player.position()/1000,autoplay=playing)

    def calibrate(self,mode):
        if self.current is None:raise ValueError('클립을 추가하고 알림이 나타나는 시점으로 이동하세요.')
        if self.preview.pending:raise ValueError('영상이 준비된 뒤 다시 시도하세요.')
        self.preview.player.pause();source=self.clips[self.current]['path'];at=self.preview.player.position()/1000
        at=min(at,max(0,self.clips[self.current]['info']['duration']-.1))
        def done(frame):
            self.calibration_frame=frame;self.preview.calibration.set_frame(frame,self.settings.damage_roi if mode=='damage_roi' else self.settings.roi,mode);self.preview.stack.setCurrentWidget(self.preview.calibration)
            self.set_status('우측 상단 데미지 표시를 감싸세요. 다른 통계·FPS 표시는 제외하세요.' if mode=='damage_roi' else ('원본 화면에서 알림 범위를 드래그하세요.' if mode=='roi' else '고정된 알림 문구만 드래그하세요. 상대 닉네임은 제외하세요.'))
        self.set_status('알림 영역 지정을 위한 원본 정지 화면을 읽는 중');self.worker(lambda:read_frame(source,at),done)

    def apply_calibration(self,mode,box):
        if self.calibration_frame is None:return
        if mode=='roi':self.settings.roi=box
        elif mode=='damage_roi':self.settings.damage_roi=list(box)
        else:
            frame=normalized_frame(self.calibration_frame);h,w=frame.shape[:2];a,b,c,d=box;crop=frame[int(b*h):int(d*h),int(a*w):int(c*w)]
            directory=DATA/'templates';directory.mkdir(exist_ok=True);path=directory/f'{mode}_{time.time_ns()}.png'
            ok,encoded=cv2.imencode('.png',crop)
            if not ok:raise ValueError('기준 이미지를 저장하지 못했습니다.')
            encoded.tofile(str(path));self.settings.templates.append(dict(kind=mode,path=str(path)))
        self.preview.calibration.roi=list(self.settings.damage_roi if mode=='damage_roi' else self.settings.roi);self.preview.calibration.update();self.update_template_label()
        self.set_status('저장했습니다. 다음 분석에 적용됩니다. 재생 화면으로 돌아가 계속 확인할 수 있습니다.')

    def update_template_label(self):self.template_label.setText(f'기준 이미지 {len(self.settings.templates)}개 · 영역 지정은 원본 정지 화면에서')
    def clear_templates(self):self.settings.templates=[];self.update_template_label()

    def reset_damage_roi(self):
        self.settings.damage_roi=list(self.defaults.damage_roi)
        if self.preview.calibration.mode=='damage_roi':self.preview.calibration.roi=list(self.settings.damage_roi);self.preview.calibration.update()
        self.set_status('데미지 영역을 기본값으로 복원했습니다.')

    def open_damage_report(self):
        clip=self.clips[self.current] if self.current is not None else (self.clips[-1] if self.clips else {})
        folder=clip.get('info',{}).get('analysis',{}).get('damage_report')
        if not folder or not Path(folder).is_dir():raise ValueError('고속 분석 완료 후 선택한 영상의 판독 기록을 열 수 있습니다.')
        os.startfile(folder)

    def _set_busy(self,busy):
        self.busy=busy
        if not busy:self.analysis_eta=None;self.eta_label.clear()
        for widget in self.locked_widgets:widget.setEnabled(not busy)

    def worker(self,function,done):
        if self.busy:return
        self.preview.player.pause();self._set_busy(True);self.cancel.clear();self.bar.setValue(0)
        def execute():
            try:self.messages.put(('done',(done,function())))
            except Cancelled:self.messages.put(('cancel',None))
            except Exception as error:
                (DATA/'last_error.txt').write_text(traceback.format_exc(),encoding='utf-8');self.messages.put(('error',str(error)))
        threading.Thread(target=execute,daemon=True).start()

    def progress(self,p,message):self.messages.put(('progress',(p,message)))

    def poll(self):
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=='analysis_progress':
                    index,stage,p,message=payload
                    if self.analysis_eta:
                        self.analysis_eta.update(index,stage,p)
                        self.bar.setValue(round(p*100));self.set_status(message)
                elif kind=='progress':self.bar.setValue(round(payload[0]*100));self.set_status(payload[1])
                elif kind=='done':self._set_busy(False);self.guard(lambda:payload[0](payload[1]))
                elif kind=='cancel':self._set_busy(False);self.set_status('작업을 취소했습니다.')
                else:self._set_busy(False);self.set_status('작업 실패');QMessageBox.warning(self,'작업 실패',payload)
        except queue.Empty:pass
        if self.analysis_eta:self.eta_label.setText(self.analysis_eta.text())

    def analyze(self):
        if not self.clips:raise ValueError('분석할 클립을 추가하세요.')
        if self.segments and not self.confirm('분석 결과로 현재 컷 목록을 바꿀까요?'):return
        settings=self.get_settings();paths=[c['path'] for c in self.clips];force=self.force.isChecked()
        self.analysis_eta=AnalysisETA([c['info']['duration'] for c in self.clips],'damage' if settings.damage_fast else False)
        def task():
            results=[]
            log_path=DATA/f'analysis_progress_{os.getpid()}.log'
            last_log=[None,0]
            for i,path in enumerate(paths):
                def report(stage,p,message,i=i):
                    self.messages.put(('analysis_progress',(i,stage,p,message)))
                    now=time.monotonic();key=(i,stage)
                    if last_log[0]!=key or now-last_log[1]>=15 or p==1:
                        try:
                            with log_path.open('a',encoding='utf-8') as log:
                                log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} · 파일 {i+1}/{len(paths)} · {stage} · {p:.1%} · {message}\n')
                        except OSError:pass
                        last_log[:]=[key,now]
                info,events=self.analyzer.analyze(path,settings,cancel=self.cancel,force=force,stage_progress=report)
                results.append(dict(path=path,info=info,events=events,analyzed=True))
            return results
        def done(results):
            self.clips=results;self.refresh_clips();self.recalculate(False)
            count=sum(len(c['events']) for c in self.clips);seconds=sum(c['info'].get('analysis',{}).get('seconds',0) for c in self.clips);note=''
            if settings.damage_fast:
                filtered=sum(c['info'].get('analysis',{}).get('damage_fast',False) for c in self.clips)
                note=f' · 고속 적용 {filtered}개 / 전체 분석 {len(self.clips)-filtered}개'
            self.set_status(f'분석 완료 · {seconds:.1f}초 · 후보 {count}개 → 구간 {len(self.segments)}개{note}')
        self.worker(task,done)

    def recalculate(self,confirm=True):
        if confirm and self.segments and not self.confirm('수동 수정한 컷 목록을 감지 결과 기준으로 다시 만들까요?'):return
        settings=self.get_settings();self.segments=[seg for c in self.clips for seg in make_segments(c['path'],c['events'],c['info']['duration'],settings)];self.refresh_segments()

    def refresh_segments(self):
        events=[e for c in self.clips for e in c['events']]
        message=''
        if events and not self.segments:
            mode=self.mode.currentData()
            if mode!='both' and not any(e.kind==mode for e in events):
                counts=' · '.join(f'{label} {sum(e.kind==kind for e in events)}개' for kind,label in [('knock','녹다운'),('kill','처치')] if any(e.kind==kind for e in events))
                message=f'감지 결과: {counts}. 현재 추출 대상 「{self.mode.currentText()}」과 달라 제외됐습니다. 대상을 변경한 뒤 「현재 결과로 컷 길이 다시 계산」을 누르세요. 재분석은 필요 없습니다.'
            else:
                message='감지 결과는 남아 있습니다. 「현재 결과로 컷 길이 다시 계산」으로 복원하세요. 구간이 계속 비면 이벤트 이전·이후 시간을 확인하세요.'
        self.segment_hint.setText(message);self.segment_hint.setVisible(bool(message))
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(self.segments))
            for i,s in enumerate(self.segments):
                for j,value in enumerate((Path(s.source).name,f'{s.start:.2f}',f'{s.end:.2f}',f'{s.end-s.start:.2f}초')):self.table.setItem(i,j,QTableWidgetItem(value))

    def selected_rows(self):return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def select_segment(self):
        rows=self.selected_rows()
        if rows:
            s=self.segments[rows[0]];self.start.setValue(s.start);self.end.setValue(s.end);index=next(i for i,c in enumerate(self.clips) if c['path']==s.source);self.show_clip(index,s.start)

    def validated_segment(self,source):
        a,b=self.start.value(),self.end.value();info=next(c['info'] for c in self.clips if c['path']==source)
        if not 0<=a<b<=info['duration']:raise ValueError(f"시작 < 끝이어야 하고 0~{info['duration']:.2f}초 안에 있어야 합니다.")
        return Segment(source,a,b)

    def edit_segment(self):
        rows=self.selected_rows()
        if len(rows)!=1:raise ValueError('수정할 구간 하나를 선택하세요.')
        i=rows[0];self.segments[i]=self.validated_segment(self.segments[i].source);self.refresh_segments()

    def add_segment(self):
        if self.current is None:raise ValueError('원본 클립을 선택하세요.')
        self.segments.append(self.validated_segment(self.clips[self.current]['path']));self.refresh_segments()

    def delete_segments(self):
        rows=set(self.selected_rows());self.segments=[s for i,s in enumerate(self.segments) if i not in rows];self.refresh_segments()

    def preview_segment(self):
        rows=self.selected_rows()
        if len(rows)!=1:raise ValueError('재생할 구간 하나를 선택하세요.')
        s=self.segments[rows[0]];index=next(i for i,c in enumerate(self.clips) if c['path']==s.source);self.show_clip(index,s.start,s.end,True);self.set_status(f'선택 구간 재생 · {s.start:.2f}~{s.end:.2f}초')

    def export(self,combined):
        if not self.segments:raise ValueError('내보낼 구간이 없습니다.')
        settings=self.get_settings();folder_name=validate_folder_name(self.output_folder.text());directory=QFileDialog.getExistingDirectory(self,'완성 영상 저장 위치')
        if not directory:return
        segments=[Segment(**asdict(s)) for s in self.segments]
        def done(files):
            self.last_output=str(Path(files[0]).parent);self.set_status(f'완료 · 영상 {len(files)}개 저장 · {self.last_output}');os.startfile(self.last_output)
        options=dict(resolution=settings.output_resolution,output_fps=settings.output_fps)
        if folder_name:options['folder_name']=folder_name
        self.worker(lambda:export_segments(segments,directory,combined,self.progress,self.cancel,**options),done)

    def save_project(self,path=None):
        settings=self.get_settings()
        if path is None:path=QFileDialog.getSaveFileName(self,'편집 프로젝트 저장','','편집 프로젝트 (*.json)')[0]
        if not path:return
        clips=[{**c,'events':[asdict(e) for e in c['events']]} for c in self.clips]
        document=dict(version=1,settings=asdict(settings),clips=clips,segments=[asdict(s) for s in self.segments],preview=dict(proxy=self.proxy.isChecked(),volume=self.preview.volume.value(),muted=self.preview.mute.isChecked(),speed=self.preview.speed.currentIndex()))
        document['output_folder']=self.output_folder.text()
        Path(path).write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8');self.set_status('프로젝트 저장 완료')

    def load_project(self,path=None):
        if path is None:path=QFileDialog.getOpenFileName(self,'편집 프로젝트 열기','','편집 프로젝트 (*.json)')[0]
        if not path:return
        doc=json.loads(Path(path).read_text(encoding='utf-8'))
        if doc.get('version')!=1:raise ValueError('지원하지 않는 프로젝트 버전입니다.')
        settings=Settings(**doc['settings']);settings.audio_fast=False;settings.validate()
        clips=[{**c,'info':metadata(c['path']),'events':[Event(**e) for e in c['events']]} for c in doc['clips']]
        for template in settings.templates:
            if not Path(template['path']).is_file():raise ValueError(f"기준 이미지가 없습니다: {template['path']}")
        segments=[Segment(**s) for s in doc['segments']]
        for s in segments:
            clip=next((c for c in clips if c['path']==s.source),None)
            if clip is None or not 0<=s.start<s.end<=clip['info']['duration']+.05:raise ValueError('프로젝트 구간 또는 원본 길이가 달라졌습니다.')
        self.preview.clear();self.settings,self.clips,self.segments=settings,clips,segments;self.current=None;self.calibration_frame=None
        self.damage_fast.setChecked(settings.damage_fast);self.damage_exclude_unknown.setChecked(settings.damage_exclude_unknown)
        for key,spin in self.vars.items():spin.setValue(getattr(settings,key))
        for combo,value in [(self.mode,settings.mode),(self.detector,settings.detector),(self.language,settings.language),(self.output_size,settings.output_resolution),(self.output_rate,settings.output_fps)]:combo.setCurrentIndex(combo.findData(value))
        self.output_folder.setText(doc.get('output_folder',''))
        prefs=doc.get('preview',{})
        with QSignalBlocker(self.proxy):self.proxy.setChecked(bool(prefs.get('proxy',False)))
        self.preview.volume.setValue(int(prefs.get('volume',50)));self.preview.mute.setChecked(bool(prefs.get('muted',False)));self.preview.speed.setCurrentIndex(max(0,min(3,int(prefs.get('speed',1)))))
        self.update_template_label();self.refresh_clips();self.refresh_segments()
        if clips:self.show_clip(0)
        self.set_status('프로젝트를 불러왔습니다.')

    def closeEvent(self,event):
        if self.busy:self.cancel.set();self.set_status('작업을 중단하는 중입니다. 완료 후 창을 닫아 주세요.');event.ignore();return
        self.preview.clear();self.poll_timer.stop();event.accept()


if __name__=='__main__':
    application=QApplication(sys.argv);application.setApplicationName('Apex Highlights')
    window=App();window.show();sys.exit(application.exec())
