"""Elapsed-time estimates based on observed stage throughput and clip duration."""
import time


def duration_text(seconds):
    seconds=max(0,round(seconds))
    if seconds<60:return f'{seconds}초'
    minutes,seconds=divmod(seconds,60)
    if minutes<60:return f'{minutes}분 {seconds}초'
    hours,minutes=divmod(minutes,60)
    return f'{hours}시간 {minutes}분'


class AnalysisETA:
    def __init__(self,durations,fast,clock=time.monotonic):
        self.clock=clock;self.started=clock();self.stage_started=self.started
        self.durations=durations;self.stages=['damage','image'] if fast=='damage' else (['audio_extract','audio_scan','image'] if fast else ['image'])
        self.rates={};self.key=None;self.fraction=0;self.index=0;self.stage='setup'

    def update(self,index,stage,fraction):
        fraction=max(0,min(1,fraction));now=self.clock()
        if self.key!=(index,stage) or fraction<self.fraction:
            self.stage_started=now
            self.key=(index,stage)
        self.index=index;self.stage=stage;self.fraction=fraction
        if stage=='image':
            for name in self.stages[:-1]:self.rates.setdefault(name,0)
        elapsed=now-self.stage_started
        if stage in self.stages and fraction>0 and (elapsed>=2 or fraction==1):
            self.rates[stage]=elapsed/(max(.001,self.durations[index])*fraction)

    def text(self):
        now=self.clock();elapsed=now-self.started
        prefix=f'경과 {duration_text(elapsed)} · 파일 {self.index+1}/{len(self.durations)}'
        if self.stage not in self.stages or not 0<self.fraction<1 or now-self.stage_started<2:
            return prefix+' · 전체 예상 남은 시간 계산 중'
        # Recalculate with wall time, so an I/O stall does not imply progress.
        rate=(now-self.stage_started)/(max(.001,self.durations[self.index])*self.fraction)
        remaining=rate*self.durations[self.index]*(1-self.fraction)
        phase=f'현재 단계 {self.fraction:.0%} · 단계 남은 시간 약 {duration_text(remaining)}'
        rates={**self.rates,self.stage:rate}
        if all(stage in rates for stage in self.stages):
            later=self.stages[self.stages.index(self.stage)+1:]
            total=remaining+self.durations[self.index]*sum(rates[s] for s in later)
            total+=sum(self.durations[self.index+1:])*sum(rates[s] for s in self.stages)
            return prefix+f' · 전체 남은 시간 약 {duration_text(total)} · '+phase
        return prefix+' · '+phase+' · 전체 예상 계산 중'
