import json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT,CFG,metrics,js,sha

scores=pd.read_csv(ROOT/'results/metrics.csv')
manifest=json.loads((ROOT/'data/manifest.json').read_text())
summary=scores.groupby('pooling')[['accuracy','macro_f1','sentence_accuracy']].mean()
summary.to_csv(ROOT/'results/summary.csv')
frames={(s,p):pd.read_csv(ROOT/f'results/{s}_{p}/predictions.csv') for s in CFG['seeds'] for p in ['mean','attention']}
ref=frames[(CFG['seeds'][0],'mean')]
for f in frames.values():assert f.case_id.tolist()==ref.case_id.tolist()
group,unique=pd.factorize(ref.sentence_id);y=ref.label.to_numpy();rng=np.random.default_rng(7919);boot=[]
def weighted_f1(pred,w):
    cm=np.bincount(y*3+pred,weights=w,minlength=9).reshape(3,3);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    return float(np.mean(np.divide(2*tp,den,out=np.zeros(3),where=den!=0)))
for _ in range(CFG['bootstrap']):
    sampled=rng.integers(0,len(unique),size=len(unique));w=np.bincount(sampled,minlength=len(unique))[group]
    boot.append(np.mean([weighted_f1(frames[(s,'attention')].prediction.to_numpy(),w)-weighted_f1(frames[(s,'mean')].prediction.to_numpy(),w) for s in CFG['seeds']]))
interval=np.quantile(boot,[.025,.975]).tolist()
js(ROOT/'results/uncertainty.json',dict(method='Paired sentence-cluster percentile bootstrap; two fitted seed pairs fixed; excludes training/split uncertainty.',repetitions=CFG['bootstrap'],macro_f1_difference_95_interval=interval))
lines=['MAMS 속성별 감성 분류: 제한된 학습량의 평균/Attention 풀링 비교','',
 '설정: 학습 4,297문장/11,186속성 사례, 검증 500문장/1,332사례, 테스트 500문장/1,336사례.',
 '공식 ATSA 분할 사용. 긍정/부정/중립 3분류이며 속성 추출은 하지 않는다.',
 '같은 정규화 문장이 분할 사이에 겹치지 않고 원문 대상 문자 위치도 일치함을 확인했다.',
 '두 모델 모두 문장 단어 벡터, 반복한 대상 벡터, 대상 위치 표식을 LSTM 입력으로 받는다.',
 '두 분류기 모두 풀링 결과와 대상 벡터를 받는다. 평균 모델에도 충분한 대상 정보를 제공한다.',
 'Attention은 대상 벡터를 이용해 은닉 상태의 점수를 계산한다. 추가 파라미터 효과를 완전히 분리하지는 못했다.',
 'GloVe Wikipedia 2014 + Gigaword 5 100차원 벡터를 동일하게 사용하고 고정했다.',
 f'학습 토큰의 사전 벡터 포함 비율: {manifest["pretrained_training_token_coverage"]*100:.2f}%.',
 '전체 문장 입력, 단방향 LSTM 은닉 64, 배치 32, AdamW lr=0.001/weight decay=0.0001.',
 '임베딩 dropout=0.10, 풀링과 대상 벡터를 합친 뒤 dropout=0.30.',
 '시드 17,42 x 두 풀링 = 총 4회. 최대 12 epoch, 검증 Macro-F1 3회 연속 미개선 시 종료.',
 '체크포인트는 최고 검증 Macro-F1, 동률이면 최소 검증 손실로 선택한다.',
 '검증 결과로 새 설정을 탐색하거나 테스트 뒤 추가 학습하지 않는다. 네 학습을 모두 마친 뒤 테스트했다.', '',
 '실제 테스트 결과: 시드별',scores.to_string(index=False),'',
 '두 시드 평균: accuracy=속성별 정확도, macro_f1=세 감성의 F1 평균, sentence_accuracy=모든 속성을 맞힌 문장 비율',
 summary.to_string(),'',
 f'Attention-평균 Macro-F1: {summary.loc["attention","macro_f1"]-summary.loc["mean","macro_f1"]:+.6f}',
 f'Attention-평균 정확도: {(summary.loc["attention","accuracy"]-summary.loc["mean","accuracy"])*100:+.4f}%p',
 f'문장 집단 부트스트랩 조건부 95% Macro-F1 차이 구간: {interval}',
 '이 구간은 고정된 학습 모델에서 평가 문장의 변동만 고려한다. 새 시드/분할의 불확실성은 포함하지 않는다.', '',
 '실제 학습량:']
for seed in CFG['seeds']:
    for pool in ['mean','attention']:
        a=json.loads((ROOT/f'results/{seed}_{pool}/complete.json').read_text())
        lines.append(f'{seed}/{pool}: 실행 {a["epochs_run"]} epoch, 선택 {a["best_epoch"]} epoch, 학습+검증 {a["seconds"]:.1f}초, 학습 파라미터 {a["trainable_parameters"]:,}개.')
lines += ['','해석의 한계:',
 '시드 2개와 최대 12 epoch로 제한한 예비 비교다. 수렴이나 최적 설정을 보장하지 않는다.',
 '문장 수와 속성 사례 수는 다르다. 같은 문장 사례를 독립 문장처럼 세지 않았다.',
 '중립 사례가 가장 많다. 정확도만으로 결론 내리지 않고 Macro-F1을 주 지표로 본다.',
 'MAMS는 상이한 감성을 가진 여러 대상을 포함하도록 구성된 자료로 일반 리뷰 전체를 대표하지 않는다.',
 '앞선 IMDB/Steam과 과제 및 정답 종류가 달라 정확도 수치를 직접 비교하지 않는다.',
 'Attention 가중치를 특정 단어가 판단의 원인이라는 증거로 해석하지 않는다.',
 '저장한 네 예측 파일에서 지표를 재계산해 일치함을 확인했다.',
 '사전학습 데이터 자체에 원문이 포함됐는지는 확인하지 못했다. 두 모델에는 같은 벡터를 사용했다.',
 '자료/코드: https://github.com/siat-nlp/MAMS-for-ABSA',
 '원 논문: https://aclanthology.org/D19-1654/',
 '단어 벡터: https://github.com/RaRe-Technologies/gensim-data']
(ROOT/'results/report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,4))
for seed in CFG['seeds']:
    for pool in ['mean','attention']:
        h=pd.read_csv(ROOT/f'results/{seed}_{pool}/history.csv');ax.plot(h.epoch,h.macro_f1,label=f'{pool}, seed {seed}',marker='o',markersize=3)
ax.set(xlabel='Epoch',ylabel='Validation Macro-F1');ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/learning.png',dpi=150);plt.close(fig)
print('\n'.join(lines),flush=True)
