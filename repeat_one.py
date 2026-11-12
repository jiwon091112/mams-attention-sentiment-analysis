"""One user-authorized replication, with the original data/training code frozen."""
import concurrent.futures,datetime,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import ROOT,CFG,Model,arrays,predict,metrics,js,sha

SEED=83
OUT=ROOT/'followup_83'

def worker(pooling):
    path=ROOT/f'results/{SEED}_{pooling}'
    assert not path.exists(),'Do not silently repeat or overwrite a completed run.'
    with (OUT/f'{pooling}_training.log').open('w') as f:
        subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'train','--seed',str(SEED),'--pooling',pooling],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Training complete:',pooling,flush=True)

def main():
    OUT.mkdir(exist_ok=True)
    plan=OUT/'plan.json'
    assert not plan.exists(),'This single-seed replication already exists.'
    original=json.loads((ROOT/'results/frozen.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in original.items())
    js(plan,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       reason='User requested training under the existing narrow topic and modest compute budget.',
       previous_seeds=CFG['seeds'],new_seed=SEED,new_training_runs=2,
       all_parameters_and_methods_unchanged=True,original_config=CFG,
       original_protocol_sha256=sha(ROOT/'protocol.json'),original_source_sha256=sha(ROOT/'experiment.py'),
       test_seen_previously=True,
       interpretation='Post-test replication using another seed, no hyperparameter/data changes. Not a new blind test.',
       stopping_rule='Two new runs only. Same maximum 12 epochs and validation Macro-F1 patience=3. No extension.'))
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(worker,['mean','attention']))
    # Test starts only after both new training runs have completed.
    assert all((ROOT/f'results/{SEED}_{p}/complete.json').exists() for p in ['mean','attention'])
    js(OUT/'evaluation_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(plan),both_training_runs_complete=True))
    f=pd.read_pickle(ROOT/'data/prepared.pkl');te=f[f.split.eq('test')].reset_index(drop=True)
    d=arrays(te);matrix=np.load(ROOT/'data/embeddings.npy');new=[]
    for pooling in ['mean','attention']:
        run=ROOT/f'results/{SEED}_{pooling}'
        m=Model(matrix,pooling)
        m.load_state_dict(torch.load(run/'best.pt',weights_only=True,map_location='cpu'))
        prob,loss=predict(m,d);pred=prob.argmax(1);score=metrics(te.label.to_numpy(),pred)
        output=te[['sentence_id','case_id','sentence_hash','text','target','label']].copy()
        output['prediction']=pred
        for c in range(3):output[f'p_{c}']=prob[:,c]
        output.to_csv(run/'predictions.csv',index=False)
        read=pd.read_csv(run/'predictions.csv')
        assert metrics(read.label.to_numpy(),read.prediction.to_numpy())==score
        new.append(dict(seed=SEED,pooling=pooling,test_loss=loss,
            sentence_accuracy=float(output.label.eq(output.prediction).groupby(output.sentence_id).all().mean()),**score))
    new=pd.DataFrame(new);new.to_csv(OUT/'new_seed_metrics.csv',index=False)
    old=pd.read_csv(ROOT/'results/metrics.csv')
    combined=pd.concat([old,new],ignore_index=True)
    combined.to_csv(OUT/'all_three_seed_metrics.csv',index=False)
    summary=combined.groupby('pooling')[['accuracy','macro_f1']].mean()
    summary.to_csv(OUT/'summary_three_seeds.csv')
    # Paired sentence bootstrap, with all three learned seed pairs fixed.
    seeds=CFG['seeds']+[SEED]
    pred={(s,p):pd.read_csv(ROOT/f'results/{s}_{p}/predictions.csv') for s in seeds for p in ['mean','attention']}
    ref=pred[(seeds[0],'mean')]
    assert all(x.case_id.tolist()==ref.case_id.tolist() for x in pred.values())
    group,unique=pd.factorize(ref.sentence_id);y=ref.label.to_numpy();rng=np.random.default_rng(7919)
    def wf1(guess,w):
        cm=np.bincount(y*3+guess,weights=w,minlength=9).reshape(3,3)
        den=cm.sum(0)+cm.sum(1);return np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den!=0))
    boot=[]
    for _ in range(CFG['bootstrap']):
        w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[group]
        boot.append(np.mean([wf1(pred[(s,'attention')].prediction.to_numpy(),w)-wf1(pred[(s,'mean')].prediction.to_numpy(),w) for s in seeds]))
    interval=np.quantile(boot,[.025,.975]).tolist()
    js(OUT/'uncertainty.json',dict(method='Paired sentence-cluster bootstrap, three fitted seed pairs fixed; excludes training and split uncertainty.',macro_f1_difference_95_interval=interval,repetitions=CFG['bootstrap']))
    lines=['MAMS 고정 조건의 추가 학습 기록','',
       '기존 시드 17,42의 결과를 보존하고 새 시드 83에서 두 모델을 각 한 번씩 추가 학습했다.',
       '데이터/분할/벡터/모델 구조/최적화/최대 epoch/체크포인트 선택 규칙은 바꾸지 않았다.',
       '이번 두 번을 포함해 총 세 시드, 여섯 모델이다. 추가 기술이나 설정 탐색은 하지 않았다.',
       '기존 테스트 결과를 본 뒤의 고정 조건 반복이며 새 독립/맹검 검증이 아니다.',
       '과제: 주어진 평가 대상의 긍정/부정/중립 분류. 속성 자동 추출은 하지 않는다.',
       '학습 4,297문장/11,186사례, 검증 500문장/1,332사례, 테스트 500문장/1,336사례.',
       '', '새 시드 83의 실제 결과:',new.to_string(index=False),'',
       '기존 결과와 합친 세 시드 평균:',summary.to_string(),'',
       f'Attention-평균 정확도 차이: {(summary.loc["attention","accuracy"]-summary.loc["mean","accuracy"])*100:+.4f}%p',
       f'Attention-평균 Macro-F1 차이: {summary.loc["attention","macro_f1"]-summary.loc["mean","macro_f1"]:+.6f}',
       f'조건부 95% Macro-F1 차이 구간: {interval}',
       '구간은 고정한 학습 모델에서 평가 문장을 재추출한 변동이며 학습 시드/분할 변동 전체를 포함하지 않는다.',
       '', '새 실행의 학습량:']
    for p in ['mean','attention']:
        meta=json.loads((ROOT/f'results/{SEED}_{p}/complete.json').read_text())
        lines.append(f'{p}: 실행 {meta["epochs_run"]} epoch, 선택 {meta["best_epoch"]} epoch, 학습+검증 {meta["seconds"]:.1f}초.')
    lines+=['','학습량 제한에 따른 예비 비교다. 최적 수렴이나 모든 조건에서의 우수성을 입증하지 않는다.',
       'Attention의 추가 학습 파라미터 효과를 분리하지 못했다. 기존 Steam/IMDB 정확도와 직접 비교하지 않는다.',
       '새 예측 파일의 지표를 재계산해 일치함을 확인했고 원래 고정 파일 해시도 재확인했다.',
       f'이번 두 실행과 평가/집계의 경과 시간: {time.perf_counter()-started:.1f}초.']
    assert all(sha(ROOT/n)==h for n,h in original.items())
    js(OUT/'checks.json',dict(original_frozen_files_unchanged=True,new_predictions_recomputed=True,both_training_before_evaluation=True))
    (OUT/'report_ko.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    torch.set_num_threads(CFG['threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    main()
