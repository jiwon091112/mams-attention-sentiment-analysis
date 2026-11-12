"""A bounded, target-aware MAMS pooling comparison. No hyperparameter search."""
import argparse,collections,concurrent.futures,gzip,hashlib,json,re,subprocess,sys,time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
CFG=dict(seeds=[17,42],embedding_dim=100,hidden_dim=64,attention_dim=32,batch_size=32,
         max_epochs=12,patience=3,learning_rate=.001,weight_decay=.0001,
         embedding_dropout=.10,pooled_dropout=.30,threads=2,bootstrap=2000)
LABELS={'negative':0,'neutral':1,'positive':2}
PATTERN=re.compile(r"\w+(?:['’]\w+)*|[^\w\s]",re.UNICODE)

def js(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    assert not (ROOT/'protocol.json').exists(),'Use frozen prepared data, or a fresh directory.'
    rows=[]
    for split in ['train','val','test']:
        for i,s in enumerate(ET.parse(ROOT/f'raw/{split}.xml').getroot().findall('sentence')):
            text=s.findtext('text');matches=list(PATTERN.finditer(text));ts=[m.group().lower() for m in matches]
            key=hashlib.sha256(' '.join(ts).encode()).hexdigest()
            for j,a in enumerate(s.findall('./aspectTerms/aspectTerm')):
                start,end=int(a.get('from')),int(a.get('to'));term=a.get('term')
                assert text[start:end]==term
                flag=[int(m.start()<end and m.end()>start) for m in matches];assert sum(flag)>0
                rows.append(dict(split=split,sentence_id=f'{split}:{i}',case_id=f'{split}:{i}:{j}',sentence_hash=key,
                    text=text,target=term,label=LABELS[a.get('polarity')],tokens=ts,
                    target_tokens=[m.group().lower() for m in PATTERN.finditer(term)],flag=flag))
    f=pd.DataFrame(rows)
    groups={s:set(f.loc[f.split.eq(s),'sentence_hash']) for s in ['train','val','test']}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    count=collections.Counter(t for ts in f.loc[f.split.eq('train'),'tokens'] for t in ts)
    words=sorted(count,key=lambda w:(-count[w],w))
    vocab={'<PAD>':0,'<UNK>':1,**{w:i+2 for i,w in enumerate(words)}}
    glove=ROOT/'raw/glove-wiki-gigaword-100.gz'
    assert hashlib.md5(glove.read_bytes()).hexdigest()=='40ec481866001177b8cd4cb0df92924f'
    matrix=np.random.default_rng(20261112).normal(0,.05,(len(vocab),100)).astype('float32');matrix[0]=0
    found=set()
    with gzip.open(glove,'rt',encoding='utf-8') as g:
        assert g.readline().strip()=='400000 100'
        for line in g:
            word,_,vec=line.rstrip().partition(' ')
            if word in vocab:
                v=np.fromstring(vec,sep=' ',dtype='float32');assert len(v)==100
                matrix[vocab[word]]=v;found.add(word)
    f['ids']=f.tokens.map(lambda ts:[vocab.get(t,1) for t in ts])
    f['aspect_ids']=f.target_tokens.map(lambda ts:[vocab.get(t,1) for t in ts])
    f.to_pickle(ROOT/'data/prepared.pkl');np.save(ROOT/'data/embeddings.npy',matrix)
    f.drop(columns=['tokens','target_tokens','ids','aspect_ids','flag']).to_csv(ROOT/'data/cases.csv',index=False)
    js(ROOT/'data/vocab.json',vocab)
    manifest=dict(counts={s:dict(sentences=int(g.sentence_id.nunique()),cases=len(g),labels=g.label.value_counts().to_dict()) for s,g in f.groupby('split')},
       vocab_size=len(vocab),pretrained_word_types=len(found),pretrained_training_token_coverage=sum(n for w,n in count.items() if w in found)/sum(count.values()),
       raw_sha256={s:sha(ROOT/f'raw/{s}.xml') for s in ['train','val','test']},sentence_overlap=0,
       pretrained_source='GloVe Wikipedia 2014 + Gigaword 5, 100d, Gensim-data word2vec conversion; frozen embeddings',
       pretrained_md5='40ec481866001177b8cd4cb0df92924f')
    js(ROOT/'data/manifest.json',manifest)
    js(ROOT/'protocol.json',dict(config=CFG,labels=LABELS,
       task='Given a sentence and its gold aspect span, classify negative/neutral/positive. No aspect extraction.',
       architecture='Both: concatenate word vector, repeated mean aspect vector, and aspect-span flag into a unidirectional LSTM. Both classifiers also receive the aspect vector. Pooling differs only.',
       primary='Macro-F1; accuracy and all-aspects-correct sentence rate secondary.',
       checkpoint='Highest validation Macro-F1, tie lowest validation loss, patience=3 epochs without new best Macro-F1.',
       test_rule='All four training runs complete before test predictions. No tuning or extra runs.',
       scope='Limited 2-seed, 12-epoch comparison; no guarantee of large benefit; not a replication of published model scores.'))
    js(ROOT/'results/frozen.json',{p:sha(ROOT/p) for p in ['experiment.py','protocol.json','data/cases.csv','data/embeddings.npy','data/vocab.json']})
    print(json.dumps(manifest,indent=2),flush=True)

class Model(nn.Module):
    def __init__(self,matrix,pooling):
        super().__init__();self.pooling=pooling
        self.embedding=nn.Embedding.from_pretrained(torch.tensor(matrix),freeze=True,padding_idx=0)
        self.dropout=nn.Dropout(CFG['embedding_dropout']);self.final_dropout=nn.Dropout(CFG['pooled_dropout'])
        self.lstm=nn.LSTM(201,CFG['hidden_dim'],batch_first=True)
        self.classifier=nn.Linear(CFG['hidden_dim']+100,3)
        if pooling=='attention':
            self.key=nn.Linear(CFG['hidden_dim'],32);self.query=nn.Linear(100,32,bias=False);self.score=nn.Linear(32,1,bias=False)
    def forward(self,x,a,flag,lengths,equal=False):
        av=self.embedding(a).sum(1)/a.ne(0).sum(1,keepdim=True)
        e=torch.cat([self.dropout(self.embedding(x)),av[:,None,:].expand(-1,x.shape[1],-1),flag[:,:,None]],dim=-1)
        h,_=self.lstm(e);mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        if self.pooling=='mean':v=(h*mask[:,:,None]).sum(1)/lengths[:,None]
        else:
            score=self.score(torch.tanh(self.key(h)+self.query(av)[:,None,:])).squeeze(-1)
            if equal:score=score*0
            weight=score.masked_fill(~mask,float('-inf')).softmax(1);v=(h*weight[:,:,None]).sum(1)
        return self.classifier(self.final_dropout(torch.cat([v,av],dim=-1)))

def pair(matrix,seed):
    torch.manual_seed(seed);a=Model(matrix,'mean');b=Model(matrix,'attention')
    b.load_state_dict(a.state_dict(),strict=False)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    return a,b

def arrays(f):
    n=len(f);lengths=np.asarray(f.ids.map(len));x=np.zeros((n,int(lengths.max())),np.int64)
    a=np.zeros((n,int(f.aspect_ids.map(len).max())),np.int64);flag=np.zeros(x.shape,np.float32)
    for i,row in enumerate(f.itertuples()):
        x[i,:len(row.ids)]=row.ids;a[i,:len(row.aspect_ids)]=row.aspect_ids;flag[i,:len(row.flag)]=row.flag
    return [torch.from_numpy(v) for v in [x,a,flag,lengths,f.label.to_numpy(np.int64)]]

def batch(data,ids):
    x,a,flag,l,y=[v[ids] for v in data];m=int(l.max());return x[:,:m],a,flag[:,:m],l,y

def metrics(y,pred):
    f=[]
    for c in range(3):
        tp=np.sum((y==c)&(pred==c));fp=np.sum((y!=c)&(pred==c));fn=np.sum((y==c)&(pred!=c));f.append(float(2*tp/max(2*tp+fp+fn,1)))
    return dict(accuracy=float(np.mean(y==pred)),macro_f1=float(np.mean(f)),class_f1=f)

@torch.no_grad()
def predict(model,data):
    model.eval();prob=[];total=0.
    for start in range(0,len(data[-1]),128):
        x,a,flag,l,y=batch(data,slice(start,start+128));z=model(x,a,flag,l)
        total+=nn.functional.cross_entropy(z,y,reduction='sum').item();prob.extend(z.softmax(1).tolist())
    return np.asarray(prob),total/len(data[-1])

def checks():
    f=pd.read_pickle(ROOT/'data/prepared.pkl');d=arrays(f[f.split.eq('train')].head(2));matrix=np.load(ROOT/'data/embeddings.npy')
    a,b=pair(matrix,17);a.eval();b.eval();x,asp,flag,l,y=d
    with torch.no_grad():
        torch.testing.assert_close(a(x,asp,flag,l),b(x,asp,flag,l,True),atol=1e-6,rtol=1e-5)
        for m in [a,b]:torch.testing.assert_close(m(x,asp,flag,l),m(nn.functional.pad(x,(0,5)),asp,nn.functional.pad(flag,(0,5)),l),atol=1e-6,rtol=1e-5)
    js(ROOT/'results/checks.json',dict(common_initialization=True,uniform_attention_equals_mean=True,padding_invariance=True,frozen_pretrained_vectors=True))

def train(seed,pooling):
    for n,h in json.loads((ROOT/'results/frozen.json').read_text()).items():assert sha(ROOT/n)==h,n
    f=pd.read_pickle(ROOT/'data/prepared.pkl');tr=arrays(f[f.split.eq('train')]);va=arrays(f[f.split.eq('val')]);matrix=np.load(ROOT/'data/embeddings.npy')
    m=pair(matrix,seed)[pooling=='attention'];torch.manual_seed(seed+777)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    run=ROOT/f'results/{seed}_{pooling}';run.mkdir(exist_ok=True);best=-1.;best_loss=float('inf');bad=0;history=[];began=time.perf_counter()
    for epoch in range(1,CFG['max_epochs']+1):
        m.train();total=0.;order=np.random.default_rng(seed*1000+epoch).permutation(len(tr[-1]));started=time.perf_counter()
        for start in range(0,len(order),CFG['batch_size']):
            ids=order[start:start+CFG['batch_size']];x,a,flag,l,y=batch(tr,ids);opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(m(x,a,flag,l),y);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();total+=loss.item()*len(ids)
        p,vl=predict(m,va);score=metrics(va[-1].numpy(),p.argmax(1));history.append(dict(epoch=epoch,train_loss=total/len(tr[-1]),validation_loss=vl,seconds=time.perf_counter()-started,**score))
        pd.DataFrame(history).to_csv(run/'history.csv',index=False);print(seed,pooling,json.dumps(history[-1]),flush=True)
        improved=score['macro_f1']>best
        if improved or (score['macro_f1']==best and vl<best_loss):
            best=score['macro_f1'];best_loss=vl;best_epoch=epoch;torch.save(m.state_dict(),run/'best.pt')
        bad=0 if improved else bad+1
        if bad>=CFG['patience']:break
    js(run/'complete.json',dict(seed=seed,pooling=pooling,best_epoch=best_epoch,epochs_run=epoch,validation_macro_f1=best,validation_loss=best_loss,
        seconds=time.perf_counter()-began,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),checkpoint_sha256=sha(run/'best.pt')))

def evaluate():
    assert all((ROOT/f'results/{s}_{p}/complete.json').exists() for s in CFG['seeds'] for p in ['mean','attention'])
    f=pd.read_pickle(ROOT/'data/prepared.pkl');te=f[f.split.eq('test')].reset_index(drop=True);d=arrays(te);matrix=np.load(ROOT/'data/embeddings.npy');results=[]
    for seed in CFG['seeds']:
        for pooling in ['mean','attention']:
            run=ROOT/f'results/{seed}_{pooling}';m=Model(matrix,pooling);m.load_state_dict(torch.load(run/'best.pt',weights_only=True,map_location='cpu'))
            prob,loss=predict(m,d);pred=prob.argmax(1);score=metrics(te.label.to_numpy(),pred)
            out=te[['sentence_id','case_id','sentence_hash','text','target','label']].copy();out['prediction']=pred
            for c in range(3):out[f'p_{c}']=prob[:,c]
            out.to_csv(run/'predictions.csv',index=False);read=pd.read_csv(run/'predictions.csv');again=metrics(read.label.to_numpy(),read.prediction.to_numpy());assert score==again
            correct=out.label.eq(out.prediction);sentence_accuracy=float(correct.groupby(out.sentence_id).all().mean())
            results.append(dict(seed=seed,pooling=pooling,test_loss=loss,sentence_accuracy=sentence_accuracy,**score))
    pd.DataFrame(results).to_csv(ROOT/'results/metrics.csv',index=False);print(json.dumps(results,indent=2),flush=True)

def run():
    checks()
    def work(task):
        seed,pool=task
        with (ROOT/f'results/{seed}_{pool}.log').open('w') as f:
            subprocess.run([sys.executable,'-u',str(ROOT/'experiment.py'),'train','--seed',str(seed),'--pooling',pool],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('Complete',seed,pool,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(work,[(s,p) for s in CFG['seeds'] for p in ['mean','attention']]))
    evaluate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','train']);p.add_argument('--seed',type=int);p.add_argument('--pooling',choices=['mean','attention']);a=p.parse_args()
    torch.set_num_threads(CFG['threads']);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if a.stage=='prepare':prepare()
    elif a.stage=='run':run()
    else:train(a.seed,a.pooling)
