from pathlib import Path
import argparse
import json
import ast
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import preprocess_drone_v51 as v

SOURCES={'DroneRFa':'DroneRFa','RF_UAVs':'Kaggle','Gent_WiFi':'Gent','RadioML_2018':'RadioML'}
original_meta=v.meta_base_dict
original_power=v.exact_window_power

def parse_name(name):
    m=re.fullmatch(r'(T[01]+)(?:_(D[01]+))?_(S[01]+)\.mat',name,re.I)
    if not m:
        raise ValueError(name)
    t,d,s=m.groups()
    return t.upper(),(d or '').upper(),s.upper(),int(s[1:],2)

def bounded_power(src,n,length):
    starts,power=original_power(src,n,length)
    keep=starts//10000000==(starts+length-1)//10000000
    return starts[keep],power[keep]

def bound_meta(meta):
    values=original_meta(meta)
    if values.get('split') in roles:
        return values
    source=SOURCES[meta.dataset]
    path=meta.source_file
    rid='RadioML|frame:'+path.split('::')[1] if source=='RadioML' else source+'|'+path
    role,label=records[rid]
    values.update(group_key=rid,split=role)
    if source in ('DroneRFa','Kaggle'):
        values['drone_type']=label
    else:
        values['modulation']=label
    return values

def configure(args):
    global records, roles, OUT
    OUT = args.out.resolve()
    d=pd.read_parquet(args.record_index, columns=['record_id', 'role', 'label'])
    roles=set(d.role)
    records=dict(zip(d.record_id,zip(d.role,d.label)))
    cfg = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    cfg = cfg.get('config', cfg)
    if isinstance(cfg.get('TARGET_MIX'), str):
        cfg['TARGET_MIX']=ast.literal_eval(cfg['TARGET_MIX'])
    cfg.update(OUTPUT_DIR=str(OUT),DRY_RUN=False)
    for key in ('DRONERFA_DIR','RF_UAVS_DIR','GENT_WIFI_DIR','RADIOML18_H5'):
        cfg[key] = str((args.data_root / cfg.get(key, getattr(v.CFG, key))).resolve())
    v.apply_cfg(cfg)
    if args.radioml_classes:
        v.RML18_CLASSES = json.loads(args.radioml_classes.read_text(encoding='utf-8'))
    v.parse_tds_from_filename=parse_name
    v.exact_window_power=bounded_power
    v.meta_base_dict=bound_meta
    v.finalize=finalize_main
    v.prepare_dirs(str(OUT))

def finalize_main(out,elapsed=0):
    OUT = Path(out)
    parts = sorted((OUT / 'index').glob('part_*.parquet'))
    seen=set()
    pool=defaultdict(list)
    counts=Counter()
    by_source=Counter()
    channels=defaultdict(set)
    for part in parts:
        f=pd.read_parquet(part,columns=['group_key','split','dataset','band','channel','start_sample','end_sample','raw_power'])
        seen.update(f.group_key)
        assert (f.split==f.group_key.map(lambda k:records[k][0])).all()
        drone=f[f.dataset.eq('DroneRFa')]
        assert (drone.start_sample//10000000==(drone.end_sample-1)//10000000).all()
        for key,g in drone.groupby('group_key'):
            channels[key].update(g.channel)
        counts.update(f.groupby(['dataset','split']).size().to_dict())
        train=f[f.split.eq('P_train')]
        by_source.update(train.dataset.value_counts().to_dict())
        for key,g in train.groupby(['dataset','band','channel'],dropna=False):
            pool[key].append(g.raw_power.to_numpy())
    if seen!=set(records) or any(len(c)!=2 for c in channels.values()):
        raise ValueError(f'record coverage: {len(seen)}/{len(records)}, or incomplete channels')
    floors={}
    for key,values in pool.items():
        p=np.concatenate(values)
        floors[key]=float(np.quantile(p[p>0],v.CFG.GLOBAL_FLOOR_Q))
    tmp=OUT/'pretrain_index.parquet.tmp'
    schema=v.FINAL_SCHEMA.append(pa.field('role',pa.string()))
    with pq.ParquetWriter(tmp,schema) as writer:
        for part in parts:
            f=pd.read_parquet(part)
            f['role']=f.split
            f['split']=f.role.map({'P_train':'train','P_val':'val'}).fillna('excl')
            floor=np.array([floors[k] for k in zip(f.dataset,f.band,f.channel)])
            f['snr_global_db']=10*np.log10(np.maximum(f.raw_power,1e-30)/np.maximum(floor,1e-30))
            f['is_active_global']=(f.snr_global_db>=v.CFG.ACTIVE_SNR_TH_DB).astype('int8')
            f['sample_weight']=f.dataset.map({s:min(v.CFG.MAX_SAMPLE_WEIGHT,v.CFG.TARGET_MIX[s]*sum(by_source.values())/n) for s,n in by_source.items()})
            writer.write_table(pa.Table.from_pandas(f,schema=schema,preserve_index=False))
    tmp.replace(OUT/'pretrain_index.parquet')
    pd.Series(counts).rename_axis(['dataset','role']).rename('windows').to_csv(OUT/'window_counts.csv')
    print('complete',flush=True)

def main():
    parser = argparse.ArgumentParser(description="Prepare the four-source IQ pretraining index")
    parser.add_argument("--record-index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--radioml-classes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    configure(parser.parse_args())
    units=v.build_units()
    assert set(u.dataset for u in units)==set(SOURCES)
    for ds in SOURCES:
        u=next(u for u in units if u.dataset==ds)
        if ds=='RF_UAVs':
            u=v.Unit(ds,'smoke',extra={'rels':u.extra['rels'][:1]})
        it=v.iter_unit_windows(u,dry=False)
        item=next(it)
        meta=bound_meta(item.meta)
        assert meta['split'] in roles
        out=v.normalize_iq_batch(item.iq[None,:])
        assert np.isfinite(out).all() and out.shape==(1,2,item.window_len)
        it.close()
        print('smoke',ds,meta['split'],out.shape,flush=True)
    print('full preprocessing',len(units),flush=True)
    v.run_full(units)


if __name__ == '__main__':
    main()
