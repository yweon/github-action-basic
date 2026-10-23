"""Read existing verified artifacts; run a small Alpha158/LGBModel smoke test only."""
import hashlib,json,os,subprocess,sys,time,traceback,venv,zipfile
from pathlib import Path
root=Path("proof");root.mkdir(exist_ok=True)
def main():
    statuses=[]
    for manifest in Path("downloads").rglob("manifest.json"):
        folder=manifest.parent; entries=json.loads(manifest.read_text())
        for item in entries:
            f=folder/"wheels"/item["filename"]
            assert hashlib.sha256(f.read_bytes()).hexdigest()==item["sha256"]
        for name in ["status.json","manifest.json","requirements.lock"]:
            text=(folder/name).read_text()
            print("PROOF_FILE "+json.dumps({"artifact":str(folder),"name":name,"text":text}),flush=True)
        statuses.append(json.loads((folder/"status.json").read_text()))
    qfolder=next(p.parent for p in Path("downloads").rglob("status.json") if json.loads(p.read_text())["target"]=="qlib" and "Linux" in json.loads(p.read_text())["platform"])
    venv.create(root/"venv",with_pip=True)
    python=(root/"venv/bin/python").resolve()
    cmd=[str(python),"-m","pip","--isolated","install","--no-index","--find-links",str((qfolder/"wheels").resolve()),
         "--require-hashes","-r",str((qfolder/"requirements.lock").resolve())]
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (root/"offline_install.txt").write_text(r.stdout)
    assert r.returncode==0,r.stdout[-4000:]
    r=subprocess.run([str(python),__file__,"--model"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (root/"model.txt").write_text(r.stdout)
    print(r.stdout,flush=True)
    assert r.returncode==0
def model():
    import numpy as np,pandas as pd,qlib
    from qlib.tests.data import GetData
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.workflow import R
    from importlib.metadata import version
    status={"scope":"SAMPLE_EXECUTION_ONLY_NOT_STRATEGY_OR_PIT_VALIDATION","formal_stock_performance":False,
            "feature_success":False,"label_success":False,"fit_success":False,"prediction_success":False,
            "data_download_success":False,"python":sys.version,"qlib":version("pyqlib"),"lightgbm":version("lightgbm"),
            "feature_time":"t close","label_expression":"Ref($close, -6)/Ref($close, -1)-1",
            "label_time":"t+1 close to t+6 close; five trading days","live":False,"order_calls":0}
    try:
        t=time.perf_counter()
        data=(root/"sample").resolve()
        GetData().qlib_data(name="qlib_data_simple",target_dir=str(data),region="cn",version="v2",delete_old=False)
        status["data_download_success"]=True
        status["download_seconds"]=time.perf_counter()-t
        status["sample_archives"]=[{"name":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in data.glob("*.zip")]
        qlib.init(provider_uri=str(data),region="cn",kernels=1)
        instruments=D.list_instruments(D.instruments("csi300"),start_time="2016-01-01",end_time="2017-12-29",as_list=True)[:20]
        assert len(instruments)>=2
        t=time.perf_counter()
        h=Alpha158(instruments=instruments,start_time="2016-01-01",end_time="2017-12-29",
                   label=(["$close"],["CURRENT_CLOSE_PLACEHOLDER_DISCARDED"]),infer_processors=[],learn_processors=[])
        features=h.fetch(col_set="feature")
        assert features.shape[1]==158
        status.update(feature_success=True,feature_shape=list(features.shape),feature_seconds=time.perf_counter()-t)
        # Features are completed before any future-label query is issued.
        t=time.perf_counter()
        labels=D.features(instruments,[status["label_expression"]],start_time="2016-01-01",end_time="2017-12-29")
        labels=labels.swaplevel().sort_index();labels.columns=["LABEL0"]
        status.update(label_success=True,label_seconds=time.perf_counter()-t)
        frame=pd.concat({"feature":features.replace([np.inf,-np.inf],np.nan),"label":labels},axis=1).sort_index()
        frame=frame.loc[frame["label"]["LABEL0"].notna()]
        segments={"train":("2016-01-01","2016-12-15"),"valid":("2017-01-03","2017-06-15"),
                  "test":("2017-07-03","2017-12-20")}
        cal=pd.DatetimeIndex(D.calendar(start_time="2016-01-01",end_time="2017-12-29"))
        for left,right in [("train","valid"),("valid","test")]:
            assert cal.searchsorted(pd.Timestamp(segments[left][1]))+6 < cal.searchsorted(pd.Timestamp(segments[right][0]))
        status["segments"]=segments;status["label_overlap_purged"]=True
        ds=DatasetH(handler=DataHandlerLP.from_df(frame),segments=segments)
        status["segment_rows"]={k:len(ds.prepare(k,col_set="feature")) for k in segments}
        model=LGBModel(num_boost_round=20,early_stopping_rounds=5,num_threads=2,num_leaves=15,seed=123)
        t=time.perf_counter()
        with R.start(experiment_name="phase21-computational-smoke"):
            model.fit(ds,verbose_eval=0)
            status["fit_success"]=True;status["fit_seconds"]=time.perf_counter()-t
            prediction=model.predict(ds)
        assert len(prediction)>0 and np.isfinite(prediction).all()
        status.update(prediction_success=True,prediction_rows=len(prediction))
    except Exception as e:
        status["error"]=repr(e);traceback.print_exc()
    finally:
        (root/"qlib_sample_status.json").write_text(json.dumps(status,indent=2))
        print("SAMPLE_STATUS "+json.dumps(status),flush=True)
    return 0 if status["prediction_success"] else 1
if __name__=="__main__":
    if "--model" in sys.argv:sys.exit(model())
    main()
