"""Build and prove offline wheelhouses. Public dependencies only; no broker code."""
import argparse, hashlib, json, os, platform, subprocess, sys, time, venv, zipfile
from pathlib import Path
from email.parser import BytesParser

p=argparse.ArgumentParser(); p.add_argument("target",choices=["bt","qlib"]); a=p.parse_args()
root=Path("bundle")/a.target; root.mkdir(parents=True,exist_ok=True)
root=root.resolve(); wheels=root/"wheels"; wheels.mkdir(exist_ok=True)
logs=root/"logs"; logs.mkdir(exist_ok=True)
status={"target":a.target,"stages":[],"files_acquired":False,"offline_install_success":False,
"import_success":False,"pip_check_success":False,"hashes_verified":False,"live":False,"order_calls":0,
"platform":platform.platform(),"python":sys.version,"executable":sys.executable}
env=os.environ.copy(); env.update(PIP_DISABLE_PIP_VERSION_CHECK="1",PYTHONUTF8="1")
def run(name,cmd,cwd=None,extra=None):
    e=env.copy(); e.update(extra or {})
    t=time.perf_counter()
    r=subprocess.run([str(x) for x in cmd],cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                     encoding="utf-8",errors="replace")
    (logs/(name+".txt")).write_text(r.stdout,encoding="utf-8")
    status["stages"].append({"name":name,"argv":[str(x) for x in cmd],"exit_code":r.returncode,
                            "wall_seconds":time.perf_counter()-t})
    print(name,r.returncode,flush=True)
    if r.returncode: raise RuntimeError(name+" failed; see log")
    return r.stdout
def pip(name,*args,python=sys.executable):
    return run(name,[python,"-m","pip","--retries","0","--timeout","45",*args])
try:
    if a.target=="bt":
        pip("download_bt","download","--only-binary=:all:","--dest",wheels,"bt==1.4.1","pip","wheel","setuptools")
    else:
        sha="54355232463878d2eebb91fe0ee5fa7fa1f5976c"
        status["source_commit"]=sha
        source=root/"qlib-source"
        run("clone",["git","clone","--branch","v0.9.8","--depth","1","https://github.com/microsoft/qlib.git",source])
        actual=run("source_commit",["git","rev-parse","HEAD"],cwd=source).strip()
        if actual!=sha: raise RuntimeError("Qlib source tag no longer points at pinned commit")
        run("checkout_exact",["git","checkout","--detach",sha],cwd=source)
        status["source_commit_verified"]=True
        pip("build_tools","install","setuptools","wheel","setuptools-scm","cython","numpy==1.26.4")
        pip("qlib_build","wheel","--no-deps","--no-build-isolation","--wheel-dir",wheels,source)
        qwheel=next(wheels.glob("pyqlib-*.whl"))
        constraints=root/"constraints.txt"; constraints.write_text("numpy==1.26.4\npandas<3\n",encoding="utf-8")
        pip("dependencies","wheel","--wheel-dir",wheels,"-c",constraints,qwheel,"lightgbm==4.7.0",
            "pip","setuptools","wheel","setuptools-scm","cython")
        shutil=__import__("shutil")
        shutil.make_archive(str(root/"source-pinned"),"zip",source)
    manifests=[]; lock=[]
    for f in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(f) as z:
            meta=BytesParser().parsebytes(z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))))
        sha=hashlib.sha256(f.read_bytes()).hexdigest()
        manifests.append({"filename":f.name,"sha256":sha,"size":f.stat().st_size,"name":meta["Name"],
                          "version":meta["Version"],"license":meta["License"],"license_expression":meta["License-Expression"]})
        lock.append(f'{meta["Name"]}=={meta["Version"]} --hash=sha256:{sha}')
    (root/"manifest.json").write_text(json.dumps(manifests,indent=2),encoding="utf-8")
    (root/"requirements.lock").write_text("\n".join(lock)+"\n",encoding="utf-8")
    status["files_acquired"]=True; status["wheel_count"]=len(manifests)
    for f in manifests:
        if hashlib.sha256((wheels/f["filename"]).read_bytes()).hexdigest()!=f["sha256"]:
            raise RuntimeError("SHA256 mismatch")
    status["hashes_verified"]=True
    offline=root/"offline-venv"; venv.create(offline,with_pip=True)
    python=offline/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    offline_env={"PIP_NO_INDEX":"1","PIP_DISABLE_PIP_VERSION_CHECK":"1"}
    run("offline_install",[python,"-m","pip","--isolated","install","--no-index","--find-links",wheels,
                           "--require-hashes","-r",root/"requirements.lock"],extra=offline_env)
    status["offline_install_success"]=True
    code=("import bt; from importlib.metadata import version; assert version('bt')=='1.4.1'; print(version('bt'))"
          if a.target=="bt" else
          "import qlib,lightgbm; from qlib.contrib.data.handler import Alpha158; from qlib.contrib.model.gbdt import LGBModel; "
          "from importlib.metadata import version; assert version('pyqlib')=='0.9.8'; assert version('lightgbm')=='4.7.0'; "
          "print(version('pyqlib'),version('lightgbm')); print(Alpha158,LGBModel)")
    run("offline_import",[python,"-B","-c",code],extra=offline_env);status["import_success"]=True
    run("offline_pip_check",[python,"-m","pip","check"],extra=offline_env);status["pip_check_success"]=True
    run("offline_freeze",[python,"-m","pip","freeze","--all"],extra=offline_env)
except Exception as e:
    status["error"]=str(e)
finally:
    (root/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    print(json.dumps(status,indent=2),flush=True)
sys.exit(0 if status["import_success"] and status["pip_check_success"] else 1)
