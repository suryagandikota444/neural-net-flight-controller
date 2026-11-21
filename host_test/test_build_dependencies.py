"""Build-dependency regression in a disposable source copy; never flashes hardware."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile


def main():
    root=Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="fc-build-regression-") as name:
        work=Path(name)
        for p in root.iterdir():
            if p.suffix in (".c",".h",".ld") or p.name=="Makefile":
                shutil.copy2(p,work/p.name)

        def build(extra=()):
            # Do not inherit an unrelated shell DEBUG value (e.g. "release").
            settings={"DEBUG":"0", "ESC_CAL":"0"}
            settings.update(arg.split("=",1) for arg in extra)
            r=subprocess.run(["make",*[f"{k}={v}" for k,v in settings.items()]],cwd=work,text=True,
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if r.returncode:raise RuntimeError(r.stdout)
            return (work/"firmware.elf").stat().st_mtime_ns

        results={}
        original=build()
        results["unchanged_build_skipped"]=build()==original
        previous=original
        for filename in ("weights.h","nn.h","linker.ld"):
            p=work/filename
            p.write_text(p.read_text()+"\n/* dependency regression */\n")
            current=build()
            results[filename+"_rebuild"]=current>previous
            previous=current
        current=build(["DEBUG=1"])
        results["flags_rebuild"]=current>previous
        results["same_flags_build_skipped"]=build(["DEBUG=1"])==current
        results["flags_revert_rebuild"]=build()>current
        print(json.dumps(results,indent=2))
        if not all(results.values()):raise SystemExit(1)


if __name__=="__main__":main()
