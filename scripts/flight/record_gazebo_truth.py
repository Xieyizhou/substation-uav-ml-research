#!/usr/bin/env python3
"""Write-only Gazebo truth recorder; never imported by flight runtime."""
import argparse, hashlib, json, os, signal, subprocess
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--topic",default="/research_camera/boxes"); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--receipt",type=Path,required=True); args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True); env={**os.environ}; env.setdefault("GZ_IP","127.0.0.1"); env.setdefault("GZ_PARTITION","substation_uav")
    process=subprocess.Popen(["gz","topic","-e","-t",args.topic,"-j"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    count=0; digest=hashlib.sha256()
    try:
        with args.output.open("w",encoding="utf-8") as target:
            for line in process.stdout:
                line=line.strip()
                if not line: continue
                row=json.loads(line); encoded=json.dumps(row,sort_keys=True,separators=(",",":"))
                target.write(encoded+"\n"); digest.update((encoded+"\n").encode()); count+=1
    except KeyboardInterrupt: pass
    finally:
        if process.poll() is None: process.send_signal(signal.SIGTERM)
        process.communicate()
    receipt={"schema_version":1,"evidence_role":"offline_truth_only","topic":args.topic,"message_count":count,"stream_sha256":digest.hexdigest()}
    receipt["identity"]=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(",",":")).encode()).hexdigest(); args.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return 0 if count else 2
if __name__=="__main__": raise SystemExit(main())
