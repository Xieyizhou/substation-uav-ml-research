"""Identity and leakage gates for v2.1 hard-example collections."""
import hashlib, json
from pathlib import Path

def load_protocol(path):
    record=json.loads(Path(path).read_text());
    if record.get("schema_version")!=1 or record.get("split_group")!="recording_seed": raise ValueError("unsupported hard-example protocol")
    if set(record["development_seeds"]) & set(record["validation_seeds"]): raise ValueError("hard-example seeds cross splits")
    return record

def audit_members(rows, protocol):
    allowed={seed:"development" for seed in protocol["development_seeds"]}|{seed:"validation" for seed in protocol["validation_seeds"]}; hashes={}; perceptual=[]; counts={"development":0,"validation":0}
    threshold=int(protocol.get("near_duplicate_hamming_threshold",6))
    for row in rows:
        seed=int(row["seed"]); split=row["split"]
        if allowed.get(seed)!=split: raise ValueError("hard-example seed appears in the wrong split")
        if row.get("source_partition") in protocol["forbidden_partitions"]: raise ValueError("validation or blind data cannot enter v2.1 training")
        digest=row["image_sha256"]
        if digest in hashes and hashes[digest]!=split: raise ValueError("exact duplicate crosses hard-example splits")
        hashes[digest]=split; counts[split]+=1
        visual=row.get("perceptual_hash")
        if visual is not None:
            value=int(visual,16)
            for previous, previous_split in perceptual:
                if previous_split!=split and (value^previous).bit_count()<=threshold:
                    raise ValueError("near duplicate crosses hard-example splits")
            perceptual.append((value,split))
    result={"schema_version":1,"protocol_id":protocol["protocol_id"],"member_count":len(rows),"split_counts":counts,"membership":[row["sample_id"] for row in sorted(rows,key=lambda x:x["sample_id"])]}
    result["identity"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest(); return result
