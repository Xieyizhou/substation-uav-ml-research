"""Close the A/B/C/D development grid without selecting or promoting a model."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json

BASE=ROOT/"data/research/ml_training_recovery_v1/visual-augmentation-training-v2"


def finalize():
    protocol_path=BASE/"protocol.json"; evaluation_path=BASE/"development-evaluation.json"
    protocol=json.loads(protocol_path.read_text()); evaluation=json.loads(evaluation_path.read_text())
    if evaluation["status"]!="development_evaluation_complete": raise ValueError("Development evaluation incomplete")
    cells=[]
    for arm in "ABCD":
        for seed in (7,17,27):
            path=BASE/f"arm-{arm}-seed-{seed}/completion.json"; row=json.loads(path.read_text())
            if row["status"]!="complete" or row["optimizer_steps"]!=100 or row["observed_draws"]!=600 or not row["exposure_verified"]: raise ValueError("Training cell incomplete")
            if row["protocol_identity"]!=protocol["identity"] or file_sha256(Path(row["weights"]))!=row["weights_sha256"]: raise ValueError("Training cell identity mismatch")
            cells.append({"arm":arm,"seed":seed,"completion_path":str(path),"completion_sha256":file_sha256(path),"weights":row["weights"],"weights_sha256":row["weights_sha256"]})
    result={"status":"complete_with_tradeoff_no_candidate_selected","training_cells":cells,"development_evaluation_identity":evaluation["identity"],"development_evaluation_sha256":file_sha256(evaluation_path),"protocol_identity":protocol["identity"],"protocol_sha256":file_sha256(protocol_path),"candidate_selected":False,"unseen_scene_status":"sealed_not_evaluated","decision":"D combines the strongest material/lighting gains with lower false positives than A/B, but reduces mean original-condition planned-instance hit rate versus A and has large seed variance. No acceptance tolerance was pre-frozen, so selecting a seed or opening the unseen test would be post-hoc.","training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(BASE/"completion.json",result);return result


if __name__=="__main__": print(json.dumps(finalize(),indent=2))
