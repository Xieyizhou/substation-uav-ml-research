"""Repeat isolated diagnostic with one optimizer update per microbatch."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,write_json


def main():
    from ultralytics import YOLO
    base=ROOT/'data/research/ml_training_recovery_v1/memorization-v1'
    weights=ROOT/'models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt'
    assert file_sha256(weights)=='820f882a5d375be0a294555c7168d49b6d68c21adce97f00b45fe7b0ebc54836'
    counts={'updates':0}
    model=YOLO(str(weights))
    def count_update(trainer): counts['updates']+=1
    model.add_callback('on_before_zero_grad',count_update)
    model.train(data=str(base/'dataset.yaml'),epochs=10,imgsz=640,batch=6,nbs=6,device='cpu',workers=0,
                optimizer='AdamW',lr0=.001,lrf=1.,warmup_epochs=0,warmup_bias_lr=0,seed=7,
                deterministic=True,patience=0,amp=False,mosaic=0,mixup=0,copy_paste=0,
                degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,
                hsv_h=0,hsv_s=0,hsv_v=0,project=str(base),name='fit-batchmatched',plots=False,save=True,val=False)
    write_json(base/'batchmatched-completion.json',{'status':'complete','optimizer_callback_count':counts['updates'],
              'script_sha256':file_sha256(Path(__file__)),'promotable':False,'training_set_only':True})


if __name__=='__main__':main()
