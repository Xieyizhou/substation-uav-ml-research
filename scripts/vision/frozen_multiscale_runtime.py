"""Shared future-training/preflight preprocessing. No optimizer or RNG."""
def preprocess(owner,batch,size):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    if size not in (320,640,960) or owner.args.multi_scale!=0:
        raise ValueError('Unfrozen size or second random multi_scale')
    if tuple(batch['img'].shape)!=(6,3,640,640) or batch['img'].dtype!=torch.uint8:
        raise ValueError('Expected original six-image uint8 640 batch')
    result=DetectionTrainer.preprocess_batch(owner,dict(batch))
    if size!=640:
        result['img']=torch.nn.functional.interpolate(result['img'],size=(size,size),mode='bilinear',align_corners=False)
    if tuple(result['img'].shape)!=(6,3,size,size):raise ValueError('Wrong effective input shape')
    if not torch.isfinite(result['img']).all() or result['img'].min()<0 or result['img'].max()>1:raise ValueError('Invalid input range')
    for field in ('cls','bboxes','batch_idx'):
        if not torch.equal(result[field],batch[field].to(result[field].device)):raise ValueError('Complete normalized labels changed')
    return result
