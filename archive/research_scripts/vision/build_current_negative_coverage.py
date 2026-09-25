"""Current-member source verification and visual coverage pages, no approvals."""
from PIL import Image,ImageDraw
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.evaluate_reviewed_negative_order import TRAIN,checked

OUT=TRAIN/'training-fit-diagnosis-v1'

def run():
    p=checked(TRAIN/'protocol.json');rows=sorted((r for r in p['pool_rows'] if r['subset']=='hard_negative'),key=lambda r:r['member_id'])
    links=Path(rows[0]['full_frame_evidence']);review=checked(links);by={r['member_id']:r for r in review['members']};records=[]
    if len(rows)!=120:raise ValueError('Unexpected negative population')
    OUT.mkdir(exist_ok=True)
    for page_index in range(4):
        page=Image.new('RGB',(1800,1440),'white');d=ImageDraw.Draw(page)
        for j,row in enumerate(rows[page_index*30:(page_index+1)*30]):
            src=by[row['member_id']];a=Image.open(row['image_path']).convert('RGB');b=Image.open(src['source_image']).convert('RGB')
            if a.size!=b.size or a.tobytes()!=b.tobytes():raise ValueError('Historical source pixels disagree')
            if file_sha256(row['image_path'])!=row['image_sha256'] or file_sha256(row['label_path'])!=row['label_sha256'] or Path(row['label_path']).read_text().strip():raise ValueError('Invalid negative labels/image')
            index=page_index*30+j;x=(j%5)*360;y=(j//5)*240;a.thumbnail((360,203));page.paste(a,(x,y+28));d.text((x,y),f'N{index:03} {row["member_id"][:22]}',fill='black')
            records.append(dict(index=index,member_id=row['member_id'],image_path=row['image_path'],image_sha256=row['image_sha256'],lineage_id=row['lineage_id'],source_image=src['source_image'],source_sha256=file_sha256(src['source_image']),source_decision=src['decision'],source_pixels_identical=True))
        path=OUT/f'negative-coverage-{page_index}.png'
        if path.exists():raise ValueError('Do not overwrite evidence')
        page.save(path)
    paths=[TRAIN/'protocol.json',links,Path(__file__)]+[OUT/f'negative-coverage-{i}.png' for i in range(4)]
    return write_record(OUT/'negative-coverage.json',dict(status='sources_verified_visual_review_pending',members=records,inputs={str(p.resolve()):file_sha256(p) for p in paths},training_admitted=False,promotable=False))

if __name__=='__main__':print(run()['status'])
