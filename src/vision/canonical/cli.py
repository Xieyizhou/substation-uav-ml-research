"""CLI adapters for offline canonical capture."""
import asyncio
from pathlib import Path

def add_parsers(commands):
    p=commands.add_parser('canonical-view-plan')
    p.add_argument('--map',dest='map_id',choices=['simple','medium','complex'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label-mode',choices=['source','visual-instance'],default='source')
    p.add_argument('--hierarchy-mode',choices=['source','top-level-equipment'],default='source')
    p=commands.add_parser('canonical-view-collect')
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['calibration','pilot'],default='calibration');p.add_argument('--review',type=Path)
    p.add_argument('--view-id',help='Repeat one calibration view only; never releases the full calibration gate')
    p.add_argument('--resume-from',type=Path,help='Resume captured views from an identity-verified receipt into a new output directory')
    p=commands.add_parser('canonical-view-audit')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path)
    p.add_argument('--development-reference',action='append',type=Path,default=[])
    p.add_argument('--protected-reference',action='append',type=Path,default=[])

def handle(args):
    if args.command=='canonical-view-plan':
        from .plan import materialize
        r=materialize(args.map_id,args.output,args.label_mode,args.hierarchy_mode)
        return {'plan':str(args.output/'plan.json'),'identity':r['identity'],'calibration_views':len(r['calibration_views']),'pilot_views':len(r['pilot_views'])}
    if args.command=='canonical-view-collect':
        from .collect import collect
        return asyncio.run(collect(args.plan,args.output,args.mode,args.review,args.view_id,args.resume_from))
    from .audit import audit
    return audit(args.input,args.output,args.review,args.development_reference,args.protected_reference)
