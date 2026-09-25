"""Preserve 2670 old slots; two paired new-source families in the remaining 30."""
import copy
from collections import Counter
from hashlib import sha256
from pathlib import Path
from scripts.vision.closed_source_training_quality import OUT,PAIRS,prior,reference,export

KEYS=tuple(f'{family}-{seed}' for family in ('E','M') for seed in (7,17,27))
VERSION='closed-source-retention-control-v1'


def digest(*parts):return sha256('|'.join(map(str,(VERSION,)+parts)).encode()).hexdigest()


def sequences(old,seed,members):
    rows={x['member_id']:x for x in old['pool_rows']};before=old['schedules'][f'T-{seed}']
    positions=[i for i,m in enumerate(before) if rows[m]['subset']=='bridge_positive' and rows[m]['variant']!='original']
    if len(positions)!=30:raise ValueError('Expected exactly 30 replaceable material slots')
    lookup={(m['pair_id'],m['variant']):m['member_id'] for m in members}
    tokens=[(pair,n) for pair in PAIRS for n in range(3 if ':transformer:' in pair else 4)]
    tokens.sort(key=lambda x:digest(seed,*x));slots=sorted(positions,key=lambda i:digest(seed,'position',i))
    e=list(before);m=list(before);records=[]
    for i,(pair,n) in zip(slots,tokens,strict=True):
        colors=sorted(('warm','cool','neutral'),key=lambda c:digest(seed,pair,c))
        color='original' if n==0 else colors[n-1]
        e[i]=lookup[pair,'original'];m[i]=lookup[pair,color]
        records.append(dict(position=i,pair_id=pair,occurrence=n,material_variant=color,reference_member=before[i]))
    return e,m,sorted(records,key=lambda r:r['position'])


def counts(seq,rows):
    return dict(images=dict(Counter(seq)),subsets=dict(Counter(rows[m]['subset'] for m in seq)),
        classes={c:sum(rows[m]['class_instances'].get(c,0) for m in seq) for c in ('transformer','switchgear','capacitor_bank','reactor')},
        lineages=dict(Counter(rows[m]['lineage_id'] for m in seq)))


def check(p,old):
    rows={r['member_id']:r for r in p['pool_rows']};new=[r for r in p['pool_rows'] if r['member_id'].startswith('new-')]
    if len(rows)!=len(p['pool_rows']):raise ValueError('Duplicate member identity')
    for r in rows.values():
        for field in ('image','label'):
            if prior.file_sha256(r[field+'_path'])!=r[field+'_sha256']:raise ValueError('Full member bytes changed')
    for seed in (7,17,27):
        expected_e,expected_m,records=sequences(old,seed,new);before=old['schedules'][f'T-{seed}']
        for family,expected in (('E',expected_e),('M',expected_m)):
            key=f'{family}-{seed}';seq=p['schedules'][key]
            if seq!=expected or p['replacement_records'][key]!=records:raise ValueError('Sequence not reproducible')
            if len(seq)!=2700 or p['brightness_factors'][key]!=old['brightness_factors'][f'T-{seed}'] or p['training_config'][key]!=old['training_config'][f'T-{seed}']:raise ValueError('Budget/config/brightness changed')
            if any(m in p['held_members'] for m in seq):raise ValueError('Held member restored')
            if counts(seq,rows)['subsets']!=counts(before,rows)['subsets']:raise ValueError('Subset budget changed')
            for i,a in enumerate(before):
                if rows[a]['variant']=='original' or rows[a]['subset']=='hard_negative':
                    if seq[i]!=a:raise ValueError('Preserved position changed')
        e=p['schedules'][f'E-{seed}'];m=p['schedules'][f'M-{seed}']
        if counts(e,rows)['classes']!=counts(m,rows)['classes'] or counts(e,rows)['lineages']!=counts(m,rows)['lineages']:raise ValueError('Paired exposure changed')
        for a,b in zip(e,m,strict=True):
            if a==b:continue
            if rows[a]['pair_id']!=rows[b]['pair_id'] or Path(rows[a]['label_path']).read_bytes()!=Path(rows[b]['label_path']).read_bytes():raise ValueError('Paired full supervision changed')
        for pair in PAIRS:
            if not any(rows[x].get('pair_id')==pair and rows[x]['variant']=='original' for x in m):raise ValueError('New-source original missing')


def protocol():
    old,_,_=reference.contract('T-7');dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);check(p,old);return p
    manifest=export()
    if manifest['gaps']:raise ValueError('Reference exclusion failed')
    p={k:copy.deepcopy(old[k]) for k in ('pool_rows','names','initialization','evaluation','held_members')}
    p['pool_rows']+=manifest['members'];rows={r['member_id']:r for r in p['pool_rows']}
    p.update(schedules={},brightness_factors={},training_config={},listings={},replacement_records={},ledger={},totals={},historical_class_delta={})
    paths=[OUT/'export/manifest.json',OUT/'quality-review.json',reference.OUT/'protocol.json',reference.OUT/'decision-protocol.json',Path(__file__).resolve()]
    # Bind the shared actual loader and all new entry points before signing preflight inputs.
    for name in ('brightness_transfer_runtime.py','order_retention_runtime.py','preflight_closed_source_control.py','train_closed_source_control.py'):
        paths.append(Path(__file__).with_name(name))
    for seed in (7,17,27):
        ref=f'T-{seed}';reference.complete(ref);paths.append(reference.OUT/'training'/ref/'completion.json')
        e,m,records=sequences(old,seed,manifest['members'])
        for family,seq in (('E',e),('M',m)):
            key=f'{family}-{seed}';p['schedules'][key]=seq;p['replacement_records'][key]=records
            p['brightness_factors'][key]=old['brightness_factors'][ref];p['training_config'][key]=old['training_config'][ref]
            listing=OUT/(key+'.txt');text=''.join(rows[x]['image_path']+'\n' for x in sorted(set(seq)))
            if listing.exists():
                if listing.read_text()!=text:raise ValueError('Listing drift')
            else:listing.write_text(text)
            p['listings'][key]=str(listing);paths.append(listing)
            p['totals'][key]=counts(seq,rows)
            p['ledger'][key]=[dict(first_step=i//6+1,**counts(seq[i:i+300],rows)) for i in range(0,2700,300)]
            p['historical_class_delta'][key]={c:p['totals'][key]['classes'][c]-counts(old['schedules'][ref],rows)['classes'][c] for c in p['names']}
    import platform,torch,ultralytics
    p.update(status='frozen_loader_preflight_required',families={'E':'new-source original control','M':'same sources with original retained plus materials'},
        acceptance_policy=prior.read(reference.OUT/'decision-protocol.json'),selected_pairs=list(PAIRS),
        exposure_design='30 new-source slots: reactor/capacitor/switchgear 4 per layout, transformer 3 per layout; E all original; M one original per source plus hash-ordered materials. No old original/negative position changes.',
        limitations=['Eight pose groups in two layouts, shared existing assets; no cross-site/asset generalization claim','Historical T comparison changes sources and full class exposure; only E/M have exact paired class/source supervision','2700 slots do not expose every candidate; unselected/held candidates remain zero exposure','Frozen brightness values retained at original positions; no tuning'],
        environment=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__),
        training_started=False,training_ready=False,training_admitted=False,promotable=False,inputs={str(x):prior.file_sha256(x) for x in paths})
    check(p,old);return prior.frozen(dest,p)


if __name__=='__main__':print(protocol()['status'],'NO_TRAINING')
