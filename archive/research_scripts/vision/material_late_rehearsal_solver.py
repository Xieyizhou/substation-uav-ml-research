"""Exact bounded subset-DP solver without environment changes."""
from collections import Counter
from hashlib import sha256
VERSION='material-late-rehearsal-v1'

def solve(members,sources,seed):
    if not members or set(members)-set(sources):raise ValueError('Unresolved source')
    mids=sorted(set(members));groups=sorted({sources[m] for m in mids})
    n,g=len(members),len(groups)
    if g>16:raise ValueError('DP bound exceeded')
    tail=members[n-g:];tc=Counter(tail);counts=Counter(members)
    def rank(i,m):return sha256(f'{VERSION}|{seed}|{i}|{m}'.encode()).hexdigest()
    dp={0:(0,(),())}
    for i,old in enumerate(tail):
        nxt={}
        for mask,(cost,ties,chosen) in dp.items():
            for j,group in enumerate(groups):
                if mask&(1<<j):continue
                m=min((m for m in mids if sources[m]==group),key=lambda m:(int(m!=old)+int(tc[m]==0),rank(i,m)))
                value=(cost+int(m!=old)+int(tc[m]==0),ties+(rank(i,m),),chosen+(m,))
                newmask=mask|(1<<j)
                if newmask not in nxt or value<nxt[newmask]:nxt[newmask]=value
        dp=nxt
    minimum,_,late=dp[(1<<g)-1]
    remaining=counts-Counter(late);early=[]
    for m in members[:n-g]:
        if remaining[m]>0:early.append(m);remaining[m]-=1
        else:early.append(None)
    filler=[m for m in sorted(remaining,key=lambda m:rank(n,m)) for _ in range(remaining[m])]
    it=iter(filler);sequence=[next(it) if m is None else m for m in early]+list(late)
    if Counter(sequence)!=counts or len({sources[m] for m in sequence[-g:]})!=g:raise ValueError('Invalid solution')
    if sum(a!=b for a,b in zip(sequence,members))!=minimum:raise ValueError('Optimality accounting conflict')
    return dict(members=sequence,minimum_changed_slots=minimum,source_count=g,
        optimal_minimum_last_exposure_slot=n-g,
        proof='Last g-1 slots cannot cover g sources; distinct last g attains bound. Exact DP minimizes late mismatches plus early deficits.',
        tie_break='Versioned SHA256 lexicographic tail choices; preserve earliest eligible early slots and hash-fill deficits.')
