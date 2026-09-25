"""LLDB callbacks for one hash-pinned arm64 library; reads target memory only.

Software breakpoints are confined to the launched diagnostic process. No target
expressions, register writes, control-flow changes, or on-disk library edits.
"""
import json
import os
import struct
import time
from pathlib import Path
import lldb

SYMBOL = '_ZN2gz9rendering2v822Ogre2BoundingBoxCamera17FullBoundingBoxesEv'
pending = {}
breakpoints = []
count = 0


def emit(record):
    folder = Path(os.environ['EDGE_TRACE_DIR'])
    with (folder/'runtime-trace.jsonl').open('a') as out:
        out.write(json.dumps(dict(record, monotonic=time.monotonic()))+'\n')


def memory(process, address, fmt):
    error = lldb.SBError()
    raw = process.ReadMemory(address, struct.calcsize(fmt), error)
    if not error.Success() or len(raw) != struct.calcsize(fmt):
        raise ValueError('Unable to read pinned layout: '+str(error))
    return struct.unpack(fmt, raw)


def label_for(process, camera, item):
    data = memory(process, camera+0x28, '<Q')[0]
    node = memory(process, data+0xf8, '<Q')[0]
    visited = set()
    while node:
        if node in visited or len(visited) > 2048:
            raise ValueError('Invalid visibility map')
        visited.add(node)
        key, value = memory(process, node+0x1c, '<II')
        if item == key:
            return value
        node = memory(process, node+(0 if item < key else 8), '<Q')[0]
    raise ValueError('Projected item missing from visible map')


def observe(frame, bp_loc, _dict):
    global count
    try:
        if not (Path(os.environ['EDGE_TRACE_DIR'])/'pose-moved.json').exists():
            return False
        process = frame.GetThread().GetProcess()
        sp = frame.FindRegister('sp').GetValueAsUnsigned()
        key = (frame.GetThread().GetThreadID(), sp)
        offset = bp_loc.GetAddress().GetLoadAddress(process.GetTarget()) - int(os.environ['EDGE_TRACE_BASE'])
        if offset == 568:
            camera = frame.FindRegister('x19').GetValueAsUnsigned()
            item = memory(process, sp+0xd4, '<I')[0]
            label = label_for(process, camera, item)
            if label != 128:
                return False
            fp = frame.FindRegister('fp').GetValueAsUnsigned()
            record = dict(event='post_mesh_projection', item_id=item, label=label,
                camera_pointer=camera, thread=key[0], stack=sp,
                min_vertex=memory(process, sp+0x54, '<3f'), max_vertex=memory(process, sp+0x48, '<3f'),
                position=memory(process, sp+0xc8, '<3f'), scale=memory(process, sp+0xac, '<3f'),
                orientation_wxyz=memory(process, sp+0xb8, '<4f'),
                view_matrix=memory(process, fp-0xc0, '<16f'), projection_matrix=memory(process, sp+0x100, '<16f'))
            pending[key] = record
            emit(record)
        elif key in pending:
            record = pending.pop(key)
            emit(dict(event='branch_observed', branch='rejected' if offset == 632 else 'accepted',
                      instruction_offset=offset, item_id=record['item_id'], label=record['label']))
            count += 1
            if count >= 3:
                for bp in breakpoints:
                    bp.SetEnabled(False)
                emit(dict(event='trace_complete', observed_branches=count))
    except Exception as error:
        emit(dict(event='trace_error', reason=repr(error)))
        for bp in breakpoints:
            bp.SetEnabled(False)
    return False


def initialize(frame, bp_loc, _dict):
    if not (Path(os.environ['EDGE_TRACE_DIR'])/'pose-moved.json').exists():
        return False
    target = frame.GetThread().GetProcess().GetTarget()
    base = frame.GetSymbol().GetStartAddress().GetLoadAddress(target)
    os.environ['EDGE_TRACE_BASE'] = str(base)
    module = frame.GetModule()
    emit(dict(event='library_loaded', path=str(module.GetFileSpec()), uuid=module.GetUUIDString(), base=base))
    # Offsets established by disassembly of the hash-pinned installed binary.
    for offset in (568, 632, 660):
        bp = target.BreakpointCreateByAddress(base+offset)
        bp.SetScriptCallbackFunction(__name__+'.observe')
        breakpoints.append(bp)
    bp_loc.GetBreakpoint().SetEnabled(False)
    return False


def __lldb_init_module(debugger, _dict):
    bp = debugger.GetSelectedTarget().BreakpointCreateByName(SYMBOL)
    bp.SetScriptCallbackFunction(__name__+'.initialize')
