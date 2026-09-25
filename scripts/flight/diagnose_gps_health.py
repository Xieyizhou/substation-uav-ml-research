"""Read-only scalar ULog diagnosis for the isolated PX4 preflight failure."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

TYPES = dict(zip(['uint64_t','int64_t','uint32_t','int32_t','uint16_t',
                 'int16_t','uint8_t','int8_t','float','double','bool','char'],
                'QqIiHhBbfd?c'))
TOPICS = {'estimator_status', 'vehicle_global_position', 'vehicle_gps_position',
          'estimator_status_flags', 'failsafe_flags'}


def read_samples(path, topics=TOPICS, allow_incomplete_tail=False):
    data = Path(path).read_bytes()
    if data[:7] != b'ULog\x01\x12\x35':
        raise ValueError('Not a ULog file')
    offset = 16
    formats, subscriptions = {}, {}
    while offset + 3 <= len(data):
        size, kind = struct.unpack_from('<HB', data, offset)
        body = data[offset+3:offset+3+size]
        if len(body) != size:
            if allow_incomplete_tail:return
            raise ValueError('Truncated ULog record')
        offset += 3 + size
        if kind == ord('F'):
            name, fields = body.decode().split(':', 1)
            formats[name] = fields
        elif kind == ord('A'):
            subscriptions[struct.unpack_from('<H', body, 1)[0]] = (body[0], body[3:].decode())
        elif kind == ord('D'):
            multi, name = subscriptions.get(struct.unpack_from('<H', body)[0], (0, ''))
            if name not in topics:
                continue
            pos, values = 2, {}
            for field in formats[name].split(';'):
                if not field:
                    continue
                dtype, key = field.split()
                # ULog omits trailing alignment padding in data records.
                if key.startswith('_padding'):
                    continue
                count = 1
                if '[' in dtype:
                    dtype, count = dtype[:-1].split('[')
                    count = int(count)
                fmt = '<' + str(count) + TYPES[dtype]
                value = struct.unpack_from(fmt, body, pos)
                pos += struct.calcsize(fmt)
                values[key] = value[0] if count == 1 else value
            yield name, multi, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ulog', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    last, flags = {}, Counter()
    for name, multi, values in read_samples(args.ulog):
        last[f'{name}:{multi}'] = values
        if name == 'estimator_status':
            flags[str(values['gps_check_fail_flags'])] += 1
    # Select finite scalar evidence only, avoiding NaN diagnostics in JSON.
    fields = {
        'estimator_status:0': ['timestamp', 'gps_check_fail_flags', 'filter_fault_flags'],
        'vehicle_global_position:0': ['timestamp', 'lat_lon_valid', 'alt_valid', 'eph', 'epv'],
        'vehicle_gps_position:0': ['timestamp', 'fix_type', 'satellites_used', 'eph', 'epv'],
        'estimator_status_flags:0': ['timestamp', 'cs_gnss_pos', 'cs_gnss_vel', 'cs_gnss_fault'],
        'failsafe_flags:0': ['timestamp', 'global_position_invalid', 'local_position_invalid']}
    evidence = {topic: {key: last[topic][key] for key in keys} for topic, keys in fields.items()}
    write_record(args.output, dict(status='source_fix_pending_build_and_runtime_verification',
        last_samples=evidence, gps_failure_flag_counts=dict(flags),
        undefined_bit_observed=bool(evidence['estimator_status:0']['gps_check_fail_flags'] & ~0x7ff),
        inputs={str(p.resolve()): file_sha256(p) for p in [args.ulog, Path(__file__)]},
        flight_tested=False, training_admitted=False, promotable=False))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
