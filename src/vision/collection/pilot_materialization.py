"""Parallel annotation materialization for recorded PNG frames."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.sensors.camera_decoder import decode_camera_payload
from src.vision.collection.synchronization import materialize_frame_annotation
from src.vision.contracts.pilot import mission_phase


def _materialize_one(arguments):
    (
        index,
        synchronized,
        output,
        recording_id,
        scenario_id,
        map_id,
        seed,
        events,
    ) = arguments
    if synchronized.truth is None:
        return None, None
    decoded = decode_camera_payload(synchronized.frame, output).image
    annotation = materialize_frame_annotation(
        synchronized,
        decoded,
        recording_id=recording_id,
        scenario_id=scenario_id,
        map_id=map_id,
        seed=seed,
        mission_phase=mission_phase(synchronized.frame, events),
        frame_order_reference=index,
    )
    return annotation, decoded.decoder_configuration_id


def materialize_recording_annotations(
    synchronized,
    output,
    *,
    recording_id,
    scenario_id,
    map_id,
    seed,
    events,
    workers=4,
):
    arguments = (
        (
            index,
            item,
            output,
            recording_id,
            scenario_id,
            map_id,
            seed,
            events,
        )
        for index, item in enumerate(synchronized)
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(_materialize_one, arguments))
    annotations = tuple(annotation for annotation, _ in results)
    decoder_ids = {
        decoder_id for _, decoder_id in results if decoder_id is not None
    }
    return annotations, decoder_ids
