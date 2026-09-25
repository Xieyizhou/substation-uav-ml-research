import json
import unittest
from pathlib import Path
from src.flight.envelope_route import envelope_route
from src.flight.tracking_envelope import check_route


class HistoricalEnvelopeEvidenceTests(unittest.TestCase):
    def test_frozen_positive_layout_has_robust_route(self):
        p=Path('data/research/material-shadow-v1/autonomy-avoidance-v1/envelope-aware-scene-v1/protocol.json')
        scene=json.loads(p.read_text())
        for case in scene['cases']:
            boxes=scene['static_boxes']+[case['harness_box']]
            route=envelope_route(boxes,(1.6,1.1),scene['goal'])
            self.assertTrue(check_route(route,boxes,map_uncertainty=.2)['passed'])

    def test_original_tracking_failure_is_not_relabelled(self):
        p=Path('data/research/material-shadow-v1/autonomy-avoidance-v1/tracking-envelope-feasibility-v1/completion.json')
        record=json.loads(p.read_text())
        self.assertFalse(record['flight_authorized'])
        self.assertFalse(record['existing_geometry_passed'])

