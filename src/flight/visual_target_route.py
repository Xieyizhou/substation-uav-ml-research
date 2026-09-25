"""Evidence-gated equipment stand-off route requests for bounded SITL trials."""

from collections import deque
import math

from src.flight.envelope_route import envelope_route
from src.ml.artifacts import object_sha256


class StableVisualTarget:
    def __init__(self, *, confirmations=3, minimum_confidence=.85, maximum_age_s=.5):
        self.confirmations = confirmations
        self.minimum_confidence = minimum_confidence
        self.maximum_age_s = maximum_age_s
        self.history = deque(maxlen=confirmations)
        self.latest = None

    def update(self, observations, evidence, *, now):
        candidates = [row for row in observations if row["confidence"] >= self.minimum_confidence
                      and row["class_name"] in {"transformer", "switchgear", "capacitor_bank", "reactor"}
                      and row.get("local_position") is not None]
        if not candidates:
            self.history.clear()
            self.latest = None
            return None
        row = max(candidates, key=lambda r: r["confidence"])
        point = tuple(row["local_position"][key] for key in ("east_m", "north_m", "altitude_m"))
        if not all(math.isfinite(value) for value in (*point, now)):
            raise ValueError("nonfinite visual target")
        timestamp = evidence["capture_timestamp"]
        if self.history:
            prior = self.history[-1]
            if (timestamp <= prior["evidence"]["capture_timestamp"]
                    or now-prior["received"] > self.maximum_age_s
                    or row["class_name"] != prior["observation"]["class_name"]
                    or math.dist(point, prior["point"]) > .3):
                self.history.clear()
                self.latest = None
        self.history.append(dict(observation=row, evidence=evidence, point=point, received=now))
        if len(self.history) < self.confirmations:
            return None
        points = [item["point"] for item in self.history]
        if any(math.dist(a, b) > .3 for a in points for b in points):
            self.latest = None
            return None
        mean = [sum(point[i] for point in points)/len(points) for i in range(3)]
        request = dict(class_name=row["class_name"], local_position=dict(zip(("east_m", "north_m", "altitude_m"), mean)),
                       confidence=min(item["observation"]["confidence"] for item in self.history),
                       evidence=[item["evidence"] for item in self.history], created_monotonic=now,
                       expires_monotonic=now+self.maximum_age_s)
        request["identity"] = object_sha256(request)
        self.latest = request
        return request

    def current(self, now):
        if self.latest is None or not self.latest["created_monotonic"] <= now <= self.latest["expires_monotonic"]:
            return None
        return self.latest


def plan_visual_standoff(request, *, now, frame, start, boxes, standoff_m=5.):
    record = dict(request)
    identity = record.pop("identity", None)
    if object_sha256(record) != identity:
        raise ValueError("visual target evidence changed")
    if not request["created_monotonic"] <= now <= request["expires_monotonic"]:
        raise ValueError("visual target expired")
    target = frame.to_map(request["local_position"]["east_m"], request["local_position"]["north_m"])
    distance = math.dist(start, target)
    if not 5.5 <= distance <= 15:
        raise ValueError("visual target outside bounded approach range")
    angle = math.atan2(start[1]-target[1], start[0]-target[0])
    choices = []
    for offset in (0, -.15, .15, -.3, .3):
        goal = (target[0]+standoff_m*math.cos(angle+offset), target[1]+standoff_m*math.sin(angle+offset))
        if math.dist(goal, start) < .5:
            continue
        try:
            route = envelope_route(boxes, start, goal)
        except ValueError:
            continue
        length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        if length <= 6:
            choices.append((length, route))
    if not choices:
        raise ValueError("no safe visual stand-off route")
    length, route = min(choices, key=lambda item: item[0])
    return dict(request_identity=identity, detected_class=request["class_name"], detected_map_position=target,
                standoff_m=standoff_m, route=route, route_length_m=length,
                selection="YOLO class and synchronized depth/estimator localization; no object truth lookup")
