"""Observe actual subscriber arrival gaps without changing sensor validity."""
from collections import deque
from src.sensors.gazebo_lidar import GazeboLidar2DSource


class ArrivalTimes(deque):
    def __init__(self):
        super().__init__(maxlen=60);self.maximum_gap_s=0.;self.long_gaps=[];self.previous=None

    def append(self,value):
        if self.previous is not None:
            gap=value-self.previous;self.maximum_gap_s=max(self.maximum_gap_s,gap)
            if gap>.1:self.long_gaps.append(dict(start=self.previous,end=value,gap_s=gap))
        self.previous=value
        super().append(value)

    def reset_measurement(self):
        self.maximum_gap_s=0.;self.long_gaps=[];self.previous=None


class ObservedLidarSource(GazeboLidar2DSource):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._received_times=ArrivalTimes()

    def diagnostics(self):
        return dict(max_subscriber_arrival_gap_s=self._received_times.maximum_gap_s,long_gaps=self._received_times.long_gaps,health=vars(self.health()),reader_done=self._reader_task.done() if self._reader_task else None,process_returncode=self._process.returncode if self._process else None)
