from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Timer:
    """
    A Timer class that keeps track of timings for different labels.

    timer = Timer()
    timer.register("step 1")
    # some code here...
    timer.register("step 2")
    # some code here...
    timer.register("step 3")
    # some code here...
    print(timer.timings())
    """

    _timings: dict[str, datetime] = field(default_factory=dict)

    def register(self, label: str) -> None:
        assert label not in self._timings, self._timings.keys()
        self._timings[label] = datetime.now()

    def since_start(self) -> timedelta:
        start_time = next(iter(self._timings.values()))
        return datetime.now() - start_time

    def timing_dict(self) -> dict[str, float]:
        result: dict[str, float] = dict()
        timings: list[tuple[str, datetime]] = [
            *self._timings.items(),
            ("end", datetime.now()),
        ]
        for (label, start_time), (_, end_time) in zip(timings[:-1], timings[1:]):
            result[f"perf/timing_secs/{label}"] = (
                end_time - start_time
            ).total_seconds()
        return result
