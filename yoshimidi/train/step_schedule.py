from pydantic import BaseModel


class StepSchedule(BaseModel, extra="forbid"):
    every_n_steps: int
    enable: bool = True
    at_begin: bool = True
    at_end: bool = True

    def should_run(self, step: int, max_steps: int) -> bool:
        if not self.enable:
            return False
        if self.at_begin and step == 0:
            return True
        if self.at_end and step == (max_steps - 1):
            return True
        return step > 0 and step % self.every_n_steps == 0
