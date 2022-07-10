# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic Schedules."""
import abc
import math
from typing import List

import numpy as np


class Schedule(abc.ABC):
  """An interface for generic schedules.."""

  @abc.abstractmethod
  def get(self, step):
    """Get the value for the given step."""
    raise NotImplementedError

  def __call__(self, step):
    return self.get(step)


class ConstantSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, value):
    super().__init__()
    self.value = value

  def get(self, step):
    """Get the value for the given step."""
    return self.value


class LinearSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    if num_steps <= 1:
        raise ValueError('num_steps need to be larger than 1.')
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    if step >= self.num_steps:
      return self.final_value
    alpha = step / self.num_steps
    return (1.0 - alpha) * self.initial_value + alpha * self.final_value


class ExponentialSchedule(Schedule):
  """Exponentially decaying scheduler."""

  def __init__(self, initial_value, final_value, num_steps, eps=1e-10):
    super().__init__()
    if initial_value <= final_value:
      raise ValueError('Final value must be less than initial value.')
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps
    self.eps = eps

  def get(self, step):
    """Get the value for the given step."""
    if step >= self.num_steps:
      return self.final_value
    final_value = max(self.final_value, self.eps)
    base = final_value / self.initial_value
    exponent = step / (self.num_steps - 1)
    return self.initial_value * base**exponent


class CosineEasingSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    alpha = min(step / self.num_steps, 1.0)
    scale = self.final_value - self.initial_value
    x = min(max(alpha, 0.0), 1.0)
    return (self.initial_value
            + scale * 0.5 * (1 + math.cos(math.pi * x + math.pi)))


class StepSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self,
               initial_value,
               decay_interval,
               decay_factor,
               max_decays,
               final_value=None):
    super().__init__()
    self.initial_value = initial_value
    self.decay_factor = decay_factor
    self.decay_interval = decay_interval
    self.max_decays = max_decays
    if final_value is None:
      final_value = self.initial_value * self.decay_factor**self.max_decays
    self.final_value = final_value

  def get(self, step):
    """Get the value for the given step."""
    phase = step // self.decay_interval
    if phase >= self.max_decays:
      return self.final_value
    else:
      return self.initial_value * self.decay_factor**phase


class PiecewiseSchedule(Schedule):
  """A piecewise combination of multiple schedules."""

  def __init__(
      self, schedules: List[Schedule], num_steps: List[int]):
    self.schedules = list(schedules)
    milestones = np.array(num_steps)
    self.milestones = np.cumsum(milestones)[:-1]

  def get(self, step):
    idx = np.searchsorted(self.milestones, step, side='right')
    schedule = self.schedules[idx]
    base_idx = self.milestones[idx - 1] if idx >= 1 else 0
    return schedule.get(step - base_idx)


class DelayedSchedule(Schedule):
  """Delays the start of the base schedule."""

  def __init__(self, base_schedule: Schedule, delay_steps, delay_mult):
    self.base_schedule = base_schedule
    self.delay_steps = delay_steps
    self.delay_mult = delay_mult

  def get(self, step):
    if self.delay_steps == 0:
        delay_rate = 1.0
    else:
        delay_rate = (
            self.delay_mult
            + (1 - self.delay_mult)
            * math.sin(0.5 * math.pi * np.clip(step / self.delay_steps, 0, 1)))
    return delay_rate * self.base_schedule(step)


SCHEDULE_MAP = {
    'constant': ConstantSchedule,
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule,
    'cosine_easing': CosineEasingSchedule,
    'step': StepSchedule,
    'piecewise': PiecewiseSchedule,
    'delayed': DelayedSchedule,
}
