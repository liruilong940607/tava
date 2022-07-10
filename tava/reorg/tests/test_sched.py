""" Example of use for schedules. """
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.create(
    [
        # {
        #     "_target_": "tava.reorg.utils.schedules.ConstantSchedule",
        #     "value": 1.0,
        # },
        # {
        #     "_target_": "tava.reorg.utils.schedules.LinearSchedule",
        #     "initial_value": 1.0,
        #     "final_value": 0.1,
        #     "num_steps": 100,
        # },
        # {
        #     "_target_": "tava.reorg.utils.schedules.ExponentialSchedule",
        #     "initial_value": 1.0,
        #     "final_value": 0.1,
        #     "num_steps": 100,
        # },
        # {
        #     "_target_": "tava.reorg.utils.schedules.CosineEasingSchedule",
        #     "initial_value": 1.0,
        #     "final_value": 0.1,
        #     "num_steps": 100,
        # },
        # {
        #     "_target_": "tava.reorg.utils.schedules.StepSchedule",
        #     "initial_value": 1.0,
        #     "decay_interval": 5,
        #     "decay_factor": 0.1,
        #     "max_decays": 3,
        # },
        # {
        #     "_target_": "tava.reorg.utils.schedules.PiecewiseSchedule",
        #     "schedules": [
        #         {
        #             "_target_": "tava.reorg.utils.schedules.ConstantSchedule",
        #             "value": 1.0,
        #         },
        #         {
        #             "_target_": "tava.reorg.utils.schedules.LinearSchedule",
        #             "initial_value": 1.0,
        #             "final_value": 0.1,
        #             "num_steps": 95,
        #         },
        #     ],
        #     "num_steps": [5, 95],
        # },
        {
            "_target_": "tava.reorg.utils.schedules.DelayedSchedule",
            "base_schedule": {
                "_target_": "tava.reorg.utils.schedules.ExponentialSchedule",
                "initial_value": 5e-4,
                "final_value": 5e-6,
                "num_steps": 100,
            },
            "delay_steps": 0,
            "delay_mult": 0.01,
        },
    ]
)

from tava.utils.training import learning_rate_decay

for cfg_sched in cfg:
    sched = instantiate(cfg_sched)
    # print ("step[10]", cfg_sched._target_, "value is", sched.get(step=10))
    for step in range(100):
        print (
            sched(step),
            learning_rate_decay(step, 5e-4, 5e-6, 100, 0, 0.01)
        )
