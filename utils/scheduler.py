from diffusers import (
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler
)

def load_scheduler(pipe, scheduler_name: str):
    scheduler_name = scheduler_name.lower()
    config = pipe.scheduler.config
    
    if scheduler_name == "euler_a":
        new_scheduler = EulerAncestralDiscreteScheduler.from_config(
            config, use_karras_sigmas=True
        )
    elif scheduler_name == "dpm++":
        new_scheduler = DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True,
            algorithm_type="dpmsolver++", solver_type="midpoint",
            final_sigmas_type="zero"
        )
    elif scheduler_name == "ddim":
        new_scheduler = DDIMScheduler.from_config(config)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' tidak dikenal. Gunakan: euler_a, dpm++, ddim")
    
    pipe.scheduler = new_scheduler
    return pipe