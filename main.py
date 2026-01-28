import torch
from tridi.model import get_model as get_diffusion_model
from tridi.model.nn_baseline import NNBaselineModel, is_nn_baseline_checkpoint


from config.config import ProjectConfig
from tridi.core.evaluator import Evaluator
from tridi.core.sampler import Sampler
from tridi.core.trainer import Trainer
#from tridi.model import get_model
from tridi.utils import training as training_utils
from tridi.utils.exp import init_exp, init_wandb, init_logging, parse_arguments

def get_model_for_sample(cfg: ProjectConfig):
    ckpt_path = getattr(cfg.resume, "checkpoint", None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if is_nn_baseline_checkpoint(ckpt):
            #  baseline: 
            return NNBaselineModel(cfg, checkpoint_path=ckpt_path)

    #  default: diffusion
    return get_diffusion_model(cfg)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')

    # Parse arguments
    arguments = parse_arguments()

    # Initialzie run
    cfg: ProjectConfig = init_exp(arguments)

    # Logging
    init_logging(cfg)
    if cfg.logging.wandb:
        init_wandb(cfg)

    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    if cfg.run.job in ['train', 'sample']:
        if cfg.run.job == 'train':
            model = get_diffusion_model(cfg)          # train always be diffusion
        else:
            model = get_model_for_sample(cfg)         # sample will choose automatically according to checkpoint 


        if cfg.run.job == 'train':
            trainer = Trainer(cfg, model)

            trainer.train()
        elif cfg.run.job == 'sample':
            sampler = Sampler(cfg, model)
            if cfg.sample.target == 'meshes':
                sampler.sample()
            elif cfg.sample.target == 'hdf5':
                sampler.sample_to_hdf5()
            else:
                raise ValueError(f"Invalid target {cfg.sample.target}")
    elif cfg.run.job == 'eval':
        evaluator = Evaluator(cfg)
        evaluator.evaluate()
    else:
        raise ValueError(f"Invalid job type {cfg.run.job}")


if __name__ == '__main__':
    main()
