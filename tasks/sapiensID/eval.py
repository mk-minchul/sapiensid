import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_

from src.models import get_model
from src.aligners import get_aligner
from src.evaluations import get_evaluator_by_name, summary
from src.pipelines import pipeline_from_name
from src.fabric.fabric import setup_dataloader_from_dataset
from general_utils.config_utils import load_config

from lightning.pytorch.loggers import WandbLogger
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from functools import partial
from omegaconf import OmegaConf
import pandas as pd
import torch

import lovely_tensors as lt
lt.monkey_patch()

def resolve_variables(config, model_name):
    config = OmegaConf.to_yaml(config)
    config = config.replace('${models.', '${')
    config = OmegaConf.create(config)
    config.yaml_path = model_name
    return config

def get_runname_and_task(ckpt_dir):
    if 'pretrained_models' in ckpt_dir:
        runname = ckpt_dir.split('/')[-1]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = 'pretrained_models'
    elif 'checkpoints' in ckpt_dir:
        runname = ckpt_dir.split('/')[-3]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = code_task
    else:
        runname = ckpt_dir.split('/')[-1]
        code_task = os.path.abspath(__file__).split('/')[-2]
        save_dir_task = code_task
    return runname, save_dir_task, code_task

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--eval_config_name', type=str, default='base')
    parser.add_argument('--pipeline_name', type=str, default='default')
    parser.add_argument('--ckpt_dir', type=str, default="checkpoints/sapiensid_wb4m")
    parser.add_argument('--aligner', type=str, default='')
    parser.add_argument('--use_wandb', type=str2bool, default='True')
    parser.add_argument('--runname_override', type=str, default='')
    parser.add_argument('--seed', type=int, default=2048)
    parser.add_argument('--sapiens_id_test_mask_override', type=str, default='')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Seed set to {args.seed}")

    # setup output dir
    runname = os.path.basename(args.ckpt_dir)
    if args.runname_override != '':
        runname = args.runname_override
    eval_config = load_config(os.path.join(root, 'tasks', 'sapiensID', f'src/evaluations/configs/{args.eval_config_name}.yaml'))
    output_dir = os.path.join(root, 'experiments', runname)

    # load model
    model_config = load_config(os.path.join(args.ckpt_dir, 'model.yaml'))
    model = get_model(model_config)
    model.load_state_dict_from_path(os.path.join(args.ckpt_dir, 'model.pth'))

    # maybe load aligner
    if args.aligner != '':
        if args.aligner.endswith('.yaml'):
            aligner_config = load_config(os.path.join(root, 'tasks', 'sapiensID', f'src/aligners/configs/{args.aligner}'))
            if hasattr(model_config, 'rgb_mean'):
                aligner_config.rgb_mean = model_config.rgb_mean
                aligner_config.rgb_std = model_config.rgb_std
            aligner = get_aligner(aligner_config)
        else:
            from transformers import AutoModel
            HF_TOKEN = os.environ['HF_TOKEN']
            aligner = AutoModel.from_pretrained(f"minchul/{args.aligner}",
                                              trust_remote_code=True,
                                              token=HF_TOKEN).model
            aligner.has_params = lambda : True
        output_dir = os.path.join(output_dir + '_' + args.aligner)
        output_dir = output_dir.replace('.yaml', '')

    else:
        raise ValueError('Aligner not found')

    # launch fabric
    csv_logger = CSVLogger(root_dir=output_dir, flush_logs_every_n_steps=1)
    wandb_logger = WandbLogger(project='sapiensid', save_dir=output_dir,
                               name=os.path.basename(output_dir),
                               log_model=False)
    fabric = Fabric(precision='32-true',
                    accelerator="auto",
                    strategy="ddp",
                    devices=args.num_gpu,
                    loggers=[csv_logger, wandb_logger],
                    )

    if args.num_gpu == 1:
        fabric.launch()
    print(f"Fabric launched with {args.num_gpu} GPUS and {args.precision}")
    fabric.setup_dataloader_from_dataset = partial(setup_dataloader_from_dataset, fabric=fabric, seed=2048)

    # prepare accelerator
    model = fabric.setup(model)
    if aligner.has_trainable_params():
        aligner = fabric.setup(aligner)
    else:
        if aligner is not None:
            aligner = aligner.to(fabric.device)


    # load pipeline
    if args.pipeline_name == 'default':
        full_config_path = os.path.join(args.ckpt_dir, 'config.yaml')
        assert os.path.isfile(full_config_path), f"config.yaml not found at {full_config_path}, try with pipeline name"
        pipeline_name = load_config(full_config_path).pipelines.eval_pipeline_name
    else:
        pipeline_name = args.pipeline_name

    eval_pipeline = pipeline_from_name(pipeline_name, model, aligner)
    eval_pipeline.integrity_check(dataset_color_space='RGB')

    # evaluation callbacks
    evaluators = []
    for name, info in eval_config.per_epoch_evaluations.items():
        assert os.environ['WEBBODY_DATA_ROOT'], 'WEBBODY_DATA_ROOT is not set'
        eval_data_path = os.path.join(eval_config.data_root, info.path)
        eval_type = info.evaluation_type
        eval_batch_size = info.batch_size
        eval_num_workers = info.num_workers
        save_artifacts_path = getattr(info, 'save_artifacts_path', '')
        evaluator = get_evaluator_by_name(eval_type=eval_type, name=name, eval_data_path=eval_data_path,
                                          transform=eval_pipeline.make_test_transform(),
                                          fabric=fabric, batch_size=eval_batch_size, num_workers=eval_num_workers,
                                          save_artifacts_path=save_artifacts_path)
        evaluator.integrity_check(info.color_space, eval_pipeline.color_space)
        evaluators.append(evaluator)

    # Evaluation
    print('Evaluation Started')
    all_result = {}
    for evaluator in evaluators:
        if fabric.local_rank == 0:
            print(f"Evaluating {evaluator.name}")
        result = evaluator.evaluate(eval_pipeline, epoch=0, step=0, n_images_seen=0)
        fabric.barrier()
        if fabric.local_rank == 0:
            print(f"{evaluator.name}")
            print(result)
        all_result.update({evaluator.name + "/" + k: v for k, v in result.items()})

    if fabric.local_rank == 0:
        os.makedirs(os.path.join(output_dir, 'result'), exist_ok=True)
        save_result = pd.DataFrame(pd.Series(all_result), columns=['val'])
        save_result.to_csv(os.path.join(output_dir, f'result/eval_final.csv'))
        mean, summary_dict = summary(save_result, epoch=0, step=0, n_images_seen=0)
        fabric.log_dict(summary_dict)
        summary_result =  pd.DataFrame(pd.Series(summary_dict), columns=['val'])
        summary_result.to_csv(os.path.join(output_dir, f'result/eval_summary_final.csv'))

    print('Evaluation Finished')
