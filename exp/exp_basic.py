import os
import torch
from models import GPT4ts,Model4F,Bert_v1,GPT4ts_v1,GPT4ts_v2,GPT4ts_v3,GPT4ts_v4,qwen4ts,qwen4ts_v1,qwen4ts_v2,qwen4ts_linear,qwen4ts_v3

from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args


        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        args.exp_name = "bert_vq_96to96_lr1e-4"
        args.log_with = ["tensorboard"]

        assert args.deepspeed_config_path.endswith('.json') and os.path.exists(args.deepspeed_config_path), \
            f"Invalid DeepSpeed config path: {args.deepspeed_config_path}"
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config_path)

        exp_log_dir = os.path.join(args.checkpoints, 'logs', args.exp_name)

        loggers = args.log_with if isinstance(args.log_with, list) else [args.log_with]

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            deepspeed_plugin=deepspeed_plugin,
            log_with=loggers,
            project_dir=exp_log_dir
        )

        self.device = self.accelerator.device


        if self.accelerator.is_local_main_process:
            print("=" * 50)
            print(f"[Accelerator] Initialized on device: {self.device}")
            print(f"[Accelerator] Logging to: {exp_log_dir}")
            print(f"[Accelerator] Log backends: {loggers}")
            print("=" * 50)

        self.model_dict = {
            'Bert4ts': Bert_v1,
            'GPT4ts': GPT4ts,
            'Model4F':Model4F,
            'GPT4ts1':GPT4ts_v1,
            'GPT4ts2':GPT4ts_v2,
            'GPT4ts3':GPT4ts_v3,
            'GPT4ts4':GPT4ts_v4,
            'qwen4ts':qwen4ts,
            'qwen4ts1':qwen4ts_v1,
            'qwen4ts2':qwen4ts_v2,
            'linear':qwen4ts_linear,
            'qwen4ts3':qwen4ts_v3
        }
        self.model,self.vq_model,self.classification_weight = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
