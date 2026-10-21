import runpy
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from packaging.requirements import Requirement
from transformers import BertConfig, BertModel

import config
from models_def import AddressTagging, load_params
from train import AddressTaggingTrainer


ROOT = Path(__file__).resolve().parents[1]


class DeviceTests(unittest.TestCase):
    def test_auto_selection(self):
        for cuda, mps, expected in [(True, True, 'cuda'), (False, True, 'mps'),
                                     (False, False, 'cpu')]:
            with self.subTest(expected=expected), patch(
                'torch.cuda.is_available', return_value=cuda
            ), patch('torch.backends.mps.is_available', return_value=mps):
                settings = runpy.run_path(str(ROOT / 'config.py'))
                self.assertEqual(settings['DEVICE'].type, expected)

    def test_installation_markers(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        source = project['tool']['uv']['sources']['torch'][0]
        source_marker = Requirement('torch; ' + source['marker']).marker
        requirements = [Requirement(line) for line in (ROOT / 'requirements.txt').read_text().splitlines()
                        if line.startswith('torch==')]
        for platform in ['darwin', 'linux', 'win32']:
            env = {'sys_platform': platform}
            matches = [req for req in requirements if req.marker.evaluate(env)]
            self.assertEqual(len(matches), 1)
            self.assertEqual('cu130' in str(matches[0].specifier), platform != 'darwin')
            self.assertEqual(source_marker.evaluate(env), platform != 'darwin')

    def test_training_prediction_and_checkpoint_on_available_devices(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        if torch.backends.mps.is_available():
            devices.append('mps')
        for device in devices:
            with self.subTest(device=device):
                bert = BertModel(BertConfig(vocab_size=21128, hidden_size=16,
                                           num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=32))
                with patch('models_def.BertModel.from_pretrained', return_value=bert):
                    model = AddressTagging(config.PRETRAINED_DIR / 'bert-base-chinese', config.LABELS)
                trainer = AddressTaggingTrainer(model, torch.device(device), 1, 1e-3, checkpoint_steps=0)
                self.assertEqual(next(model.parameters()).device.type, device)
                self.assertEqual({id(p) for group in trainer.optimizer.param_groups for p in group['params']},
                                 {id(p) for p in model.parameters()})
                batch = model.tokenizer([list('浙江省')], is_split_into_words=True, return_tensors='pt')
                inputs = {key: batch[key] for key in ['input_ids', 'attention_mask']}
                inputs['labels'] = torch.tensor([[-100, config.LABELS.index('B-prov'),
                                                 config.LABELS.index('I-prov'),
                                                 config.LABELS.index('E-prov'), -100]])
                trainer.dataloader = {'train': [inputs]}
                trainer.global_step = 0
                trainer.writer = None
                metrics = trainer.run_epoch('train')
                self.assertTrue(torch.isfinite(torch.tensor(metrics['loss'])))
                with patch.object(config, 'DEVICE', torch.device(device)):
                    self.assertEqual(len(model.predict('浙江省')), 3)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'weights.pt'
                    torch.save(model.state_dict(), path)
                    model.to('cpu')
                    with patch('models_def.torch.load', wraps=torch.load) as loader:
                        load_params(model, path)
                        self.assertEqual(loader.call_args.kwargs['map_location'], 'cpu')
                        self.assertTrue(loader.call_args.kwargs['weights_only'])
                print(f'训练、推理、权重加载检查通过：{device}')


if __name__ == '__main__':
    unittest.main()
