import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, BertConfig, BertModel

import config
from address_alignment import address_alignment
from models_def import AddressTagging, load_params
from preprocess import AddressTaggingProcessor


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            config.PRETRAINED_DIR / 'bert-base-chinese', local_files_only=True
        )

    def processor(self, data_path=None, save_dir=None):
        return AddressTaggingProcessor(
            data_path or config.RAW_DATA_DIR / 'data.jsonl',
            save_dir or config.PROCESSED_DATA_DIR,
            self.tokenizer, 4, config.LABELS, max_seq_len=config.MAX_SEQ_LEN,
        )

    def test_all_source_records_match_config(self):
        examples = list(self.processor()._generate_examples())
        self.assertEqual(len(examples), 10826)
        self.assertTrue(all(len(x['text']) == len(x['labels']) for x in examples))
        self.assertTrue(all(len(char) == 1 for x in examples for char in x['text']))
        longest = max(examples, key=lambda x: len(x['text']))
        mapped = self.processor()._map_fn({
            'text': [longest['text']], 'labels': [longest['labels']]
        })
        self.assertEqual(sum(x != -100 for x in mapped['labels'][0]), len(longest['text']))

    def test_first_character_and_subwords(self):
        self.assertEqual(
            self.processor()._align_labels_with_tokens([3, 4], [None, 0, 0, 1, None]),
            [-100, 3, -100, 4, -100],
        )

    def test_jsonl_to_saved_dataloaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'sample.jsonl'
            lines = (config.RAW_DATA_DIR / 'data.jsonl').read_text().splitlines()[:20]
            source.write_text('\n'.join(lines) + '\n')
            processor = self.processor(source, root / 'processed')
            processor.process()
            sizes = {name: len(load_from_disk(root / 'processed' / name))
                     for name in ['train', 'valid', 'test']}
            self.assertEqual(sizes, {'train': 16, 'valid': 2, 'test': 2})
            batch = next(iter(processor.get_dataloader('train')))
            self.assertEqual(batch['input_ids'].shape, (4, config.MAX_SEQ_LEN))
            self.assertTrue((batch['labels'][:, 1] != -100).all())

    def test_invalid_labels_report_source_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'bad.jsonl'
            source.write_text(json.dumps({'text': ['甲'], 'labels': ['B-assist']}))
            with self.assertRaisesRegex(ValueError, r'bad.jsonl:1'):
                list(self.processor(source)._generate_examples())

    def test_forward_backward_and_position_preserving_prediction(self):
        small_bert = BertModel(BertConfig(
            vocab_size=self.tokenizer.vocab_size, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        ))
        with patch('models_def.BertModel.from_pretrained', return_value=small_bert):
            model = AddressTagging(config.PRETRAINED_DIR / 'bert-base-chinese', config.LABELS)
        mapped = self.processor()._map_fn({
            'text': [list('浙江省')],
            'labels': [[config.LABELS.index(x) for x in ['B-prov', 'I-prov', 'E-prov']]],
        })
        batch = {key: torch.tensor(value) for key, value in mapped.items()}
        loss = model(**batch)['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.bias[config.LABELS.index('B-detail')] = 10
        text = '甲 0000号\t乙'
        result = model.predict(text, device=torch.device("cpu"))
        self.assertEqual(len(result), len(text))
        self.assertEqual(result[1], '')
        self.assertEqual(result[-1], 'detail')
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / 'model.pt'
            torch.save(model.state_dict(), weights)
            self.assertTrue(load_params(model, weights))
        self.assertFalse(load_params(model, None))

    def test_training_entrypoint_saves_reusable_best_weights(self):
        import main
        small_bert = BertModel(BertConfig(
            vocab_size=self.tokenizer.vocab_size, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        ))
        with patch('models_def.BertModel.from_pretrained', return_value=small_bert):
            model = AddressTagging(config.PRETRAINED_DIR / 'bert-base-chinese', config.LABELS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw'
            raw.mkdir()
            lines = (config.RAW_DATA_DIR / 'data.jsonl').read_text().splitlines()[:20]
            (raw / 'data.jsonl').write_text('\n'.join(lines) + '\n')
            weights = root / 'finetuned' / 'address_tagging.pt'
            with patch.multiple(config, RAW_DATA_DIR=raw,
                                PROCESSED_DATA_DIR=root / 'processed',
                                FINETUNED_DIR=weights.parent, MODEL_PARAMS_PATH=weights,
                                LOGS_DIR=root / 'logs'), patch(
                'main.AddressTagging', return_value=model
            ), patch('main.device', torch.device('cpu')):
                main.model_go(train=True, test=True, inference=True, epochs=1, batch_size=4)
            self.assertTrue(weights.is_file())
            self.assertTrue(load_params(model, weights))

    def test_alignment_ignores_contacts_and_joins_details(self):
        text = '张三 138浙江省杭州市余杭区东街27号 甲楼'
        tags = (['name'] * 2 + [''] + ['phone'] * 3 + ['prov'] * 3 +
                ['city'] * 3 + ['district'] * 3 + ['town'] * 2 +
                ['detail'] * 3 + [''] + ['detail'] * 2)
        self.assertEqual(len(text), len(tags))
        class Model:
            def predict(self, value):
                return tags
        def query(region_type, name, region_id=None):
            return {'name': name, 'parent_name': {3: '浙江省', 4: '杭州市', 5: '余杭区'}.get(region_type), 'parent_id': 1}
        with patch('address_alignment.query_parent', side_effect=query):
            result = address_alignment(text, Model())
        self.assertEqual(result, {'省份': '浙江省', '城市': '杭州市', '区县': '余杭区',
                                  '街道': '东街', '详细地址': '27号甲楼'})


if __name__ == '__main__':
    unittest.main()
