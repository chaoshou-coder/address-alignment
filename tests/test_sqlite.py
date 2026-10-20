import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import config
from address_alignment import address_alignment, query_parent
from init_db import initialize_database


class SQLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.database = Path(cls.temp.name) / 'region.db'
        cls.count = initialize_database(db_path=cls.database)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.setting = patch.object(config, 'SQLITE_DB_PATH', self.database)
        self.setting.start()
        self.addCleanup(self.setting.stop)

    def test_import_preserves_rows_and_integrity(self):
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(self.count, 44699)
            self.assertEqual(connection.execute('PRAGMA quick_check').fetchone()[0], 'ok')
            self.assertEqual(connection.execute('SELECT name FROM region WHERE id=1').fetchone()[0], '中国')

    def test_query_exact_fuzzy_parent_and_id(self):
        self.assertEqual(query_parent(2, '浙江')['name'], '浙江省')
        self.assertEqual(query_parent(3, '杭州市')['parent_name'], '浙江省')
        self.assertEqual(query_parent(4, '余杭区')['parent_name'], '杭州市')
        self.assertEqual(query_parent(3, '市辖区', 3)['parent_name'], '北京市')
        self.assertIsNone(query_parent(4, '余杭区', 3))
        self.assertIsNone(query_parent(2, "' OR 1=1 --"))

    def test_real_database_completes_missing_ancestors(self):
        class Model:
            def predict(self, text):
                return ['town'] * 4 + ['detail'] * 3
        result = address_alignment('永乐店镇27号', Model())
        self.assertEqual(result, {'省份': '北京市', '城市': '市辖区', '区县': '通州区',
                                  '街道': '永乐店镇', '详细地址': '27号'})

    def test_existing_database_is_not_reimported(self):
        self.assertEqual(initialize_database(sql_path='/nonexistent.sql', db_path=self.database), self.count)

    def test_failed_import_does_not_leave_partial_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'bad.sql', root / 'new.db'
            source.write_text("insert into region values (1, '坏数据');\n")
            with self.assertRaises(sqlite3.Error):
                initialize_database(source, target)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob('*.db')), [])

    def test_missing_database_is_not_created_by_query(self):
        target = self.database.parent / 'missing.db'
        with patch.object(config, 'SQLITE_DB_PATH', target):
            with self.assertRaises(sqlite3.OperationalError):
                query_parent(2, '浙江省')
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
