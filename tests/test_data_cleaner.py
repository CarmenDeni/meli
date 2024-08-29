import unittest
import pandas as pd
from data_experts.data_pipeline import DataCleaner  # Asegúrate de ajustar el import según tu estructura

class TestDataCleaner(unittest.TestCase):

    def setUp(self):
        # Datos crudos simulados como si fueran devueltos por la API de Mercado Libre
        self.raw_data = [
            ('{"site_id":"MLA","results":[{"id":"1","title":"Product 1","price":100}]}', "query1"),
            ('{"site_id":"MLA","results":[{"id":"2","title":"Product 2","price":200}]}', "query2")
        ]
        self.cleaner = DataCleaner(["query1", "query2"], "test_token")

    def test_clean_data(self):
        # Test para asegurar que clean_data genera correctamente un DataFrame
        raw_data, query = self.raw_data[0]
        df = self.cleaner.clean_data(raw_data, query)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("id", df.columns)
        self.assertIn("query", df.columns)

    def test_process_data(self):
        # Simula la obtención de datos crudos
        self.cleaner.fetch_data = lambda: self.raw_data  # Mock fetch_data

        df_result = self.cleaner.process_data()
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(len(df_result), 2)  # Asegura que los datos combinados tienen el número correcto de filas
        self.assertIn("query", df_result.columns)

if __name__ == "__main__":
    unittest.main()
