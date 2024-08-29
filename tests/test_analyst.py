import unittest
import pandas as pd
from data_experts.analyst import Analyst  # Asegúrate de ajustar el import según tu estructura

class TestAnalyst(unittest.TestCase):

    def setUp(self):
        # Configuración de un DataFrame de prueba básico
        self.df = pd.DataFrame({
            "brand": ["BrandA", "BrandB", "BrandA", "BrandC"],
            "price_quartile": ["Q1", "Q2", "Q1", "Q3"],
            "discount": [10, 20, 15, 5]
        })
        self.analyst = Analyst("test_product", "test_token")
        self.analyst.dataframe = self.df

    def test_analyze_by_single_feature(self):
        result = self.analyst.analyze_by_features(["brand"])
        self.assertIn("discount", result.columns)
        self.assertEqual(result.loc["BrandA"]["discount"], 12.5)  # Promedio de 10 y 15

    def test_analyze_by_multiple_features(self):
        result = self.analyst.analyze_by_features(["brand", "price_quartile"])
        self.assertIn("discount", result.columns)
        self.assertEqual(result.loc[("BrandA", "Q1")]["discount"], 12.5)
        self.assertEqual(result.loc[("BrandB", "Q2")]["discount"], 20.0)
        self.assertEqual(result.loc[("BrandC", "Q3")]["discount"], 5.0)

    def test_analyze_with_missing_features(self):
        with self.assertRaises(ValueError):
            self.analyst.analyze_by_features(["non_existent_column"])

if __name__ == "__main__":
    unittest.main()
