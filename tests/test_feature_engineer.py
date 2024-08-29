import unittest
import pandas as pd
from datetime import datetime
import pytz

from data_experts.feature_engineer import FeatureEngineer  # Asegúrate de ajustar el import según tu estructura

class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        # Configuración de un DataFrame de prueba básico con los nuevos features en mente
        self.df = pd.DataFrame({
            "original_price": [1000, 2000, 3000],
            "price": [900, 1800, 2500],
            "stop_time": ["2024-12-31T23:59:59.000Z"]*3,
            "attributes": [
                [{"id": "BRAND", "value_name": "BrandA"}],
                [{"id": "BRAND", "value_name": "BrandB"}],
                [{"id": "BRAND", "value_name": "BrandC"}]
            ],
            "shipping": [
                {"free_shipping": True, "store_pick_up": False},
                {"free_shipping": False, "store_pick_up": True},
                {"free_shipping": True, "store_pick_up": True}
            ],
            "official_store_name": [None, "Official Store 1", None],
            "domain_id": ["MLA-DOMAIN-1", "MLA-DOMAIN-2", "MLA-DOMAIN-3"],
            "available_quantity": [10, 20, 30],
            "listing_type_id": ["gold_pro", "gold_special", "silver"],
            "installments": [
                {"quantity": 12, "rate": 0},
                {"quantity": 6, "rate": 5},
                None  # Caso sin información de plazos
            ],
            "sale_price": [
                {"conditions": {"end_time": "2024-12-31T23:59:59.000Z"}},
                {"conditions": {"end_time": "2024-12-25T23:59:59.000Z"}},
                None  # Caso sin datos de descuento
            ]
        })
        self.engineer = FeatureEngineer("test_product", "test_token")
        self.engineer.dataframe = self.df

    def test_add_discount_feature(self):
        self.engineer.add_discount_feature()
        self.assertIn("discount", self.engineer.dataframe.columns)

    def test_add_days_until_discount_end(self):
        self.engineer.add_days_until_discount_end()
        self.assertIn("days_until_discount_end", self.engineer.dataframe.columns)
        self.assertIsNotNone(self.engineer.dataframe['days_until_discount_end'].iloc[0])
        self.assertIsNone(self.engineer.dataframe['days_until_discount_end'].iloc[2])

    def test_add_timestamp(self):
        self.engineer.add_timestamp()
        self.assertIn("data_timestamp", self.engineer.dataframe.columns)

    def test_add_price_quartile(self):
        self.engineer.add_price_quartile()
        self.assertIn("price_quartile", self.engineer.dataframe.columns)

    def test_extract_brand(self):
        self.engineer.extract_brand()
        self.assertIn("brand", self.engineer.dataframe.columns)

    def test_add_shipping_types_booleans(self):
        self.engineer.add_shipping_types_booleans()
        self.assertIn("free_shipping", self.engineer.dataframe.columns)
        self.assertIn("store_pick_up", self.engineer.dataframe.columns)

    def test_add_is_official_store(self):
        self.engineer.add_is_official_store()
        self.assertIn("is_official_store", self.engineer.dataframe.columns)

    def test_add_domain_id(self):
        self.engineer.add_domain_id()
        self.assertIn("domain_id", self.engineer.dataframe.columns)

    def test_add_available_quantity(self):
        self.engineer.add_available_quantity()
        self.assertIn("available_quantity", self.engineer.dataframe.columns)

    def test_add_listing_type(self):
        self.engineer.add_listing_type()
        self.assertIn("listing_type_id", self.engineer.dataframe.columns)

    def test_add_installment_features(self):
        self.engineer.add_installment_features()
        self.assertIn("installment_quantity", self.engineer.dataframe.columns)
        self.assertIn("has_msi", self.engineer.dataframe.columns)
        self.assertEqual(self.engineer.dataframe['installment_quantity'].iloc[0], 12)
        self.assertTrue(self.engineer.dataframe['has_msi'].iloc[0])
        self.assertFalse(self.engineer.dataframe['has_msi'].iloc[1])
        self.assertIsNone(self.engineer.dataframe['installment_quantity'].iloc[2])

    def test_apply_all_features(self):
        # Lista de los nuevos features para verificar en el DataFrame resultante
        new_features = [
            "discount",               # Porcentaje de descuento calculado
            "days_until_discount_end",# Días restantes hasta el fin del descuento
            "data_timestamp",         # Timestamp del momento en que se obtuvieron los datos
            "price_quartile",         # Cuartil de precio del producto
            "brand",                  # Marca del producto
            "free_shipping",          # Indica si el producto tiene envío gratuito
            "store_pick_up",          # Indica si el producto tiene opción de recogida en tienda
            "is_official_store",      # Indica si el producto es vendido por una tienda oficial
            "domain_id",              # ID del dominio del producto
            "available_quantity",     # Cantidad disponible de unidades
            "listing_type_id",        # Tipo de listado del producto
            "installment_quantity",   # Cantidad de plazos en los que se puede pagar el producto
            "has_msi"                 # Booleano que indica si el producto tiene meses sin intereses (MSI)
        ]

        df_result = self.engineer.apply_all_features()
        for feature in new_features:
            self.assertIn(feature, df_result.columns)

if __name__ == "__main__":
    unittest.main()
