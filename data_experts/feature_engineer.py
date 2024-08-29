import pandas as pd
import json
from datetime import datetime, timedelta
import requests
from typing import Any, Dict
import pytz

from .data_pipeline import DataPipeline

class FeatureEngineer(DataPipeline):
    """
    Clase para realizar el feature engineering a los datos limpios.
    
    Métodos:
    - add_discount_feature(): Añade una columna de descuento calculado.
    - add_days_until_discount_end(): Añade una columna de días restantes para el fin del descuento.
    - add_timestamp(): Añade una columna con el timestamp actual.
    - extract_brand(): Extrae la marca (brand) desde los atributos y crea una nueva columna 'brand'.
    - add_shipping_types_booleans(): Añade columnas booleanas para los tipos de shipping 'free_shipping' y 'store_pick_up'.
    - add_is_official_store(): Añade una columna que indica si el producto es de una tienda oficial.
    - add_domain_id(): Añade una columna con el ID del dominio.
    - add_available_quantity(): Añade una columna con la cantidad disponible de un producto.
    - add_listing_type(): Añade una columna con el tipo de listado.
    - add_installment_features(): Añade columnas para la cantidad de plazos y si tiene MSI (Meses Sin Intereses).
    - apply_all_features(): Aplica todas las funciones de feature engineering en una sola llamada.
    """

    def add_discount_feature(self) -> pd.DataFrame:
        """
        Añade una columna al DataFrame con el porcentaje de descuento calculado.
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna de descuento.
        """
        self.dataframe['discount'] = (self.dataframe['original_price'] - self.dataframe['price']) / self.dataframe['original_price'] * 100
        self.dataframe['discount'].fillna(0, inplace= True)
        return self.dataframe

    def add_days_until_discount_end(self) -> pd.DataFrame:
        """
        Añade una columna al DataFrame con los días restantes hasta que finalice el descuento,
        buscando la información dentro de sale_price['conditions']['end_time'].

        Returns:
            pd.DataFrame: DataFrame con la nueva columna de días restantes para el fin del descuento.
        """
        def calculate_days_until_end(sale_price):
            if sale_price and sale_price.get('conditions') and sale_price['conditions'].get('end_time'):
                end_time = pd.to_datetime(sale_price['conditions']['end_time'])
                current_time = datetime.now(pytz.UTC)
                return (end_time - current_time).days
            return 0

        self.dataframe['days_until_discount_end'] = self.dataframe['sale_price'].apply(calculate_days_until_end)
        return self.dataframe

    def add_timestamp(self) -> pd.DataFrame:
        """
        Añade una columna al DataFrame con el timestamp del momento en que se obtuvieron los datos.
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna de timestamp.
        """
        self.dataframe['data_timestamp'] = self.timestamp
        return self.dataframe
    
  #def add_price_quartile(self) -> pd.DataFrame:
  #    """
  #    Añade una columna categórica al DataFrame indicando en qué cuartil de precio se encuentra el producto.
  #    
  #    Returns:
  #        pd.DataFrame: DataFrame con la nueva columna de cuartil de precio.
  #    """
  #    self.dataframe['price_quartile'] = pd.qcut(self.dataframe['price'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
  #    return self.dataframe


    def extract_brand(self) -> pd.DataFrame:
        """
        Extrae la marca (brand) de la columna 'attributes' si está presente y crea una nueva columna 'brand'.
        
        Returns:
            pd.DataFrame: DataFrame con una nueva columna 'brand'.
        """
        def get_brand(attributes: list) -> str:
            for attribute in attributes:
                if attribute.get('id') == 'BRAND':
                    return attribute.get('value_name', 'Unknown')
            return 'Unknown'

        # Aplicar la extracción de la marca a cada fila
        self.dataframe['brand'] = self.dataframe['attributes'].apply(get_brand)
        return self.dataframe
    
    def add_shipping_types_booleans(self) -> pd.DataFrame:
        """
        Añade columnas booleanas para 'free_shipping' y 'store_pick_up' desde la columna 'shipping'.
        
        Returns:
            pd.DataFrame: DataFrame con las nuevas columnas de booleanos para tipos de shipping.
        """
        self.dataframe['free_shipping'] = self.dataframe['shipping'].apply(lambda x: x.get('free_shipping', False))
        self.dataframe['store_pick_up'] = self.dataframe['shipping'].apply(lambda x: x.get('store_pick_up', False))
        return self.dataframe

    def add_is_official_store(self) -> pd.DataFrame:
        """
        Añade una columna que indica si el producto es de una tienda oficial.
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna que indica si es una tienda oficial.
        """
        self.dataframe['is_official_store'] = self.dataframe['official_store_name'].notnull()
        return self.dataframe
    
    def add_domain_id(self) -> pd.DataFrame:
        """
        Añade una columna con el ID del dominio.
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna de domain_id.
        """
        self.dataframe['domain_id'] = self.dataframe['domain_id']
        return self.dataframe
    
    def add_available_quantity(self) -> pd.DataFrame:
        """
        Añade una columna con la cantidad disponible de un producto.
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna de available_quantity.
        """
        self.dataframe['available_quantity'] = self.dataframe['available_quantity']
        return self.dataframe

    def add_listing_type(self) -> pd.DataFrame:
        """
        Añade una columna con el tipo de listado.
        
        Returns:
            pd.DataFrame: DataFrame con la nueva columna de listing_type.
        """
        self.dataframe['listing_type_id'] = self.dataframe['listing_type_id']
        return self.dataframe
    
    
    def add_installment_features(self) -> pd.DataFrame:
        """
        Añade dos columnas al DataFrame relacionadas con los plazos de pago:
        - 'installment_quantity': Cantidad de plazos en los que se puede pagar el producto.
        - 'has_msi': Booleano que indica si el producto tiene meses sin intereses (tasa de interés = 0 o sin installments).
        - 'installment_rate': float que indica en que proporción aumenta el costo si se paga a meses.
        - 'msi': Cantidad de plazos en los que se puede pagar el producto sin cargos adicionales.

        Returns:
            pd.DataFrame: DataFrame con las nuevas columnas 'installment_quantity', 'has_msi' y 'installment_rate'.
        """
        def get_installment_quantity(installments):
            if installments and installments.get('quantity'):
                return installments['quantity']
            return 1

        def check_msi(installments):
            if installments and installments.get('rate') is not None:
                return installments['rate'] == 0
            return False
        
        def get_installment_rate(installments):
            if installments and installments.get('rate') is not None:
                return installments['rate']
            return 0
        
        def get_msi(row):
            if row.has_msi:
                return row.installment_quantity
            return 1
            

        # Aplicar las funciones a la columna 'installments'
        self.dataframe['installment_quantity'] = self.dataframe['installments'].apply(get_installment_quantity)
        self.dataframe['installment_rate'] = self.dataframe['installments'].apply(get_installment_rate)
        self.dataframe['has_msi'] = self.dataframe['installments'].apply(check_msi)
        self.dataframe['msi'] = self.dataframe.apply(get_msi, axis = 1)

        return self.dataframe
    
    def apply_all_features(self) -> pd.DataFrame:
        """
        Aplica todas las funciones de feature engineering en el DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame con todas las columnas de features añadidas.
        """
        self.extract_brand()
        self.add_discount_feature()
        self.add_days_until_discount_end()
        self.add_timestamp()
        #self.add_price_quartile()
        self.add_shipping_types_booleans()
        self.add_is_official_store()
        self.add_domain_id()
        self.add_available_quantity()
        self.add_listing_type()
        self.add_installment_features()
        return self.dataframe

