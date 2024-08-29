import pandas as pd
import json
from datetime import datetime, timedelta
import requests
from typing import Any, Dict
import pytz
from .data_pipeline import DataPipeline

class DataCleaner(DataPipeline):
    """
    Clase para limpiar y preparar los datos heredando de DataPipeline.

    Métodos:
    - clean_and_prepare(): Limpia los datos y los devuelve como DataFrame.
    """

    def clean_and_prepare(self) -> pd.DataFrame:
        """
        Limpia y prepara los datos llamando al método `process_data` y devuelve los datos limpios.
        
        Returns:
            pd.DataFrame: DataFrame con los datos limpios.
        """
        cleaned_df = self.process_data()
        print("Data cleaned successfully")
        return cleaned_df
