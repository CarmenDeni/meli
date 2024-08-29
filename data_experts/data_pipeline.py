import pandas as pd
import json
from datetime import datetime
import requests
import pytz

class DataPipeline:
    """
    Clase para manejar la obtención, limpieza y preparación de datos desde la API de Mercado Libre.

    Atributos:
    - queries (list): Lista de palabras clave para realizar las búsquedas.
    - access_token (str): Token de acceso para la API de Mercado Libre.
    - dataframe (pd.DataFrame): DataFrame que almacena los datos combinados de todas las consultas.
    - timestamp (datetime): Marca temporal del momento en que se obtuvieron los datos.
    """

    def __init__(self, queries: list, access_token: str):
        """
        Inicializa la clase con las consultas y el token de acceso.

        Args:
            queries (list): Lista de palabras clave para las búsquedas (valores ?q= en la API).
            access_token (str): Token de acceso para la API de Mercado Libre.
        """
        self.queries = queries
        self.access_token = access_token
        self.raw_data = None
        self.dataframe = pd.DataFrame()
        self.timestamp = datetime.now(pytz.UTC)

    def fetch_data(self) -> str:
        """
        Realiza múltiples consultas a la API de Mercado Libre y obtiene los datos crudos sobre los productos.

        Returns:
            str: La respuesta cruda del API en formato JSON como cadena de texto.
        """
        combined_raw_data = []
        for query in self.queries:
            url = f"https://api.mercadolibre.com/sites/MLM/search?q={query}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                self.raw_data = response.text
                combined_raw_data.append((self.raw_data, query))  # Guardar datos crudos con su consulta
            else:
                raise ValueError(f"Error fetching data from API for query: {query}")

        return combined_raw_data

    def clean_data(self, raw_data: str, query: str) -> pd.DataFrame:
        """
        Limpia los datos obtenidos y los convierte en un DataFrame de pandas. Añade una columna 'query' para 
        identificar de qué búsqueda provienen los datos.

        Args:
            raw_data (str): Datos crudos obtenidos de la API en formato JSON.
            query (str): La consulta asociada a los datos crudos (para la columna 'query').

        Returns:
            pd.DataFrame: DataFrame con los datos limpios y la columna 'query'.
        """
        json_start_index = raw_data.find('{"site_id"')
        cleaned_data = raw_data[json_start_index:]
        parsed_data = json.loads(cleaned_data)
        
        # Convertir los resultados en DataFrame y añadir la columna 'query'
        df = pd.DataFrame(parsed_data['results'])
        df['query'] = query  # Añadir la columna 'query'
        return df

    def process_data(self) -> pd.DataFrame:
        """
        Realiza la obtención, limpieza y combinación de datos de todas las consultas.

        Returns:
            pd.DataFrame: DataFrame con los datos combinados de todas las consultas y la columna 'query'.
        """
        combined_data = []
        raw_data_with_queries = self.fetch_data()

        for raw_data, query in raw_data_with_queries:
            cleaned_df = self.clean_data(raw_data, query)
            combined_data.append(cleaned_df)

        # Combinar todos los DataFrames en uno solo y quitar los items que salen repetidos en búsquedas de múltiples queries
        self.dataframe = pd.concat(combined_data, ignore_index=True)
        self.dataframe.drop_duplicates(subset=["catalog_product_id", "official_store_id"], inplace=True)
        return self.dataframe

    def clean_and_prepare(self) -> pd.DataFrame:
        """
        Limpia y prepara los datos llamando al método `process_data` y devuelve los datos limpios.

        Returns:
            pd.DataFrame: DataFrame con los datos limpios.
        """
        cleaned_df = self.process_data()
        print("Data cleaned successfully")
        return cleaned_df
