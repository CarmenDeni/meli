import pandas as pd
from typing import List, Optional


class Analyst:
    """
    Clase para realizar análisis agrupados por uno o más features categóricos.

    Métodos:
    - analyze_by_features(): Agrupa los datos por uno o más features categóricos y calcula una métrica especificada.
    - general_analysis(): Realiza un análisis descriptivo general con la opción de incluir o excluir NaN y agrupar por un feature específico.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa la clase con un DataFrame.

        Args:
            dataframe (pd.DataFrame): El DataFrame que contiene los datos a analizar.
        """
        self.dataframe = dataframe

    def general_analysis(self, group_by_feature: Optional[str] = None, include_nan: bool = False) -> pd.DataFrame:
        """
        Realiza un análisis descriptivo general del DataFrame. Puede agrupar los datos por una característica específica
        y decidir si incluir o no los valores NaN.

        Args:
            group_by_feature (str, optional): Nombre de la columna por la cual agrupar los datos. Si es None, no agrupa.
            include_nan (bool, optional): Si True, incluye valores NaN en el análisis. Por defecto es False.

        Returns:
            pd.DataFrame: DataFrame con las estadísticas descriptivas.
        """
        if group_by_feature and group_by_feature not in self.dataframe.columns:
            raise ValueError(f"La columna '{group_by_feature}' no existe en el DataFrame.")

        if group_by_feature:
            grouped = self.dataframe.groupby(group_by_feature, dropna=not include_nan)
            return grouped.describe(include='all')
        else:
            return self.dataframe.describe(include='all')

    def analyze_by_features(self, features: List[str], metric: str = "mean") -> pd.DataFrame:
        """
        Agrupa los datos por uno o más features categóricos y calcula la medida de tendencia central especificada para discount.

        Args:
            features (list): Lista de nombres de columnas por las cuales agrupar los datos.
            metric (str): La métrica de tendencia central a calcular. Opciones: 'mean', 'median', 'mode'.
                          Por defecto es 'mean'.

        Returns:
            pd.DataFrame: DataFrame con los valores agrupados y su métrica calculada.
        """
        metrics_map = {
            "mean": "mean",
            "median": "median",
            "mode": lambda x: x.mode().iloc[0] if not x.mode().empty else float('nan')
        }

        if metric not in metrics_map:
            raise ValueError(f"Métrica '{metric}' no soportada. Use 'mean', 'median', o 'mode'.")

        missing_features = [feature for feature in features if feature not in self.dataframe.columns]
        if missing_features:
            raise ValueError(f"Las siguientes columnas no existen en el DataFrame: {', '.join(missing_features)}")

        group = self.dataframe.groupby(features).agg({'discount': metrics_map[metric]})
        return group
