import pandas as pd
import numpy as np
import os


class Data:

    def paths(self, data_path):
        # Paths to data files directly in Data directory
        self.connections = os.path.join("Data", "organization_connections.csv")
        self.data = os.path.join("Data", "top_200_companies_3days_2025.csv")
        self.embeddings = os.path.join("Data", "organization_embeddings.csv")
        self.avg_esg = os.path.join("Data", "average_esg_scores.csv")
        self.daily_esg = os.path.join("Data", "overall_daily_esg_scores.csv")
        self.e_score = os.path.join("Data", "daily_E_score.csv")
        self.s_score = os.path.join("Data", "daily_S_score.csv")
        self.g_score = os.path.join("Data", "daily_G_score.csv")

    def read(self, start_day="jan6", end_day="jan12"):
        # Check if Data directory exists
        if not os.path.exists("Data"):
            raise NameError("Data directory not found")
        
        # Check if required files exist
        required_files = [
            "organization_connections.csv",
            "top_200_companies_3days_2025.csv",
            "organization_embeddings.csv",
            "average_esg_scores.csv",
            "overall_daily_esg_scores.csv",
            "daily_E_score.csv",
            "daily_S_score.csv",
            "daily_G_score.csv"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join("Data", f))]
        if missing_files:
            raise NameError(f"Missing required files in Data directory: {missing_files}")

        self.paths("Data")
        data = {
            "conn": pd.read_csv(self.connections),
            "data": pd.read_csv(self.data, parse_dates=["DATE"],
                             infer_datetime_format=True),
            "embed": pd.read_csv(self.embeddings),
            "overall_score": pd.read_csv(self.daily_esg,
                              index_col="date", parse_dates=["date"],
                             infer_datetime_format=True),
            "E_score": pd.read_csv(self.e_score, parse_dates=["date"],
                             infer_datetime_format=True, index_col="date"),
            "S_score": pd.read_csv(self.s_score, parse_dates=["date"],
                             infer_datetime_format=True, index_col="date"),
            "G_score": pd.read_csv(self.g_score, parse_dates=["date"],
                             infer_datetime_format=True, index_col="date"),
            "ESG": pd.read_csv(self.avg_esg),
        }

        # Convert DATE column to date
        data["data"]["DATE"] = data["data"]["DATE"].dt.date

        # Multiply tones by large number
        esg_tables = ["E_score", "S_score", "G_score", "overall_score", "ESG"]
        for t in esg_tables:
            num_cols = data[t].select_dtypes(include=["number"]).columns
            data[t][num_cols] *= 10000

        return data