import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine():
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    driver = os.getenv("DB_DRIVER")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION", "yes")
    trust_server_certificate = os.getenv(
        "DB_TRUST_SERVER_CERTIFICATE", "yes")

    connection_string = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Trusted_Connection={trusted_connection};"
        f"TrustServerCertificate={trust_server_certificate};")

    params = quote_plus(connection_string)
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}")

    return engine