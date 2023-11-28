import psycopg2
import pandas as pd

from config import AppConfig


def getConnection():

    # Connect to postgres with a copy of the MIMIC-III database
    con = psycopg2.connect(
        dbname=AppConfig.db_details["sql_db_name"],
        user=AppConfig.db_details["sql_user_name"],
        host=AppConfig.db_details["sql_host_name"],
        port=AppConfig.db_details["sql_port_number"],
        password=AppConfig.db_details["sql_password"]
        )

    return con


def getDatamatrix():
    con = getConnection()
    sql = 'select * from ' + AppConfig.db_details["sql_schema_name"] + '.data_matrix'
    df = pd.read_sql_query(sql=sql, con=con)
    return df


def getDatamatrixForTraining(windowEnd):
    con = getConnection()
    sql = '''select
            *
            from
            ''' + AppConfig.db_details["sql_schema_name"] + '''.data_matrix
            where
            (death_datetime is not null and (death_datetime::timestamp > (anchor_time::timestamp + interval ' ''' + str(windowEnd) + ''' hour')))
            or
            (death_datetime is null)
    '''
    df = pd.read_sql_query(sql=sql, con=con)
    return df


def getDatamatrixForId(id):
    con = getConnection()
    sql = 'select * from ' + AppConfig.db_details["sql_schema_name"] + '.data_matrix where person_id = ' + id
    df = pd.read_sql_query(sql=sql, con=con)
    return df
