from mage_ai.data_preparation.repo_manager import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.snowflake import Snowflake
from pandas import DataFrame
from os import path



if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter




@data_exporter
def export_data_to_snowflake(tables: dict, **kwargs) -> None:
    """
    Template for exporting data to a Snowflake warehouse.
    Specify your configuration settings in 'io_config.yaml'.

    Docs: https://docs.mage.ai/design/data-loading#snowflake

    """
    for tablename, table in tables.items():


        table_name = tablename
        database = 'PROYECTO1'
        schema = 'RAW'
        config_path = path.join(get_repo_path(), 'io_config.yaml')
        config_profile = 'default'

        with Snowflake.with_config(ConfigFileLoader(config_path, config_profile)) as loader:
            loader.export(
                table,
                table_name,
                database,
                schema,
                if_exists='replace',  
            )
