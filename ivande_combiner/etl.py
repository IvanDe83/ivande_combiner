import io

import pandas as pd
from sqlalchemy import create_engine


def quick_df_to_sql(df: pd.DataFrame, con: str, schema: str, name: str, save_args: dict) -> None:
    """
    quick function to save a DataFrame to a SQL table
    """
    engine = create_engine(con)
    pg_connection = engine.connect()

    try:
        with pg_connection.begin() as transaction:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=False)
            df.sort_index(inplace=True)
            df_ = df.truncate(before=-1, after=-1)
            df_.to_sql(con=pg_connection, name=name, schema=schema, **save_args)
            csv_buffer.seek(0)
            cursor = pg_connection.connection.cursor()
            copy_table_query = f"copy {schema}.{name} from stdin with (format csv)"
            cursor.copy_expert(copy_table_query, csv_buffer)
            transaction.commit()
    finally:
        pg_connection.close()
        engine.dispose()
