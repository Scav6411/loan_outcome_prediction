import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

def get_data_from_table(table_name):
    
    load_dotenv()
    
    db_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME')
    }
    
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

def save_df_to_csv(df: pd.DataFrame, file_path):
    try:
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

if __name__ == "__main__":
    table_names_list = ["loan_outcomes_predict", "loan_outcomes_train", "features", "events", "gps"]
    os.makedirs("dataset", exist_ok=True)
    for table_name in table_names_list:
        data = get_data_from_table(table_name)
        if data is not None:
            print(data.head())
            save_df_to_csv(data, f"dataset/{table_name}.csv")