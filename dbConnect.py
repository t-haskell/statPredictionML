import mysql.connector
import pandas as pd



def dataRetrieve(year):
    # Credentials to the database full of stats
    db_config = {
        'host': 'plldatabase-1.cpzseqbr5kxy.us-east-1.rds.amazonaws.com',
        'port': 3306,
        'user': 'admin',
        'password': 'plldatabase',
        'database': 'player_stats'  # Replace 'your_database_name' with the actual database name
    }

    try:
        # Create a connection to the database
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            print("Connected to the MySQL database!")
            # Create a cursor to execute SQL queries
            cursor = connection.cursor()
            #
            # # Execute the ALTER statement to normalize all column names
            # query = "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = %s;"
            # cursor.execute(query, (db_config['database'],))
            # for table_name, column_name in cursor.fetchall():
            #     new_column_name = column_name.lower().replace(' ', '_')
            #     if '_' in new_column_name:
            #         column_name = f"`{column_name}`"
            #     alter_query = f"ALTER TABLE {table_name} RENAME COLUMN {column_name} TO {new_column_name};"
            #     # Execute the ALTER query here
            #     try:
            #         cursor.execute(alter_query)
            #         connection.commit()
            #         print(f"Column name changed in table {table_name}: {column_name} -> {new_column_name}")
            #     except mysql.connector.Error as err:
            #         print(f"Error altering table {table_name} - {err}")
            # cursor.close()

            # Gathers all player stats from a given year
            selectALL_query = f"SELECT * FROM {year};"



            cursor.execute(selectALL_query)

            # Fetch all the rows and create a pandas DataFrame
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=cursor.column_names)

            # Print the DataFrame
            return df


    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
            print("Connection closed.")
