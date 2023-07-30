import mysql.connector
import pandas as pd

#Credentials to the database full of stats
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

        # Execute the SELECT statement to fetch all players with jersey number 8
        jersey_number = 8
        select_query = f"SELECT * FROM stats WHERE Jersey = {jersey_number};"
        # Replace 'your_table_name' with the actual table name containing player information
        cursor.execute(select_query)

        # Fetch all the rows and create a pandas DataFrame
        data = cursor.fetchall()
        df = panda.DataFrame(data, columns=cursor.column_names)

        # Print the DataFrame
        print(df)

        # Fetch all the rows and print them
        result = cursor.fetchall()
        for row in result:
            print(row)


except Exception as e:
    print("An error occurred:", e)

finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if connection and connection.is_connected():
        connection.close()
        print("Connection closed.")
