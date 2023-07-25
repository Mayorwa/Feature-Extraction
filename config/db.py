import MySQLdb

connection = MySQLdb.connect(
    host="localhost",
    user="root",
    db="ccbfe",
    unix_socket="/tmp/mysql.sock",
    port=3306
)


def create_db_tables():
    create_tags_table_sql = """
    CREATE TABLE IF NOT EXISTS dicom_tags (
        id INT AUTO_INCREMENT PRIMARY KEY,
        tag VARCHAR(255),
        value VARCHAR(255)
    )
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute(create_tags_table_sql)
        connection.commit()
        print("DICOM tags table created.")
    except MySQLdb.Error as e:
        print(f"Error creating DICOM tags table: {e}")
