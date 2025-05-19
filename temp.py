import re
import sqlite3


# def _extract_schema_with_sqlite(schema_str):
#     """Extracts table and column information from a SQLite schema string"""
#     tables = {}

#     # Remove comments
#     schema_str = re.sub(r"--.*$", "", schema_str, flags=re.MULTILINE)

#     # Remove trailing commas within CREATE TABLE statements
#     schema_str = re.sub(r",\s*\)", ")", schema_str, flags=re.MULTILINE, count=0)

#     # Quote table names in CREATE TABLE statements
#     schema_str = re.sub(r"CREATE TABLE (\S+)", r"CREATE TABLE '\1'", schema_str)

#     conn = sqlite3.connect(":memory:")
#     cursor = conn.cursor()
#     cursor.executescript(schema_str)

#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#     table_names = [row[0] for row in cursor.fetchall()]

#     for table_name in table_names:
#         cursor.execute(f"PRAGMA table_info(`{table_name}`);")
#         columns = [row[1] for row in cursor.fetchall()]
#         tables[table_name.lower()] = columns

#     conn.close()
#     return tables
