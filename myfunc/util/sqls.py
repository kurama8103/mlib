from sqlite3 import connect
from pandas import read_sql


def get_sql_master(db_path, strSQL="SELECT * FROM sqlite_master WHERE type='table'"):
    conn = connect(db_path)
    return read_sql(strSQL, conn)


def insert_sql(db_path, sql):
    con = connect(db_path)
    cur = con.cursor()
    cur.executescript(sql)
    con.close()


def execute_sql(db_path, sql):
    con = connect(db_path)
    cur = con.cursor()
    cur.execute(sql)
    try:
        return cur.fetchall(), [i[0] for i in cur.description]
    except:
        return cur.fetchall()
