import sqlite3

from Services.User import User


class UserService:
    def createTable(self):
        sqliteConnection=None
        try:
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            sqlite_create_table_query = '''CREATE TABLE User (
                 id INTEGER PRIMARY KEY,
                 login TEXT NOT NULL,                 
                 password TEXT NOT NULL,
                 mail TEXT
                     
                );'''

            cursor = sqliteConnection.cursor()
            print("Successfully Connected to SQLite")
            cursor.execute(sqlite_create_table_query)
            sqliteConnection.commit()
            print("SQLite table created")

            cursor.close()

        except sqlite3.Error as error:
            print("Error while creating a sqlite table", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("sqlite connection is closed")

    def checkIfTableExist(self):
        val=True
        try:
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")
            cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='User' ''')

            # if the count is 1, then table exists
            if cursor.fetchone()[0] == 0:
                self.createTable()
                val=False
            cursor.close()

        except sqlite3.Error as error:
            print("Failed to insert Python variable into sqlite table", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("The SQLite connection is closed")
        return val

    def insertIntoTable(self,user):
        try:
            self.checkIfTableExist()
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")


            sqlite_insert_with_param = """INSERT INTO User
                              (login, password, mail) 
                              VALUES (?, ?, ?);"""

            data_tuple = (user.login, user.mdp, user.email)
            cursor.execute(sqlite_insert_with_param, data_tuple)
            sqliteConnection.commit()
            print("Python Variables inserted successfully into User table")

            cursor.close()

        except sqlite3.Error as error:
            print("Failed to insert Python variable into sqlite table", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("The SQLite connection is closed")


    # FOR READING ONE RECORD FUNCTION DEFINITION
    def readAll(self):
        results = []
        try:
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            cursor = sqliteConnection.cursor()

            sqlite_select_query = """SELECT * from User"""
            cursor.execute(sqlite_select_query)
            records = cursor.fetchall()


            for row in records:
                user=User(row[0],row[1],row[2],row[3])
                results.append(user)
            print("Total rows are:  ", len(records))


            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("The SQLite connection is closed")
        return results


    def read_one(self):
        con = sqlite3.connect("data.db")
        cursor = con.cursor()
        ids = int(input("Enter Your ID: "))
        query = "SELECT * from USERS WHERE id = ?"
        result = cursor.execute(query, (ids,))
        if (result):
            for i in result:
                print(f"Name is: {i[1]}")
                print(f"Age is: {i[2]}")
                print(f"Salary is: {i[4]}")
        else:
            print("Roll Number Does not Exist")
            cursor.close()


    # FOR READING ALL RECORDS FUNCTION DEFINITION
    def read_all(self):
        con = sqlite3.connect("data.db")
        cursor = con.cursor()
        query = "SELECT * from USERS"
        result = cursor.execute(query)
        if (result):
            print("\n<===Available Records===>")
            for i in result:
                print(f"Name is : {i[1]}")
                print(f"Age is : {i[2]}")
                print(f"Salary is : {i[4]}\n")
        else:
            pass


    # FOR UPDATING RECORDS FUNCTION DEFINITION
    def update(self,user):
        con = sqlite3.connect("SQLite_Python.db")
        cursor = con.cursor()
        data = (user.login, user.mdp, user.email, user.id)
        query = "UPDATE User set login = ?, password = ?, mail = ? WHERE id = ?"
        result = cursor.execute(query, data)
        con.commit()
        cursor.close()
        if (result):
            print("Records Updated")
        else:
            print("Something Error in Updation")


    # FOR DELETING RECORDS FUNCTION DEFINITION
    def delete(self):
        con = sqlite3.connect("data.db")
        cursor = con.cursor()
        idd = int(input("Enter ID: "))
        query = "DELETE from USERS where ID = ?"
        result = cursor.execute(query, (idd,))
        con.commit()
        cursor.close()
        if (result):
            print("One record Deleted")
        else:
            print("Something Error in Deletion")