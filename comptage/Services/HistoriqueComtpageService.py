import sqlite3

from Services.Historique_comptage import Historique_comptage


class HistoriqueComtpageService:

    def createTable(self):
        sqliteConnection=None
        try:
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            sqlite_create_table_query = '''CREATE TABLE Comptage (
                 id INTEGER PRIMARY KEY,
                 nom TEXT NOT NULL,
                 date DATETIME DEFAULT  (DATETIME('now', 'localtime')) ,
                 duree TEXT NOT NULL,
                 lieu TEXT,
                 nbCar INTEGER NOT NULL,
                 nbBus INTEGER NOT NULL,
                 nbTruck INTEGER NOT NULL,
                 nbMotorBike INTEGER NOT NULL,
                 entree INTEGER NOT NULL,
                 sortie INTEGER NOT NULL             
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
        try:
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")
            cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Comptage' ''')

            # if the count is 1, then table exists
            if cursor.fetchone()[0] == 0:
                self.createTable()

            cursor.close()

        except sqlite3.Error as error:
            print("Failed to insert Python variable into sqlite table", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("The SQLite connection is closed")
    def insertIntoTable(self,comptage):
        try:
            self.checkIfTableExist()
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")
            cursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='students' ''')



            sqlite_insert_with_param = """INSERT INTO Comptage
                              (nom, duree, lieu, nbCar, nbBus, nbTruck, nbMotorBike, entree, sortie) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);"""

            data_tuple = (comptage.nom, comptage.duree, comptage.lieu, comptage.nbCar, comptage.nbBus, comptage.nbTruck, comptage.nbMotorBike,comptage.entree,comptage.sortie)
            cursor.execute(sqlite_insert_with_param, data_tuple)
            sqliteConnection.commit()
            print("Python Variables inserted successfully into Comptage table")

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
            print("Connected to SQLite")

            sqlite_select_query = """SELECT * from Comptage"""
            cursor.execute(sqlite_select_query)
            records = cursor.fetchall()


            for row in records:
                cmpt=Historique_comptage(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10])
                results.append(cmpt)
            print("Total rows are:  ", len(records))
            print("Printing each row")


            cursor.close()

        except sqlite3.Error as error:
            print("Failed to read data from sqlite table", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("The SQLite connection is closed")
        return results

    def findByCriteria(self,critere):
        results = []
        try:
            sqliteConnection = sqlite3.connect('SQLite_Python.db')
            cursor = sqliteConnection.cursor()
            print("Connected to SQLite")
            print("Comptage",critere)
            sqlite_select_query = """SELECT * from Comptage"""
            if critere.lieu:
                sqlite_select_query+=" where lieu LIKE '%"+critere.lieu+"%' "
                if (critere.dateDebut == critere.dateFin):
                    sqlite_select_query += " and date >= date('" + critere.dateDebut + "')"
                else:
                    if critere.dateDebut:
                        sqlite_select_query += " and date >= date('" + critere.dateDebut + "')"
                    if critere.dateFin:
                        sqlite_select_query += " and date <= date('" + critere.dateFin + "')"
            else:
                if(critere.dateDebut==critere.dateFin):
                    sqlite_select_query += " where date >= date('" + critere.dateDebut + "')"
                else:
                    if critere.dateDebut:
                        sqlite_select_query+=" where date >= date('"+critere.dateDebut+"')"
                    if critere.dateFin:
                        sqlite_select_query+=" and date <= date('"+critere.dateFin+"')"
            print("Query ",sqlite_select_query)
            cursor.execute(sqlite_select_query)
            records = cursor.fetchall()

            for row in records:
                cmpt = Historique_comptage(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                                           row[9], row[10])
                results.append(cmpt)
            print("Total rows are:  ", len(records))
            print("Printing each row")

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
    def update(self):
        con = sqlite3.connect("data.db")
        cursor = con.cursor()
        idd = int(input("Enter ID: "))
        name = input("Enter Name: ")
        age = int(input("Enter Age: "))
        gender = input("Enter Gender: ")
        salary = int(input("Enter Salary: "))
        data = (name, age, gender, salary, idd,)
        query = "UPDATE USERS set name = ?, age = ?, gender = ?, salary = ? WHERE id = ?"
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



