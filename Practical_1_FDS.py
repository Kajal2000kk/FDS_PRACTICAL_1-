#!/usr/bin/env python
# coding: utf-8

# # Practical 1: Data Collection, Modelling and Compilation

# In[1]:


my_dict={'Name':["a","b","c","d","e","f","g"],'Age':[20,27,35,45,55,43,35],'Designation':["VP","CEO","CFO","VP","VP","CEO","MD"]}

import pandas as pd
import numpy as np
df=pd.DataFrame(my_dict)
df


# In[2]:


df.to_csv('Csv_example')
df


# df_csv=pd.read_csv('Csv_example')
# df_csv

# In[5]:


df.to_csv('Csv_Ex',index=False)
df_csv=pd.read_csv('Csv_Ex')
df_csv


# Load data from csv file and display data without headers

# In[7]:


import pandas as pd
Location= "D:/FDS Prac/student-mat.csv"
df= pd.read_csv(Location,header=None)
df.head()


# In[8]:


import pandas as pd
Location= "D:/FDS Prac/student-mat.csv"
df= pd.read_csv(Location)
df.head()


# In[9]:


import pandas as pd
Location= "D:/FDS Prac/student-mat.csv"
# To add headers as we load the data
df= pd.read_csv(Location, names=['RollNo','Names','Grades'])
# To add headers to a dataframe
df.columns = ['RollNo','Names','Grades']
df.head()


# In[10]:


import pandas as pd
name = ['Akshata','Vish','Pragati','Mamta','Ak']
grade = [85,78,67,84,54]
bsc = [1,1,0,1,0]
msc = [1,2,0,1,0]
phd = [0,0,1,0,0]
Degrees = zip(name,grade,bsc,msc,phd)
columns = ['Names','Grades','BSC','MSC','PHD']
df = pd.DataFrame(data = Degrees, columns=columns)
df


# Loading Data from Excel File and changing column names

# In[14]:


import pandas as pd
Location= "D:/FDS Prac/gradedata.xlsx"
df = pd.read_excel(Location)

#Changing column names
df.columns = ['First','Last','Sex','Age','Exer','Hrs','Grd','Address']
df.head()


# In[17]:


import pandas as pd
name = ['Akshata','Vish','Pragati','Mamta','Ak']
grade = [85,78,67,84,54]
GradeList = zip(name,grade)
df = pd.DataFrame(data = GradeList,columns=['Names','Grades'])

writer = pd.ExcelWriter('dataframe.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()


# Load Data from sqlite

# In[1]:


import sqlite3
con = sqlite3.connect("D:/FDS Prac/portal_mammals.sqlite")
cur = con.cursor()

for row in cur.execute('SELECT * FROM species;'):
    print(row)
con.close()    


# In[4]:


import sqlite3
con = sqlite3.connect("D:/FDS Prac/portal_mammals.sqlite")
cur = con.cursor()
cur.execute('SELECT plot_id FROM plots WHERE plot_type="Control"')
print(cur.fetchall())
cur.execute('SELECT species FROM species WHERE taxa="Bird"')
print(cur.fetchone())
con.close()


# In[5]:


import pandas as pd
import sqlite3

con = sqlite3.connect("D:/FDS Prac/portal_mammals.sqlite")
df = pd.read_sql_query("SELECT * from surveys", con)

print(df.head())
con.close()


# Saving Data to SQL

# In[8]:


from pandas import DataFrame
Cars = {'Brand':['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price':[22000,25000,27000,35000]} 
df = DataFrame(Cars,columns=['Brand','Price'])
print(df)


# In[9]:


import sqlite3
conn= sqlite3.connect('TestDB1.db')
c=conn.cursor()


# In[10]:


c.execute('CREATE TABLE CARS(Brand text, Price number)')
conn.commit()


# In[11]:


df.to_sql('CARS',conn,if_exists='replace',index=False)
df


# In[12]:


c.execute('''
SELECT Brand,max(Price) from CARS
''')


# In[13]:


df= DataFrame(c.fetchall(),columns=['Brand','Price'])
df


# # Example 1

# In[14]:


import pandas as pd
import os
import sqlite3 as lite
from sqlalchemy import create_engine


# In[15]:


Student_id=[101,102,103,104]
SName=["Akshata","Vish","Pragati","Mamta"]
LName=["Khedekar","Desai","Mahadik","Karandikar"]
Department=["DSAI","BScIT","BMS","BCom"]
Email=["Akshata@gmail.com","vish23@gmail.com","pragati02@gmail.com","mamta1602@gmail.com"]


# In[16]:


Studata = zip(Student_id,SName,LName,Department,Email)


# In[17]:


df = pd.DataFrame(data = Studata, columns=['Student_id','SName','LName','Department','Email'])
df


# In[18]:


df1 = df.to_csv('Studata.csv',index=False,header=True)
df1


# In[19]:


df2 = df.to_excel('Studata.xlsx',index=False,header=True)
df2


# In[21]:


db_filename = r'Studata.db'
con = lite.connect(db_filename)
df.to_sql('student',
con,
schema=None,
if_exists='replace',
index=True,
index_label=None,
chunksize=None,
dtype=None
)
con.close()

db_file = r'Studata.db'
engine = create_engine(r"sqlite:///{}" .format(db_file))
sql = 'SELECT * from Student'

studf = pd.read_sql(sql, engine)
studf


# Data Preprocessing

# In[24]:


import pandas as pd
import numpy as np


# In[28]:


state=pd.read_csv("D:/FDS Prac/US_violent_crime.csv")
state.head()


# In[29]:


def some_func(x):
 return x*2
state.apply(some_func)
state.apply(lambda n: n*2)


# In[30]:


state.transform(func = lambda x: x*10)


# In[33]:


mean_purchase = state.groupby('State')["Murder"].mean().rename("User_mean").reset_index()
print(mean_purchase)


# In[34]:


mer=state.merge(mean_purchase)
mer


# In[35]:


#checking missing values
print(state.isnull().sum())


# # Example 2

# In[38]:


import pandas as pd
import numpy as np
cols=['col0', 'col1', 'col2', 'col3', 'col4']
rows=['row0', 'row1', 'row2', 'row3', 'row4']
data=np.random.randint(0, 100, size=(5,5))
df=pd.DataFrame(data, columns=cols, index=rows)
df.head()


# In[39]:


df.iloc[4,2]


# In[ ]:




