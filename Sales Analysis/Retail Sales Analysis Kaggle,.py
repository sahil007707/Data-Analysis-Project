#!/usr/bin/env python
# coding: utf-8

# # **🧰 Section: Importing Libraries**

# In[147]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
pd.set_option("display.max_colwidth",300)
pd.set_option("display.max_column",30)


# ## **Storytelling:**
# #### *Every great data story starts with gathering the right tools. Here, you're importing essential libraries for data handling and visualization:*
# 
# #### *Pandas for data manipulation,*
# 
# #### *Seaborn and Matplotlib for static visualizations,*
# 
# #### *Plotly Express for interactive plots.*

# # **<p style="color:Orange;">Project:</p>**

# ## **<p style="color:#3498db;">📈 Section: Sales Analysis**

# ![May 27, 2025, 05_20_09 PM.png](attachment:ab8ae6ed-e1fe-4c56-a9a7-192d30d4f14c.png)

# In[148]:


df=pd.read_csv(r"c:\Users\user\Desktop\Cleaned Dataset CSV\Sales Dataset.csv")


# ### `First And Last 5 rows`

# In[149]:


df.head()


# In[150]:


df.tail()


# ## **<p style="color:#3498db;">🧹 Section: Data Handling and Cleaning**

# In[151]:


df.info()


# In[152]:


df["Date"]=pd.to_datetime(df["Date"])


# In[153]:


df.shape


# ### *<p style="color:Orange;">The Dataset contains 1000 rows and 8 columns*

# In[154]:


df.isna().sum()
df.duplicated().sum()


# In[155]:


df.drop(columns=["Unnamed: 0"],inplace=True)


# In[156]:


df.rename(columns={"Total Amount": "Total Sales"}, inplace=True)


# ## **<p style="color:#3498db;">EDA & Visualization**

# In[157]:


gender_counts=df["Gender"].value_counts()
plt.figure(figsize=(8,6))
plt.pie(gender_counts,labels=gender_counts.index,autopct="%1.1f%%",startangle=90,colors=["pink","red"])
plt.title("Gender Classification",fontsize=15)
plt.show()


# ### `Average Age Of Customers By Their Gender`

# In[158]:


classifi_age=df.groupby("Gender")["Age"].mean().reset_index()
classifi_age
plt.figure(figsize=(8,6))
sns.set_style("whitegrid")
sns.catplot(data=classifi_age,x="Gender",y="Age",hue="Gender")
plt.title("Gender Classification By Customers By Their Average Age",fontsize=13)
plt.show()


# ### *<p style="color:Orange;">As we can see almost the same average age between two genders*

# In[159]:


classifi_g=df.groupby(["Gender","Product Category"]).agg({
    "Age":"mean",
    "Quantity":"sum"
}).reset_index()
classifi_g.style.background_gradient(cmap="Oranges")


# ### `How Many Product People Bought By Category And The Average Age Of Each Gender`

# In[160]:


classifi_g=df.groupby(["Gender","Product Category"]).agg({
    "Age":"mean",
    "Quantity":"sum"
}).reset_index()
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.barplot(data=classifi_g,x="Product Category",y="Quantity",hue="Gender",palette="PiYG")
plt.title("How Many Product People Bought By Category ",fontsize=13)
plt.xlabel("Product Categories")
plt.ylabel("Quantity")


plt.subplot(1,2,2)
sns.barplot(data=classifi_g,x="Product Category",y="Age",hue="Gender",palette="plasma")
plt.title("The Average Age Of People Who Bought The Products Per Category",fontsize=13)
plt.xlabel("Product Categories")
plt.ylabel("Average Age")
plt.show()


# ### `Total Sales Per Product Categorry`

# In[161]:


classifi_sale=df.groupby(["Gender","Product Category"])["Total Sales"].sum().sort_values(ascending=False).reset_index()
classifi_sale.style.background_gradient(cmap="Greens")


# In[162]:


classifi_sale=df.groupby(["Gender","Product Category"])["Total Sales"].sum().reset_index()
fig=px.bar(classifi_sale,x="Product Category",y="Total Sales",color="Gender",title="Total Sales Per Product Category By Gender Ratio")
fig.update_layout(height=500,width=600)
fig.show()

plt.figure(figsize=(8,6))
sns.lineplot(data=classifi_sale,x="Product Category",y="Total Sales")
plt.title("Total Sales Per Product Category",fontsize=13)
plt.show()


# ### `Total Sales,Price Per Unit & Category Classification`

# In[163]:


sns.histplot(data=df,x="Product Category",kde=True)

plt.figure(figsize=(8,6))
sns.set_color_codes("pastel")
sns.barplot(data=df,x="Price per Unit",y="Product Category")
plt.show()

fig=px.bar(df,x="Product Category",y="Price per Unit")
fig.show()


# ### `Time-Series Analysis`

# In[ ]:


plt.figure(figsize=(20,6))
sns.lineplot(data=df, x='Date', y='Total Sales', marker='o',color="red")
plt.title('Daily Total Sales Over Time', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### `Correlation Heatmap`

# In[172]:


plt.figure(figsize=(12,8))
cor=df.select_dtypes(include="number").corr()
sns.heatmap(cor,annot=True,fmt=".2f")
plt.show()


# ## **<p style="color:#3498db;">The Motive And The Bigger Picture**
# ### *<p style="color:Orange;">My analysis showed Clothing drove volume, Electronics high revenue, and Beauty balanced both. This could guide stocking or marketing strategies.*
# 
# ### *<p style="color:Orange;"> Why: I focused on Product Category and Price per Unit to understand sales drivers and profitability.*
# 
# ### *<p style="color:Orange;"> Story: I uncovered the marketplace’s rhythm—Clothing drew crowds, Electronics brought riches, Beauty kept things steady. My findings could steer the market to thrive*

# ## **<p style="color:#3498db;"> More To Explore**
# 
# ### *<p style="color:Orange;">My journey revealed key patterns, but the scroll holds more. I could analyze Total Sales by Gender, Age, or Date, or plot trends over time. As I rolled up the scroll, I knew I’d return to uncover more of the marketplace’s secrets with my data tools.*
