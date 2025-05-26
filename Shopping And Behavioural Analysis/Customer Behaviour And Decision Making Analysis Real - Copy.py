#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px


# # **<p style="color:Orange;">Project:</p>**
# ## <p style="color:#3498db;">Shopping Trends & Customer Behaviour Analysis</p>

# ![ChatGPT Image May 26, 2025, 08_08_07 PM.png](attachment:03ee7bbc-50cd-4e43-bef1-e318ae9dac07.png)

# # **<p style="color:#3498db;">🛍️ Project: Shopping Trends & Customer Behavior Analysis**
# ## **<p style="color:Red;">✨ Goal:**
# ###  <p style="color:Orange;"> To uncover insights from consumer shopping data — exploring trends, behaviors, and patterns that could inform marketing or business strategy.

# In[3]:


df=pd.read_csv(r"c:\Users\user\Desktop\Kaggle Datasets\Shopping Trends And Customer Behaviour Dataset.csv")


# ## Data Handling & Cleaning 

# In[4]:


df.isna().sum()
df.duplicated().sum()


# In[5]:


df.drop(columns=["Unnamed: 0"],inplace=True) # Dropping Column For A Clean Dataset


# In[6]:


df.head()


# In[7]:


df["Subscription Status"].value_counts()
df["Subscription Status"]=df["Subscription Status"].replace({"Yes":1,"No":0}).astype("int64") # Converting Column Data Type and Replacing Value, yes=1 & No=0 


# In[8]:


df["Shipping Type"].value_counts()
df.drop(columns=["Shipping Type"],inplace=True)


# In[9]:


df.shape


# ## EDA & Visualization

# ### `Which Item Purchased By Most & In Which Category`

# ![image.png](attachment:image.png)

# ## **<p style="color:Green;">Gender classification based on different types of purchase making decisions,behaviour and frequency.**
# ### *<p style="color:#3498db;">In this way we can follow the Gender analysis pattern, which type of gender has which type of criteria to buy which types of product and also how freqent they are or how their decision making ability convince them to buy a product. This visualization analysis shows and helps us to understand the customer and their behaviour and pattern when they purchase a product.*

# In[10]:


col=df.select_dtypes(include=["object"]).columns
plt.figure(figsize=(12,6))
for cols in col:
    plt.figure(figsize=(12,6))
    sns.histplot(data=df,x=cols,kde=True,bins=30,hue="Gender",multiple="stack")
    plt.title(f"The Gender Clssification Based On {cols}")
    plt.grid(True)
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.show()


# In[11]:


classifi_pur=df.groupby(["Gender","Item Purchased","Category"])["Purchase Amount (USD)"].sum().reset_index().sort_values("Purchase Amount (USD)",ascending=False)
classifi_pur.style.background_gradient(cmap="Reds")


# In[12]:


plt.figure(figsize=(15,8))
fig=px.bar(classifi_pur,x="Category",y="Purchase Amount (USD)", color="Gender",color_discrete_sequence=["#FFA15A", "#FF6692"])
fig.show()


# In[13]:


loc_counts=df.groupby("Location").size().sort_values(ascending=False)
top_location=loc_counts.idxmax()
top_count=loc_counts.max()
print(f"Location with the highest number of purchases: {top_location} ({top_count})")


# #### *<p style="color:#FFA15A;">Location with the highest number of purchases: Montana (96)*

# ### `Classification Of Gender, How Frequent They Purchase From Which Location And How Much They Have Spent`

# In[14]:


classifi_lo=df.groupby(["Gender","Location","Frequency of Purchases"])["Purchase Amount (USD)"].sum().reset_index().sort_values("Purchase Amount (USD)",ascending=False)
classifi_lo


# ### `How Frequent They Are With Their Purchase`

# In[15]:


classifi_lo=df.groupby(["Gender","Location","Frequency of Purchases"]).size().unstack()
classifi_lo
# Sum across frequencies to get total count per Gender + Location
total_counts = classifi_lo.sum(axis=1)

# Find the index (i.e., Gender + Location) with the highest total
top_combination = total_counts.idxmax()
top_value = total_counts.max()

print(f"Most frequent Gender + Location combo: {top_combination} ({top_value} purchases)")


# #### *<p style="color:#FECB52 ;">From this analysis we can notice the most frequent customer from Male category is from California with 66.0 purchases*

# In[16]:


df.head()


# ### `Classification By Gender And Age To Notice The Average Rating Review And Highest Previous Purchases`

# In[17]:


col=df.select_dtypes(include=["object"]).columns
plt.figure(figsize=(12,6))
for cols in col:
    plt.figure(figsize=(12,6))
    sns.histplot(data=df,x=cols,kde=True,bins=30,hue="Gender",multiple="stack")
    plt.title(f"The Gender Clssification Based On {cols}")
    plt.grid(True)
    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()
    plt.show()


# In[18]:


classifi_age=df.groupby(["Gender","Category","Item Purchased","Payment Method"]).agg({
    "Age":"mean",
    "Review Rating":"mean",
    "Previous Purchases":"sum"
}).reset_index()
classifi_age=classifi_age.sort_values("Previous Purchases",ascending=False)
classifi_age.head(25).style.background_gradient(cmap="Oranges")


# In[19]:


classifi_age=df.groupby(["Gender","Category","Item Purchased","Payment Method"]).agg({
    "Age":"mean",
    "Review Rating":"mean",
    "Previous Purchases":"sum"
}).reset_index()
classifi_age=classifi_age.sort_values("Previous Purchases",ascending=False)
classifi_age
plt.figure(figsize=(15,6))
for col in classifi_age:
    if col not in ["Gender","Category","Item Purchased","Payment Method"]:
            plt.figure(figsize=(15,6))
            sns.histplot(data=classifi_age,x=col,hue="Gender",kde=True,multiple="stack")
            plt.title(f"The Classification Based On {col}",fontweight="bold",fontsize=13)
            plt.tight_layout()
            plt.show()


# In[20]:


df["Category"].value_counts().plot.pie(autopct="%1.1f%%",figsize=(8,8))
plt.title("Products By Categories")
plt.tight_layout()
plt.show()


# ### `Classification Based On Seasons,Frequency And Sales`

# In[21]:


classifi_sea=df.groupby(["Gender","Item Purchased","Category","Season","Frequency of Purchases"])["Purchase Amount (USD)"].sum().reset_index()
classifi_sea


# In[22]:


plt.figure(figsize=(20,12))
plt.subplot(2,2,1)
sns.barplot(data=classifi_sea,x="Season",y="Purchase Amount (USD)",hue="Category")
plt.title("Classification On Category By Season And Sales",fontweight="bold",fontsize=15)
plt.xlabel("Seasons")

plt.subplot(2,2,2)
sns.barplot(data=classifi_sea,x="Season",y="Purchase Amount (USD)",hue="Gender")
plt.title("Classification On Gender By Season And Sales",fontweight="bold",fontsize=15)
plt.xlabel("Seasons")

plt.subplot(2,2,3)
sns.barplot(data=classifi_sea,x="Category",y="Purchase Amount (USD)",hue="Season")
plt.title("Classification On Season By Sales And Category",fontweight="bold",fontsize=15)
plt.xlabel("Categories")

plt.subplot(2,2,4)
sns.barplot(data=classifi_sea,x="Item Purchased",y="Purchase Amount (USD)",hue="Gender")
plt.xticks(rotation=45,ha="right")
plt.title("Classification On Gender By Product Names And Sales",fontweight="bold",fontsize=15)
plt.xlabel("Product Names")
plt.tight_layout()
plt.show()


# ### `Interactive Visualization`

# In[23]:


import plotly.express as px

fig = px.bar(
    classifi_sea,
    x="Category",
    y="Purchase Amount (USD)",
    color="Season",
    title="Classification On Sales By Category And Season"
)
fig.show()


# ### `How Many People Used The Promo Code And If They Purchased Discount Applied Product Or Not And Which Age Group Is Most Likely To Apply The Codes`

# In[24]:


classifi_dis=df.groupby(["Gender","Discount Applied","Promo Code Used"])["Age"].mean().reset_index()
classifi_dis.style.background_gradient(cmap="Greens")


# In[25]:


x=df["Promo Code Used"].value_counts().reset_index()
plt.figure(figsize=(10,6))
plt.bar(x["Promo Code Used"],x["count"],color=["Purple","Green"])
# Labels and title
plt.xlabel("Promo Code Used")
plt.ylabel("Count")
plt.title("Promo Code Usage Distribution",fontsize=14)
plt.show()


# ### `Relation Between Age & Ratings By Products And Categories`

# In[26]:


#Products and categories by highest to lowest ratings value
classifi_ratings=df.groupby(["Gender","Item Purchased","Category"]).agg({
    "Age":"mean",
    "Review Rating":"mean"
}).sort_values("Review Rating",ascending=False).reset_index()
classifi_ratings.style.background_gradient(cmap="Reds")


# In[27]:


classifi_ratings=df.groupby(["Gender","Item Purchased","Category"]).agg({
    "Age":"mean",
    "Review Rating":"mean"
}).reset_index().sort_values("Review Rating",ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=classifi_ratings,x="Category",y="Review Rating",hue="Gender",palette="Set1")
plt.show()


# ### `Alternate Visualization For Better Understanding`

# In[28]:


pivot_table = classifi_ratings.pivot_table(index="Category", columns="Gender", values="Review Rating")
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Heatmap of Review Ratings by Gender & Category")
plt.show()


# ### `Which Clothing Colour People Buys The Most`

# In[29]:


x=df[df["Category"]=="Clothing"]
classifi_co = x.groupby(["Gender", "Color"]).size().reset_index(name="Count")
classifi_co.sort_values("Count",ascending=False).style.background_gradient(cmap="Blues")


# #### *<p style="color:Orange;">As we can see Teal,Green,Violet and Maroon in the Male Category and Black,Pink,Orange,Maroon and Teal in the Feamle category, are the most poplar colours among customers when it comes to clothing category.*

# In[30]:


x=df[df["Category"]=="Accessories"]
classifi_col=x.groupby(["Gender","Color"]).size().reset_index(name="Count")
classifi_col.sort_values("Count",ascending=False).style.background_gradient(cmap="Greens")


# #### *<p style="color:Orange;">As we can see Olive,Gray,Black,Charcoal and Yellow in the Male Category and Green,Blue,Magenta,Peach and Lavender in the Feamle category, are the most poplar colours among customers when it comes to Accessories category.*

# In[31]:


df["Category"].value_counts()


# In[32]:


x=df[df["Category"]=="Footwear"]
classifi_colo=x.groupby(["Gender","Color"]).size().reset_index(name="Count")
classifi_colo.sort_values("Count",ascending=False).style.background_gradient(cmap="Purples")


# #### *<p style="color:Orange;">As we can see Olive,Violet,Teal & Cyan in the Male Category and Gray,Brown,Silver & Blue in the Feamle category, are the most poplar colours among customers when it comes to Footwear category.*

# In[33]:


dt=df.select_dtypes(include="number").corr()
plt.figure(figsize=(10,8))
sns.heatmap(dt,annot=True,fmt=".2f")
plt.show()


# ## <p style="color:Orange;">🧠 Overall Story
# ### **The dataset like a detective at a scene — first surveying the territory, then cleaning and preparing the clues. I used descriptive statistics to sketch the customer portrait, visual tools to uncover behavioral trends, and segmentation to hint at actionable insights. By the end, my analysis paints a picture of who the customers are, what they buy, when, and how much they spend — a full-cycle customer intelligence report.**
