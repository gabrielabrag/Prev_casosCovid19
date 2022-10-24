#!/usr/bin/env python
# coding: utf-8

# Projeto evolução COVID-19
# 
#     Digital Innovation One

# In[4]:


import pandas as pd 
import numpy as np
from datetime import datetime 
import plotly.express as px 
import plotly.graph_objects as go 


# In[7]:


#import dados 
df= pd.read_csv("covid_19_data.csv",parse_dates=['ObservationDate','Last Update'])
df


# In[8]:


#conferir tipo dados
df.dtypes


# In[9]:


#alterar nome coluna
import re 

def corrigi_colunas(col_name):
    return re.sub(r"[/|]","",col_name).lower()


# In[10]:


corrigi_colunas("Province/State")#teste funcao


# In[12]:


#corrige nome todas colunas 
df.columns = [corrigi_colunas(col)for col in df.columns]


# In[13]:


df


# #Brasil 
# 

# In[14]:


df.loc[df.countryregion =='Brazil']


# In[15]:


brasil = df.loc[(df.countryregion =='Brazil') & 
                (df.confirmed >0)]


# In[16]:


brasil


# Casos confirmados
# 

# In[18]:


#grafico da evolução dos casos confirmados
px.line(brasil,'observationdate','confirmed',title= 'Casos confirmados no Brasil')


# Novos casos por dia 

# In[24]:


#tecnoca de programacao funcional 
brasil ['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil ['confirmed'].iloc[x-1],
    np.arange(brasil.shape[0])
))


# In[27]:


#vizualizar 
px.line(brasil, x='observationdate',y='novoscasos',title="Novos casos por dia")


# Mortes 
# 

# In[29]:


fig = go.Figure()
fig.add_trace(
    go.Scatter(x=brasil.observationdate, y= brasil.deaths, name='Mortes',
              mode= 'lines+markers', line={'color':'red'})
)
#Layout
fig.update_layout(title = 'Mortes por covid no Brasil')

fig.show()


# Taxa de Crescimento
# 
# taxa_crescimento = (presente/passado)**(1/n)-1

# In[51]:


def taxa_crescimento(data,variable,data_inicio= None, data_fim=None):
    if data_inicio ==None:
        data_inicio = data.observationdate.loc[data[variable]>0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
            
    if data_fim==None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    #define valores do presente e passado
    passado = data.loc[data.observationdate == data_inicio,variable].values[0]
    presente= data.loc[data.observationdate == data_fim,variable].values[0]
    
    #define n° de pontos no tempo q vamos avaliar 
    n= (data_fim - data_inicio).days
    
    #calcula taxa
    taxa= (presente/passado)**(1/n) - 1
    return taxa * 100
            


# In[52]:


#taxa crescimento medio covid em todo perioso
taxa_crescimento(brasil,'confirmed')


# In[64]:


def taxa_crescimento_diaria(data, variable, data_inicio=None):

  if data_inicio == None: 
    data_inicio = data.observationdate.loc[data[variable] > 0].min()
  else:
    data_inicio = pd.to_datetime(data_inicio)

  data_fim = data.observationdate.max()
  n = (data_fim - data_inicio).days

  taxas = list(map(
      lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
      range(1, n+1) 
  ))
  return np.array(taxas) * 100


# In[65]:


tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')
tx_dia


# In[66]:


primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0 ]. min()

px.line(x=pd.date_range(primeiro_dia,brasil.observationdate.max())[1:],
       y=tx_dia,title="Taxa de crescimento de casos confirmados no Brasil")


# Predicões

# In[67]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt 


# In[69]:


confirmados= brasil.confirmed
confirmados.index = brasil.observationdate
confirmados


# In[70]:


res=seasonal_decompose(confirmados)


# In[73]:


fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle= 'dashed', c= 'black')
plt.show


# Arima
# aprender com passado,prever o futuro
# 

# In[78]:


get_ipython().system('pip install pmdarima')


# In[77]:


from pmdarima.arima import auto_arima
modelo= auto_arima(confirmados)


# In[79]:


fig= go.Figure(go.Scatter(
    x=confirmados.index , y= confirmados, name='Observados'
))

fig.add_trace(go.Scatter(
    x=confirmados.index, y= modelo.predict_in_sample(),name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y= modelo.predict(31),name= 'Forecast'
))

fig.update_layout(title='Previsão de casoso confirmados brasil prox 31 dias')
fig.show()


# Modelo de Crescimento

# In[ ]:




