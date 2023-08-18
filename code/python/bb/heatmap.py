#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 23:47:06 2023

@author: chenglinlin
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt

# f1='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/virtual_2d_l2cs_cal/virtual_2d_p1.csv'
# f2='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/virtual_2d_l2cs/virtual_2d_p1.csv'
# root='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/processed_data/virtual_2d_l2cs_cal/'

f1 = 'E:/Master/MasterThesis/L2CS-Net-main/output/simple_mytest6.csv'
# f2='E:/Master/MasterThesis/L2CS-Net-main/output/virtual_2d_p1.csv'
root = 'E:/Master/MasterThesis/L2CS-Net-main/output/'
"""
df1=pd.read_csv(f1)
# df2=pd.read_csv(f2)
plt.plot(df1['yaw'],df1['pitch'],'g.')
# plt.plot(df2['virtual2d_x'],df2['virtual2d_y'],'r.')
plt.show()
"""

# df = pd.DataFrame()
f = root + 'simple_mytest6.csv'
df1 = pd.read_csv(f)[['yaw', 'pitch']]
df = df1.copy()
df.rename(columns={'yaw': 'pitch', 'pitch': 'yaw'}, inplace=True)
print(df.head())
# for i in range(1):
#     x='p'+str(i+1)
#     f=root+'virtual_2d_'+x+'.csv'
#     df1=pd.read_csv(f)[['virtual2d_x','virtual2d_y']]
#     df=df.append(df1)

# df1_1=pd.read_csv(f)[['virtual2d_x','virtual2d_y']].iloc[150*0:150*1]
# df1_2=pd.read_csv(f)[['virtual2d_x','virtual2d_y']].iloc[150*3:150*4]
# df1_3=pd.read_csv(f)[['virtual2d_x','virtual2d_y']].iloc[150*6:150*7]
# df=df.append(df1_1)
# df=df.append(df1_2)
# df=df.append(df1_3)

plt.plot(df['yaw'],df['pitch'],'g.')
x=df['yaw']
y=df['pitch']
plt.scatter(x, y, s=70, alpha=0.3)
# plt.show()


#df1=df[df['pos_number']==5][['virtual_l2cs_cal_x','virtual_l2cs_cal_y']]
# df1=df1['yaw','pitch']
df1=df.dropna()
samples = df1.to_numpy()

kde = stats.gaussian_kde(samples.T)

# Regular grid to evaluate kde upon
x_flat = np.r_[samples[:,0].min():samples[:,0].max():200j]
y_flat = np.r_[samples[:,1].min():samples[:,1].max():200j]
x,y = np.meshgrid(x_flat,y_flat)
grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

z = kde(grid_coords.T)
z = z.reshape(200,200)

glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
sns.heatmap(glue)
plt.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),extent=[-50,50,-50,50])
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde as kde
# from matplotlib.colors import Normalize
# from matplotlib import cm


# df1=df[abs(df)<100]
# df1=df1.dropna()
# samples = df1.to_numpy()
# densObj = kde( samples.T )

# def makeColours( vals ):
#     colours = np.zeros( (len(vals),1) )
#     norm = Normalize( vmin=vals.min(), vmax=vals.max() )

#     #Can put any colormap you like here.
#     colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

#     return colours

# colours = makeColours( densObj.evaluate( samples ) )

# plt.scatter( samples[0], samples[1], color=colours )
# plt.show()

"""
# Create some dummy data
df1=df[abs(df)<80]
# df1=df[abs(df['virtual2d_x'])<100]
# df1=df1[df1['virtual2d_y']<80]
# df1=df1[df1['virtual2d_y']>-120]
df1=df1.dropna()

samples = df1.to_numpy()

kde = stats.kde.gaussian_kde(samples.T)

# Regular grid to evaluate kde upon
x_flat = np.r_[samples[:,0].min():samples[:,0].max():200j]
y_flat = np.r_[samples[:,1].min():samples[:,1].max():200j]
x,y = np.meshgrid(x_flat,y_flat)
grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

z = kde(grid_coords.T)
z = z.reshape(200,200)

f='/Users/chenglinlin/ownCloud/BETA_AI_Gaze_Estimation_in_Human-Robot_Interaction (Projectfolder)/original_data/formal_experiment/p1_2022_11_14_17_35_52_78206780_Header.csv'
ideal=pd.read_csv(f)[['x','y']]
ix=ideal['x']*142/3840*2
iy=ideal['y']*80/2160*2
plt.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),extent=[-100,100,-100,100])
plt.plot(ix,iy,'r.')
plt.show()
"""
