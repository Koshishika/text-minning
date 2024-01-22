import pandas as pd
from sklearn.cluster import KMeans
import japanize_matplotlib
from matplotlib import pyplot as plt
import numpy as np

score_df = pd.read_csv("/Users/shishishikakou/Downloads/text_mininngⅡ/NLPⅡ_1 クラスタリング/fortravel_bow.csv", encoding='ms932',index_col=0)
print(score_df)
vec = KMeans(n_clusters=5)
pred_vec =vec.fit_predict(score_df.iloc[:,3:])
print(pred_vec)
score_df["class"]=pred_vec
print(score_df)
centers = vec.cluster_centers_
print(centers)

headers=score_df.columns.tolist()[3:-1]
print(headers)
fig = plt.figure()
x=np.arange(len(headers))

ax1=fig.add_subplot(511,title='cluster0 center')
ax1.bar(x,centers[0])
ax1.axes.xaxis.set_visible(False)#目盛りを消す

ax2=fig.add_subplot(512,title='cluster1 center')
ax2.bar(x,centers[1])
ax2.axes.xaxis.set_visible(False)

ax3=fig.add_subplot(513,title='cluster2 center')
ax3.bar(x,centers[2],tick_label=headers)
ax3.axes.xaxis.set_visible(False)

ax4=fig.add_subplot(514,title='cluster3 center')
ax4.bar(x,centers[3],tick_label=headers)
ax4.axes.xaxis.set_visible(False)

ax5=fig.add_subplot(515,title='cluster4 center')
ax5.bar(x,centers[4],tick_label=headers)
ax5.set_xticklabels(headers, rotation=45, ha='right',fontsize=6)

plt.subplots_adjust(hspace =0.5,bottom=0.2)   
plt.show()