import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca():
    sensors = ['MQ2','MQ3','MQ4','MQ6','MQ7','MQ8','MQ135']

    df = pd.read_csv('dataset.csv')
    sensor_values = df.loc[:, sensors].values
    target_values = df.loc[:,['target']].values

    sensor_values = StandardScaler().fit_transform(sensor_values)
    pca = PCA(n_components=2)
    components = pca.fit_transform(sensor_values)
    principalDf = pd.DataFrame(data=components, columns=['pc1', 'pc2'])
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20)


    targets = ['healthy', 'azotemic']
    colors = ['g', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'], c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    pca()    