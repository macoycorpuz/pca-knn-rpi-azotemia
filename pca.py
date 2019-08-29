import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def onClickExit(e):
    plt.close()

def pca():
    sensors = ['MQ2','MQ3','MQ4','MQ6','MQ7','MQ8','MQ135']

    df = pd.read_csv('_dataset.csv', error_bad_lines=False)
    sensor_values = df.loc[:, sensors].values

    sensor_values = StandardScaler().fit_transform(sensor_values)
    pca = PCA(n_components=2)
    components = pca.fit_transform(sensor_values)
    principalDf = pd.DataFrame(data=components, columns=['pc1', 'pc2'])
    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)


    fig = plt.figure()
    fig.canvas.manager.full_screen_toggle()
    ax = fig.add_axes([0.13, 0.25, 0.80, 0.68])
    ax_close = plt.axes([0.85, 0.04, 0.13, 0.05])
    ax.set_xlabel('PC 1', fontsize = 10)
    ax.set_ylabel('PC 2', fontsize = 10)
    targets = ['healthy', 'azotemic']
    colors = ['g', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'], c = color, s = 50)
    ax.legend(targets)
    ax.grid()

    btnExit = Button(ax_close, 'Exit')
    btnExit.label.set_fontsize(9)
    btnExit.on_clicked(onClickExit)
    plt.show()

if __name__ == "__main__":
    pca()    