import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import os

from main import get_args




def daily_energy_requirment_plotting():

    args = get_args()

    colors_xkcd = ['very dark purple', "windows blue", "amber", "faded green", "darkish red", "greyish",
                     "dusty purple", 'pastel blue', "pumpkin orange" ]

    fig, ax = plt.subplots()


    colors_xkcd = ['very dark purple', "windows blue", "amber", "faded green", "darkish red", "greyish",
                   "dusty purple", 'pastel blue', "pumpkin orange"]
    linetypes = ['-', '-.']

    # cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
    # sns.set_palette(sns.xkcd_palette(colors_xkcd))
    # sns.set_style("whitegrid")

    num_cities = 14

    csv_filename = 'irrig_elec_results_year_2018_irriglb_80_h20req_6.csv'
    csv_pathname = os.path.join(args.elec_results_dir, csv_filename)
    raw_data = pd.read_csv(csv_pathname, usecols= range(1,num_cities+1))
    print(raw_data.shape)
    x_domain = range(365)

    ax.plot([1, 2, 3], [1, 2, 3])
    #
    # for i in range(num_cities):
    #     # print(np.mod(i,7))
    #     # linestyle =  linetypes[int(np.floor_divide(i, 7))]
    #     # color = cmap[int(np.remainder(i, 7))]
    #
    #
    #     ax.plot(x_domain, x_domain) #, linestyle = linestyle) #, color = color) #, label = raw_data.columns[i])


    plt.show()


if __name__ == '__main__':
    # print('a')

    daily_energy_requirment_plotting()
