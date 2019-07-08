import numpy as np
import time

# Plotting
import matplotlib.pyplot as plt

# Custom functions
from generic_functions import *
from multivariate_images_tools import *
from data_management import *
from change_detection_functions import *
from monte_carlo_tools import compute_several_statistics

if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    

    enable_multi = True
    # These two variables serves to split the original image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 3
    number_of_threads_columns = 2

    # Dataset
    dataset_choice = 'UAVSAR Scene 3'
    path_to_data = '../Data/'

    statistic_list = [covariance_equality_glrt_gaussian_statistic]
    statistic_names = [r'covariance_equality_glrt_gaussian_statistic']
    args_list = ['log']
    number_of_statistics = len(statistic_list)


    # Sliding windows mask used
    windows_mask = np.ones((9,9))
    m_r, m_c = windows_mask.shape

    dataset = fetch_dataset(dataset_choice, path_to_data)
    function_to_compute = compute_several_statistics
    function_args = [statistic_list, args_list]

    for t in range(2,dataset.number_dates):

        results = sliding_windows_treatment_image_time_series_parallel(dataset.data[:,:,:,:t], windows_mask, function_to_compute, 
                    function_args, multi=enable_multi, number_of_threads_rows=number_of_threads_rows,
                    number_of_threads_columns=number_of_threads_columns)

        image_temp = np.nan*np.ones((dataset.number_rows, dataset.number_columns))
        image_temp[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)] = results[:,:,0] - results[:,:,0].min()
        plt.imsave('./Results/CD_Gauss_%d_omnibus.png'%t, image_temp)



