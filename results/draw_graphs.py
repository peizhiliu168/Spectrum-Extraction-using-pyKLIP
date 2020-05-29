import os
import glob
import math
import numpy as np
import pandas as pd
import pysynphot as S
from typing import NamedTuple
from uncertainties import ufloat
from uncertainties import unumpy
import matplotlib.pyplot as plt

class data_file(NamedTuple):
    name: str = ""
    movement: int = -1
    spect: list = []
    error: list = []
    num_basis: list = []
    wvs: list = []

################################ INITIALIZE DATA DIRECTORY AND PARAMETERS ################################
data_dir = "hd1160"

#ref_file = "hd1160b_published.csv"
ref_file = "hd1160b_cobi.csv"
ref_author = "Currie et al"
numbasis = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
movements = [3,5,7,9]
spectral_lib = "kurucz"

#magnitudes for j,h, and k bands
auto_calc_mag = False
#ref_magnitudes = [7.72,7.20,6.91]
#ref_magnitudes_error = [0.02,0.02,0.10]
ref_magnitudes = [11.58,10.70,10.05]
ref_magnitudes_error = [0.0985,0.086,0.086]
ref_magnitudes_wvs = np.array([1.250,1.635,2.150])
corresponding_slices = [2,10,18]
#corresponding_slices = [2,9,15]
masking = False
mask = np.array([False,True,True,True,
                True,False,False,False,
                True,True,True,True,
                True,False,False,True,
                True,True, True,True,
                True,False])

# pickels
spect_file = "/home/PeizhiLiu/cdbs/grid/pickles/dat_uvk/pickles_uk_9.fits"
star_mag = np.array([ufloat(1.91,0.1), # absolute magnitude
                    ufloat(1.91,0.1),
                    ufloat(1.91,0.1),
                    ufloat(1.91,0.1),
                    ufloat(1.91,0.1),
                    ufloat(1.91,0.1),
                    ufloat(1.91,0.1),
                    ufloat(1.91,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.94,0.1),
                    ufloat(1.97,0.1),
                    ufloat(1.97,0.1),
                    ufloat(1.97,0.1),
                    ufloat(1.97,0.1),
                    ufloat(1.97,0.1),
                    ufloat(1.97,0.1),
                    ufloat(1.97,0.1)])
# kurucz
eff_temp = 11327
log_g = 4.174
metallicity = -0.36
R_star = ufloat(2.29, 0.06) * 6.957 * (10**5)
D_star = ufloat(50.0, 0.1) * 3.086e13
'''eff_temp = 9011
log_g = 4.5
metallicity = -0.30
R_star = 1.7747 * 6.957 * (10**5)
D_star = 3.19367298 * (10**15)'''
# units
wv_unit = "micron"
spect_unit = "mJy"
#graph
#individual_graph_title = "Kappa And b Spectrum (KL={basis})"
#group_graph_title = "Kappa And b Spectra"
individual_graph_title = "HD1160 b Spectrum (KL={basis})"
group_graph_title = "HD1160 b Spectra"


################################ READ IN DATA ################################
data_files = glob.glob(data_dir+"/*.csv")

# reads in data and stores them in a data_file struct, creates a list of 
# data_file structs called data
data = []
num_basis_string = [str(val) for val in numbasis]
for mov in movements:
    files = []
    for file in data_files:
        if str(mov) in file:
            files.append(file)
    for file in files:
        if "spectrum" in file:
            df = pd.read_csv(file)
            wvs = df["wvs"].to_numpy()
            spect = df[num_basis_string].to_numpy().transpose()
            name = file
        elif "error" in file:
            df = pd.read_csv(file)
            error = df[num_basis_string].to_numpy().transpose()
    d = data_file(name, mov, spect, error, numbasis, wvs)
    data.append(d)
    
#reads reference file if one is specified
if ref_file != "":
    df = pd.read_csv(ref_file)
    ref_wvs = df.to_numpy().transpose()[0]
    ref_val = df.to_numpy().transpose()[1]
    try:
        ref_err = df.to_numpy().transpose()[2]
    except:
        pass
    
    
################################ PERFORM SPECTRAL CALIBRATION ################################
os.chdir("/home/PeizhiLiu/cdbs")
calibrated_data = []
nl = len(data[0].wvs)
#satellite spot to star ratio
spot_to_star_ratio = np.array([2.72*(10**-3)/(lamb/1.55)**2 for lamb in data[0].wvs])

if spectral_lib == "pickles":
    #star spectrum
    sp = S.FileSpectrum(spect_file)
    sp.convert(wv_unit)
    sp.convert(spect_unit)
    star_spectrum = np.array([sp.sample(data[0].wvs[i]) for i in range(nl)])
    star_spectrum = star_spectrum * 10**((star_mag)/-2.5)
    #loop through every dataset in data
    for spect in data:  
        #convert to ufloat
        exspect = spect.spect
        e = spect.error
        new_exspect = np.ndarray(shape=np.shape(exspect), dtype=object)
        for i in range(len(exspect)):
            for j in range(nl):
                new_exspect[i][j] = ufloat(exspect[i,j],e[i,j])
        #perform actual calibration
        exspect_flux = new_exspect * star_spectrum * spot_to_star_ratio
        #separate values
        actual_values = np.zeros(np.shape(exspect))
        error_bars = np.zeros(np.shape(exspect))
        for i in range(len(exspect)):
            for j in range(nl):
                actual_values[i][j] = exspect_flux[i][j].nominal_value
                error_bars[i][j] = exspect_flux[i][j].std_dev
        
        d = data_file(spect.name, spect.movement, actual_values, error_bars, 
                      spect.num_basis, spect.wvs)
        calibrated_data.append(d)    
        
        
elif spectral_lib == "kurucz":
    #star spectrum
    sp = S.Icat('ck04models', eff_temp, metallicity, log_g)
    sp.convert(wv_unit)
    sp.convert(spect_unit)
    star_spectrum = np.array([sp.sample(data[0].wvs[i]) for i in range(nl)]) * (R_star / D_star)**2
    #loop through every dataset in data
    for spect in data:  
        #convert to ufloat
        exspect = spect.spect
        e = spect.error
        new_exspect = np.ndarray(shape=np.shape(exspect), dtype=object)
        for i in range(len(exspect)):
            for j in range(nl):
                new_exspect[i][j] = ufloat(exspect[i,j],e[i,j])
        #perform actual calibration
        exspect_flux = new_exspect * star_spectrum * spot_to_star_ratio
        #separate values
        actual_values = np.zeros(np.shape(exspect))
        error_bars = np.zeros(np.shape(exspect))
        for i in range(len(exspect)):
            for j in range(nl):
                actual_values[i][j] = exspect_flux[i][j].nominal_value
                error_bars[i][j] = exspect_flux[i][j].std_dev
        
        d = data_file(spect.name, spect.movement, actual_values, error_bars, 
                      spect.num_basis, spect.wvs)
        calibrated_data.append(d)    
    
os.chdir("/home/PeizhiLiu/Documents/Synced/pyklip_tests/subaru_data")

################################ PLOT SPECTRUM ################################
# display uncalibrated data
for dataset in data:
    fig = plt.figure(figsize=(20,20), )
    for i in range(len(numbasis)):
        ax = fig.add_subplot(math.ceil(len(numbasis)/4)+1,4,i+1)
        title = individual_graph_title
        ax.set_title(title.format(basis=numbasis[i]))
        y_axis = "flux (with respect to satellite spots)"
        x_axis = "wavelength ({wv_units})"
        ax.set(xlabel=x_axis.format(wv_units=sp.waveunits), ylabel=y_axis.format(flux_units=sp.fluxunits))
        ax.errorbar(dataset.wvs, dataset.spect[i], yerr = dataset.error[i], label='pyKLIP', capsize=3)
        ax.errorbar(ref_wvs, ref_val, label='Cobi', capsize=3)
        ax.legend()
        
    ax = fig.add_subplot(math.ceil(len(numbasis)/4) + 1,1,5)
    ax.set_title(group_graph_title)
    ax.set(xlabel=x_axis.format(wv_units=sp.waveunits), ylabel=y_axis.format(flux_units=sp.fluxunits))
    for i in range(len(numbasis)):
        ax.plot(dataset.wvs, dataset.spect[i], label=str(numbasis[i]))
        ax.legend()
    
    plt.tight_layout()

# display calibrated data with reference if specified
mask = np.invert(mask)
for dataset in calibrated_data:
    data_wvs = dataset.wvs
    if masking == True:
        data_wvs = np.ma.masked_array(dataset.wvs, mask=mask)
    fig = plt.figure(figsize=(20,20), )
    for i in range(len(numbasis)):
        ax = fig.add_subplot(math.ceil(len(numbasis)/4)+1,4,i+1)
        title = individual_graph_title
        ax.set_title(title.format(basis=numbasis[i]))
        y_axis = "flux ({flux_units})"
        x_axis = "wavelength ({wv_units})"
        ax.set(xlabel=x_axis.format(wv_units=sp.waveunits), ylabel=y_axis.format(flux_units=sp.fluxunits))
        ax.errorbar(data_wvs, dataset.spect[i], yerr=dataset.error[i], label='pyKLIP', capsize=3)
        if ref_file != "":
            try:
                ax.errorbar(ref_wvs, ref_val, yerr=ref_err, label=ref_author, capsize=3)
            except:
                ax.errorbar(ref_wvs, ref_val, label=ref_author, capsize=3)
        ax.legend()
        
    ax = fig.add_subplot(math.ceil(len(numbasis)/4) + 1,1,5)
    ax.set_title(group_graph_title)
    ax.set(xlabel=x_axis.format(wv_units=sp.waveunits), ylabel=y_axis.format(flux_units=sp.fluxunits))
    for i in range(len(numbasis)):
        ax.plot(data_wvs, dataset.spect[i], label=str(numbasis[i]))
        ax.legend()
    
    plt.tight_layout()

################################ PLOT MAGNITUDES ################################
# magnitude plots
if auto_calc_mag:
    pass
for dataset in data: 
    #convert to ufloat
    exspect = dataset.spect
    e = dataset.error
    new_exspect = np.ndarray(shape=np.shape(exspect), dtype=object)
    for i in range(len(exspect)):
        for j in range(nl):
            new_exspect[i][j] = ufloat(exspect[i,j],e[i,j])
    mag_val = new_exspect[:,corresponding_slices]
    mag_wvs = np.array([dataset.wvs[corresponding_slices[0]], dataset.wvs[corresponding_slices[1]], dataset.wvs[corresponding_slices[2]]])
    
    mag_val = -2.5*unumpy.log10(mag_val*spot_to_star_ratio[corresponding_slices])
    
    #separate values
    actual_mag_val = np.zeros(np.shape(mag_val))
    mag_error_bars = np.zeros(np.shape(mag_val))
    for i in range(len(mag_val)):
        for j in range(3):
            actual_mag_val[i][j] = mag_val[i][j].nominal_value
            mag_error_bars[i][j] = mag_val[i][j].std_dev
    
    fig, ax = plt.subplots()
    ax.set_title(r'$\Delta$J, $\Delta$H, $\Delta$K Comparison')
    ax.set(xlabel=x_axis.format(wv_units=sp.waveunits), ylabel="Difference in planet and star magnitude")
    for i in range(len(numbasis)):
        ax.errorbar(mag_wvs, actual_mag_val[i], yerr=mag_error_bars[i], label='pyKLIP', capsize=3, color="orange")
    ax.errorbar(ref_magnitudes_wvs, ref_magnitudes, yerr=ref_magnitudes_error, label=ref_author, capsize=3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    #ax.set_ylim(6.5,10)
    ax.set_ylim(9,13)



# all other useless code
'''movements = []
for file in data_files:
    for sec in re.split("_|\.", file):
        if sec.isdigit() and int(sec) not in movements:
            movements.append(int(sec))'''