import numpy as np
import matplotlib.pyplot as matplt

def generate_utility_function(resolution=100, min_val=0, max_val=1,):
    No_U_Curve = True
    No_Clip_Curve = False
    while True:
        # random select point between low and high, and build the function
        fx = np.array([min_val,np.random.uniform(min_val,max_val), max_val])
        for i in range(resolution-3): # since we have 3 point at the beginning
            idx = np.random.choice(len(fx)-1)
            mid = np.random.uniform(fx[idx], fx[idx+1])
            fx = np.insert(fx, idx+1, mid)

        # use the 2-deg poly function to fit the previous function
        coeff = np.polyfit(np.arange(len(fx)),np.log(fx+1),2)
        yy = np.poly1d(coeff)
        fit_fx = np.zeros(len(fx))
        for j in range(len(fx)):
            fit_fx[j] = yy(j)

        #matplt.plot(fit_fx)
        if coeff[0] > 0 and No_U_Curve:  # this means U curve
            continue

        # to ensure non-decreasing, we flat the curve
        min_idx, max_idx = [], []
        tmp_min, tmp_max = fit_fx[0], fit_fx[-1]
        # find if there is a decreasing trend
        for i in range(len(fit_fx)):
            if (fit_fx[i] < tmp_min):
                min_idx.append(i)

            if (fit_fx[i] > tmp_max):
                max_idx.append(i)
        # if we found it, then we flatten it
        if min_idx:
            fit_fx[min_idx] = tmp_min
            if No_Clip_Curve:
                continue  # if it is U curve, we re-generate one

        if max_idx:
            fit_fx[max_idx] = tmp_max
            if No_Clip_Curve:
                continue  # if it is cliped N curve, we re-generate one

        break # if not bad U curve, then we think we get the curve
    clip_fx = fit_fx

    # we normalize the fx
    clip_fx = min_val + clip_fx - min(clip_fx)  # normalize to min_val
    clip_fx = max_val * clip_fx/max(clip_fx)   # normalize to max_val

    final_fx = clip_fx
    return final_fx

def generate_alpha_fairness_function(resolution=1000, min_val=0, max_val=1,):
    x = np.linspace(0,1,resolution)
    alpha = np.random.uniform(0,1)
    fx = (x**alpha)/alpha

    fx = min_val + fx - min(fx)  # normalize to min_val
    fx = max_val * fx/max(fx)   # normalize to max_val

    final_fx = fx
    return final_fx
