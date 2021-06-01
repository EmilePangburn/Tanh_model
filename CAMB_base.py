import numpy as np
import camb



def cl_from_CAMB(tau,model=None):
    cmb_params = {'omch2': 0.120, 
                  'ombh2': 0.0224,'H0': 67.4,'As': 10**-10*np.exp(3.047),'ns': 0.965,'tau': tau}
    
    if model is not None:
        cmb_params.pop('tau')
        cmb_params['Reion']=model
        
    camb_params = camb.set_params(**cmb_params)
    results = camb.get_results(camb_params)
    powers =results.get_cmb_power_spectra(camb_params, CMB_unit='muK')
    totCl = powers['total']
    return totCl

def xe(z, tau=0.0544, dz=0.5, f=1.08, with_helium=False,zre=None):
    cmb_params = {'omch2': 0.120, 'ombh2': 0.0224,'H0': 67.4,'As': 10**-10*np.exp(3.047),'ns': 0.965,'tau': tau}

    He_fraction = 8.1884281020483535E-002
    y = (1+z)**(3/2)
    camb_params = camb.set_params(**cmb_params)
    if zre is not None:
        zr = zre
    if zre is None :
        zr = camb.get_zre_from_tau(camb_params, tau)
    yre = (1+zr)**(3/2)
    dy = 3/2 * (1 + z)**(1/2) * dz   
    xe = (1+He_fraction)/2 * (1 + np.tanh((yre - y) / dy))  
    # Effect of Helium becoming fully ionized is small so details not important
    
    He_redshift = 3.5
    He_delta = 0.4
    xe += He_fraction / 2 * (1 + np.tanh((He_redshift - z)/He_delta))   
    #xe += He_fraction  * (1 + np.tanh((He_redshift - z)/He_delta))   
    return xe



def get_deriv(cmb_params,parameter,model=None):
    if parameter == 'As':
        stepsize = 10**(-11)
    if parameter != 'As':
        stepsize = 0.01
    cmb_params_l = cmb_params.copy()
    cmb_params_r = cmb_params.copy()
    cmb_params_l[parameter] = cmb_params[parameter]-stepsize
    cmb_params_r[parameter] = cmb_params[parameter]+stepsize
    pars_left = camb.set_params(**cmb_params_l,Reion=model)
    pars_right = camb.set_params(**cmb_params_r,Reion=model)
    result_left = camb.get_results(pars_left)
    result_right = camb.get_results(pars_right)
    powers_left =result_left.get_cmb_power_spectra(pars_left, CMB_unit='muK')
    powers_right =result_right.get_cmb_power_spectra(pars_right, CMB_unit='muK')
    cl_left = powers_left['total']
    cl_right = powers_right['total']
    cl_tt_left = cl_left[:,0]
    cl_tt_right = cl_right[:,0]
    cl_te_left = cl_left[:,3]
    cl_te_right = cl_right[:,3]
    cl_ee_left = cl_left[:,1]
    cl_ee_right = cl_right[:,1]
    cl_bb_left = cl_left[:,2]
    cl_bb_right = cl_right[:,2]
    
    dCltt_dh = (cl_tt_right - cl_tt_left) / (2 * stepsize)
    dClte_dh = (cl_te_right - cl_te_left) / (2 * stepsize)
    dClee_dh = (cl_ee_right - cl_ee_left) / (2 * stepsize)
    dClbb_dh = (cl_bb_right - cl_bb_left) / (2 * stepsize)
    return dCltt_dh,dClte_dh,dClee_dh,dClbb_dh



def Fisher_noise_ij(cmb_params,param1,param2,f_sky): 
    F_ij = 0
    if param1==param2:
        res1 = get_deriv_new(param1)
        res2 = res1
    if param1 != param2:
        res1 = get_deriv_new(param1)
        res2 = get_deriv_new(param2)
    results = camb.get_results(cmb_params)
    powers =results.get_cmb_power_spectra(cmb_params, CMB_unit='muK')
    totCL=powers['total']
    for l in range(2,2000):
        mat_cl = np.zeros((2,2))
        matder1 = np.zeros((2,2))
        matder2 = np.zeros((2,2))
        mat_cl[0,0] = totCL[l,0]+noise(sT,l)
        mat_cl[0,1] = totCL[l,3]
        mat_cl[1,0] = totCL[l,3]
        mat_cl[1,1] = totCL[l,1]+noise(sP,l)
        
        matder1[0,0] = res1[0][l]
        matder1[0,1] = res1[1][l]
        matder1[1,0] = res1[1][l]
        matder1[1,1] = res1[2][l]
        
        matder2[0,0] = res2[0][l]
        matder2[0,1] = res2[1][l]
        matder2[1,0] = res2[1][l]
        matder2[1,1] = res2[2][l]
        inv_cl = np.linalg.inv(mat_cl)
        F_ij += (2*l+1)*f_sky/2*np.trace(np.dot(np.dot(inv_cl,matder1),np.dot(inv_cl,matder2)))
    return F_ij

def Full_Fisher_noise():
    f_sky = 1
    liste = ['omch2','ombh2','H0','As','ns','optical_depth']
    Fisher = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            Fisher[i,j] = Fisher_noise_ij(liste[i],liste[j],f_sky)
            print(i,j)

    return Fisher


