
import numpy as np
from scipy import optimize

def schmidtco2(tinc):# from the newer Wanninkhof 2014
    return 1923.6 -125.06*tinc+4.3773*tinc**2-0.085681*tinc**3+0.00070284 *tinc**4

def schmidtrn(tinc):# from the newer Wanninkhof 2014
    return 3171-224.28*tinc+8.2809 *tinc**2 -0.16699 *tinc**3+0.0013915 *tinc**4

def henryco2(tinc): # this is copied from Conroy 2023
    A = 108.3865# constant 
    B = 0.01985076 # constant 
    C = -6919.53 # constant 
    D = -40.4515 # constant 
    E = 669365 # constant 
    
    tempk =tinc+273.15
    
    return 10**(A+B*tempk+C/tempk+D*np.log10(tempk)+E/tempk**2)


def r2delta(r):
    pdb = 0.011180 
    output = ((r/pdb)-1)*1000
    return output

def delta2r(delta):
    pdb = 0.011180 
    output = pdb*(delta/1000+1)
    return output
   
        

def phfromalk(DIC,alk,tinc):
    tink = tinc + 273.15

   # % k1 values for CO2(d) and HCO3- Harned and Davis Jr 1943
    logk1 = -3404.71/tink + 14.8435 - 0.032786*tink
    
    #% k2 values for HCO3- and CO3-- Harned and Scholes Jr 1941
    logk2 = -2902.39/tink + 6.4980 - 0.02379*tink
    
 #   % Ionozation constant of H2O Millero et al. 1987 and Millero 1995
    pKw = -np.log10(np.exp(148.9802 - 13847.26/tink - 23.6521*np.log(tink))) # assuming zero ionic strength and zero salinity
    
    k1 = 10**logk1
    k2 = 10**logk2
    
    def henryco2(tinc): # this is copied from Conroy 2023
        A = 108.3865# constant 
        B = 0.01985076 # constant 
        C = -6919.53 # constant 
        D = -40.4515 # constant 
        E = 669365 # constant 

        tempk =tinc+273.15
    
        return 10**(A+B*tempk+C/tempk+D*np.log10(tempk)+E/tempk**2)



    
    def getph (modelph):
        pH = modelph
        pOH = pKw - pH
        H = 10**-pH
        OH = 10**-pOH
        modelAlk = DIC * (k1*H) / (H**2 + k1*H + k1*k2) + 2*DIC * (k1*k2) / (H**2 + k1*H + k1*k2) + OH - H
        ErrorVector = modelAlk - alk
        sse = ErrorVector ** 2
        modelpH = pH
        modelpOH = pOH
        
        return sse

    ph_solution = optimize.fmin(func=getph, x0=[6],disp=False)
    
    dco2_solution = alk/(k1/(10**-ph_solution)+2*k1*k2/((10**-ph_solution)**2))
    
    bicarb_solution = dco2_solution*k1/10**-ph_solution
    
    carb_solution = dco2_solution*k1*k2/((10**-ph_solution)**2)

    pco2_solution=1e6*dco2_solution/henryco2(tinc)
        
    return [ph_solution,dco2_solution,bicarb_solution,carb_solution,pco2_solution]


def epsilon_dic_co2(dic,alk,d13c,tinc): ## From Campeau et al., 2017
    [ph,co2,bicarb,carb,pco2] = phfromalk(dic,alk,tinc)
    
    e_bg = -0.1141*tinc+10.78
    e_dg = 0.0049*tinc-1.31
    e_db = e_dg-e_bg
    
    e_cg = -0.052*tinc+7.22
    e_cb = e_cg-e_bg/(1+e_db*10**-3)
    
    d13c_bicarb =(d13c*(dic)-(e_db*co2+e_cb*carb))/((1+e_db*10**-3)*co2+bicarb+(1+e_cb*10**-3)*carb)
      
    d13c_co2 = d13c_bicarb-(1+e_db*10**-3)+e_db
    
    return d13c_co2[0]

def epsilon_co2_dic(dic,alk,d13c,tinc): ## From Campeau et al., 2017
    [ph,co2,bicarb,carb,pco2] = phfromalk(dic,alk,tinc)
    
    e_bg = -0.1141*tinc+10.78
    e_dg = 0.0049*tinc-1.31
    e_db = e_dg-e_bg
    
    e_cg = -0.052*tinc+7.22
    e_cb = e_cg-e_bg/(1+e_db*10**-3)
    
    d13c_bicarb = d13c+(1+e_db*10**-3)-e_db
    d13c_carb = d13c_bicarb-(1+e_cb*10**-3)+e_cb
    d13c_total = (co2*d13c+bicarb*d13c_bicarb+carb*d13c_carb)/dic
    
    return d13c_total
