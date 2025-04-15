import numpy as np
from math import pi,cos,sin,sqrt,exp
import time
import random
from Free_fermions_functions import *
import sys

scriptname,L,t1,t2,t12,p2,Nmax,NRseries,multiplier=sys.argv #pass argument from the function call in the bash script

L=int(L) #size of the system (each chain)
t1=float(t1) #chain 1 hopping
t2=float(t2) #chain 2 hopping
t12=float(t12) #inter chain hopping
p2=float(p2) #noise probability on chain 2
Nmax=int(Nmax) #number of cycles
NRseries=int(NRseries) #number of trajectories to simulate in series (one after the other)
multiplier=float(multiplier) #multiplier for the probability (if it is passed as an integer value using the JOB ARRAY feature)
p2=p2*multiplier #actual noise probability to use

#array of noise probabilities on chain 1
p1v=0.05*np.arange(1,21) 

#calculate once the matrix R. The first argument is 0.5 because for each cycle we perform two time evolutions with R for half a cycle each (the first evolution includes measurements at the end, the second evolution is purely unitary)
R = Rmn(0.5,t1,t2,t12,L)

#define the number of cycles between two evaluations of the correlations
t_step = 1

folder='/g100_work/IscrC_EMEND-Q/Correlations/Cr/t2_'+str(np.round(t2,2))+'/L'+str(L)#define folder where to store results. Change it to your path. make sure you actually created the folders

#loop over the number of serial trajectories
for i in range(NRseries):
    #loop over the different values of p1
    for p1 in p1v:
        #the order of the loops can be inverted. With this order we have to define the file names for each trajectory.
        #However if the simulations take more time than expected we have the same number of trajectories for each p1 (only smaller than NRseries)
        #if the loop over p1 is called first then we can define the file names just once, but we risk the job to terminate before the p1 loop is finished, so we'll have some values of p1 with no trajectories simulated.
        
        #define file names for the correlations C_11. A different one for every value p1
        file11A = folder + '/Cr11/Cr11A_L_'+str(L)+'t1'+str(np.round(t1,2))+'_t2'+str(np.round(t2,2))+'_t12'+str(np.round(t12,2))+'_p1'+str(np.round(p1,2))+'_p2'+str(np.round(p2,2))+'.dat' 
        file11B = folder + '/Cr11/Cr11B_L_'+str(L)+'t1'+str(np.round(t1,2))+'_t2'+str(np.round(t2,2))+'_t12'+str(np.round(t12,2))+'_p1'+str(np.round(p1,2))+'_p2'+str(np.round(p2,2))+'.dat'
                
            #define file names for the correlations C_22. A different one for every value p1
        file22A = folder + '/Cr22/Cr22A_L_'+str(L)+'t1'+str(np.round(t1,2))+'_t2'+str(np.round(t2,2))+'_t12'+str(np.round(t12,2))+'_p1'+str(np.round(p1,2))+'_p2'+str(np.round(p2,2))+'.dat' 
        file22B = folder + '/Cr22/Cr22B_L_'+str(L)+'t1'+str(np.round(t1,2))+'_t2'+str(np.round(t2,2))+'_t12'+str(np.round(t12,2))+'_p1'+str(np.round(p1,2))+'_p2'+str(np.round(p2,2))+'.dat'
               
            #define file names for the correlations C_12. A different one for every value p1
        file12A = folder + '/Cr12/Cr12A_L_'+str(L)+'t1'+str(np.round(t1,2))+'_t2'+str(np.round(t2,2))+'_t12'+str(np.round(t12,2))+'_p1'+str(np.round(p1,2))+'_p2'+str(np.round(p2,2))+'.dat' 
        file12B = folder + '/Cr12/Cr12B_L_'+str(L)+'t1'+str(np.round(t1,2))+'_t2'+str(np.round(t2,2))+'_t12'+str(np.round(t12,2))+'_p1'+str(np.round(p1,2))+'_p2'+str(np.round(p2,2))+'.dat'
        
        try:
            #Run the simulation and extract the arrays that give the correlation functions vs time and r.
            C11A,C12A,C22A,C11B,C12B,C22B,tvec = Transient_Cr_avg(L,t1,t2,t12,p2,p1,Nmax,t_step=t_step,R=R)
            #save them to file. Use 'ab+' to append, so that we save all trajectories into one file.
            #the protocol is: (i) save the tvec vector as a row (1,Nmax/t_step), so we know the actual times t at which Cr is calculated
            #(ii) save the correlations array as a 2Darray with size (L/2+1,Nmax/t_step) for Cr11, Cr22 -- size (L+1,Nmax/t_step) for Cr12.
            with open(file11A,'ab+') as f:
                np.savetxt(f,np.reshape(tvec,(1,len(tvec))))
                np.savetxt(f,C11A)
            with open(file11B,'ab+') as f:
                np.savetxt(f,np.reshape(tvec,(1,len(tvec))))
                np.savetxt(f,C11B)

            with open(file22A,'ab+') as f:
                np.savetxt(f,np.reshape(tvec,(1,len(tvec))))
                np.savetxt(f,C22A)
            with open(file22B,'ab+') as f:
                np.savetxt(f,np.reshape(tvec,(1,len(tvec))))
                np.savetxt(f,C22B)
                
            with open(file12A,'ab+') as f:
                np.savetxt(f,np.reshape(tvec,(1,len(tvec))))
                np.savetxt(f,C12A)
            with open(file12B,'ab+') as f:
                np.savetxt(f,np.reshape(tvec,(1,len(tvec))))
                np.savetxt(f,C12B)                
                
        except Exception as error:
            print("An exception occurred:", error)
