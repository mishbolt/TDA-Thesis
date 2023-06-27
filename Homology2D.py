# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:21:23 2022

@author: Misha
"""

import numpy as np
import matplotlib.pyplot as p
import string
import galois
from matplotlib.patches import Polygon
from math import comb
np.set_printoptions(threshold=np.inf)

# Computes the index of the lowest nonzero entry
def Low(A,j):
    if all(A[:,j]==0):
        i=-1
    else: 
        i=np.max(np.nonzero(A[:,j]))
    return(i)

# Compares matrices A and B col by col, outputs a matrix with "new" columns 
def Compare(A,B):
    index=[]
    if len(B)==0:
        C=A
    if len(A)==0:
        C=[]
    else:
        for i in range(len(A[0])):
            for j in range(len(B[0])):
                if np.all(A[:,i]==B[:,j]):
                    index=np.append(index,i)
        if len(index)==0:
           C=A 
        else:
            index=index.astype(int)        
        C=np.delete(A,index,1)
    return(C)

#Performs reduction algorithm described on page 70 of CTDA book
def Reduce(D):
    A = D.astype(int)
    GF = galois.GF(2)           
    A = GF(A)
    i=[]
    c=[]
    m=max(np.nonzero(np.any(A != 0, axis=0))[0])+1
    for j in range(m):
        for k in range(j):
            if Low(A,k)==Low(A,j) and not(Low(A,k)==-1):
                A[:,j]=A[:,k]+A[:,j]
        if not(Low(A,j)==-1):
            i=np.append(i,[Low(A,j)])
            c=np.append(c,[j])
    return(A,i,c)
        
            
# Input points
S = np.array([[0,0],
    [-1,-3],
    [2,-2],
    [2,0],
    [1,3],
    [5,1]])
#S= 10*np.random.rand(15,2)

# Distance matrix for the given coordinates of points 
R = np.zeros((len(S),len(S)))

for i in range(len(S)):
    for j in range(len(S)):
        R[i][j]=1/2*np.linalg.norm(S[i]-S[j])

#Sort the R matrix in increasing order (induces filtration)
UNr = sorted(list(set(i for j in R for i in j)))

# Set the initial D matrix as being zero matrix of appropriate dimension
D =np.zeros((len(S)+comb(len(S),2)+comb(len(S),3),len(S)+comb(len(S),2)+comb(len(S),3)))

# UNr=range(3)

# The big loop for varying radii              
for r in UNr:
    
    #Preliminary definitions
    H_0=len(S)
    H_1=0
    H_2=0
    
    #print(S)
    alphabet_string = string.ascii_lowercase
    
    Alphabet = list(alphabet_string)
    
    SV=[Alphabet[i] for i in range(len(S))]
    
    
    figure, axes = p.subplots(1)
    p.grid()
    ax = p.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    #p.xlim([-5, 7])
    #p.ylim([-5,5])
    
    #Plot points in S, with balls of radius r
    for i in range(len(S)):
        
        circles = p.Circle( (S[i][0],S[i][1]), r ,fill = False, linestyle ='-.' )
        axes.add_artist(circles)
        p.plot(S[i][0],S[i][1],'rD')
        axes.set_aspect( 1 )
        p.annotate(SV[i], (S[i][0],S[i][1]))
    
    #Basis for 1-chains
    C1=[]
        
    for i in range(len(S)):
        for j in range (i+1,len(S)):
            if np.linalg.norm(S[i]-S[j])<=2*r: #distance requirement
                C1=np.append(C1, [np.concatenate([S[i],S[j]])]);
                p.plot([S[i][0],S[j][0]],[S[i][1],S[j][1]],'-b')
    del_1=[]
    
    #Boundary Matrix
    if not(len(C1)==0):            
        C1=np.split(C1,len(C1)/4)
        del_1=np.zeros((len(S),len(C1)));
        for i in range(len(S)):
            for j in range(len(C1)):
                if all(S[i]== C1[j][0:2]) or all(S[i]==C1[j][2:4]):
                    del_1[i][j]=1;
        # Changing inputs of del_1 to integer type            
        del_1 = del_1.astype(int)
        # Converting into a matrix over Z_2
        GF = galois.GF(2)           
        del_1 = GF(del_1)
        #Computing rank $ kernel
        B0 = np.linalg.matrix_rank(del_1)
        Z1 = len(del_1[0])-B0
        H_0 = len(S)-B0
    
    
    
    #Similarly for 2 chains, this is where the problem is, there is no precise
    #ordering of 2 simplices
    C2=[]
    
    for i in range(len(S)):
        for j in range (i+1,len(S)):
            for k in range(j+1,len(S)):
                if np.linalg.norm(S[i]-S[j])<=2*r and np.linalg.norm(S[j]-S[k])<=2*r and np.linalg.norm(S[i]-S[k])<=2*r:
                    C2=np.append(C2, [np.concatenate([S[i],S[j],S[k]])])
                    poly = Polygon([S[i], S[j], S[k]],facecolor='pink')
                    axes.add_patch(poly)
    
    #Higher dimention boundary matrix
    del_2=[];
    if not(len(C2)==0):
        C2=np.split(C2,len(C2)/6)
        del_2=np.zeros((len(C1),len(C2)));
        for i in range(len(C1)):
            for j in range(len(C2)):
                if all(C1[i]== C2[j][0:4]) or all(C1[i]==C2[j][2:6]) or all(C1[i]==C2[j][[0,1,4,5]]):
                    del_2[i][j]=1;
        del_2 = del_2.astype(int)
        del_2 = GF(del_2)
        B1 = np.linalg.matrix_rank(del_2)
        Z2 = len(del_2[0])-B1
        H_1 = Z1-B1
        
        
        
    # C3=[]
    
    # for i in range(len(S)):
    #     for j in range (i+1,len(S)):
    #         for k in range(j+1,len(S)):
    #             for l in range(k+1,len(S)):
    #                 if np.linalg.norm(S[i]-S[j])<=2*r and np.linalg.norm(S[j]-S[k])<=2*r \
    #                 and np.linalg.norm(S[i]-S[k])<=2*r and np.linalg.norm(S[i]-S[l])<=2*r \
    #                 and np.linalg.norm(S[j]-S[l])<=2*r and np.linalg.norm(S[k]-S[l])<=2*r:
    #                     C3=np.append(C3, [np.concatenate([S[i],S[j],S[k],S[l]])])
    #                     poly = Polygon([S[i], S[j], S[k],S[l]],facecolor='purple')
    #                     axes.add_patch(poly)
    # if not(len(C3)==0):
    #     C3=np.split(C3,len(C3)/8)
    #     del_3=np.zeros((len(C2),len(C3)));
    #     for i in range(len(C2)):
    #         for j in range(len(C3)):
    #             if all(C2[i] == C3[j][0:6]) or all(C2[i]==C3[j][2:8])\
    #                 or all(C2[i] == C3[j][[0,1,2,3,6,7]]) or all(C2[i] == C3[j][[0,1,4,5,6,7]]):
    #                 del_3[i][j]=1;
                    
    #     del_3 = del_3.astype(int)
    #     del_3 = GF(del_3)
    #     B2 = np.linalg.matrix_rank(del_3)
    #     Z3 = len(del_3[0])-B2
    #     H_2= Z2-B2
    
    
    
    #Calculating the direct sum of boundary matrices with the filtration 
    #imposed by appearence of simpleces
    A=[]
    B=[]
    
    if len(del_1)==0:
        m1=0
        n1=0
    else:
        [m1,n1]=np.shape(del_1)
    if len(del_2)==0:
        m2=0
        n2=0
    else:
        [m2,n2]=np.shape(del_2)
        
    D_cols1 = np.nonzero(np.any(D[0:len(S),:] != 0, axis=0))[0] #nonzero columns of D
    D_cols2 = np.nonzero(np.any(D[len(S):len(S)+n1,:] != 0, axis=0))[0]
    if len(D_cols1)==0:
        D_cols1=[0]
    if len(D_cols2)==0:
        D_cols2=[0]
    if not(m1==0):
        A=Compare(del_1,D[0:m1,D_cols1])#"new" columns, not present in the D matrix
        B=Compare(del_2,D[len(S):len(S)+m2,D_cols2])
        if (len(A)==0):
            m_a=0
            n_a=0
        else:
            [m_a,n_a]=np.shape(A)
        if len(B)==0:
            m_b=0
            n_b=0
        else:
            [m_b,n_b]=np.shape(B)
        m=max(D_cols1[-1],D_cols2[-1])
        if m==0:
            m=len(S)
        else:
            m=m+1
        if not(m_a==0):
            D[0:m_a,m:m+n_a]=A
        if not(m_b==0):
            m_new=max(np.nonzero(np.any(D != 0, axis=0))[0])+1 #first nonzero column
            #D[m1:m1+m_b,m_new:m_new+n_b]=B           
            
    
    # May disable(uncomment) to see the output at every step
    #input("Press Enter to continue...")        
    
    
    #Plots the title
    p.rc('text', usetex=True)
    title="$H_0 \cong Z_2^ {H_0}, H_1 \cong Z_2^{H_1}, H_2 \cong Z_2^{H_2} $".format(H_0=H_0,H_1=H_1,H_2=H_2)
    p.title(title)        
    
# Reduce the obtained matrix, plot the persistance diagram    
[D_reduced,i,j]=Reduce(D)
f,ax=p.subplots(figsize=(10, 10))
p.plot(i,j,'rD')    
ax.axline((1,1), slope=1,linestyle='--')

  

    



    