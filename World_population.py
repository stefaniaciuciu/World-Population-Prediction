from numpy import math
import matplotlib.pyplot as plt
import numpy as np
import math

m = 8 # nr de linii
n1 = 2 # nr de coloane
A = np.array([[1, 1950], [1, 1955], [1, 1960], [1, 1965], [1, 1970], [1, 1975], [1, 1980], [1, 1985]])
A1 = np.array([[1, 1950], [1, 1955], [1, 1960], [1, 1965,], [1, 1970], [1, 1975], [1, 1980], [1, 1985]])
n2 = 3 # nr de coloane
A2 = np.array([[1, 1950, 1950**2], [1, 1955, 1955**2], [1, 1960, 1960**2], [1, 1965, 1965**2], [1, 1970, 1970**2], [1, 1975, 1975**2], [1, 1980, 1980**2], [1, 1985, 1985**2]])
n3 = 4 # nr de coloane
A3 = np.array([[1, 1950, 1950**2, 1950**3], [1, 1955, 1955**2, 1950**3], [1, 1960, 1960**2, 1960**3], [1, 1965, 1965**2, 1965**3], [1, 1970, 1970**2, 1970**3], [1, 1975, 1975**2, 1975**3], [1, 1980, 1980**2, 1980**3], [1, 1985, 1985**2, 1985**3]])
A1 = A1.astype(float)
A2 = A2.astype(float)
A3 = A3.astype(float)
b1 = np.array([[2.53], [2.77], [3.05], [3.36], [3.72], [4.01], [4.47], [4.87]])
b1 = b1.astype(float)
b2 = np.array([[2.53], [2.77], [3.05], [3.36], [3.72], [4.01], [4.47], [4.87]])
b2 = b2.astype(float)
b3 = np.array([[2.53], [2.77], [3.05], [3.36], [3.72], [4.01], [4.47], [4.87]])
b3 = b3.astype(float)

#-----------------------------------------------

print('------------A-------------\n', A)
print('------------b-------------\n', b1)

sol = np.linalg.pinv(np.copy(A1))@np.copy(b1)

#Triangularizarea ortogonala cu reflectori
def  TORT (A,n):
    m,n  = np.shape(A)
    p = min(m-1,n)
    U = np.zeros((m,n))
    R = np.zeros((m,n))
    tau=0
    sigma=0
    beta = np.zeros((p, 1))
    for k in range(p):
       sum1=0
       for i in range(k,m):
         sum1=sum1+(A[i][k]*A[i][k])
       sigma=np.sign(A[k][k])*np.sqrt(sum1)
       if(sigma == 0):
          beta[k]=0
       else:
         U[k][k]=A[k][k]+sigma
         for i in range(k+1,m):
               U[i][k]=A[i][k]
         beta[k]=U[k][k]*sigma
         R[k][k]=-sigma
         A[k][k]=R[k][k]
         for i in range(k+1,m):
             A[i][k]=0
         for j in range(k+1,n):
             sum2=0
             for z in range(k,m):
               sum2=sum2+(U[z][k]*A[z][j])
             tau=sum2/beta[k]
             for i in range(k,m):
               A[i][j]=A[i][j]-tau*U[i][k]
               if(i<j):
                 R[i][j]=A[i][j] 
  
    return  U, R, beta

# Algoritmul UTRIS
def Utris(U, b, n):
    n = len(U)
    x = np.zeros((n,1))
    for i in range(n-1,-1, -1):
        s = b[i]
        for j in range(i+1,n):
            s = s -U[i][j]*x[j]
        x[i] = s/U[i][i]
    return x



print('-----------Grad 1------------\n')



#1. Triangularizarea ortogonala a lui A
U,R, beta  = TORT(A1,n1)


#2. Aplicarea Reflectorilor asupra lui b
for k in range(n1):
      t=0
      for i in range(k,m):
        t = t+U[i][k]*b1[i]
      tau = t/beta[k]
      for i in range(k,m):
          b1[i] = b1[i] - tau*U[i][k]

#3. Calcularea solutiei CMMP
x = Utris(R[0:n1,:],b1[0:n1,0], n1)

# Verificare
#print('-----------x------------\n',x)
#print('----------sol------------\n', sol)

#print('-----------Polinom------------')
print('pol =',x[1],'*t +',x[0],'\n')

#4. Predictia populatiei planetei
an_dorit = np.array([[1986], [1987], [1988], [1989], [1990], [1991], [1992], [1993], [1994], [1995], [1996], [1997], [1998], [1999], [2000], [2001], [2002], [2003], [2004], [2005], [2006], [2007], [2008], [2009], [2010], [2011], [2012], [2013], [2014], [2015], [2016], [2017], [2018], [2019], [2020]])
valoare1 = np.zeros((35,1))
for i in range(35):
    for k in range(n1):
        valoare1[i] = valoare1[i] + x[k]*(an_dorit[i]**k)




print('-----------Grad 2------------\n')


sol = np.linalg.pinv(np.copy(A2))@np.copy(b2)

#1. Triangularizarea ortogonala a lui A
U,R, beta  = TORT(A2,n2)


#2. Aplicarea Reflectorilor asupra lui b
for k in range(n2):
      t=0
      for i in range(k,m):
        t = t+U[i][k]*b2[i]
      tau = t/beta[k]
      for i in range(k,m):
          b2[i] = b2[i] - tau*U[i][k]

#3. Calcularea solutiei CMMP
x = Utris(R[0:n2,:],b2[0:n2,0],n2)

# Verificare
#print('-----------x------------\n',x)
#print('----------sol------------\n', sol)

#print('-----------Polinom------------')
print('pol =',x[2],'*t^2 +',x[1],'*t +',x[0],'\n')

#4. Predictia populatiei planetei
valoare2 = np.zeros((35,1))
for i in range(35):
    for k in range(n2):
        valoare2[i] = valoare2[i] + x[k]*(an_dorit[i]**k)



print('-----------Grad 3------------\n')



sol = np.linalg.pinv(np.copy(A3))@np.copy(b3)

#1. Triangularizarea ortogonala a lui A
U,R, beta  = TORT(A3,n3)

#print('----------U------------\n',U)
#print('-----------R-----------\n',R)

#2. Aplicarea Reflectorilor asupra lui b
for k in range(n3):
      t=0
      for i in range(k,m):
        t = t+U[i][k]*b3[i]
      tau = t/beta[k]
      for i in range(k,m):
          b3[i] = b3[i] - tau*U[i][k]

#3. Calcularea solutiei CMMP
x = Utris(R[0:n3,:],b3[0:n3,0],n3)

# Verificare
#print('-----------x------------\n',x)
#print('----------sol------------\n', sol)

#print('-----------Polinom------------')
print('pol =',x[3],'*t^3 +',x[2],'*t^2 +',x[1],'*t +',x[0],'\n')


#4. Predictia populatiei planetei
valoare3 = np.zeros((35,1))
for i in range(35):
    for k in range(n3):
        valoare3[i] = valoare3[i] + x[k]*(an_dorit[i]**k)

anul_dorit = 1990
i = anul_dorit - 1986
print('Predictie pentru anul',anul_dorit,'de grad 1:',valoare1[i])
print('Predictie pentru anul',anul_dorit,'de grad 2:',valoare2[i])
print('Predictie pentru anul',anul_dorit,'de grad 3:',valoare3[i],'\n')


# Valori reale
valori_reale = np.array([[4.926], [5.014], [5.102], [5.191], [5.281], [5.369], [5.453], [5.538], [5.623], [5.708], [5.79], [5.873], [5.955], [6.035], [6.115], [6.194], [6.274], [6.353], [6.432], [6.513], [6.594], [6.675], [6.758], [6.841], [6.923], [7.004], [7.07], [7.171], [7.256], [7.341], [7.426], [7.511], [7.594], [7.674], [7.838]]) 


# Calcularea erorilor si eroarea medie
eroare1 = np.zeros((35,1))
err1 = 0
eroare2 = np.zeros((35,1))
err2 = 0
eroare3 = np.zeros((35,1))
err3 = 0
for i in range(35):
    eroare1[i] = abs(valori_reale[i] - valoare1[i])
    err1 = err1 + eroare1[i]
    eroare2[i] = abs(valori_reale[i] - valoare2[i])
    err2 = err2 + eroare2[i]
    eroare3[i] = abs(valori_reale[i] - valoare3[i])
    err3 = err3 + eroare3[i]
err1 = err1/35
err2 = err2/35
err3 = err3/35
print('Valorile medii ale erorilor sunt:')
print(err1,' ',err2,' ',err3,' ')
    

#Plotarea in grafic
plt.plot(an_dorit, valoare1, 'r')  
plt.plot(an_dorit, valoare2, 'b') 
plt.plot(an_dorit, valoare3, 'g')
plt.plot(an_dorit, valori_reale, 'y')
plt.title('Predictia populatiei in functie de gradul polinomului')
plt.xlabel('An')
plt.ylabel('Populatie')
plt.show()
