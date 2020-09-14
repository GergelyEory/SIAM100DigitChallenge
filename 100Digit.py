from __future__ import absolute_import
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator, cg, bicgstab, gmres, spsolve
import pyopencl as cl
import time

FINALSOL = 0.424011387033

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

platform = cl.get_platforms()[0]    # Select the first platform [0]
device = platform.get_devices()[0]  # Select the first device on this platform [0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)  # Create (non-custom) command queue
mf = cl.mem_flags

### Create a kernel which carries out the CSR multiplication
### Build kernel in the init method NOT in matvec
matvec_kernel ="""
        __kernel void matvec( const int N,
                          __global const float * data,
                          __global const int * index,
                          __global const int * indptr,
                          __global const float * vec,
                          __global float * res )
        {
        
        int gid = get_global_id(0);
        
        float acc = 0.0f;
        
        for (int k = indptr[gid]; k < indptr[gid + 1]; k++)
            {
            acc += data[k] * vec[index[k]];
            }
        
        res[gid] = acc;
        } """

matvec_simd_kernel ="""
        __kernel void mult(__global const float * data,
                            __global const int * index,
                            __global const float * vec,
                            __global float * resTmp )
        {
        
            int gid = get_global_id(0);
            
            double4 dataSlice;
            double4 vecSlice;
            double4 resSlice;

            dataSlice = (double4)(data[gid*4+0],
                                data[gid*4+1],
                                data[gid*4+2],
                                data[gid*4+3]);

            vecSlice = (double4)(vec[index[gid*4+0]],
                                vec[index[gid*4+1]],
                                vec[index[gid*4+2]],
                                vec[index[gid*4+3]]);

            resSlice = dataSlice * vecSlice;

            resTmp[gid*4+0] = resSlice.s0;
            resTmp[gid*4+1] = resSlice.s1;
            resTmp[gid*4+2] = resSlice.s2;
            resTmp[gid*4+3] = resSlice.s3;
        
        }

        __kernel void add(__global const int* indptr,
                            __global const float* resTmp,
                            __global float* res) {

            int gid = get_global_id(0);
            float acc = 0.0f;

            for(uint j = indptr[gid]; j < indptr[gid+1]; j++){
                acc += resTmp[j];
            }

            res[gid] = acc;
        }
        """


prgCL = cl.Program(ctx,matvec_kernel).build()
prgSIMD = cl.Program(ctx,matvec_simd_kernel).build()

class Sparce(LinearOperator):
    '''
    Class derived from LinearOperator, initialised with a sparce matrix in
    the CSR format. It overrides the _matvec function to implement 
    algorithm for CSR multiplication, accelerated by OpenCL.
    '''

    def __init__(self, inMat):
        '''
        Initiaises Sparce matrix with input type scipy.sparse.csr_matrix.
        shape and dtype fields are required by LinearOperator. 
        '''
        self.shape = inMat.get_shape()
        self.dtype = 'float32'
        self.data = inMat.data
        self.indices = inMat.indices
        self.indptr = inMat.indptr      
        
    def _matvec(self, v):
        '''
        OpenCL implementation of product of sparce matrix object with vector v.
        Returns A*v. Dimensions of v should match first dimension of the sparce matrix
        '''
          
        vec =  v.astype(np.float32)
        data = self.data.astype(np.float32)    # Data vector
        ind = self.indices.astype(np.int32)   # Vector of column indices
        indptr  = self.indptr.astype(np.int32)    # Index pointer of column


        #Create buffers
        data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        ind_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ind)
        indptr_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indptr)
        vec_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec)
        res_buff = cl.Buffer(ctx,mf.WRITE_ONLY,vec.nbytes)

        N = np.float32(len(indptr) - 1)    # Length of the output vector   

        res_np = np.empty_like(vec).astype(np.float32)
        prgCL.matvec(queue,vec.shape,None,np.int32(N),
            data_buff,ind_buff,indptr_buff,vec_buff,res_buff).wait()
        cl.enqueue_copy(queue, res_np, res_buff).wait()

        return res_np

    def _matvecsimd(self, v):
        '''
        Vectorised sparse matrix product.
        Breaks problem down into double4 vector types, carries out multiplications
        in vector form, then accumulates the values corresponding to each matrix
        row, therefore to each result element.
        '''
        
        # 4 compute units available on my CPU (2 cores + HT)
        # AVX2 -> 256 bit AVX --> double4

        vec =  v.astype(np.float32)
        data = self.data.astype(np.float32)    # Data vector
        ind = self.indices.astype(np.int32)   # Vector of column indices
        indptr  = self.indptr.astype(np.int32)    # Index pointer of column

        ### Pad data and indices to be divisible by 4
        remainder = np.uint32(np.size(data)%4)
        if remainder != 0:
            data = np.concatenate([data, np.zeros(4 - remainder)]).astype(np.float32)
            ind = np.concatenate([ind, np.zeros(4 - remainder)]).astype(np.uint32)

        iteration_len = np.size(data)

        #Create buffers
        data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        ind_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ind)
        indptr_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indptr)
        vec_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec)
        res_buff = cl.Buffer(ctx,mf.READ_WRITE,vec.nbytes)
        res_tmp_buff = cl.Buffer(ctx, mf.READ_WRITE, data.nbytes)


        prgSIMD.mult(queue, (int(iteration_len/4), ), None,
                    data_buff, ind_buff, vec_buff, res_tmp_buff)

        prgSIMD.add(queue, (np.size(indptr) - 1,), None, 
                    indptr_buff, res_tmp_buff, res_buff)

        res = np.empty_like(v, dtype = np.float32)
        cl.enqueue_copy(queue, res, res_buff).wait()

        # Trim result so only the original size vector is returned without padding.
        return res[0:np.size(v)]

@njit
def forwardMatrix(N, dx, dt):
    
    # Initialising the arrays for the COO matrix.
    row = []
    col = []
    data = []   
    
    for i in np.arange(N*N):
        x = i%N
        y = i//N
        
        if (x == 0) or (x == N-1) or (y == 0) or (y == N-1): 
            row.append(i)
            col.append(i)
            data.append(1)
        
        # All other are interior values
        else:
            # Centre
            row.append(i)
            col.append(i)
            data.append(-4*dt/(dx*dx) + 1)
            
            # Left
            row.append(i)
            col.append(i-1)
            data.append(dt/(dx*dx))
            
            # Right
            row.append(i)
            col.append(i+1)
            data.append(dt/(dx*dx))
            
            # Top
            row.append(i)
            col.append(i-N)
            data.append(dt/(dx*dx))
            
            # Bottom
            row.append(i)
            col.append(i+N)
            data.append(dt/(dx*dx))
            
    return (np.float32(data), (row, col))

forwardMatrix(5, 0.2, 0.01)

@njit
def backwardMatrix(N, dx, dt):
    
    # Initialising the arrays for the COO matrix.
    row = []
    col = []
    data = []   
    
    for i in np.arange(N*N):
        x = i%N
        y = i//N
        
        if (x == 0) or (x == N-1) or (y == 0) or (y == N-1):
            row.append(i)
            col.append(i)
            data.append(1)
        
        # All other are interior values
        else:
            # Centre
            row.append(i)
            col.append(i)
            data.append(4*dt/(dx*dx) + 1)
            
            # Left
            row.append(i)
            col.append(i-1)
            data.append(-dt/(dx*dx))
            
            # Right
            row.append(i)
            col.append(i+1)
            data.append(-dt/(dx*dx))
            
            # Top
            row.append(i)
            col.append(i-N)
            data.append(-dt/(dx*dx))
            
            # Bottom
            row.append(i)
            col.append(i+N)
            data.append(-dt/(dx*dx))
            
    return (np.float32(data), (row, col))

def forwardEuler(N = 51, dt = 1e-4):

    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2

    A = sparse.coo_matrix(forwardMatrix(N, dx, dt), shape=(N*N, N*N))
    Amat = sparse.csr_matrix(A)

    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()

    flatGrid = np.zeros(N*N)
    flatGrid[:N] = 5
    
    while (flatGrid[centre] < 1):
        timeElapsed += dt
        iterations += 1
        
        flatGrid = Amat.dot(flatGrid)

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

def forwardEulerLin(N = 51, dt = 1e-4):

    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2

    A = sparse.coo_matrix(forwardMatrix(N, dx, dt), shape=(N*N, N*N))
    Amat = Sparce(sparse.csr_matrix(A))

    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()

    flatGrid = np.zeros(N*N)
    flatGrid[:N] = 5
    
    while (flatGrid[centre] < 1):
        timeElapsed += dt
        iterations += 1
        
        flatGrid = Amat._matvec(flatGrid)

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

def forwardEulerSimd(N=51, dt = 1e-4):

    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2

    A = sparse.coo_matrix(forwardMatrix(N, dx, dt), shape=(N*N, N*N))
    Amat = Sparce(sparse.csr_matrix(A))

    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()

    flatGrid = np.zeros(N*N)
    flatGrid[:N] = 5
    
    while (flatGrid[centre] < 1):
        timeElapsed += dt
        iterations += 1
        
        flatGrid = Amat._matvecsimd(flatGrid)

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

def backwardEulerCG(N = 21, dt = 1e-5):
    
    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2
    
    A = sparse.coo_matrix(backwardMatrix(N, dx, dt), shape=(N*N, N*N))
    Amat = sparse.csr_matrix(A)
    
    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()
    
    centre = (N*N)//2
    
    time = 0
    
    iterations = 0
    
    while flatGrid[centre] < 1:
        
        flatGrid, _ = cg(Amat, flatGrid)

        timeElapsed += dt
        iterations += 1

    return (timeElapsed, timeElapsed - FINALSOL, iterations)


def backwardEuler(N = 21, dt = 1e-5):
    
    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2
    
    A = sparse.coo_matrix(backwardMatrix(N, dx, dt), shape=(N*N, N*N))
    Amat = sparse.csr_matrix(A)
    
    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()
    
    centre = (N*N)//2
    
    time = 0
    
    iterations = 0
    
    while flatGrid[centre] < 1:
        
        flatGrid = spsolve(Amat, flatGrid)

        timeElapsed += dt
        iterations += 1

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

def crankNicholsonM(N, dx, dt):
    # Backward matrix with factor of 2 on the diagonal element
    # Initialising the arrays for the COO matrix.
    row = []
    col = []
    data = []   
    
    for i in np.arange(N*N):
        x = i%N
        y = i//N
        
        if (x == 0) or (x == N-1) or (y == 0) or (y == N-1): 
            row.append(i)
            col.append(i)
            data.append(1)
        
        # All other are interior values
        else:
            # Centre
            row.append(i)
            col.append(i)
            data.append(4*dt/(dx*dx) + 2)
            
            # Left
            row.append(i)
            col.append(i-1)
            data.append(-dt/(dx*dx))
            
            # Right
            row.append(i)
            col.append(i+1)
            data.append(-dt/(dx*dx))
            
            # Top
            row.append(i)
            col.append(i-N)
            data.append(-dt/(dx*dx))
            
            # Bottom
            row.append(i)
            col.append(i+N)
            data.append(-dt/(dx*dx))
            
    return (np.float32(data), (row, col))

def crankNicholsonN(N, dx, dt):
    # Forward matrix with factor of 2 on the diagonal element
    # Initialising the arrays for the COO matrix.
    row = []
    col = []
    data = []   
    
    for i in np.arange(N*N):
        x = i%N
        y = i//N
        #print(dt/(dx*dx))
        
        # Boundary values, in order:
        # Left Most
        # Right Most
        # Top Most
        # Bottom and Right
        if x == 0 \
        or x == N-1 \
        or y == 0 \
        or y == N-1: 
            row.append(i)
            col.append(i)
            data.append(1)
        
        # All other are interior values
        else:
            # Centre
            row.append(i)
            col.append(i)
            data.append(-4*dt/(dx*dx) + 2)
            
            # Left
            row.append(i)
            col.append(i-1)
            data.append(dt/(dx*dx))
            
            # Right
            row.append(i)
            col.append(i+1)
            data.append(dt/(dx*dx))
            
            # Top
            row.append(i)
            col.append(i-N)
            data.append(dt/(dx*dx))
            
            # Bottom
            row.append(i)
            col.append(i+N)
            data.append(dt/(dx*dx))
            
    return (np.float32(data), (row, col))

def crankNicholson(N = 51, dt = 1e-4):

    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2
    
    A = sparse.coo_matrix(crankNicholsonM(N, dx, dt), shape=(N*N, N*N))

    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()

    Nmat = sparse.coo_matrix(crankNicholsonN(N, dx, dt), shape=(N*N, N*N))
    

    while flatGrid[centre] < 1:

        b = Nmat @ flatGrid
        grid_new = spsolve(A.tocsc(), b)
        flatGrid = np.copy(grid_new)

        timeElapsed += dt
        iterations += 1

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

def crankNicholsonInv(N = 21, dt = 1e-5):

    dx = np.float32(2/(N-1))
    timeElapsed = 0
    iterations = 0
    centre = (N*N)//2
    
    grid = np.zeros((N,N), dtype = np.float32)
    grid[:,N-1] = np.ones(N)*5
    flatGrid = grid.flatten()

    Nmat = sparse.coo_matrix(crankNicholsonN(N, dx, dt), shape=(N*N, N*N))
    Mmat = sparse.coo_matrix(crankNicholsonM(N, dx, dt), shape=(N*N, N*N))

    Minv = sparse.linalg.inv(Mmat.tocsr())
    A = Minv @ Nmat

    while flatGrid[centre] < 1:

        flatGrid = spsolve(A.tocsr(), flatGrid)

        timeElapsed += dt
        iterations += 1

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

<<<<<<< HEAD
def cheb(M):
    '''
    Creates Differntial Matrix D and Chebyshev function space discretisation.
    '''
    M = M-1
    if M==0:
        D = np.array([[0.]]); x = np.array([1.])
    else:
        n = np.arange(0,M + 1)
        x = np.cos(np.pi* n / M).reshape(M + 1, 1) 
        c = (np.hstack(( [2.], np.ones(M - 1), [2.]))*(-1)**n).reshape(M + 1,1)
        X = np.tile(x,(1,M + 1))
        dX = X - X.T
        D = np.dot(c,1./c.T)/(dX + np.eye(M + 1))
        D -= np.diag( np.sum( D.T, axis=0))
    return D, x.reshape(M + 1)

def spectral_explicit_solve(M, dt):
    '''
    Solves the given heat spread problem using spectral differentiation
    '''
    
    h = np.float64(2/(M-1))
    
    D, x = cheb(M)
    D2 = D @ D
    I = np.eye(M)
    A = (np.kron(D2, I) + np.kron(I, D2))
    I2 = np.eye(M**2)
    
    A = sparse.csr_matrix(A)
    I2 = sparse.csr_matrix(I2)

    u = np.zeros(M**2)
    u[:M] = 5
    
    midpoint = (M*M)//2
    timeElapsed = 0
    iterations = 0
    
    while u[midpoint] < 1:
        
        u_old = np.copy(u)        
        u = (I2 + dt*A).dot(u_old)
        
        u[:M] = 5
        u[M:M*(M-1):M] = 0
        u[M+M-1:M*(M-1):M] = 0
        u[M*(M-1):] = 0

        timeElapsed += dt
        iterations += 1

    return (timeElapsed, timeElapsed - FINALSOL, iterations)

'''
with Timer() as t0:
    oof = forwardEulerSimd(21, 1e-5)

print(oof)
print('SIMD - Time taken to run: {0}'.format(t0.interval))


with Timer() as t1:
    foo = forwardEulerLin(21, 1e-5)

print(foo)
print('OpenCL - Time taken to run: {0}'.format(t1.interval))


with Timer() as t2:
    bar = forwardEuler(21, 1e-5)

print(bar)
print('Python - Time taken to run: {0}'.format(t2.interval))

with Timer() as t3:
    tar = backwardEuler(21, 1e-5)

print(tar)
print('Backward - Time taken to run: {0}'.format(t3.interval))

with Timer() as t4:
    oof = backwardEulerCG(21, 1e-5)

print(oof)
print('CG - Time taken to run: {0}'.format(t3.interval))

with Timer() as t5:
    oof = crankNicholson(21, 1e-5)

print(oof)
print('CN - Time taken to run: {0}'.format(t3.interval))
'''


=======
>>>>>>> origin/master
