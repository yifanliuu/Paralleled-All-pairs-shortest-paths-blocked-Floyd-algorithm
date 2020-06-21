#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define BLOCK_SIZE 8
#define TILE_SIZE 32
#define MATRIX_SIZE 8096

#define CUDA_SAFE_CALL(call)                                              \
    {                                                                     \
        cudaError err = call;                                             \
        if (cudaSuccess != err)                                           \
        {                                                                 \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

double timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void showMatrix(unsigned int *mat, const size_t N)
{
    for (int i = 0; i < N * N; i++)
    {
        printf("mat[%d] = %d, ", i, mat[i]);
    }
    printf("done\n");
}


void Floyd_sequential(unsigned int *mat, const size_t N)
{
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                int i0 = i * N + j;
                int i1 = i * N + k;
                int i2 = k * N + j;
                mat[i0] = (mat[i0] < mat[i1] + mat[i2]) ? mat[i0] : (mat[i1] + mat[i2]);
            }
}

__global__ void gpu_primary(unsigned int *result, int N, int start)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int row = y + start;
    int col = x + start;

    __shared__ unsigned int s_mat[TILE_SIZE][TILE_SIZE];
    //printf("x: %d, y: %d\n", x, y);
    int i0 = row * N + col;
    //printf("i0: %d\n", i0);
    s_mat[y][x] = result[i0];
    __syncthreads();
    //printf("s_mat[%d][%d] = %d\n", y,x, s_mat[y][x]);

    for (int k = 0; k < TILE_SIZE; k++)
    {

        s_mat[y][x] = (s_mat[y][x] < s_mat[y][k] + s_mat[k][x]) ? s_mat[y][x] : (s_mat[y][k] + s_mat[k][x]);
        /*
        if(k == 1){
            printf("s_mat[%d][%d]=%d\n",row,col, s_mat[y][x]);
        }
        */
    }
    if (row < N && col < N){
        result[i0] = s_mat[y][x];
    }
}

__global__ void gpu_phase2(unsigned int *result, int N, int start, int k)
{
    int skip = blockIdx.x < k ? (blockIdx.x - k): (blockIdx.x - k + 1);
    int row_start = 0;
    int col_start = 0;
    //printf("skip: %d\n",skip);
    if(blockIdx.y == 0){
        row_start = start;
        col_start = skip * TILE_SIZE + start;
    }
    else{
        row_start = skip * TILE_SIZE + start;
        col_start = start;
    }
    
    //printf("row_start: %d, col_start: %d\n", row_start, col_start);
    int row = threadIdx.y + row_start;
    int col = threadIdx.x + col_start;

    __shared__ int unsigned s_mat[TILE_SIZE][TILE_SIZE];
    __shared__ int unsigned sp_mat[TILE_SIZE][TILE_SIZE];
    //printf("row: %d, col: %d\n", row, col);
    int i0 = row * N + col;
    //printf("i0: %d\n", i0);
    s_mat[threadIdx.y][threadIdx.x] = result[i0];
    sp_mat[threadIdx.y][threadIdx.x] = result[(start + threadIdx.y) * N + (start + threadIdx.x)];
    __syncthreads();

    for (int idx = 0; idx < TILE_SIZE; idx++)
    {
        if(blockIdx.y == 1){
            
            s_mat[threadIdx.y][threadIdx.x] = (s_mat[threadIdx.y][threadIdx.x] < sp_mat[idx][threadIdx.x] + s_mat[threadIdx.y][idx]) ? 
                s_mat[threadIdx.y][threadIdx.x] : (sp_mat[idx][threadIdx.x] + s_mat[threadIdx.y][idx]);
            //printf("block%d: sp_mat[%d][%d] = %d\n", blockIdx.x, idx, threadIdx.x, sp_mat[idx][threadIdx.x]);
            /*
            printf("result[%d][%d]=%d, s_mat[%d][%d]=%d, result[%d][%d] = %d + s_mat[%d][%d]=%d\n", row, col, result[i0],
            threadIdx.y, threadIdx.x, s_mat[threadIdx.y][threadIdx.x],
                idx + start, col, result[(idx + start) * N + col], row, idx, s_mat[threadIdx.y][idx]);
            */
        }
        else {
            s_mat[threadIdx.y][threadIdx.x] = (s_mat[threadIdx.y][threadIdx.x] < s_mat[idx][threadIdx.x] + sp_mat[threadIdx.y][idx]) ? 
                s_mat[threadIdx.y][threadIdx.x] : (s_mat[idx][threadIdx.x] + sp_mat[threadIdx.y][idx]);
        }

    }
    __syncthreads();

    if(row < N && col <N){
        result[i0] = s_mat[threadIdx.y][threadIdx.x];
        //printf("idx=%d, mat[%d][%d]=%d\n ",idx, row, col, result[i0]);    
    }            
}

 
__global__ void gpu_phase3(unsigned int *result, int N, int start, int k)
{
    int skip_x = blockIdx.x < k ? (blockIdx.x - k): (blockIdx.x - k + 1);
    int skip_y = blockIdx.y < k ? (blockIdx.y - k): (blockIdx.y - k + 1);
    int row_start = skip_y * TILE_SIZE + start;
    int col_start = skip_x * TILE_SIZE + start;
    int row = threadIdx.y + row_start;
    int col = threadIdx.x + col_start;

    __shared__ unsigned int s_mat[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int sh_mat[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int sv_mat[TILE_SIZE][TILE_SIZE];
    //printf("row: %d, col: %d\n", row, col);
    int i0 = row * N + col;
    s_mat[threadIdx.y][threadIdx.x] = result[i0];
    sv_mat[threadIdx.y][threadIdx.x] = result[row * N + (start + threadIdx.x)];
    sh_mat[threadIdx.y][threadIdx.x] = result[(start + threadIdx.y) * N + col];
    __syncthreads();

    for (int idx = 0; idx < TILE_SIZE; idx++)
    {
        //printf("result[%d][%d]: %d, result[%d][%d]: %d, result[%d][%d]: %d\n", x, y,result[i0], x, k, result[i1],k, y, result[i2]);
        /*
        printf("sv_mat[%d][%d]=%d, sh_mat[%d][%d]=%d while computing s_mat[%d][%d]=%d\n", 
            row, start + threadIdx.x, sv_mat[threadIdx.y][idx], start + threadIdx.y, col,
            sh_mat[idx][threadIdx.x], row, col,  s_mat[threadIdx.y][threadIdx.x]);
        */
        s_mat[threadIdx.y][threadIdx.x] = (s_mat[threadIdx.y][threadIdx.x] < sv_mat[threadIdx.y][idx] + sh_mat[idx][threadIdx.x]) ? 
            s_mat[threadIdx.y][threadIdx.x] : (sv_mat[threadIdx.y][idx] + sh_mat[idx][threadIdx.x]);
        
    }

    __syncthreads();
    if(row < N && col < N){
        result[i0] = s_mat[threadIdx.y][threadIdx.x];
        //printf("phase3: idx=%d, mat[%d][%d]=%d\n ",idx, row, col, result[i0]); 
    }
}

void GenMatrix(unsigned int *mat, const size_t N)
{
    /*
    mat[0] = 0;
    mat[1] = 17;
    mat[2] = 21;
    mat[3] = 12;
    mat[4] = 5;
    mat[5] = INT_MAX / 2;
    mat[6] = 6;
    mat[7] = 0;
    mat[8] = INT_MAX / 2;
    mat[9] = INT_MAX / 2;
    mat[10] = INT_MAX / 2;
    mat[11] = 3;
    mat[12] = 10;
    mat[13] = INT_MAX / 2;
    mat[14] = 0;
    mat[15] = 14;
    mat[16] = INT_MAX / 2;
    mat[17] = INT_MAX / 2;
    mat[18] = INT_MAX / 2;
    mat[19] = 11;
    mat[20] = INT_MAX / 2;
    mat[21] = 0;
    mat[22] = INT_MAX / 2;
    mat[23] = 4;
    mat[24] = INT_MAX / 2;
    mat[25] = 4;
    mat[26] = 13;
    mat[27] = INT_MAX / 2;
    mat[28] = 0;
    mat[29] = INT_MAX / 2;
    mat[30] = 9;
    mat[31] = INT_MAX / 2;
    mat[32] = INT_MAX / 2;
    mat[33] = INT_MAX / 2;
    mat[34] = 20;
    mat[35] = 0;
    */
    
    for (int i = 0; i < N * N; i++)
    {
        mat[i] = rand() % 32;
        if (mat[i] == 0)
        {
            mat[i] = INT_MAX / 2;
        }
        if (i % N == i / N)
        {
            mat[i] = 0;
        }
    }
    
}


bool CmpArray(const unsigned int *l, const unsigned int *r, const size_t eleNum)
{
    for (int i = 0; i < eleNum; i++)
        if (l[i] != r[i])
        {
            printf("ERROR: l[%d] = %d, r[%d] = %d\n", i, l[i], i, r[i]);
            return false;
        }
    return true;
}

int main(int argc, char **argv)
{

    // generate a random matrix.
    size_t N = MATRIX_SIZE;
    unsigned int *mat = (unsigned int *)malloc(sizeof(int) * N * N);
    GenMatrix(mat, N);

    // compute the reference result.
    unsigned int *ref = (unsigned int *)malloc(sizeof(int) * N * N);
    memcpy(ref, mat, sizeof(int) * N * N);

    // ---------------- primary module test ----------------- 
    /*
    unsigned int *ref_primary = (unsigned int *)malloc(sizeof(int) * 3 * 3);
    ref_primary[0] = mat[0];
    ref_primary[1] = mat[1];
    ref_primary[2] = mat[N];
    ref_primary[3] = mat[N + 1];
    Floyd_sequential(ref_primary, 2);
    showMatrix(ref_primary, 2);
    */
    // ------------------------------------------------------

    //printf("mat\n");
    //showMatrix(mat, N);

    // ------------- sequential ---------------
    
    double time1 = timestamp();
    Floyd_sequential(ref, N);
    double time2 = timestamp();

    // ----------------------------------------

    //printf("ref\n");
    //showMatrix(ref, N);

    //CUDA Portion
    unsigned int *result = (unsigned int *)malloc(sizeof(int) * N * N);
    memcpy(result, mat, sizeof(int) * N * N);
    unsigned int *d_result;
    // compute your results

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, sizeof(int) * N * N));

    CUDA_SAFE_CALL(cudaMemcpy(d_result, result, sizeof(int) * N * N, cudaMemcpyHostToDevice));

    double time3 = timestamp();
    int block_num = N / TILE_SIZE;
    printf("block_num: %d\n", block_num);
    int k = 0;
    double p_cost = 0, s_cost = 0, o_cost = 0;

    for (; k < block_num; k += 1)
    {
        //if(k)break;
        // primary modules
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(1,1);
        int start = k * TILE_SIZE;
        double time_p = timestamp();
        gpu_primary<<<grid, block>>>(d_result, N, start);
        double time_p_done = timestamp();
        p_cost += time_p_done - time_p;
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            printf("Cuda error in file '%s' in line %i : %s.\n",
                    __FILE__, __LINE__, cudaGetErrorString(err));
        }
        
        // phase2 modules
        dim3 block2(TILE_SIZE, TILE_SIZE);
        dim3 grid2(block_num - 1, 2);
        double time_s = timestamp();
        gpu_phase2<<<grid2, block2>>>(d_result, N, start, k);
        double time_s_done = timestamp();
        s_cost += time_s_done - time_s;
        cudaError_t err2 = cudaGetLastError();
        if (cudaSuccess != err2)
        {
            printf("Cuda error in file '%s' in line %i : %s.\n",
                    __FILE__, __LINE__, cudaGetErrorString(err2));
        }

        //phase3 modules
        dim3 block3(TILE_SIZE, TILE_SIZE);
        dim3 grid3(block_num - 1, block_num - 1);
        double time_o = timestamp();
        gpu_phase3<<<grid3, block3>>>(d_result, N, start, k);
        double time_o_done = timestamp();
        o_cost += time_o_done - time_o;
        cudaError_t err3 = cudaGetLastError();
        if (cudaSuccess != err3)
        {
            printf("Cuda error in file '%s' in line %i : %s.\n",
                    __FILE__, __LINE__, cudaGetErrorString(err3));
        }
    }
    double time4 = timestamp();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, sizeof(int) * N * N, cudaMemcpyDeviceToHost));

    //printf("result\n");
    //showMatrix(result, N);
    printf("p_cost: %f, s_cost: %f, o_cost: %f\n", p_cost, s_cost, o_cost);
    printf("cuda compute time use: %f\n", p_cost + s_cost + o_cost);

    printf("sequential time use: %f, cuda time use: %f\nspeedup-rate: %f\n", time2 - time1, time4 - time3, (time2 - time1) / (time4 - time3));

    // compare cuda result with reference result
    if (CmpArray(result, ref, N * N))
        printf("The matrix matches.\n");
    else
        printf("The matrix do not match.\n");

    free(ref);
    free(mat);
    free(result);
    cudaFree(d_result);
}
