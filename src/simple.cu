#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#define BLOCK_SIZE 8
#define MATRIX_SIZE 1000

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

void Floyd_count(unsigned int *ref, const size_t N, unsigned int *reference_cnt)
{
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                int i0 = i * N + j;
                int i1 = i * N + k;
                int i2 = k * N + j;
                if(k == 0) {
                    reference_cnt[i0] += 2;
                    reference_cnt[i1] += 1;
                    reference_cnt[i2] += 1;
                }

                //mat[i0] = (mat[i0] < mat[i1] + mat[i2]) ? mat[i0] : (mat[i1] + mat[i2]);
            }
}

__global__ void gpu_Floyd(unsigned int *result, int N, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("row: %d, col: %d\n", row, col);

    if (row < N && col < N)
    {
        int i0 = row * N + col;
        int i1 = row * N + k;
        int i2 = k * N + col;
        //printf("result[%d][%d]: %d, result[%d][%d]: %d, result[%d][%d]: %d\n", row, col,result[i0], row, k, result[i1],k, col, result[i2]);
        result[i0] = (result[i0] < result[i1] + result[i2]) ? result[i0] : (result[i1] + result[i2]);
    }
}

void GenMatrix(unsigned int *mat, const size_t N)
{
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

void showMatrix(unsigned int *mat, const size_t N)
{
    for (int i = 0; i < N * N; i++)
    {
        printf("mat[%d] = %d, ", i, mat[i]);
    }
    printf("done\n");
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
    unsigned int *reference_cnt = (unsigned int *)malloc(sizeof(int) * N * N);
    memset(reference_cnt, 0, sizeof(int) * N * N);
    GenMatrix(mat, N);

    // compute the reference result.
    unsigned int *ref = (unsigned int *)malloc(sizeof(int) * N * N);
    memcpy(ref, mat, sizeof(int) * N * N);

    //printf("mat\n");
    //showMatrix(mat, N);

    double time1 = timestamp();
    Floyd_sequential(ref, N);
    double time2 = timestamp();

    //Floyd_count(ref, N, reference_cnt);
    //showMatrix(reference_cnt, N);

    //printf("ref\n");
    //showMatrix(ref, N);

    //CUDA Portion
    unsigned int *result = (unsigned int *)malloc(sizeof(int) * N * N);
    memcpy(result, mat, sizeof(int) * N * N);
    unsigned int *d_result;
    // compute your results

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, sizeof(int) * N * N));

    CUDA_SAFE_CALL(cudaMemcpy(d_result, result, sizeof(int) * N * N, cudaMemcpyHostToDevice));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((size_t)ceil(((float)N) / ((float)block.x)), (size_t)ceil(((float)N) / ((float)block.y)));
    double time3 = timestamp();
    for (int k = 0; k < N; k++)
    {
        gpu_Floyd<<<grid, block>>>(d_result, N, k);
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
                    __FILE__, __LINE__, cudaGetErrorString(err));
        }
    }
    double time4 = timestamp();
    CUDA_SAFE_CALL(cudaMemcpy(result, d_result, sizeof(int) * N * N, cudaMemcpyDeviceToHost));

    //printf("result\n");
    //showMatrix(result, N);

    printf("sequential time use: %f, cuda time use: %f\nspeedup-rate: %f\n", time2 - time1, time4 - time3, (time2 - time1) / (time4 - time3));

    // compare your result with reference result
    if (CmpArray(result, ref, N * N))
        printf("The matrix matches.\n");
    else
        printf("The matrix do not match.\n");

    free(ref);
    free(mat);
    free(result);
    cudaFree(d_result);
}
