#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <sys/mman.h>
using namespace std;

#define tdatap "../trainingData/dataBinaries"
#define NNdatap "../trainingData/NNbinaries"
#define TestDataPath "../trainingData/TestdataBinaries"

#define ReLUalpha 0.1
#define learningRate 0.001

#define fullSize 109386

extern "C" void* readF(const char* filename, int elemSize, long* nElWriteB);
extern "C" void writeF(void* d1, int size, int elSize, const char* filepath);

__global__ void finalActivation(float* A, float* C, int size, int tId);
__global__ void tailLaunchBackBiasAddition(float* A, float* B, float* C, int nLayers, int curLayer, int* layerSizes, int tId);

//A=weights, B=prevActivation, C=writeback
__global__ void tailLaunchBackMatMult(float* A , float* B, float* C, int nLayers, int curLayer, int* layerSizes, int tId){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    int AcBr = layerSizes[curLayer-1];
    int Arows = layerSizes[curLayer];
    int Bcols = 1;

    if(i < Arows && k < Bcols){
        C[i + k*Arows] = 0;
        for(int j = 0; j < AcBr; j++){
            C[i + k*Arows] += A[i + Arows*j] * B[AcBr*k + j];
        }
    }
    else{
        return;
    }
    __syncthreads();
    if(i + k == 0){
        A += AcBr*Arows;
        if(curLayer >= (nLayers-1)){
            finalActivation<<<1, Arows>>>(A, C, Arows, tId);
            return;
        }
        else{
            tailLaunchBackBiasAddition<<<1, layerSizes[curLayer]>>>(A, B, C, nLayers, curLayer, layerSizes, tId);
            return;
        }
    }
}

//A=biases, B=curActivation, C=writeback
__global__ void tailLaunchBackBiasAddition(float* A, float* B, float* C, int nLayers, int curLayer, int* layerSizes, int tId){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int tSize = layerSizes[curLayer];

    if(i < tSize){
        C[i] = A[i] + C[i];
        if(C[i] <= 0.0){
            C[i] *= ReLUalpha;       //leaky ReLU
        }
    }
    else{
        return;
    }
    __syncthreads();
    if(i == 0){
        C += tSize;
        A += tSize;
        curLayer++;
        dim3 threadspB(layerSizes[curLayer], 1);
        dim3 blockSize1(1, 1);
        tailLaunchBackMatMult<<<blockSize1, threadspB>>>(A, (C-tSize), C, nLayers, curLayer, layerSizes, tId);
        return;
    }
}

__global__ void finalActivation(float* A, float* C, int size, int tId){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    C[i] = A[i] + C[i];
    __syncthreads();
    double totalE = 0;
    int max = -9999999999;
    for(int k = 0; k < size; k++){
        if(C[k] > max){
            max = C[k];
        }
    }
    for(int k = 0; k < size; k++){
        totalE += exp(double(C[k] - max));
    }
    __syncthreads();
    C[i] = (exp(double(C[i] - max)))/(totalE);
    //printf("done: %f ; %d\n", C[i], tId);
    return;
}

__global__ void mlStartup(float* Ndat, float* tData, float* writeBack, int nLayers, int* layerN, int trainOffset, int trainInitialOffset, int totalTrain){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= totalTrain){
        return;
    }
    int wrtBOffset = 0;
    for(int k = 1; k < nLayers; k++){
        wrtBOffset += layerN[k];
    }
    tData += trainInitialOffset + (trainOffset*i);
    writeBack += wrtBOffset*i;
    tailLaunchBackMatMult<<<1, layerN[1]>>>(Ndat, tData, writeBack, nLayers, 1, layerN, i);
}

//hardcoded values
__global__ void gradientCalculator(float* NNdata,float* tData,float* returnData,float* writeBack,int* weightAmmount,int* biasAmmount,int* layerSizes,int totalSize,int tDataOffset){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= totalSize){
        return;
    }
    int f1rstL = weightAmmount[0]+biasAmmount[0];
    int s2ndL = f1rstL + weightAmmount[1]+biasAmmount[1];
    if(i < f1rstL){
        int t = i%layerSizes[1];
        int y = i/layerSizes[1];
        float a1d = 1;

        if(i < weightAmmount[0]){
            a1d = tData[y+tDataOffset];
            if(a1d <= 0){
                writeBack[i] = 0;
                return;
            }
            if(returnData[t] <= 0){
                a1d *= ReLUalpha;
            }
        }

        int wgBOffset0 = (weightAmmount[0] + biasAmmount[0]);
        int wgBOffset1 = wgBOffset0 + (weightAmmount[1] + biasAmmount[1]);
        //int wgBOffset2 = wgBOffset1 + (weightAmmount[2] + biasAmmount[2]);
        int tOffset = layerSizes[2]*t + wgBOffset0;
        int thLOffset = layerSizes[1];
        int totLOffset = thLOffset + layerSizes[2];

        float* zd = (float*)malloc(layerSizes[3]*sizeof(float));

        for(int k = 0; k < layerSizes[3]; k++){
            float finalOuterSum = 0;
            for(int j = 0; j < layerSizes[2]; j++){
                float w2d = NNdata[(j + tOffset)];
                float w3d = NNdata[k + j*layerSizes[3] + wgBOffset1];
                if(returnData[thLOffset] <= 0){
                    w3d *= ReLUalpha;
                }
                finalOuterSum += w3d*w2d;
            }
            zd[k] = finalOuterSum;
        }
        float realValuePD = zd[(int)tData[0]];
        float finalPartialDer = 0;
        for(int k = 0; k < layerSizes[3]; k++){
            finalPartialDer += returnData[totLOffset + k]*zd[k];
        }
        finalPartialDer -= realValuePD;
        writeBack[i] = finalPartialDer;
        free(zd);
        return;
    }
    else if(i < s2ndL){
        int j = (i-f1rstL)%layerSizes[2];
        int t = (i-f1rstL)/layerSizes[2];
        float a2td = 1;

        if(i < f1rstL+weightAmmount[1]){
            a2td = returnData[t];
            if(returnData[layerSizes[1] + j] <= 0){
                a2td *= ReLUalpha;
            }
        }

        float* zd = (float*)malloc(layerSizes[3]*sizeof(float));

        for(int k = 0; k < layerSizes[3]; k++){
            zd[k] = NNdata[k + j*layerSizes[3] + s2ndL]*a2td;
        }

        float realValuePD = zd[(int)tData[0]];
        float finalPartialDer = 0;
        for(int k = 0; k < layerSizes[3]; k++){
            finalPartialDer += returnData[layerSizes[1] + layerSizes[2] + k]*zd[k];
        }
        finalPartialDer -= realValuePD;
        writeBack[i] = finalPartialDer;
        free(zd);
        return;
    }
    else{
        int k = (i - s2ndL)%layerSizes[3];
        int j = (i - s2ndL)/layerSizes[3];

        float a2jd = 1;

        if(i < s2ndL+weightAmmount[2]){
            a2jd = returnData[layerSizes[1] + j];
        }

        if(k == (int)tData[0]){
            writeBack[i] = (returnData[layerSizes[1] + layerSizes[2] + k] - 1.0)*a2jd;
        }
        else{
            writeBack[i] = returnData[layerSizes[1] + layerSizes[2] + k]*a2jd;
        }
        return;
    }
}


float* derivationCPUSideLaunch(float* NNdata, float* tData, float* returnData, int* layerVals, int totalIterations, int tDataOffset, int tDatainitiatOffset){
    int wAM[3];
    int bAM[3];
    int totalNS = 0;
    int retSize = 0;
    for(int i = 0; i < 3; i++){
        wAM[i] = layerVals[i]*layerVals[i+1];
        bAM[i] = layerVals[i+1];
        totalNS += bAM[i] + wAM[i];
        retSize += bAM[i];
    }

    float* DeviceNND;
    float* DevicetD;
    float* DeviceRetV;
    int* DeviceLayerV;
    int* Device_wAM;
    int* Device_bAM;
    float* DeviceGradient;

    cudaMalloc(&DeviceNND, totalNS*sizeof(float));
    cudaMalloc(&DevicetD, totalIterations*tDataOffset*sizeof(float));
    cudaMalloc(&DeviceRetV, retSize*totalIterations*sizeof(float));
    cudaMalloc(&DeviceLayerV, 4*sizeof(int));
    cudaMalloc(&Device_wAM, 3*sizeof(int));
    cudaMalloc(&Device_bAM, 3*sizeof(int));
    cudaMalloc(&DeviceGradient, totalIterations*totalNS*sizeof(float));

    cudaMemcpy(DeviceNND, NNdata, totalNS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DevicetD, tData, totalIterations*tDataOffset*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceRetV, returnData, retSize*totalIterations*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceLayerV, layerVals, 4*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Device_wAM, wAM, 3*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Device_bAM, bAM, 3*sizeof(int), cudaMemcpyHostToDevice);

    float* baseDtD = DevicetD;
    float* baseDRetV = DeviceRetV;
    float* baseDGrad = DeviceGradient;

    for(int i = 0; i < totalIterations; i++){
        gradientCalculator<<<1024, 112>>>(DeviceNND, DevicetD, DeviceRetV, DeviceGradient, Device_wAM, Device_bAM, DeviceLayerV, totalNS, tDatainitiatOffset);
        DevicetD += tDataOffset;
        DeviceRetV += retSize;
        DeviceGradient += totalNS;
        //cudaDeviceSynchronize();
    }

    DeviceGradient = baseDGrad;

    float* hGradient = (float*)malloc(totalIterations*totalNS*sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpy(hGradient, DeviceGradient, totalIterations*totalNS*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(DeviceNND);
    cudaFree(baseDtD);
    cudaFree(baseDRetV);
    cudaFree(DeviceLayerV);
    cudaFree(Device_wAM);
    cudaFree(Device_bAM);
    cudaFree(DeviceGradient);

    for(int i = 1; i < totalIterations; i++){
        int shift = i*totalNS;
        for(int k = 0; k < totalNS; k++){
            hGradient[k] += hGradient[k + shift];
        }
    }

    float* retGradient = (float*)malloc(totalNS*sizeof(float));

    for(int i = 0; i < totalNS; i++){
        retGradient[i] = hGradient[i]/totalIterations;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(hGradient);
    return retGradient;
}


float* mlCPUSideLaunch(float* Ndat, float* tData, int nLayers, int* layerN, int trainOffset, int trainInitialOffset, int totalTrain){
    int totalNeuralOffset = 0;
    int totalReturnSize = 0;
    for(int i = 0; i < nLayers - 1; i++){
        totalNeuralOffset += (layerN[i]+1)*(layerN[i+1]);
        totalReturnSize += layerN[i+1];
    }
    int totalTrainSize = (trainOffset+trainInitialOffset)*totalTrain;
    totalReturnSize *= totalTrain;

    float* DeviceNND;
    float* DeviceRetVal;
    float* DevicetData;
    int* DeviceLayerVals;

    cudaMalloc(&DeviceNND, totalNeuralOffset*sizeof(float));
    cudaMalloc(&DeviceRetVal, totalReturnSize*sizeof(float));
    cudaMalloc(&DevicetData, totalTrainSize*sizeof(float));
    cudaMalloc(&DeviceLayerVals, nLayers*sizeof(int));

    cudaMemcpy(DeviceNND, Ndat, totalNeuralOffset*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DevicetData, tData, totalTrainSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DeviceLayerVals, layerN, nLayers*sizeof(int), cudaMemcpyHostToDevice);

    mlStartup<<<ceil(((float)totalTrain)/16.0), 16>>>(DeviceNND, DevicetData, DeviceRetVal, nLayers, DeviceLayerVals, trainOffset, trainInitialOffset, totalTrain);
    float* retVal = (float*)malloc(totalReturnSize*sizeof(float));
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaMemcpy(retVal, DeviceRetVal, totalReturnSize*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(DeviceNND);
    cudaFree(DeviceRetVal);
    cudaFree(DevicetData);
    cudaFree(DeviceLayerVals);

    return retVal;
}

float* costCalc(float* NNdata, float* tData, int nLayers, int* layerN, int trainOffset, int trainInitialOffset, int totalTrain){
    int rOff = 0;
    for(int i = 1; i < nLayers; i++){
        rOff += layerN[i];
    }
    int rStdOffset = rOff-layerN[nLayers-1];
    float* retVals = mlCPUSideLaunch(NNdata, tData, nLayers, layerN, trainOffset, trainInitialOffset, totalTrain);

    float cost = 0;
    float accuracy = 0;

    for(int i = 0; i < totalTrain; i++){
        cost -= log(retVals[rStdOffset + i*rOff + (int)tData[i*trainOffset]]);
        float max = 0.0;
        int mInd = 0;
        for(int k = 0; k < 10; k++){
            if(retVals[rStdOffset + i*rOff + k] > max){
                max = retVals[rStdOffset + i*rOff + k];
                mInd = k;
            }
        }
        if(mInd == (int)tData[i*trainOffset]){
            accuracy++;
        }
    }

    cost = cost/(float)totalTrain;
    accuracy = accuracy/(float)totalTrain;

    float* retArr = (float*)malloc(2*sizeof(float));
    retArr[0] = cost;
    retArr[1] = accuracy;
    return retArr;
}

void TrainDataInitialization(float* TrainData, int size, int tiOffset, int initialOffset, float maxValue){
    for(int i = 0; i < size; i++){
        for(int k = initialOffset; k < tiOffset; k++){
            TrainData[i*tiOffset + k] = TrainData[i*tiOffset + k]/maxValue;
        }
    }
    random_device rd1;
    mt19937 gen1(rd1());

    int* positions = (int*)malloc(size*sizeof(int));
    int* finalPositions = positions;
    for(int i = 0; i < size; i++){
        positions[i] = i;
    }
    for(int i = 0; i < size; i++){
        uniform_real_distribution<> dis1(0, ((size-i)-0.00000000001));
        int posPicked = (int)dis1(gen1) + i;
        int tempH = positions[i];
        positions[i] = positions[posPicked];
        positions[posPicked] = tempH;
    }
    float* FinalTrainData = (float*)malloc(tiOffset*size*sizeof(float));
    for(int i = 0; i < size; i++){
        memcpy((FinalTrainData+(tiOffset*i)),(TrainData+(tiOffset*finalPositions[i])), tiOffset*sizeof(float));
    }
    memcpy(TrainData, FinalTrainData, tiOffset*size*sizeof(float));
    free(FinalTrainData);
    free(finalPositions);
}

float* randomWiehgtHeGeneration(int* layerVals, int totalLayers){
    int fullsize = 0;
    for(int i = 1; i < totalLayers; i++){
        fullsize += (layerVals[i-1] + 1)*layerVals[i];
    }
    float* retVal = (float*)mmap(NULL, fullsize*sizeof(float), 0x3, 0x22, -1, 0);
    float* movR = retVal;
    random_device rd1;
    mt19937 gen1(rd1());
    for(int i = 1; i < totalLayers; i++){
        normal_distribution<> dis1(0, sqrt(2.0/(layerVals[i-1] + layerVals[i])));
        for(int k = 0; k < layerVals[i-1]*layerVals[i]; k++){
            movR[k] = dis1(gen1);
        }
        for(int k = layerVals[i-1]*layerVals[i]; k < (layerVals[i-1]+1)*layerVals[i]; k++){
            movR[k] = 0;
        }
        movR += (layerVals[i-1] + 1)*layerVals[i];
    }
    return retVal;
}

int main(int argc, char** argv){
    long nElems = 0;
    float* trainData;
    float* NNdata;
    float* testData;
    trainData = (float*)readF(tdatap, sizeof(float), &nElems);
    testData = (float*)readF(TestDataPath, sizeof(float), &nElems);
    NNdata = (float*)readF(NNdatap, sizeof(float), &nElems);
    int LayerVals[4] = {784, 128, 64, 10};
    //NNdata = randomWiehgtHeGeneration(LayerVals, 4);

    TrainDataInitialization(testData, 10000, 785, 1, 255);

    float cost = 100;
    float* movTrainData = trainData;
    int iterN = 0;
    int rott1 = 0;

    const float movAvg = 0.9;
    const float decay2 = 0.999;

    float aggregate[fullSize];
    for(int i = 0; i < fullSize; i++){
        aggregate[i] = 0;
    }
    
    float sqrSum[fullSize];
    for(int i = 0; i < fullSize; i++){
        sqrSum[i] = 0;
    }

    TrainDataInitialization(trainData, 60000, 785, 1, 255);
    float* accC = costCalc(NNdata, testData, 4, LayerVals, 785, 1, 5000);
    cost = accC[0];
    printf("cost=%f\naccuracy=%f\n", cost, accC[1]);
    free(accC);


    while(true){
        float* retD = mlCPUSideLaunch(NNdata, movTrainData, 4, LayerVals, 785, 1, 128);
        float* gradient = derivationCPUSideLaunch(NNdata, movTrainData, retD, LayerVals, 128, 785, 1);

        for(int i = 0; i < fullSize; i++){
            aggregate[i] = ((movAvg*aggregate[i]) + ((1.0 - movAvg)*gradient[i]));
            sqrSum[i] = ((decay2*sqrSum[i]) + ((1.0 - decay2)*pow(gradient[i], 2.0)));
            gradient[i] = (aggregate[i]/(1.0-(pow(movAvg, (float)(rott1+1)))))*(1.0/(sqrt(sqrSum[i]/(1.0-(pow(decay2, (float)(rott1+1))))) + 0.000000000001));
            NNdata[i] -= learningRate*gradient[i];
        }

        free(gradient);
        free(retD);

        movTrainData += 128*785;
        iterN++;
        if(iterN >= 468){   //468 bcs its 60000/128 rounded down
            TrainDataInitialization(trainData, 60000, 785, 1, 1);
            movTrainData = trainData;
            float* accC = costCalc(NNdata, testData, 4, LayerVals, 785, 1, 5000);
            cost = accC[0];
            printf("cost=%f\naccuracy=%f\n", cost, accC[1]);
            free(accC);

            writeF(NNdata, fullSize, sizeof(float), NNdatap);
            printf("epoch\n");
            iterN = 0;
        }
        rott1++;
        if(rott1 >= 50){
            printf("reset\n");
            for(int i = 0; i < fullSize; i++){
                aggregate[i] = 0;
            }
            for(int i = 0; i < fullSize; i++){
                sqrSum[i] = 0;
            }
            rott1 = 0;
        }
    }



}