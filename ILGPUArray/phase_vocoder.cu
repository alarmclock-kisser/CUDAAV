extern "C" __global__ void PhaseVocoder(float2* spectrum, float2* stretchedSpectrum, int numFrames, int newNumFrames, int fftSize, float stretchFactor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= newNumFrames * fftSize) return;

    int oldIndex = (int)(tid / stretchFactor);
    if (oldIndex >= numFrames * fftSize) return;

    stretchedSpectrum[tid].x = spectrum[oldIndex].x;
    stretchedSpectrum[tid].y = spectrum[oldIndex].y;
}
