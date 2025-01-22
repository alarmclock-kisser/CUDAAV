using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Reflection;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;

namespace GPUAV
{
	public class CudaHandling
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public int DeviceId = -1;
		public PrimaryContext? Ctx = null;

		public ProgressBar BarUsage;
		public Label LabelUsage;
		public ComboBox ComboDevices;

		public const int FFT_SIZE = 2048;
		public const int HOP_SIZE = 512;



		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTOR ~~~~~ ~~~~~ ~~~~~ \\
		public CudaHandling(ProgressBar pbarVram, Label labelVram, ComboBox comboDevices)
		{
			BarUsage = pbarVram;
			LabelUsage = labelVram;
			ComboDevices = comboDevices;

			// Register events
			ComboDevices.SelectedIndexChanged += (sender, e) => Init(ComboDevices.SelectedIndex);

			// Fill combo devices
			FillComboDevices();
		}




		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public string[] GetDeviceNames()
		{
			string[] names = new string[CudaContext.GetDeviceCount()];

			for (int i = 0; i < CudaContext.GetDeviceCount(); i++)
			{
				names[i] = CudaContext.GetDeviceName(i);
			}

			return names;
		}

		public void FillComboDevices()
		{
			ComboDevices.Items.Clear();
			ComboDevices.Items.AddRange(GetDeviceNames());
			ComboDevices.Items.Add("- No device -");
		}

		public int GetStrongestDeviceId()
		{
			int id = -1;
			int max = 0;

			for (int i = 0; i < CudaContext.GetDeviceCount(); i++)
			{
				int cc = CudaContext.GetDeviceComputeCapability(i).Major * 10 + CudaContext.GetDeviceComputeCapability(i).Minor;

				if (cc > max)
				{
					max = cc;
					id = i;
				}
			}

			return id;
		}

		public void Init(int id = -2)
		{
			// If id is out of bounds, no Ctx
			if (id >= CudaContext.GetDeviceCount())
			{
				Ctx?.Dispose();
				Ctx = null;
				UpdateVramUi();
				return;
			}

			// Get strongest if id is -2
			if (id == -2)
			{
				id = GetStrongestDeviceId();
			}

			DeviceId = id;

			Ctx = new PrimaryContext(id);
			Ctx.SetCurrent();

			UpdateVramUi();
		}

		public void UpdateVramUi()
		{
			// Set progressbar max & value
			BarUsage.Maximum = (int) GetVramTotal(true);
			BarUsage.Value = (int) (GetVramTotal(true) - GetVramUsed(true));

			// Set label text
			LabelUsage.Text = $"VRAM: {GetVramTotal(true) - GetVramUsed(true)} / {GetVramTotal(true)} MB";
		}

		public long GetVramTotal(bool readable = false)
		{
			long total = Ctx?.GetTotalDeviceMemorySize() ?? 0;
			if (readable)
			{
				total = total / 1024 / 1024;
			}
			return total;
		}

		public long GetVramUsed(bool readable = false)
		{
			long used = Ctx?.GetFreeDeviceMemorySize() ?? 0;
			if (readable)
			{
				used = used / 1024 / 1024;
			}
			return used;
		}

		public float[] GetFromCuda(TrackObject track)
		{
			// If no Ctx or Pointer, return empty array
			if (Ctx == null || track.Pointer < 1)
			{
				return [];
			}

			// Get data from Cuda (create Pointer from long)
			CUdeviceptr ptr = new(track.Pointer);
			float[] data = new float[track.Length];
			Ctx.CopyToHost(data, ptr);

			// Free memory
			Ctx.FreeMemory(ptr);

			// Update UI & return data
			UpdateVramUi();
			return data;
		}

		public long SendToCuda(TrackObject track)
		{
			// If no Ctx or Data, return 0
			if (Ctx == null || track.Data.Length == 0)
			{
				return 0;
			}

			// Allocate Pointer
			CUdeviceptr ptr = Ctx.AllocateMemory(track.Length * sizeof(float));

			// Copy data to Cuda
			Ctx.CopyToDevice(ptr, track.Data);

			// Update UI & return Pointer
			UpdateVramUi();
			return ptr.Pointer;
		}

		public long StretchFftOnCuda(long pointer = 0, float factor = 1.0f)
		{
			// If no Ctx, return 0
			if (Ctx == null || pointer < 1)
			{
				return 0;
			}

			// Create Pointer from long
			CUdeviceptr ptr = new(pointer);

			// Perform SFFT on Cuda forward
			CudaFFTPlan1D plan = new(1024, cufftType.R2C, 1);

			plan.SetAutoAllocation(true);


			plan.Exec(ptr, ptr);

			// Return Pointer
			return ptr.Pointer;
		}

		public float[] StretchSfft(float[] input, float factor = 1.0f)
		{
			// Abort if Ctx is null or input is empty
			if (Ctx == null || input.Length == 0)
			{
				return [];
			}

			int numFrames = input.Length / HOP_SIZE;
			int newNumFrames = (int) (numFrames * factor);

			// Allocate device memory
			CudaDeviceVariable<float> d_input = new(input.Length);
			CudaDeviceVariable<float2> d_spectrum = new(numFrames * FFT_SIZE);
			CudaDeviceVariable<float2> d_stretchedSpectrum = new(newNumFrames * FFT_SIZE);
			CudaDeviceVariable<float> d_output = new(newNumFrames * HOP_SIZE);

			// Copy data to GPU
			d_input.CopyToDevice(input);

			// Create cuFFT plan
			CudaFFTPlan1D fftPlan = new(FFT_SIZE, cufftType.R2C, numFrames);
			CudaFFTPlan1D ifftPlan = new(FFT_SIZE, cufftType.C2R, newNumFrames);

			// Launch FFT kernel
			fftPlan.Exec(d_input.DevicePointer, d_spectrum.DevicePointer, TransformDirection.Forward);

			// Phase Vocoder Kernel
			ApplyPhaseVocoderKernel(d_spectrum.DevicePointer, d_stretchedSpectrum.DevicePointer, numFrames, newNumFrames, FFT_SIZE, factor);

			// Inverse FFT
			ifftPlan.Exec(d_stretchedSpectrum.DevicePointer, d_output.DevicePointer, TransformDirection.Inverse);

			// Copy output back
			float[] output = new float[newNumFrames * HOP_SIZE];
			d_output.CopyToHost(output);

			// Free memory (cleanup)
			d_input.Dispose();
			d_spectrum.Dispose();
			d_stretchedSpectrum.Dispose();
			d_output.Dispose();

			// Return output
			return output;
		}

		private void ApplyPhaseVocoderKernel(CUdeviceptr d_spectrum, CUdeviceptr d_stretchedSpectrum, int numFrames, int newNumFrames, int fftSize, float stretchFactor)
		{
			// Abort if Ctx is null
			if (Ctx == null)
			{
				return;
			}

			// CUDA Kernel for Phase Vocoder (Implemented in a separate .cu file)
			string kernelSource = @"
        extern ""C"" __global__ void PhaseVocoder(float2* spectrum, float2* stretchedSpectrum, int numFrames, int newNumFrames, int fftSize, float stretchFactor) 
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= newNumFrames * fftSize) return;

            int oldIndex = (int)(tid / stretchFactor);
            if (oldIndex >= numFrames * fftSize) return;

            stretchedSpectrum[tid].x = spectrum[oldIndex].x;
            stretchedSpectrum[tid].y = spectrum[oldIndex].y;
        }";

			CudaKernel kernel = Ctx.LoadKernelPTX(kernelSource, "PhaseVocoder");
			kernel.BlockDimensions = new dim3(256, 1, 1);
			kernel.GridDimensions = new dim3((newNumFrames * fftSize + 255) / 256, 1, 1);
			kernel.Run(d_spectrum, d_stretchedSpectrum, numFrames, newNumFrames, fftSize, stretchFactor);
		}
	}
}
