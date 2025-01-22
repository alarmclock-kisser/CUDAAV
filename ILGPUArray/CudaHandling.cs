using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Reflection;

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

		internal float[] GetFromCuda(TrackObject track)
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

		internal long SendToCuda(TrackObject track)
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
	}
}
