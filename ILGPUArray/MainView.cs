namespace GPUAV
{
	public partial class MainView : Form
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Repopath;

		public AudioHandling AudioH;
		public CudaHandling CudaH;




		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTOR ~~~~~ ~~~~~ ~~~~~ \\
		public MainView()
		{
			InitializeComponent();

			// Get Repopath
			Repopath = GetRepopath(true);

			// Window position & initial image
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);
			pictureBox_waveform.Image = Image.FromFile(Repopath + @"Resources\importInfo.bmp");

			// Init. classes
			AudioH = new AudioHandling();
			CudaH = new CudaHandling(progressBar_vram, label_vram, comboBox_devices);

			// Register events
			pictureBox_waveform.Click += ImportAudios;
		}







		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public string GetRepopath(bool root = false)
		{
			string repo = AppDomain.CurrentDomain.BaseDirectory;

			if (root)
			{
				repo += @"..\..\..\";
			}

			repo = Path.GetFullPath(repo);

			return repo;
		}

		public void UpdateTrackView()
		{
			// Set max id & offset
			numericUpDown_id.Maximum = AudioH.Tracks.Count;

			// Clear waveform
			pictureBox_waveform.Image = null;

			// If no tracks
			if (AudioH.Tracks.Count == 0 || numericUpDown_id.Value == 0)
			{
				label_pointer.Text = @"No track selected";
				pictureBox_waveform.Image = Image.FromFile(Repopath + @"Resources\importInfo.bmp");
				return;
			}

			// Set maximum offset
			numericUpDown_offset.Maximum = AudioH.Tracks[(int) numericUpDown_id.Value - 1].Length;

			// Set buton toCuda "to Host"
			if (AudioH.Tracks[(int) numericUpDown_id.Value - 1].Data.Length == 0)
			{
				button_toCuda.Text = "to Host";
			}
			else
			{
				button_toCuda.Text = "to CUDA";
			}

			// Update label pointer
			label_pointer.Text = $"[{AudioH.Tracks[(int) numericUpDown_id.Value - 1].Pointer}]";
			if (label_pointer.Text == "[0]")
			{
				label_pointer.Text = "[0] - currently on Host";
			}
			else
			{
				label_pointer.Text = $"[{AudioH.Tracks[(int) numericUpDown_id.Value - 1].Pointer}] - pointer on CUDA";
			}

			// Draw waveform
			pictureBox_waveform.Image = AudioH.Tracks[(int) numericUpDown_id.Value - 1].DrawWaveformSmooth(pictureBox_waveform, (long) numericUpDown_offset.Value, (int) numericUpDown_zoom.Value);
		}




		// ~~~~~ ~~~~~ ~~~~~ EVENTS ~~~~~ ~~~~~ ~~~~~ \\
		private void ImportAudios(object? sender, EventArgs e)
		{
			string initial = string.Empty;
			// If SHIFT down, open file dialog at Repopath
			if (Control.ModifierKeys == Keys.Shift)
			{
				initial = Repopath + @"Resources\";
			}
			else if (Control.ModifierKeys == Keys.Control)
			{
				initial = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);
			}
			else
			{
				return;
			}

			// Open file dialog at initial, audio multi-select
			OpenFileDialog ofd = new();
			ofd.Title = "Select audio files";
			ofd.Filter = "Audio files|*.wav;*.mp3;*.flac";
			ofd.Multiselect = true;
			ofd.InitialDirectory = initial;

			// If files selected (OK)
			if (ofd.ShowDialog() == DialogResult.OK)
			{
				// For each file
				foreach (string file in ofd.FileNames)
				{
					// Add track with AudioH
					AudioH.AddTrack(file);
				}
			}

			// Update track view
			UpdateTrackView();

			// Select first track
			numericUpDown_id.Value = 1;
		}

		private void numericUpDown_offset_ValueChanged(object sender, EventArgs e)
		{
			UpdateTrackView();
		}

		private void numericUpDown_id_ValueChanged(object sender, EventArgs e)
		{
			UpdateTrackView();
		}

		private void numericUpDown_zoom_ValueChanged(object sender, EventArgs e)
		{
			UpdateTrackView();
		}

		private void button_toCuda_Click(object sender, EventArgs e)
		{
			// Abort if no Ctx
			if (CudaH.Ctx == null)
			{
				return;
			}

			// Get index
			int id = (int) numericUpDown_id.Value - 1;

			// If no Data, GetFromCuda, Pointer to 0
			if (AudioH.Tracks[id].Pointer > 0)
			{
				AudioH.Tracks[id].Data = CudaH.GetFromCuda(AudioH.Tracks[id]);
				AudioH.Tracks[id].Pointer = 0;
			}
			else
			{
				// If Data, SendToCuda, set Pointer
				AudioH.Tracks[id].Pointer = CudaH.SendToCuda(AudioH.Tracks[id]);
				AudioH.Tracks[id].Data = [];
			}

			// Update track view
			UpdateTrackView();
		}
	}
}
