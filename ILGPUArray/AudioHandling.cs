using NAudio.Wave;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;

namespace GPUAV
{
	public class AudioHandling
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\


		public List<TrackObject> Tracks = [];




		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTOR ~~~~~ ~~~~~ ~~~~~ \\
		public AudioHandling()
		{
		}





		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public void AddTrack(string filepath)
		{
			// Abort if file not valid (.mp3, .wav, .flac)
			if (!filepath.EndsWith(".mp3") && !filepath.EndsWith(".wav") && !filepath.EndsWith(".flac"))
			{
				return;
			}

			Tracks.Add(new TrackObject(filepath));
		}




	}





	public class TrackObject
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Filepath;
		public string Name;

		public float[] Data;

		public int Samplerate;
		public int Bitdepth;
		public int Channels;

		public long Length;
		public double Duration;

		public WaveOutEvent Player;
		public RawSourceWaveStream? CurrentStream = null;

		public long Pointer = 0;



		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTOR ~~~~~ ~~~~~ ~~~~~ \\
		public TrackObject(string filepath)
		{
			// New Player
			Player = new WaveOutEvent();

			// Set Filepath & Name
			Filepath = filepath;
			Name = Path.GetFileNameWithoutExtension(filepath);

			// Create reader
			AudioFileReader reader = new(filepath);

			// Set attributes
			Samplerate = reader.WaveFormat.SampleRate;
			Bitdepth = reader.WaveFormat.BitsPerSample;
			Channels = reader.WaveFormat.Channels;
			Length = reader.Length;
			Duration = reader.TotalTime.TotalSeconds;

			// Read data
			Data = new float[Length];
			reader.Read(Data, 0, (int) Length);

			// Close reader
			reader.Close();
			reader.Dispose();
		}



		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public byte[] GetBytes()
		{
			int bytesPerSample = Bitdepth / 8;
			byte[] bytes = new byte[Data.Length * bytesPerSample];

			for (int i = 0; i < Data.Length; i++)
			{
				byte[] byteArray;
				float sample = Data[i];

				switch (Bitdepth)
				{
					case 16:
						short shortSample = (short) (sample * short.MaxValue);
						byteArray = BitConverter.GetBytes(shortSample);
						break;
					case 24:
						int intSample24 = (int) (sample * (1 << 23));
						byteArray = new byte[3];
						byteArray[0] = (byte) (intSample24 & 0xFF);
						byteArray[1] = (byte) ((intSample24 >> 8) & 0xFF);
						byteArray[2] = (byte) ((intSample24 >> 16) & 0xFF);
						break;
					case 32:
						int intSample32 = (int) (sample * int.MaxValue);
						byteArray = BitConverter.GetBytes(intSample32);
						break;
					default:
						throw new ArgumentException("Unsupported bit depth");
				}

				Buffer.BlockCopy(byteArray, 0, bytes, i * bytesPerSample, bytesPerSample);
			}

			return bytes;
		}

		public Bitmap DrawWaveformSmooth(PictureBox wavebox, long offset = 0, int samplesPerPixel = 1)
		{
			// Überprüfen, ob floats und die PictureBox gültig sind
			if (Data.Length == 0 || wavebox.Width <= 0 || wavebox.Height <= 0)
			{
				return new Bitmap(1, 1);
			}

			// Farben aus der PictureBox übernehmen
			Color waveformColor = Color.OrangeRed;
			Color backgroundColor = Color.White;

			Bitmap bmp = new Bitmap(wavebox.Width, wavebox.Height);
			using Graphics gfx = Graphics.FromImage(bmp);
			using Pen pen = new Pen(waveformColor);
			gfx.SmoothingMode = SmoothingMode.AntiAlias;
			gfx.Clear(backgroundColor);

			float centerY = wavebox.Height / 2f;
			float yScale = wavebox.Height / 2f;

			for (int x = 0; x < wavebox.Width; x++)
			{
				long sampleIndex = offset + (long) x * samplesPerPixel;

				if (sampleIndex >= Data.Length)
				{
					break;
				}

				float maxValue = float.MinValue;
				float minValue = float.MaxValue;

				for (int i = 0; i < samplesPerPixel; i++)
				{
					if (sampleIndex + i < Data.Length)
					{
						maxValue = Math.Max(maxValue, Data[sampleIndex + i]);
						minValue = Math.Min(minValue, Data[sampleIndex + i]);
					}
				}

				float yMax = centerY - maxValue * yScale;
				float yMin = centerY - minValue * yScale;

				// Überprüfen, ob die Werte innerhalb des sichtbaren Bereichs liegen
				if (yMax < 0) yMax = 0;
				if (yMin > wavebox.Height) yMin = wavebox.Height;

				// Zeichne die Linie nur, wenn sie sichtbar ist
				if (Math.Abs(yMax - yMin) > 0.01f)
				{
					gfx.DrawLine(pen, x, yMax, x, yMin);
				}
				else if (samplesPerPixel == 1)
				{
					// Zeichne einen Punkt, wenn samplesPerPixel 1 ist und die Linie zu klein ist
					gfx.DrawLine(pen, x, centerY, x, centerY - Data[sampleIndex] * yScale);
				}
			}

			return bmp;
		}

		public void Play()
		{
			// Erst stoppen, falls schon am Laufen
			if (Player.PlaybackState == PlaybackState.Playing)
			{
				Player.Stop();
			}

			WaveFormat waveFormat = new(Samplerate, Bitdepth, Channels);
			MemoryStream memoryStream = new(GetBytes());

			// Hier den RawStream in CurrentStream speichern
			CurrentStream = new RawSourceWaveStream(memoryStream, waveFormat);
			Player.Init(CurrentStream);

			Player.Play();
		}

		public void Stop()
		{
			Player.Stop();
			Player.Dispose();
			Player = new WaveOutEvent();
		}

		public long GetCurrentSamplePosition()
		{
			if (CurrentStream == null)
			{
				return 0;
			}

			if (Player.PlaybackState == PlaybackState.Playing)
			{
				long current = Player.GetPosition() / (Bitdepth / 8 * Channels);
				return current;
			}
			else
			{
				return 0;
			}
		}

	}
}