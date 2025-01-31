﻿namespace GPUAV
{
    partial class MainView
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainView));
			pictureBox_waveform = new PictureBox();
			numericUpDown_id = new NumericUpDown();
			numericUpDown_offset = new NumericUpDown();
			numericUpDown_zoom = new NumericUpDown();
			comboBox_devices = new ComboBox();
			progressBar_vram = new ProgressBar();
			label_vram = new Label();
			button_toCuda = new Button();
			label_pointer = new Label();
			button_playStop = new Button();
			groupBox_fft = new GroupBox();
			numericUpDown_factor = new NumericUpDown();
			button_stretch = new Button();
			((System.ComponentModel.ISupportInitialize) pictureBox_waveform).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_id).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_offset).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_zoom).BeginInit();
			groupBox_fft.SuspendLayout();
			((System.ComponentModel.ISupportInitialize) numericUpDown_factor).BeginInit();
			SuspendLayout();
			// 
			// pictureBox_waveform
			// 
			pictureBox_waveform.BackColor = Color.White;
			pictureBox_waveform.InitialImage = (Image) resources.GetObject("pictureBox_waveform.InitialImage");
			pictureBox_waveform.Location = new Point(12, 520);
			pictureBox_waveform.Name = "pictureBox_waveform";
			pictureBox_waveform.Size = new Size(680, 120);
			pictureBox_waveform.TabIndex = 0;
			pictureBox_waveform.TabStop = false;
			// 
			// numericUpDown_id
			// 
			numericUpDown_id.Location = new Point(12, 646);
			numericUpDown_id.Maximum = new decimal(new int[] { 0, 0, 0, 0 });
			numericUpDown_id.Name = "numericUpDown_id";
			numericUpDown_id.Size = new Size(40, 23);
			numericUpDown_id.TabIndex = 1;
			numericUpDown_id.ValueChanged += numericUpDown_id_ValueChanged;
			// 
			// numericUpDown_offset
			// 
			numericUpDown_offset.Location = new Point(592, 646);
			numericUpDown_offset.Maximum = new decimal(new int[] { 0, 0, 0, 0 });
			numericUpDown_offset.Name = "numericUpDown_offset";
			numericUpDown_offset.Size = new Size(100, 23);
			numericUpDown_offset.TabIndex = 2;
			numericUpDown_offset.ValueChanged += numericUpDown_offset_ValueChanged;
			// 
			// numericUpDown_zoom
			// 
			numericUpDown_zoom.Location = new Point(632, 491);
			numericUpDown_zoom.Maximum = new decimal(new int[] { 99999, 0, 0, 0 });
			numericUpDown_zoom.Minimum = new decimal(new int[] { 1, 0, 0, 0 });
			numericUpDown_zoom.Name = "numericUpDown_zoom";
			numericUpDown_zoom.Size = new Size(60, 23);
			numericUpDown_zoom.TabIndex = 3;
			numericUpDown_zoom.Value = new decimal(new int[] { 128, 0, 0, 0 });
			numericUpDown_zoom.ValueChanged += numericUpDown_zoom_ValueChanged;
			// 
			// comboBox_devices
			// 
			comboBox_devices.FormattingEnabled = true;
			comboBox_devices.Location = new Point(12, 12);
			comboBox_devices.Name = "comboBox_devices";
			comboBox_devices.Size = new Size(200, 23);
			comboBox_devices.TabIndex = 4;
			comboBox_devices.Text = "Select CUDA device to initialize";
			// 
			// progressBar_vram
			// 
			progressBar_vram.Location = new Point(12, 41);
			progressBar_vram.Name = "progressBar_vram";
			progressBar_vram.Size = new Size(200, 10);
			progressBar_vram.TabIndex = 5;
			// 
			// label_vram
			// 
			label_vram.AutoSize = true;
			label_vram.Location = new Point(12, 54);
			label_vram.Name = "label_vram";
			label_vram.Size = new Size(90, 15);
			label_vram.TabIndex = 6;
			label_vram.Text = "VRAM: 0 / 0 MB";
			// 
			// button_toCuda
			// 
			button_toCuda.Location = new Point(317, 489);
			button_toCuda.Name = "button_toCuda";
			button_toCuda.Size = new Size(75, 23);
			button_toCuda.TabIndex = 7;
			button_toCuda.Text = "to CUDA";
			button_toCuda.UseVisualStyleBackColor = true;
			button_toCuda.Click += button_toCuda_Click;
			// 
			// label_pointer
			// 
			label_pointer.AutoSize = true;
			label_pointer.Location = new Point(58, 646);
			label_pointer.Name = "label_pointer";
			label_pointer.Size = new Size(124, 15);
			label_pointer.TabIndex = 8;
			label_pointer.Text = "[0] (currently on Host)";
			// 
			// button_playStop
			// 
			button_playStop.Location = new Point(12, 491);
			button_playStop.Name = "button_playStop";
			button_playStop.Size = new Size(75, 23);
			button_playStop.TabIndex = 9;
			button_playStop.Text = "Play";
			button_playStop.UseVisualStyleBackColor = true;
			button_playStop.Click += button_playStop_Click;
			// 
			// groupBox_fft
			// 
			groupBox_fft.Controls.Add(button_stretch);
			groupBox_fft.Controls.Add(numericUpDown_factor);
			groupBox_fft.Location = new Point(492, 298);
			groupBox_fft.Name = "groupBox_fft";
			groupBox_fft.Size = new Size(200, 142);
			groupBox_fft.TabIndex = 10;
			groupBox_fft.TabStop = false;
			groupBox_fft.Text = "CUDA FFT";
			// 
			// numericUpDown_factor
			// 
			numericUpDown_factor.DecimalPlaces = 8;
			numericUpDown_factor.Increment = new decimal(new int[] { 1, 0, 0, 196608 });
			numericUpDown_factor.Location = new Point(114, 113);
			numericUpDown_factor.Maximum = new decimal(new int[] { 5, 0, 0, 0 });
			numericUpDown_factor.Minimum = new decimal(new int[] { 5, 0, 0, 196608 });
			numericUpDown_factor.Name = "numericUpDown_factor";
			numericUpDown_factor.Size = new Size(80, 23);
			numericUpDown_factor.TabIndex = 11;
			numericUpDown_factor.Value = new decimal(new int[] { 10, 0, 0, 65536 });
			// 
			// button_stretch
			// 
			button_stretch.Location = new Point(6, 113);
			button_stretch.Name = "button_stretch";
			button_stretch.Size = new Size(75, 23);
			button_stretch.TabIndex = 11;
			button_stretch.Text = "Stretch";
			button_stretch.UseVisualStyleBackColor = true;
			button_stretch.Click += button_stretch_Click;
			// 
			// MainView
			// 
			AutoScaleDimensions = new SizeF(7F, 15F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(704, 681);
			Controls.Add(groupBox_fft);
			Controls.Add(button_playStop);
			Controls.Add(label_pointer);
			Controls.Add(button_toCuda);
			Controls.Add(label_vram);
			Controls.Add(progressBar_vram);
			Controls.Add(comboBox_devices);
			Controls.Add(numericUpDown_zoom);
			Controls.Add(numericUpDown_offset);
			Controls.Add(numericUpDown_id);
			Controls.Add(pictureBox_waveform);
			MaximizeBox = false;
			MaximumSize = new Size(720, 720);
			MinimumSize = new Size(720, 720);
			Name = "MainView";
			Text = "GPU AUDIO VISUALIZER";
			((System.ComponentModel.ISupportInitialize) pictureBox_waveform).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_id).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_offset).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_zoom).EndInit();
			groupBox_fft.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize) numericUpDown_factor).EndInit();
			ResumeLayout(false);
			PerformLayout();
		}

		#endregion

		private PictureBox pictureBox_waveform;
		private NumericUpDown numericUpDown_id;
		private NumericUpDown numericUpDown_offset;
		private NumericUpDown numericUpDown_zoom;
		private ComboBox comboBox_devices;
		private ProgressBar progressBar_vram;
		private Label label_vram;
		private Button button_toCuda;
		private Label label_pointer;
		private Button button_playStop;
		private GroupBox groupBox_fft;
		private Button button_stretch;
		private NumericUpDown numericUpDown_factor;
	}
}
