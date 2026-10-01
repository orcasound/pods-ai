// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for SamplesGridControl.xaml
    /// </summary>
    public partial class SamplesGridControl : UserControl
    {
        public event EventHandler<SampleRecord>? InferenceRequested;

        private void ListenButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not Button button)
            {
                return;
            }

            if (button.DataContext is not SampleRecord sample)
            {
                return;
            }

            Process.Start(new ProcessStartInfo
            {
                FileName = sample.URI,
                UseShellExecute = true
            });
        }

        private void InferButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button &&
            button.DataContext is SampleRecord sample)
            {
                InferenceRequested?.Invoke(this, sample);
            }
        }

        public SamplesGridControl()
        {
            InitializeComponent();
        }
    }
}
