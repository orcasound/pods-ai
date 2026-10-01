// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for SamplesGridControl.xaml
    /// </summary>
    public partial class SamplesGridControl : UserControl
    {
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

        public SamplesGridControl()
        {
            InitializeComponent();
        }
    }
}
