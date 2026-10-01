// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TrainingSamplesPage.xaml
    /// </summary>
    public partial class TrainingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;

        public TrainingSamplesPage(List<SampleRecord> samples, string category)
            : base(samples, category)
        {
            InitializeComponent();
            SamplesGridControl.InferenceRequested += SamplesGridControl_InferenceRequested;
            DataContext = this;
        }
    }
}
