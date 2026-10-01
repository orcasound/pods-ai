// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TestingSamplesPage.xaml
    /// </summary>
    public partial class TestingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;

        public TestingSamplesPage(List<SampleRecord> samples, string category)
            : base(samples, category)
        {
            InitializeComponent();
            SamplesGridControl.InferenceRequested += SamplesGridControl_InferenceRequested;
            DataContext = this;
        }
    }
}
