// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using System.Windows;

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TrainingSamplesPage.xaml
    /// </summary>
    public partial class TrainingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\wav";

        public TrainingSamplesPage(List<SampleRecord> samples, string category)
            : base(samples, category)
        {
            InitializeComponent();
            DataContext = this;
        }

        protected async void MoreMispredictions_Click(object sender, RoutedEventArgs e)
        {
            await RunPythonAsync(@"src\process_testing_set_mispredictions.py", $"--category {Category}");
        }
    }
}
