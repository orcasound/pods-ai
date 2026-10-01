// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
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
    }
}
