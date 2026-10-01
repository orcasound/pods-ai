// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for TestingSamplesPage.xaml
    /// </summary>
    public partial class TestingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => LogViewerControl;
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\testing-wav";

        public TestingSamplesPage(List<SampleRecord> samples, string category)
            : base(samples, category)
        {
            InitializeComponent();
            DataContext = this;
        }
    }
}
