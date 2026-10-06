// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for AddTestingSamplesPage.xaml
    /// </summary>
    public partial class AddTestingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => new LogViewerControl();
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\testing-wav";

        private void SamplesGrid_AcceptRequested(object? sender, SampleRecord sample)
        {
            Repository.AcceptTestingSample(sample);
        }

        private void SamplesGrid_RejectRequested(object? sender, SampleRecord sample)
        {
            Repository.RejectTestingSample(sample);
        }

        public AddTestingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.ProposedTestingSamples, category)
        {
            InitializeComponent();
        }
    }
}
