// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

namespace DatasetManager
{
    /// <summary>
    /// Interaction logic for AddTrainingSamplesPage.xaml
    /// </summary>
    public partial class AddTrainingSamplesPage : SamplesPageBase
    {
        protected override LogViewerControl LogViewer => new LogViewerControl();
        protected override SamplesGridControl SamplesGrid => SamplesGridControl;
        protected override string WavFolderPath => @"output\training-wav";

        private void SamplesGrid_AcceptRequested(object? sender, SampleRecord sample)
        {
            Repository.AcceptTrainingSample(sample);
        }

        private void SamplesGrid_RejectRequested(object? sender, SampleRecord sample)
        {
            Repository.RejectTrainingSample(sample);
        }

        public AddTrainingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.ProposedTrainingSamples, category)
        {
            InitializeComponent();
        }
    }
}
