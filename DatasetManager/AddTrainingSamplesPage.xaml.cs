// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
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
            UpdateFilteredSamples(Repository.ProposedTrainingSamples);
        }

        private void SamplesGrid_RejectRequested(object? sender, SampleRecord sample)
        {
            Repository.RejectTrainingSample(sample);
            UpdateFilteredSamples(Repository.ProposedTrainingSamples);
        }

        public AddTrainingSamplesPage(DatasetRepository repository, string category)
            : base(repository, repository.ProposedTrainingSamples, category)
        {
            InitializeComponent();
        }
    }
}
